"""Run a real provider-backed benchmark and write training JSONL.

The runner uses an OpenAI-compatible chat completions API via the Python
standard library. Configure your tier models, prompts, and judge model in
``examples/benchmark.config.example.json``, then run:

    PYTHONPATH=src python3 examples/benchmark_pipeline.py --config benchmark.config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smol_router import (
    BenchmarkTask,
    ModelTarget,
    ProviderError,
    RouterConfig,
    run_benchmark,
    write_benchmark_jsonl,
)


@dataclass(frozen=True)
class ChatProvider:
    base_url: str
    api_key: str
    timeout_seconds: float = 60.0

    def chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"provider request failed: {exc}", retryable=True) from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"provider response did not contain chat content: {body!r}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("provider response content was empty")
        return content


def load_tasks(path: str | Path) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                prompt = record["prompt"]
                reference = record.get("reference")
                group = record.get("group")
                weight = record.get("weight", 1.0)
                metadata = record.get("metadata", {})
                tasks.append(
                    BenchmarkTask(
                        prompt=prompt,
                        reference=reference,
                        group=group,
                        weight=weight,
                        metadata=metadata,
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SystemExit(f"{source}:{line_number}: invalid benchmark prompt: {exc}") from exc
    if not tasks:
        raise SystemExit(f"{source}: prompt file is empty")
    return tasks


def load_provider(payload: dict[str, Any]) -> ChatProvider:
    provider = payload.get("provider")
    if not isinstance(provider, dict):
        raise SystemExit("benchmark config must contain provider")
    api_key_env = provider.get("api_key_env", "OPENAI_API_KEY")
    if not isinstance(api_key_env, str) or not api_key_env:
        raise SystemExit("provider.api_key_env must be a non-empty string")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"set {api_key_env} before running the benchmark")
    base_url = provider.get("base_url", "https://api.openai.com/v1")
    if not isinstance(base_url, str) or not base_url:
        raise SystemExit("provider.base_url must be a non-empty string")
    timeout = provider.get("timeout_seconds", 60)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise SystemExit("provider.timeout_seconds must be positive")
    return ChatProvider(base_url=base_url, api_key=api_key, timeout_seconds=float(timeout))


def config_path_for(payload: dict[str, Any], key: str, default: str, base: Path) -> Path:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{key} must be a non-empty string")
    return base / value


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise TypeError("judge JSON must be an object")
    return payload


def judge_score(provider: ChatProvider, judge_model: str, task: BenchmarkTask, answer: str) -> float:
    rubric = task.reference if task.reference is not None else task.metadata.get("rubric")
    if not isinstance(rubric, str) or not rubric.strip():
        raise ProviderError("each prompt needs reference or metadata.rubric for judge scoring")
    content = provider.chat(
        judge_model,
        [
            {
                "role": "system",
                "content": (
                    "Score the answer against the rubric. Return only compact JSON "
                    'with a numeric "score" field from 0 to 1.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Prompt:\n{task.prompt}\n\nRubric/reference:\n{rubric}\n\n"
                    f"Answer:\n{answer}\n\nJSON:"
                ),
            },
        ],
    )
    try:
        payload = parse_json_object(content)
        score = payload["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError
        return float(score)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ProviderError(f"judge did not return JSON with numeric score: {content!r}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/benchmark.config.example.json")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("benchmark config must be a JSON object")

    router_config = RouterConfig.load(
        config_path_for(payload, "router_config", "../router.example.json", config_path.parent)
    )
    prompt_path = config_path_for(payload, "prompts", "benchmark_prompts.example.jsonl", config_path.parent)
    output_path = config_path_for(payload, "output", "../data/benchmark.jsonl", config_path.parent)
    judge = payload.get("judge")
    if not isinstance(judge, dict) or not isinstance(judge.get("model"), str):
        raise SystemExit("benchmark config must contain judge.model")

    provider = load_provider(payload)
    system_prompt = payload.get("system_prompt", "Answer the user request accurately and concisely.")
    if not isinstance(system_prompt, str):
        raise SystemExit("system_prompt must be a string")

    def invoke(target: ModelTarget, prompt: str) -> str:
        return provider.chat(
            target.model,
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        )

    records = run_benchmark(
        load_tasks(prompt_path),
        router_config,
        invoke=invoke,
        judge=lambda task, answer: judge_score(provider, judge["model"], task, answer),
    )
    write_benchmark_jsonl(records, output_path)
    print(f"wrote {len(records)} benchmark records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
