from __future__ import annotations

import inspect
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Iterable, Mapping

from .config import ModelTarget, RouterConfig
from .data import Example, parse_record
from .errors import DatasetError, ProviderError
from .types import Tier


@dataclass(frozen=True)
class BenchmarkTask:
    prompt: str
    reference: Any = None
    group: str | None = None
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Reuse the same prompt, group, and weight invariants as training records.
        Example(self.prompt, Tier.LOW, self.weight, self.group)
        if not isinstance(self.metadata, Mapping):
            raise DatasetError("benchmark metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class BenchmarkRecord:
    task: BenchmarkTask
    scores: tuple[float, float, float]
    latency_ms: tuple[float, float, float]
    errors: tuple[str | None, str | None, str | None]

    def __post_init__(self) -> None:
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in self.scores):
            raise DatasetError("benchmark scores must be finite and in [0, 1]")
        if any(not math.isfinite(value) or value < 0 for value in self.latency_ms):
            raise DatasetError("benchmark latency must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "prompt": self.task.prompt,
            "scores": {tier.label: self.scores[int(tier)] for tier in Tier},
            "latency_ms": {tier.label: self.latency_ms[int(tier)] for tier in Tier},
            "errors": {tier.label: self.errors[int(tier)] for tier in Tier},
            "weight": self.task.weight,
        }
        if self.task.group is not None:
            result["group"] = self.task.group
        if self.task.metadata:
            result["metadata"] = dict(self.task.metadata)
        return result

    def training_example(self, acceptable_score: float = 0.8) -> Example:
        return parse_record(self.to_dict(), acceptable_score)


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError("judge must return a numeric score")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise DatasetError("judge score must be finite and in [0, 1]")
    return result


def run_benchmark(
    tasks: Iterable[BenchmarkTask],
    config: RouterConfig,
    invoke: Callable[[ModelTarget, str], Any],
    judge: Callable[[BenchmarkTask, Any], float],
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for task in tasks:
        scores: list[float] = []
        latencies: list[float] = []
        errors: list[str | None] = []
        for tier in Tier:
            started = time.perf_counter()
            try:
                output = invoke(config.target_for(tier), task.prompt)
                scores.append(_score(judge(task, output)))
                errors.append(None)
            except ProviderError as exc:
                scores.append(0.0)
                errors.append(str(exc))
            latencies.append((time.perf_counter() - started) * 1000)
        records.append(BenchmarkRecord(task, tuple(scores), tuple(latencies), tuple(errors)))  # type: ignore[arg-type]
    return records


async def run_benchmark_async(
    tasks: Iterable[BenchmarkTask],
    config: RouterConfig,
    invoke: Callable[[ModelTarget, str], Awaitable[Any]],
    judge: Callable[[BenchmarkTask, Any], float | Awaitable[float]],
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for task in tasks:
        scores: list[float] = []
        latencies: list[float] = []
        errors: list[str | None] = []
        for tier in Tier:
            started = time.perf_counter()
            try:
                output = await invoke(config.target_for(tier), task.prompt)
                verdict = judge(task, output)
                scores.append(_score(await verdict if inspect.isawaitable(verdict) else verdict))
                errors.append(None)
            except ProviderError as exc:
                scores.append(0.0)
                errors.append(str(exc))
            latencies.append((time.perf_counter() - started) * 1000)
        records.append(BenchmarkRecord(task, tuple(scores), tuple(latencies), tuple(errors)))  # type: ignore[arg-type]
    return records


def write_benchmark_jsonl(records: Iterable[BenchmarkRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            for record in records:
                handle.write(json.dumps(record.to_dict(), separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)

