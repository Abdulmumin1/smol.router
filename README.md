# Smol Capability Router

A dependency-free Python classifier and SDK that routes a prompt to the cheapest model tier likely to answer it correctly.

**[Documentation](https://router.ai-query.dev)** · **[Quickstart](https://router.ai-query.dev/quickstart/)** · **[Python SDK](https://router.ai-query.dev/sdk/)**

```text
prompt -> calibrated classifier -> cost/risk policy -> model target
```

The router predicts the minimum capable tier: `low`, `medium`, or `high`. It does not call model providers by itself; your app keeps control of credentials, retries, budgets, and observability.

## Quick Start

```bash
make test
make train
PYTHONPATH=src python3 -m smol_router init
PYTHONPATH=src python3 -m smol_router route \
  --model model.json --config router.json \
  "Find the race condition and justify the fix"
```

Install the CLI:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
smol-router --version
```

## Build Training Data

Do not label prompt difficulty by intuition. Benchmark your actual tiers, score each answer, and train from the cheapest tier that passes.

```bash
cp router.example.json router.json
cp examples/benchmark.config.example.json benchmark.config.json
cp examples/benchmark_prompts.example.jsonl benchmark_prompts.jsonl
```

Edit `router.json` with your `low`, `medium`, and `high` model IDs. Add prompts and rubrics to `benchmark_prompts.jsonl`, then run the OpenAI-compatible benchmark runner:

```bash
OPENAI_API_KEY=... PYTHONPATH=src python3 examples/benchmark_pipeline.py \
  --config benchmark.config.json

smol-router train data/benchmark.jsonl --output model.json --acceptable-score 0.8
```

See [Training](https://router.ai-query.dev/training/) for prompt JSONL format, judge config, and evaluation guidance.

## Python SDK

```python
from smol_router import Router

router = Router.from_files("model.json", "router.json")
result = router.route("Prove this concurrent queue is linearizable")

print(result.tier.label, result.model, result.decision.confidence)
```

To invoke a provider and optionally escalate failed responses:

```python
result = router.execute(
    prompt,
    invoke=lambda target, prompt: client.generate(model=target.model, prompt=prompt),
    validate=lambda answer: answer_passes_task_checks(answer),
)
```

## CLI

```bash
smol-router init [--output router.json]
smol-router train DATASET [--output model.json]
smol-router evaluate DATASET --model model.json
smol-router route --model model.json [--config router.json] [PROMPT]
smol-router serve --model model.json [--config router.json]
```

## Development

Runtime dependencies: none. Python 3.10+.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```
