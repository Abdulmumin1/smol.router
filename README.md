# Tiny Capability Router

A small, inspectable classifier that sends a prompt to the cheapest model tier likely to answer it correctly. It has no runtime dependencies and does not call any model provider itself.

The classifier predicts the minimum required capability (`low`, `medium`, or `high`). A separate cost-aware policy selects a tier by minimizing:

```text
model cost + under-routing penalty × expected capability shortfall
```

Keeping prediction and policy separate means costs and risk tolerance can change without retraining.

## Quick start

```bash
make test
make train
PYTHONPATH=src python3 -m tiny_router route --model model.json \
  "Find the race condition and give a correctness proof"
```

Install the CLI if preferred:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
tiny-router train data/seed.jsonl --output model.json
tiny-router route --model model.json "Rewrite this email politely"
```

Serve it over HTTP:

```bash
tiny-router serve --model model.json --port 8080
curl -s http://127.0.0.1:8080/route \
  -H 'content-type: application/json' \
  -d '{"prompt":"Prove this algorithm is correct"}'
```

## Dataset

Training accepts JSONL with either a direct label:

```json
{"prompt":"Translate hello to French","label":"low"}
```

or benchmark scores. The loader labels each prompt with the cheapest tier meeting `--acceptable-score` (default `0.8`):

```json
{"prompt":"Find the concurrency bug","scores":{"low":0.2,"medium":0.6,"high":0.94}}
```

The included `data/seed.jsonl` only demonstrates the pipeline. A real router should be trained from representative production prompts scored by all candidate models using human review, task-specific tests, or a carefully validated judge. Split related prompts by task family to avoid evaluation leakage.

## Policy controls

- `--costs LOW MEDIUM HIGH`: relative price or latency of each tier.
- `--underroute-penalty`: damage assigned per missed capability level.
- `--confidence-threshold`: below this confidence, do not route beneath the fallback tier.
- `--uncertain-tier`: fallback for uncertain predictions.

Measure accuracy, under-routing rate, average selected cost, and the confusion matrix:

```bash
tiny-router evaluate data/held-out.jsonl --model model.json
```

The router returns a tier, class probabilities, expected costs, confidence, and the reason for the decision. Your application maps tiers to actual provider/model identifiers and can retry one tier higher when downstream validation fails.

## Fuzzing

`make test` runs deterministic randomized tests over arbitrary printable, Unicode, control-character, empty, and long prompts, as well as thousands of random probability distributions. Installing the test extra also enables Hypothesis to generate an additional Unicode corpus.

