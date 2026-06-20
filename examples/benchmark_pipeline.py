"""Runnable benchmark-data example using deterministic fake model outputs."""

from pathlib import Path

from tiny_router import BenchmarkTask, RouterConfig, run_benchmark, write_benchmark_jsonl


config = RouterConfig.load(Path(__file__).parents[1] / "router.example.json")
tasks = [
    BenchmarkTask("Extract the email", reference="low", group="extract-1"),
    BenchmarkTask("Rewrite the title", reference="low", group="rewrite-1"),
    BenchmarkTask("Implement and test an API", reference="medium", group="code-1"),
    BenchmarkTask("Compare two database designs", reference="medium", group="design-1"),
    BenchmarkTask("Prove the concurrent algorithm", reference="high", group="proof-1"),
    BenchmarkTask("Audit the cryptographic protocol", reference="high", group="security-1"),
]

tier_by_model = {
    config.target_for("low").model: 0,
    config.target_for("medium").model: 1,
    config.target_for("high").model: 2,
}
required = {"low": 0, "medium": 1, "high": 2}


def invoke(target, prompt):
    return {"tier": tier_by_model[target.model], "text": f"answer to: {prompt}"}


def judge(task, output):
    return 1.0 if output["tier"] >= required[task.reference] else 0.0


records = run_benchmark(tasks, config, invoke, judge)
output = Path(__file__).parents[1] / "data" / "example-benchmark.jsonl"
write_benchmark_jsonl(records, output)
print(f"wrote {len(records)} benchmark records to {output}")

