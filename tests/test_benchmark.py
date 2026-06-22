import tempfile
import unittest
from pathlib import Path

from smol_router import (
    BenchmarkTask,
    ProviderError,
    RouterConfig,
    Tier,
    run_benchmark,
    run_benchmark_async,
    write_benchmark_jsonl,
)
from smol_router.data import load_jsonl


def config():
    return RouterConfig.from_dict({"models": {"low": "low", "medium": "medium", "high": "high"}})


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_outputs_training_compatible_records(self) -> None:
        task = BenchmarkTask("solve", reference="correct", group="math-1")
        scores = {"low": 0.3, "medium": 0.85, "high": 1.0}
        records = run_benchmark([task], config(), lambda target, prompt: target.model, lambda task, output: scores[output])
        self.assertEqual(records[0].training_example().label, Tier.MEDIUM)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.jsonl"
            write_benchmark_jsonl(records, path)
            loaded = load_jsonl(path)
            self.assertEqual((loaded[0].label, loaded[0].group), (Tier.MEDIUM, "math-1"))

    def test_provider_failures_are_recorded_as_zero(self) -> None:
        def invoke(target, prompt):
            if target.model == "low":
                raise ProviderError("timeout", retryable=True)
            return "ok"

        record = run_benchmark([BenchmarkTask("solve")], config(), invoke, lambda task, output: 1.0)[0]
        self.assertEqual(record.scores[0], 0)
        self.assertEqual(record.errors[0], "timeout")

    def test_invalid_judge_score_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "judge"):
            run_benchmark([BenchmarkTask("solve")], config(), lambda target, prompt: "x", lambda task, output: float("nan"))


class AsyncBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_benchmark_accepts_async_judge(self) -> None:
        async def invoke(target, prompt):
            return target.model

        async def judge(task, output):
            return {"low": 0.1, "medium": 0.2, "high": 0.9}[output]

        records = await run_benchmark_async([BenchmarkTask("solve")], config(), invoke, judge)
        self.assertEqual(records[0].training_example().label, Tier.HIGH)


if __name__ == "__main__":
    unittest.main()
