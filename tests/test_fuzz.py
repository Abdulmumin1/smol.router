from __future__ import annotations

import math
import random
import string
import unittest
import json

from tiny_router.features import extract_features
from tiny_router.model import RouterModel
from tiny_router.policy import RoutingPolicy
from tiny_router.types import Tier
from tiny_router.config import RouterConfig
from tiny_router.errors import ArtifactError, ConfigurationError


def random_text(rng: random.Random) -> str:
    alphabet = string.printable + "éλ中🙂\x00\n\t"
    return "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 4000)))


def random_json_value(rng: random.Random, depth: int = 0):
    scalars = [None, True, False, rng.randint(-1000, 1000), rng.random(), random_text(rng)[:20]]
    if depth >= 3:
        return rng.choice(scalars)
    choice = rng.randrange(3)
    if choice == 0:
        return rng.choice(scalars)
    if choice == 1:
        return [random_json_value(rng, depth + 1) for _ in range(rng.randrange(5))]
    return {f"key-{index}": random_json_value(rng, depth + 1) for index in range(rng.randrange(5))}


class FuzzTests(unittest.TestCase):
    def test_random_json_artifacts_fail_with_artifact_errors_only(self) -> None:
        rng = random.Random(712)
        for _ in range(2000):
            payload = json.loads(json.dumps(random_json_value(rng)))
            try:
                model = RouterModel.from_dict(payload)
                self.assertEqual(len(model.weights), 3)
            except ArtifactError:
                pass

    def test_random_json_configs_fail_with_configuration_errors_only(self) -> None:
        rng = random.Random(713)
        for _ in range(2000):
            payload = json.loads(json.dumps(random_json_value(rng)))
            try:
                config = RouterConfig.from_dict(payload)
                self.assertEqual(set(config.models), set(Tier))
            except ConfigurationError:
                pass

    def test_hostile_policy_types_fail_with_value_errors(self) -> None:
        invalid = (None, "1", True, float("nan"), float("inf"), [], {})
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RoutingPolicy(underroute_penalty=value)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    RoutingPolicy().expected_costs((value, 0.0, 1.0))  # type: ignore[arg-type]
    def test_random_prompts_never_break_probability_or_decision_invariants(self) -> None:
        rng = random.Random(20260620)
        model = RouterModel.empty(257)
        for _ in range(750):
            prompt = random_text(rng)
            probabilities = model.predict_proba(prompt)
            self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities))
            self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
            decision = RoutingPolicy().route(model, prompt)
            self.assertIn(decision.tier, Tier)

    def test_random_distributions_produce_finite_costs_and_valid_tiers(self) -> None:
        rng = random.Random(441)
        for _ in range(5000):
            raw = [rng.expovariate(1) for _ in range(3)]
            total = sum(raw)
            probabilities = tuple(value / total for value in raw)
            costs = RoutingPolicy(underroute_penalty=rng.random() * 1000).expected_costs(probabilities)  # type: ignore[arg-type]
            self.assertTrue(all(math.isfinite(value) and value >= 0 for value in costs))
            self.assertIn(Tier(min(range(3), key=costs.__getitem__)), Tier)

    def test_more_failure_penalty_never_selects_a_lower_tier(self) -> None:
        rng = random.Random(991)
        for _ in range(5000):
            raw = [rng.expovariate(1) for _ in range(3)]
            total = sum(raw)
            probabilities = tuple(value / total for value in raw)
            previous = Tier.LOW
            for penalty in (0, 1, 5, 10, 25, 100, 1000):
                costs = RoutingPolicy(underroute_penalty=penalty).expected_costs(probabilities)  # type: ignore[arg-type]
                selected = Tier(min(range(3), key=costs.__getitem__))
                self.assertGreaterEqual(selected, previous)
                previous = selected

    def test_feature_extraction_is_deterministic_and_bounded(self) -> None:
        rng = random.Random(818)
        for _ in range(500):
            prompt = random_text(rng)
            first = extract_features(prompt, 64)
            self.assertEqual(first, extract_features(prompt, 64))
            self.assertTrue(all(0 <= index < 64 for index in first))
            self.assertLessEqual(len(first), 64)


# Hypothesis extends the built-in fuzz corpus when installed via `pip install -e .[test]`.
try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ImportError:
    pass
else:
    class HypothesisFuzzTests(unittest.TestCase):
        @settings(max_examples=500, deadline=None)
        @given(st.text(max_size=20_000))
        def test_arbitrary_unicode_prompt(self, prompt: str) -> None:
            probabilities = RouterModel.empty(64).predict_proba(prompt)
            self.assertAlmostEqual(sum(probabilities), 1.0)
            self.assertTrue(all(math.isfinite(value) for value in probabilities))


if __name__ == "__main__":
    unittest.main()
