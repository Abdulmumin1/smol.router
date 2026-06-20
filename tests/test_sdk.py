import unittest

from tiny_router import InvalidPromptError, Router, RouterConfig, RouterModel, Tier


def make_router() -> Router:
    config = RouterConfig.from_dict(
        {
            "models": {
                "low": "test/tiny",
                "medium": {"model": "test/regular", "provider": "test"},
                "high": "test/frontier",
            },
            "policy": {
                "underroute_penalty": 0,
                "confidence_threshold": 0,
                "high_probability_threshold": 1,
            },
        }
    )
    return Router(RouterModel.empty(32), config, max_prompt_chars=100)


class SdkTests(unittest.TestCase):
    def test_route_resolves_model_target(self) -> None:
        result = make_router().route("hello")
        self.assertEqual(result.tier, Tier.LOW)
        self.assertEqual(result.model, "test/tiny")
        self.assertEqual(result.to_dict()["model"], "test/tiny")

    def test_request_can_override_tier_floor(self) -> None:
        result = make_router().route("hello", minimum_tier="high")
        self.assertEqual(result.tier, Tier.HIGH)
        self.assertEqual(result.model, "test/frontier")

    def test_invalid_request_bounds_are_domain_errors(self) -> None:
        with self.assertRaisesRegex(InvalidPromptError, "tier bounds"):
            make_router().route("hello", minimum_tier="high", maximum_tier="low")

    def test_invalid_prompts_fail_before_feature_extraction(self) -> None:
        router = make_router()
        for prompt in ("", "  \n", "x" * 101, None):
            with self.subTest(prompt=prompt):
                with self.assertRaises(InvalidPromptError):
                    router.route(prompt)  # type: ignore[arg-type]

    def test_route_many_preserves_order(self) -> None:
        results = make_router().route_many(["one", "two", "three"])
        self.assertEqual([result.model for result in results], ["test/tiny"] * 3)
        with self.assertRaises(TypeError):
            make_router().route_many("not a batch")


if __name__ == "__main__":
    unittest.main()
