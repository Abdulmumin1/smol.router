import unittest

from smol_router.errors import ProviderError, RouterError


class ErrorTests(unittest.TestCase):
    def test_provider_error_exposes_retryability(self) -> None:
        error = ProviderError("timeout", retryable=True)
        self.assertIsInstance(error, RouterError)
        self.assertTrue(error.retryable)


if __name__ == "__main__":
    unittest.main()
