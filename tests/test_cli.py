import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tiny_router.cli import main
from tiny_router.model import RouterModel


class CliTests(unittest.TestCase):
    def test_init_route_and_inspect_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "router.json"
            model_path = root / "model.json"
            RouterModel.empty(32).save(model_path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["init", "--output", str(config_path)]), 0)
                self.assertEqual(main(["inspect", "--model", str(model_path)]), 0)
                self.assertEqual(
                    main([
                        "route", "--model", str(model_path), "--config", str(config_path),
                        "--minimum-tier", "high", "hello",
                    ]),
                    0,
                )
            self.assertTrue(config_path.exists())
            self.assertIn('"sha256"', output.getvalue())
            self.assertIn('"model": "provider/frontier"', output.getvalue())

    def test_domain_errors_return_two_without_traceback(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main(["route", "--model", "/definitely/missing", "hello"])
        self.assertEqual(status, 2)
        self.assertTrue(stderr.getvalue().startswith("error:"))
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
