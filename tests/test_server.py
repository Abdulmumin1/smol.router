import json
import threading
import unittest
import urllib.error
import urllib.request

from smol_router import Router, RouterConfig, RouterModel
from smol_router.server import create_server


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = RouterConfig.from_dict(
            {
                "models": {"low": "tiny", "medium": "regular", "high": "frontier"},
                "policy": {"underroute_penalty": 0, "confidence_threshold": 0, "high_probability_threshold": 1},
            }
        )
        cls.server = create_server(Router(RouterModel.empty(32), config), port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path: str, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        headers = {} if data is None else {"Content-Type": "application/json"}
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status, json.load(response)

    def test_health_and_models(self) -> None:
        self.assertEqual(self.request("/health"), (200, {"status": "ok"}))
        status, payload = self.request("/models")
        self.assertEqual(status, 200)
        self.assertEqual(payload["models"]["high"]["model"], "frontier")

    def test_single_and_batch_routing(self) -> None:
        status, payload = self.request("/route", {"prompt": "hello", "minimum_tier": "medium"})
        self.assertEqual((status, payload["tier"], payload["model"]), (200, "medium", "regular"))
        status, payload = self.request("/route/batch", {"prompts": ["one", "two"]})
        self.assertEqual([result["model"] for result in payload["results"]], ["tiny", "tiny"])

    def test_invalid_request_has_stable_error_shape(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.request("/route", {"prompt": 42})
        self.assertEqual(raised.exception.code, 400)
        payload = json.load(raised.exception)
        raised.exception.close()
        self.assertEqual(payload["error"]["code"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
