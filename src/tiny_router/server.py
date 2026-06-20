from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .errors import RouterError
from .sdk import Router
from .types import Tier

MAX_BODY_BYTES = 1_000_000
MAX_BATCH_SIZE = 1_000


def create_server(router: Router, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "tiny-router"
        sys_version = ""

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/health":
                self._json(200, {"status": "ok"})
            elif path == "/models":
                self._json(
                    200,
                    {"models": {tier.label: router.config.target_for(tier).to_dict() for tier in Tier}},
                )
            else:
                self._error(404, "not_found", "endpoint not found")

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            try:
                payload = self._read_json()
                if path == "/route":
                    if not isinstance(payload, dict):
                        raise ValueError("request body must be an object")
                    prompt = payload.get("prompt")
                    result = router.route(
                        prompt,  # type: ignore[arg-type]
                        minimum_tier=payload.get("minimum_tier"),
                        maximum_tier=payload.get("maximum_tier"),
                    )
                    self._json(200, result.to_dict())
                elif path == "/route/batch":
                    if not isinstance(payload, dict) or not isinstance(payload.get("prompts"), list):
                        raise ValueError("prompts must be an array")
                    prompts = payload["prompts"]
                    if len(prompts) > MAX_BATCH_SIZE:
                        raise ValueError(f"batch exceeds maximum size {MAX_BATCH_SIZE}")
                    self._json(200, {"results": [result.to_dict() for result in router.route_many(prompts)]})
                else:
                    self._error(404, "not_found", "endpoint not found")
            except (RouterError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._error(400, "invalid_request", str(exc))

        def _read_json(self) -> Any:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("Content-Length is required")
            length = int(raw_length)
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError(f"request body must be between 0 and {MAX_BODY_BYTES} bytes")
            content_type = self.headers.get_content_type()
            if content_type != "application/json":
                raise ValueError("Content-Type must be application/json")
            return json.loads(self.rfile.read(length))

        def _error(self, status: int, code: str, message: str) -> None:
            self._json(status, {"error": {"code": code, "message": message}})

        def _json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)

