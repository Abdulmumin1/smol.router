from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .data import load_jsonl, split_examples
from .evaluate import evaluate
from .model import RouterModel
from .policy import RoutingPolicy
from .types import Tier


def _policy(args: argparse.Namespace) -> RoutingPolicy:
    return RoutingPolicy(
        tier_costs=tuple(args.costs),
        underroute_penalty=args.underroute_penalty,
        confidence_threshold=args.confidence_threshold,
        uncertain_tier=Tier.parse(args.uncertain_tier),
    )


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--costs", nargs=3, type=float, default=(1.0, 4.0, 15.0), metavar=("LOW", "MEDIUM", "HIGH"))
    parser.add_argument("--underroute-penalty", type=float, default=25.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--uncertain-tier", choices=("low", "medium", "high"), default="medium")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tiny-router", description="Train and run a tiny capability router")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train from a JSONL dataset")
    train.add_argument("dataset")
    train.add_argument("--output", default="model.json")
    train.add_argument("--dimensions", type=int, default=1024)
    train.add_argument("--epochs", type=int, default=35)
    train.add_argument("--learning-rate", type=float, default=0.35)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--acceptable-score", type=float, default=0.8)
    train.add_argument("--seed", type=int, default=17)
    _add_policy_arguments(train)

    route = subparsers.add_parser("route", help="route a prompt")
    route.add_argument("--model", default="model.json")
    route.add_argument("prompt", nargs="?")
    _add_policy_arguments(route)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a model")
    evaluate_parser.add_argument("dataset")
    evaluate_parser.add_argument("--model", default="model.json")
    evaluate_parser.add_argument("--acceptable-score", type=float, default=0.8)
    _add_policy_arguments(evaluate_parser)

    serve = subparsers.add_parser("serve", help="serve POST /route")
    serve.add_argument("--model", default="model.json")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    _add_policy_arguments(serve)
    return parser


def _serve(model: RouterModel, policy: RoutingPolicy, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/route":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1_000_000:
                    raise ValueError("request body too large")
                payload = json.loads(self.rfile.read(length))
                prompt = payload.get("prompt") if isinstance(payload, dict) else None
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be a string")
                body = json.dumps(policy.route(model, prompt).to_dict()).encode()
                self.send_response(200)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            print(format % args, file=sys.stderr)

    print(f"listening on http://{host}:{port}", file=sys.stderr)
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = _policy(args)
    if args.command == "train":
        examples = load_jsonl(args.dataset, args.acceptable_score)
        training, validation = split_examples(examples, args.validation_fraction, args.seed)
        model = RouterModel.train(
            training,
            dimensions=args.dimensions,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        model.save(args.output)
        result: dict[str, object] = {"model": str(Path(args.output)), "training_examples": len(training)}
        if validation:
            result["validation"] = evaluate(model, validation, policy).to_dict()
        print(json.dumps(result, indent=2))
    elif args.command == "route":
        prompt = args.prompt if args.prompt is not None else sys.stdin.read()
        print(json.dumps(policy.route(RouterModel.load(args.model), prompt).to_dict(), indent=2))
    elif args.command == "evaluate":
        metrics = evaluate(RouterModel.load(args.model), load_jsonl(args.dataset, args.acceptable_score), policy)
        print(json.dumps(metrics.to_dict(), indent=2))
    elif args.command == "serve":
        _serve(RouterModel.load(args.model), policy, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
