from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import load_jsonl, split_examples
from .evaluate import evaluate
from .model import RouterModel
from .policy import RoutingPolicy
from .types import Tier
from .config import ModelTarget, RouterConfig
from .sdk import Router
from .server import create_server
from .errors import RouterError
from . import __version__


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
    parser = argparse.ArgumentParser(prog="smol-router", description="Train and run a smol capability router")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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
    train.add_argument("--no-calibrate", action="store_true", help="skip held-out temperature calibration")
    _add_policy_arguments(train)

    route = subparsers.add_parser("route", help="route a prompt")
    route.add_argument("--model", default="model.json")
    route.add_argument("--config", help="JSON model registry and policy")
    route.add_argument("--minimum-tier", choices=("low", "medium", "high"))
    route.add_argument("--maximum-tier", choices=("low", "medium", "high"))
    route.add_argument("prompt", nargs="?")
    _add_policy_arguments(route)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a model")
    evaluate_parser.add_argument("dataset")
    evaluate_parser.add_argument("--model", default="model.json")
    evaluate_parser.add_argument("--acceptable-score", type=float, default=0.8)
    _add_policy_arguments(evaluate_parser)

    serve = subparsers.add_parser("serve", help="serve POST /route")
    serve.add_argument("--model", default="model.json")
    serve.add_argument("--config", help="JSON model registry and policy")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    _add_policy_arguments(serve)

    init = subparsers.add_parser("init", help="write a documented starter configuration")
    init.add_argument("--output", default="router.json")
    init.add_argument("--force", action="store_true")

    inspect = subparsers.add_parser("inspect", help="inspect a model artifact")
    inspect.add_argument("--model", default="model.json")
    return parser


def _serve(router: Router, host: str, port: int) -> None:
    print(f"listening on http://{host}:{port}", file=sys.stderr)
    server = create_server(router, host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _default_config(policy: RoutingPolicy) -> RouterConfig:
    return RouterConfig({tier: ModelTarget(tier.label) for tier in Tier}, policy)


def _configured_router(args: argparse.Namespace) -> Router:
    model = RouterModel.load(args.model)
    config = RouterConfig.load(args.config) if args.config else _default_config(_policy(args))
    return Router(model, config)


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        policy = _policy(args)
        examples = load_jsonl(args.dataset, args.acceptable_score)
        training, validation = split_examples(examples, args.validation_fraction, args.seed)
        model = RouterModel.train(
            training,
            dimensions=args.dimensions,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        if validation and not args.no_calibrate:
            model.calibrate(validation)
        model.save(args.output)
        result: dict[str, object] = {
            "model": str(Path(args.output)),
            "training_examples": len(training),
            "temperature": model.temperature,
        }
        if validation:
            result["validation"] = evaluate(model, validation, policy).to_dict()
        print(json.dumps(result, indent=2))
    elif args.command == "route":
        prompt = args.prompt if args.prompt is not None else sys.stdin.read()
        router = _configured_router(args)
        result = router.route(prompt, minimum_tier=args.minimum_tier, maximum_tier=args.maximum_tier)
        print(json.dumps(result.to_dict(), indent=2))
    elif args.command == "evaluate":
        policy = _policy(args)
        metrics = evaluate(RouterModel.load(args.model), load_jsonl(args.dataset, args.acceptable_score), policy)
        print(json.dumps(metrics.to_dict(), indent=2))
    elif args.command == "serve":
        _serve(_configured_router(args), args.host, args.port)
    elif args.command == "init":
        destination = Path(args.output)
        if destination.exists() and not args.force:
            raise RouterError(f"{destination} already exists; pass --force to replace it")
        template = RouterConfig.from_dict(
            {"models": {"low": "provider/tiny", "medium": "provider/standard", "high": "provider/frontier"}}
        )
        destination.write_text(json.dumps(template.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {destination}")
    elif args.command == "inspect":
        model = RouterModel.load(args.model)
        payload = model.to_dict()
        print(json.dumps({
            "format": payload["format"],
            "dimensions": model.dimensions,
            "temperature": model.temperature,
            "sha256": payload["sha256"],
        }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (RouterError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
