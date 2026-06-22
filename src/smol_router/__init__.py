"""Tiny cost-aware model capability router."""

from .model import RouterModel
from .policy import RoutingPolicy
from .types import RouteDecision, Tier
from .errors import (
    ArtifactError,
    ConfigurationError,
    DatasetError,
    ExhaustedError,
    InvalidPromptError,
    ProviderError,
    RouterError,
)
from .config import ModelTarget, RouterConfig
from .sdk import ExecutionAttempt, ExecutionResult, Router, RoutingResult
from .benchmark import BenchmarkRecord, BenchmarkTask, run_benchmark, run_benchmark_async, write_benchmark_jsonl

__all__ = [
    "ArtifactError",
    "BenchmarkRecord",
    "BenchmarkTask",
    "ConfigurationError",
    "DatasetError",
    "ExhaustedError",
    "ExecutionAttempt",
    "ExecutionResult",
    "InvalidPromptError",
    "ModelTarget",
    "ProviderError",
    "RouteDecision",
    "RouterError",
    "RouterConfig",
    "Router",
    "RouterModel",
    "RoutingPolicy",
    "RoutingResult",
    "Tier",
    "run_benchmark",
    "run_benchmark_async",
    "write_benchmark_jsonl",
]
__version__ = "0.1.0"
