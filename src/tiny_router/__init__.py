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
from .sdk import Router, RoutingResult

__all__ = [
    "ArtifactError",
    "ConfigurationError",
    "DatasetError",
    "ExhaustedError",
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
]
__version__ = "0.1.0"
