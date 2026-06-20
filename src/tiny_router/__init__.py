"""Tiny cost-aware model capability router."""

from .model import RouterModel
from .policy import RoutingPolicy
from .types import RouteDecision, Tier
from .errors import (
    ArtifactError,
    ConfigurationError,
    DatasetError,
    ExhaustedError,
    ProviderError,
    RouterError,
)
from .config import ModelTarget, RouterConfig

__all__ = [
    "ArtifactError",
    "ConfigurationError",
    "DatasetError",
    "ExhaustedError",
    "ModelTarget",
    "ProviderError",
    "RouteDecision",
    "RouterError",
    "RouterConfig",
    "RouterModel",
    "RoutingPolicy",
    "Tier",
]
__version__ = "0.1.0"
