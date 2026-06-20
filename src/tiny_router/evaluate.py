from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .data import Example
from .model import RouterModel
from .policy import RoutingPolicy
from .types import Tier


@dataclass(frozen=True)
class Metrics:
    total: int
    accuracy: float
    underroute_rate: float
    average_tier_cost: float
    confusion_matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "accuracy": self.accuracy,
            "underroute_rate": self.underroute_rate,
            "average_tier_cost": self.average_tier_cost,
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


def evaluate(model: RouterModel, examples: Sequence[Example], policy: RoutingPolicy) -> Metrics:
    if not examples:
        raise ValueError("evaluation examples cannot be empty")
    matrix = [[0, 0, 0] for _ in Tier]
    correct = under = 0
    total_cost = 0.0
    for example in examples:
        predicted = policy.route(model, example.prompt).tier
        matrix[int(example.label)][int(predicted)] += 1
        correct += predicted == example.label
        under += predicted < example.label
        total_cost += policy.tier_costs[int(predicted)]
    count = len(examples)
    return Metrics(
        count,
        correct / count,
        under / count,
        total_cost / count,
        tuple(tuple(row) for row in matrix),  # type: ignore[arg-type]
    )

