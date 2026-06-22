from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .data import Example
from .model import RouterModel
from .policy import RoutingPolicy
from .types import Tier


@dataclass(frozen=True)
class Metrics:
    total: int
    accuracy: float
    classifier_accuracy: float
    underroute_rate: float
    overroute_rate: float
    mean_shortfall: float
    average_tier_cost: float
    average_realized_cost: float
    average_regret: float
    savings_vs_high: float
    log_loss: float
    brier_score: float
    per_tier_recall: tuple[float, float, float]
    confusion_matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "accuracy": self.accuracy,
            "classifier_accuracy": self.classifier_accuracy,
            "underroute_rate": self.underroute_rate,
            "overroute_rate": self.overroute_rate,
            "mean_shortfall": self.mean_shortfall,
            "average_tier_cost": self.average_tier_cost,
            "average_realized_cost": self.average_realized_cost,
            "average_regret": self.average_regret,
            "savings_vs_high": self.savings_vs_high,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "per_tier_recall": {tier.label: self.per_tier_recall[int(tier)] for tier in Tier},
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


def evaluate(model: RouterModel, examples: Sequence[Example], policy: RoutingPolicy) -> Metrics:
    if not examples:
        raise ValueError("evaluation examples cannot be empty")
    matrix = [[0, 0, 0] for _ in Tier]
    correct = classifier_correct = under = over = shortfall = 0
    total_cost = realized_cost = oracle_cost = log_loss = brier = 0.0
    for example in examples:
        probabilities = model.predict_proba(example.prompt)
        classifier_tier = Tier(max(range(3), key=probabilities.__getitem__))
        predicted = policy.route(model, example.prompt).tier
        matrix[int(example.label)][int(predicted)] += 1
        correct += predicted == example.label
        classifier_correct += classifier_tier == example.label
        under += predicted < example.label
        over += predicted > example.label
        missed_levels = max(0, int(example.label) - int(predicted))
        shortfall += missed_levels
        total_cost += policy.tier_costs[int(predicted)]
        realized_cost += policy.tier_costs[int(predicted)] + policy.underroute_penalty * missed_levels
        oracle_cost += policy.tier_costs[int(example.label)]
        log_loss -= math.log(max(probabilities[int(example.label)], 1e-15))
        brier += sum((probability - int(index == example.label)) ** 2 for index, probability in enumerate(probabilities))
    count = len(examples)
    recall = tuple(
        matrix[int(tier)][int(tier)] / max(1, sum(matrix[int(tier)])) for tier in Tier
    )
    return Metrics(
        count,
        correct / count,
        classifier_correct / count,
        under / count,
        over / count,
        shortfall / count,
        total_cost / count,
        realized_cost / count,
        (realized_cost - oracle_cost) / count,
        (policy.tier_costs[int(Tier.HIGH)] * count - total_cost) / count,
        log_loss / count,
        brier / count,
        recall,  # type: ignore[arg-type]
        tuple(tuple(row) for row in matrix),  # type: ignore[arg-type]
    )
