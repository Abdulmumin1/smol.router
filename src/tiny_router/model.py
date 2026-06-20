from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .data import Example
from .features import extract_features
from .types import Tier

MODEL_FORMAT = "tiny-router-v1"


def _softmax(logits: Sequence[float]) -> tuple[float, float, float]:
    maximum = max(logits)
    exponents = [math.exp(max(-60.0, min(60.0, value - maximum))) for value in logits]
    total = sum(exponents)
    return tuple(value / total for value in exponents)  # type: ignore[return-value]


@dataclass
class RouterModel:
    dimensions: int
    weights: list[list[float]]
    temperature: float = 1.0

    @classmethod
    def empty(cls, dimensions: int = 1024) -> "RouterModel":
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        return cls(dimensions, [[0.0] * dimensions for _ in Tier])

    def predict_proba(self, prompt: str) -> tuple[float, float, float]:
        features = extract_features(prompt, self.dimensions)
        temperature = max(self.temperature, 1e-3)
        logits = [
            sum(row[index] * value for index, value in features.items()) / temperature
            for row in self.weights
        ]
        return _softmax(logits)

    def predict(self, prompt: str) -> Tier:
        probabilities = self.predict_proba(prompt)
        return Tier(max(range(3), key=probabilities.__getitem__))

    @classmethod
    def train(
        cls,
        examples: Sequence[Example],
        *,
        dimensions: int = 1024,
        epochs: int = 35,
        learning_rate: float = 0.35,
        l2: float = 1e-5,
        seed: int = 17,
        underroute_weight: float = 2.0,
    ) -> "RouterModel":
        if not examples:
            raise ValueError("training examples cannot be empty")
        if epochs < 1 or learning_rate <= 0 or l2 < 0 or underroute_weight < 1:
            raise ValueError("invalid training hyperparameters")
        model = cls.empty(dimensions)
        rng = random.Random(seed)
        order = list(range(len(examples)))

        for epoch in range(epochs):
            rng.shuffle(order)
            rate = learning_rate / math.sqrt(1.0 + epoch * 0.15)
            for position in order:
                example = examples[position]
                features = extract_features(example.prompt, dimensions)
                probabilities = model.predict_proba(example.prompt)
                # Mistaking medium/high requirements for low is progressively more costly.
                sample_weight = 1.0 + underroute_weight * int(example.label)
                for class_index, row in enumerate(model.weights):
                    error = (probabilities[class_index] - int(class_index == example.label)) * sample_weight
                    for feature_index, value in features.items():
                        row[feature_index] -= rate * (error * value + l2 * row[feature_index])
        return model

    def save(self, path: str | Path) -> None:
        payload = {
            "format": MODEL_FORMAT,
            "dimensions": self.dimensions,
            "temperature": self.temperature,
            "weights": self.weights,
        }
        Path(path).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RouterModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != MODEL_FORMAT:
            raise ValueError("unsupported model format")
        dimensions = int(payload["dimensions"])
        weights = payload["weights"]
        if (
            dimensions < 8
            or not isinstance(weights, list)
            or len(weights) != 3
            or any(not isinstance(row, list) or len(row) != dimensions for row in weights)
        ):
            raise ValueError("invalid model dimensions")
        clean_weights = [[float(value) for value in row] for row in weights]
        if any(not math.isfinite(value) for row in clean_weights for value in row):
            raise ValueError("model contains non-finite weights")
        temperature = float(payload.get("temperature", 1.0))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("model temperature must be finite and positive")
        return cls(dimensions, clean_weights, temperature)
