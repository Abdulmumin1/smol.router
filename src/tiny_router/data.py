from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .types import Tier


@dataclass(frozen=True)
class Example:
    prompt: str
    label: Tier


def label_from_scores(scores: dict[str, float], acceptable_score: float) -> Tier:
    for tier in Tier:
        score = float(scores.get(tier.label, float("-inf")))
        if score >= acceptable_score:
            return tier
    return Tier.HIGH


def parse_record(record: dict[str, object], acceptable_score: float = 0.8) -> Example:
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("record.prompt must be a non-empty string")
    if "label" in record:
        label = Tier.parse(record["label"])  # type: ignore[arg-type]
    else:
        scores = record.get("scores")
        if not isinstance(scores, dict):
            raise ValueError("record needs either label or scores")
        label = label_from_scores(scores, acceptable_score)  # type: ignore[arg-type]
    return Example(prompt=prompt, label=label)


def load_jsonl(path: str | Path, acceptable_score: float = 0.8) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                examples.append(parse_record(record, acceptable_score))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"{path}: dataset is empty")
    return examples


def split_examples(
    examples: Iterable[Example], validation_fraction: float = 0.2, seed: int = 17
) -> tuple[list[Example], list[Example]]:
    import random

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    items = list(examples)
    random.Random(seed).shuffle(items)
    validation_size = int(len(items) * validation_fraction)
    if validation_fraction and len(items) > 1:
        validation_size = max(1, validation_size)
    return items[validation_size:], items[:validation_size]

