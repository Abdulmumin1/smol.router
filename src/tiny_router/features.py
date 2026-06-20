from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]", re.UNICODE)


def _bucket(value: str, dimensions: int) -> tuple[int, float]:
    digest = hashlib.blake2b(value.encode("utf-8", "replace"), digest_size=8).digest()
    number = int.from_bytes(digest, "little")
    sign = 1.0 if number & 1 else -1.0
    return (number >> 1) % dimensions, sign


def extract_features(text: str, dimensions: int) -> dict[int, float]:
    if dimensions < 8:
        raise ValueError("dimensions must be at least 8")
    if not isinstance(text, str):
        raise TypeError("prompt must be a string")

    normalized = text.lower()
    tokens = TOKEN_RE.findall(normalized)[:4096]
    raw: Counter[int] = Counter()

    def add(name: str, value: float = 1.0) -> None:
        index, sign = _bucket(name, dimensions)
        raw[index] += sign * value

    add("bias")
    for token in tokens:
        add(f"w:{token}")
    for left, right in zip(tokens, tokens[1:]):
        add(f"b:{left}\x1f{right}")

    length = len(text)
    add(f"length:{min(length // 80, 20)}")
    add(f"tokens:{min(len(tokens) // 12, 20)}")
    add("has_code_fence", float("```" in text))
    add("has_question", float("?" in text))
    add("newline_bucket", min(text.count("\n"), 20) / 20.0)
    add("digit_ratio", sum(ch.isdigit() for ch in text) / max(length, 1))

    norm = math.sqrt(sum(value * value for value in raw.values())) or 1.0
    return {index: value / norm for index, value in raw.items() if value}

