"""Program-composition diversity without rewarding frame-to-frame noise."""

from __future__ import annotations

from collections import Counter
from typing import Any
import math


def operation_diversity(operations_by_frame: list[list[dict[str, Any]]]) -> dict[str, float]:
    names = [item["op"] for frame in operations_by_frame for item in frame]
    if not names:
        return {"score": 0.0, "entropy": 0.0, "unique_operations": 0.0}
    counts = Counter(names)
    probabilities = [count / len(names) for count in counts.values()]
    entropy = -sum(value * math.log(value) for value in probabilities)
    normalized = entropy / math.log(max(len(counts), 2))
    coverage = min(1.0, len(counts) / 7.0)
    return {"score": float(0.65 * normalized + 0.35 * coverage), "entropy": float(entropy), "unique_operations": float(len(counts))}
