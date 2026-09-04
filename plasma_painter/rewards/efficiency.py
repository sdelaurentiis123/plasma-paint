"""Bounded cost score for source length, operations, and runtime."""

from __future__ import annotations

import math


def efficiency_score(*, code_bytes: int, operation_count: float, runtime_ms: float) -> dict[str, float]:
    code = math.exp(-max(0, code_bytes - 2000) / 16000)
    operations = math.exp(-max(0.0, operation_count - 80) / 800)
    runtime = math.exp(-max(0.0, runtime_ms - 25) / 1200)
    return {"score": float((code + operations + runtime) / 3), "code": code, "operations": operations, "runtime": runtime}
