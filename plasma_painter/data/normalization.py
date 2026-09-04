"""Robust statistics fitted on an explicitly identified training interval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class RobustStats:
    median: float
    mad: float
    low: float
    high: float
    clip_z: float
    fit_start: int
    fit_stop: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        scale = max(1.4826 * self.mad, np.finfo(np.float32).eps)
        result = (np.asarray(values, dtype=np.float32) - self.median) / scale
        return np.clip(result, -self.clip_z, self.clip_z).astype(np.float32)

    def unit_interval(self, values: np.ndarray) -> np.ndarray:
        span = max(self.high - self.low, np.finfo(np.float32).eps)
        return np.clip((np.asarray(values, dtype=np.float32) - self.low) / span, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_robust(
    arrays: Iterable[np.ndarray],
    *,
    fit_start: int,
    fit_stop: int,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    clip_z: float = 5.0,
) -> RobustStats:
    flattened = [np.asarray(array, dtype=np.float32).reshape(-1) for array in arrays]
    if not flattened:
        raise ValueError("normalization requires at least one array")
    values = np.concatenate(flattened)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("normalization input has no finite values")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    low, high = np.percentile(values, [low_percentile, high_percentile])
    if not float(high) > float(low):
        raise ValueError("robust percentile interval is degenerate")
    return RobustStats(
        median=median,
        mad=max(mad, float(np.finfo(np.float32).eps)),
        low=float(low),
        high=float(high),
        clip_z=float(clip_z),
        fit_start=int(fit_start),
        fit_stop=int(fit_stop),
    )

