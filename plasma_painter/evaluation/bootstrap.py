"""Bootstrap helpers that resample independent clips, never individual frames."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_interval(values: list[float], *, samples: int = 1000, seed: int = 1701) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "independent_clips": 0}
    rng = np.random.default_rng(seed)
    means = np.asarray([np.mean(rng.choice(array, size=len(array), replace=True)) for _ in range(samples)])
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(np.mean(array)), "low": float(low), "high": float(high), "independent_clips": int(len(array))}


def paired_bootstrap_interval(left: list[float], right: list[float], *, samples: int = 1000, seed: int = 1701) -> dict[str, float | int]:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or not len(a):
        raise ValueError("paired bootstrap needs equal nonempty clip vectors")
    return bootstrap_mean_interval((a - b).tolist(), samples=samples, seed=seed)
