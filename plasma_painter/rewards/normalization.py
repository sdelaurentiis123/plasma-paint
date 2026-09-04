"""Robust train-reference normalization for continuous reward components."""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_normalization(records: list[dict[str, float]], *, fit_split: str = "art_train") -> dict[str, Any]:
    if not records:
        raise ValueError("normalization needs at least one training record")
    output = {"method": "training_reference_05_95_quantile", "fit_split": fit_split, "components": {}}
    for name in records[0]:
        values = np.asarray([record[name] for record in records], dtype=float)
        low, high = np.quantile(values, [0.05, 0.95])
        if high - low < 1e-8:
            low, high = float(np.min(values)), float(np.max(values) + 1e-8)
        output["components"][name] = {
            "q05": float(low),
            "q95": float(high),
            "median": float(np.median(values)),
            "mad": float(np.median(np.abs(values - np.median(values)))),
            "count": int(len(values)),
        }
    return output


def apply_normalization(values: dict[str, float], record: dict[str, Any]) -> dict[str, float]:
    output = {}
    for name, value in values.items():
        stats = record["components"].get(name)
        if stats is None:
            output[name] = float(value)
            continue
        output[name] = float(np.clip((value - stats["q05"]) / max(stats["q95"] - stats["q05"], 1e-8), 0.0, 1.0))
    return output
