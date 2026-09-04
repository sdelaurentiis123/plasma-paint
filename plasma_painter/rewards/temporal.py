"""Clip-level temporal response and stability metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from plasma_painter.features.pipeline import decode_unit_raster


def _gray(image: Image.Image, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(image.convert("L").resize((shape[1], shape[0]), Image.Resampling.BOX), dtype=np.float32) / 255.0


def temporal_clip(frames: list[dict[str, Any]], images: list[Image.Image]) -> dict[str, Any]:
    if len(frames) < 2:
        return {"score": 1.0, "change_response": 1.0, "stable_region_score": 1.0, "static_failure": False}
    shape = tuple(reversed(frames[0]["rasters"]["density"]["shape"]))
    densities = [decode_unit_raster(frame["rasters"]["density"]).T for frame in frames]
    rendered = [_gray(image, shape) for image in images]
    data_changes = np.asarray([np.mean(np.abs(b - a)) for a, b in zip(densities, densities[1:])])
    image_changes = np.asarray([np.mean(np.abs(b - a)) for a, b in zip(rendered, rendered[1:])])
    if np.std(data_changes) > 1e-8 and np.std(image_changes) > 1e-8:
        correlation = float(spearmanr(data_changes, image_changes).statistic)
    else:
        correlation = 0.0
    change_response = float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))
    stable_scores = []
    for density_a, density_b, image_a, image_b in zip(densities, densities[1:], rendered, rendered[1:]):
        data_delta = np.abs(density_b - density_a)
        image_delta = np.abs(image_b - image_a)
        stable = data_delta <= np.quantile(data_delta, 0.35)
        changing = data_delta >= np.quantile(data_delta, 0.75)
        stable_change = float(np.mean(image_delta[stable]))
        changing_change = float(np.mean(image_delta[changing]))
        stable_scores.append(float(np.clip(0.5 + 0.5 * (changing_change - stable_change) / max(changing_change + stable_change, 1e-6), 0, 1)))
    static_failure = bool(float(np.mean(image_changes)) < 0.001 and float(np.mean(data_changes)) > 0.003)
    stable_score = float(np.mean(stable_scores))
    score = 0.55 * change_response + 0.45 * stable_score
    if static_failure:
        score *= 0.25
    return {
        "score": float(score),
        "change_response": change_response,
        "change_spearman_raw": correlation,
        "stable_region_score": stable_score,
        "mean_data_change": float(np.mean(data_changes)),
        "mean_rendered_change": float(np.mean(image_changes)),
        "static_failure": static_failure,
    }
