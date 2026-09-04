"""Low-level scientific correspondence metrics for a rendered clip."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, sobel
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from plasma_painter.features.pipeline import decode_unit_raster


def _unit_image(image: Image.Image, shape: tuple[int, int]) -> np.ndarray:
    resized = image.convert("RGB").resize((shape[1], shape[0]), Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.float32) / 255.0


def pigment_density(image: Image.Image, shape: tuple[int, int], paper: str = "#f2ede2") -> np.ndarray:
    rgb = _unit_image(image, shape)
    clean = paper.lstrip("#")
    paper_rgb = np.asarray([int(clean[index : index + 2], 16) for index in (0, 2, 4)], dtype=np.float32) / 255.0
    return np.linalg.norm(rgb - paper_rgb[None, None, :], axis=-1) / np.sqrt(3.0)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return 0.0
    value = spearmanr(left.ravel(), right.ravel()).statistic
    return float(value) if np.isfinite(value) else 0.0


def _contour_mask(frame: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    for contour in frame["contours"]:
        for x, z in contour["points"]:
            row = min(height - 1, max(0, round(z * (height - 1))))
            col = min(width - 1, max(0, round(x * (width - 1))))
            mask[row, col] = True
    return mask


def _contour_score(frame: dict[str, Any], ink: np.ndarray) -> float:
    mask = _contour_mask(frame, ink.shape)
    if not np.any(mask):
        return 1.0
    edge = np.hypot(sobel(ink, axis=0), sobel(ink, axis=1))
    threshold = float(np.quantile(edge, 0.80))
    rendered_edges = edge >= threshold
    distance = distance_transform_edt(~rendered_edges)
    scale = max(2.0, 0.04 * min(ink.shape))
    return float(np.mean(np.exp(-distance[mask] / scale)))


def _periodic_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dx = left[:, None, 0] - right[None, :, 0]
    dz_raw = np.abs(left[:, None, 1] - right[None, :, 1])
    dz = np.minimum(dz_raw, 1.0 - dz_raw)
    return np.hypot(dx, dz)


def _filament_score(frame: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, float]:
    expected = frame["filaments"]
    actual = [item["args"] for item in operations if item["op"] == "dab" and item["args"].get("source") == "filament"]
    if not expected and not actual:
        return {"precision": 1.0, "recall": 1.0, "position": 1.0, "size_order": 1.0, "score": 1.0}
    if not expected or not actual:
        return {"precision": 0.0, "recall": 0.0, "position": 0.0, "size_order": 0.0, "score": 0.0}
    truth = np.asarray([item["centroid"] for item in expected], dtype=float)
    predicted = np.asarray([item["center"] for item in actual], dtype=float)
    cost = _periodic_distance(truth, predicted)
    rows, cols = linear_sum_assignment(cost)
    accepted = [(row, col) for row, col in zip(rows, cols) if cost[row, col] <= 0.08]
    matched = len(accepted)
    precision = matched / len(actual)
    recall = matched / len(expected)
    position = float(np.mean([max(0.0, 1.0 - cost[row, col] / 0.08) for row, col in accepted])) if accepted else 0.0
    if len(accepted) >= 2:
        expected_sizes = np.asarray([expected[row]["area_fraction"] for row, _ in accepted])
        predicted_sizes = np.asarray([actual[col]["radius"] for _, col in accepted])
        size_order = max(0.0, _safe_spearman(expected_sizes, predicted_sizes))
    else:
        size_order = 1.0 if accepted else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "position": position,
        "size_order": float(size_order),
        "score": float(0.5 * f1 + 0.3 * position + 0.2 * size_order),
    }


def _orientation_score(frame: dict[str, Any], operations: list[dict[str, Any]]) -> float:
    vectors = frame["vectors"]["density_gradient"]
    paths = [item["args"] for item in operations if item["op"] == "dryBrushPath" and len(item["args"].get("points", [])) >= 2]
    if not vectors:
        return 1.0
    if not paths:
        return 0.0
    starts = np.asarray([item["points"][0] for item in paths], dtype=float)
    directions = np.asarray(
        [np.asarray(item["points"][-1], dtype=float) - np.asarray(item["points"][0], dtype=float) for item in paths]
    )
    norms = np.linalg.norm(directions, axis=1)
    scores = []
    for vector in vectors:
        start = np.asarray([vector["x"], vector["z"]], dtype=float)
        delta = starts - start
        delta[:, 1] = np.minimum(np.abs(delta[:, 1]), 1.0 - np.abs(delta[:, 1]))
        nearest = int(np.argmin(np.linalg.norm(delta, axis=1)))
        if np.linalg.norm(delta[nearest]) > 0.04 or norms[nearest] < 1e-9:
            continue
        actual = directions[nearest] / norms[nearest]
        expected = np.asarray([vector["dx"], vector["dz"]], dtype=float)
        scores.append(max(0.0, float(np.dot(actual, expected))))
    return float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.0


def fidelity_frame(
    frame: dict[str, Any], image: Image.Image, operations: list[dict[str, Any]], *, paper: str = "#f2ede2"
) -> dict[str, Any]:
    density = decode_unit_raster(frame["rasters"]["density"]).T
    ink = pigment_density(image, density.shape, paper)
    correlation = _safe_spearman(ink, density)
    coarse = float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))
    density_high = density >= np.quantile(density, 0.90)
    ink_high = ink >= np.quantile(ink, 0.75)
    extrema = float(np.count_nonzero(density_high & ink_high) / max(np.count_nonzero(density_high), 1))
    filament = _filament_score(frame, operations)
    orientation = _orientation_score(frame, operations)
    transport_available = bool(frame["transport"]["available"])
    transport = None
    structural_marks = sum(
        item["op"] in {"washRegion", "strokePath", "dryBrushPath", "dab", "poolPigment"}
        for item in operations
    )
    components = {
        "coarse_intensity": coarse,
        "coarse_spearman_raw": correlation,
        "contour": _contour_score(frame, ink),
        "extrema": extrema,
        "filament": filament["score"],
        "filament_detail": filament,
        "orientation": orientation,
        "transport": transport,
        "transport_available": transport_available,
        "nonempty_fraction": float(np.mean(ink > 0.08)),
        "structural_marks": int(structural_marks),
    }
    weights = {"coarse_intensity": 0.30, "contour": 0.20, "extrema": 0.15, "filament": 0.20, "orientation": 0.10}
    if transport_available and transport is not None:
        weights["transport"] = 0.05
    total = sum(weights.values())
    components["score"] = float(sum(weights[name] * float(components[name]) for name in weights) / total)
    return components


def fidelity_clip(
    frames: list[dict[str, Any]], images: list[Image.Image], operations_by_frame: list[list[dict[str, Any]]], *, paper: str = "#f2ede2"
) -> dict[str, Any]:
    per_frame = [fidelity_frame(frame, image, operations, paper=paper) for frame, image, operations in zip(frames, images, operations_by_frame)]
    keys = ("score", "coarse_intensity", "coarse_spearman_raw", "contour", "extrema", "filament", "orientation", "nonempty_fraction", "structural_marks")
    result = {key: float(np.mean([record[key] for record in per_frame])) for key in keys}
    result["transport"] = None
    result["transport_reason"] = frames[0]["transport"]["reason"]
    result["per_frame"] = per_frame
    return result
