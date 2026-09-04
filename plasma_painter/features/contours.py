"""Contour extraction in normalized logical-domain coordinates."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from skimage.measure import find_contours


def _decimate(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=int)
    return points[indices]


def _split_periodic_jumps(points: np.ndarray) -> list[np.ndarray]:
    if len(points) < 2:
        return []
    jumps = np.flatnonzero(np.abs(np.diff(points[:, 1])) > 0.5) + 1
    return [part for part in np.split(points, jumps) if len(part) >= 3]


def extract_contours(
    field: np.ndarray,
    levels: Iterable[float],
    *,
    max_paths_per_level: int = 16,
    max_points: int = 128,
) -> list[dict]:
    """Extract paths while respecting periodicity of the second axis."""

    values = np.asarray(field, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("contour field must be two-dimensional")
    nx, nz = values.shape
    tiled = np.concatenate([values, values, values], axis=1)
    output: list[dict] = []
    for level in levels:
        candidates: list[np.ndarray] = []
        for raw in find_contours(tiled, float(level)):
            centre = float(np.mean(raw[:, 1]))
            if not nz <= centre < 2 * nz:
                continue
            normalized = np.column_stack(
                [
                    raw[:, 0] / max(nx - 1, 1),
                    # A periodic cell coordinate occupies [0, nz), unlike the
                    # closed radial interval.  Dividing by nz therefore keeps
                    # every wrapped point strictly inside normalized bounds.
                    np.mod(raw[:, 1] - nz, nz) / max(nz, 1),
                ]
            )
            for part in _split_periodic_jumps(normalized):
                candidates.append(_decimate(part, max_points))
        candidates.sort(key=len, reverse=True)
        for path in candidates[:max_paths_per_level]:
            output.append(
                {
                    "level": float(level),
                    "sign": 1 if level > 0 else -1,
                    "points": [[round(float(x), 5), round(float(z), 5)] for x, z in path],
                }
            )
    return output
