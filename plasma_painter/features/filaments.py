"""Signed connected-component filaments with periodic topology."""

from __future__ import annotations

from collections import deque

import numpy as np


def periodic_distance(a: float, b: float, period: float = 1.0) -> float:
    delta = abs(float(a) - float(b)) % period
    return min(delta, period - delta)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    nx, nz = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    groups: list[list[tuple[int, int]]] = []
    for x in range(nx):
        for z in range(nz):
            if not mask[x, z] or visited[x, z]:
                continue
            queue = deque([(x, z)])
            visited[x, z] = True
            group: list[tuple[int, int]] = []
            while queue:
                cx, cz = queue.popleft()
                group.append((cx, cz))
                for nx2, nz2 in ((cx - 1, cz), (cx + 1, cz), (cx, (cz - 1) % nz), (cx, (cz + 1) % nz)):
                    if 0 <= nx2 < nx and mask[nx2, nz2] and not visited[nx2, nz2]:
                        visited[nx2, nz2] = True
                        queue.append((nx2, nz2))
            groups.append(group)
    return groups


def _component_record(
    field: np.ndarray, cells: list[tuple[int, int]], sign: int, component_id: int
) -> dict:
    nx, nz = field.shape
    x_index = np.asarray([item[0] for item in cells], dtype=np.float64)
    z_index = np.asarray([item[1] for item in cells], dtype=np.float64)
    weights = np.abs(field[x_index.astype(int), z_index.astype(int)]).astype(np.float64)
    weights = np.maximum(weights, np.finfo(np.float64).eps)
    x_centroid = float(np.average(x_index, weights=weights))
    angles = 2.0 * np.pi * z_index / nz
    vector = np.sum(weights * np.exp(1j * angles))
    z_angle = float(np.angle(vector) % (2.0 * np.pi))
    z_centroid = z_angle * nz / (2.0 * np.pi)
    z_delta = ((z_index - z_centroid + nz / 2.0) % nz) - nz / 2.0
    points = np.column_stack([x_index - x_centroid, z_delta])
    covariance = np.cov(points.T, aweights=weights) if len(cells) > 1 else np.eye(2) * 1e-6
    eigenvalues, eigenvectors = np.linalg.eigh(np.nan_to_num(covariance, nan=0.0))
    order = np.argsort(eigenvalues)[::-1]
    major = max(float(eigenvalues[order[0]]), 1e-9)
    minor = max(float(eigenvalues[order[-1]]), 1e-9)
    direction = eigenvectors[:, order[0]]
    orientation = float(np.arctan2(direction[1], direction[0]))
    signed_values = field[x_index.astype(int), z_index.astype(int)]
    peak = float(np.max(signed_values) if sign > 0 else np.min(signed_values))
    return {
        "component_id": int(component_id),
        "track_id": None,
        "sign": int(sign),
        "centroid": [round(x_centroid / max(nx - 1, 1), 6), round(z_centroid / nz, 6)],
        "area_cells": int(len(cells)),
        "area_fraction": round(len(cells) / float(nx * nz), 7),
        "peak": round(peak, 6),
        "mean_abs_intensity": round(float(np.mean(np.abs(signed_values))), 6),
        "elongation": round(float(np.sqrt(major / minor)), 5),
        "orientation_radians": round(orientation, 6),
        "bbox": [
            round(float(np.min(x_index)) / max(nx - 1, 1), 6),
            round(float(np.max(x_index)) / max(nx - 1, 1), 6),
        ],
    }


def extract_filaments(
    field: np.ndarray,
    *,
    threshold: float,
    min_cells: int = 4,
) -> list[dict]:
    values = np.asarray(field, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("filament field must be two-dimensional")
    records: list[dict] = []
    component_id = 0
    for sign, mask in ((1, values >= threshold), (-1, values <= -threshold)):
        for cells in _components(mask):
            if len(cells) < min_cells:
                continue
            records.append(_component_record(values, cells, sign, component_id))
            component_id += 1
    records.sort(key=lambda item: item["area_cells"], reverse=True)
    return records

