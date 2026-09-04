"""Auditable scalar-gradient visualization vectors."""

from __future__ import annotations

import numpy as np


def periodic_gradient(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(field, dtype=np.float32)
    gx = np.gradient(values, axis=0)
    gz = 0.5 * (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1))
    magnitude = np.hypot(gx, gz)
    safe = np.maximum(magnitude, np.finfo(np.float32).eps)
    return gx / safe, gz / safe, magnitude


def sample_gradient_vectors(
    field: np.ndarray,
    *,
    stride: int = 4,
    label: str = "density_gradient",
    percentile_floor: float = 55.0,
) -> list[dict]:
    ux, uz, magnitude = periodic_gradient(field)
    threshold = float(np.percentile(magnitude[np.isfinite(magnitude)], percentile_floor))
    nx, nz = field.shape
    scale = max(float(np.percentile(magnitude, 99.0)), np.finfo(np.float32).eps)
    vectors: list[dict] = []
    for x in range(stride // 2, nx, stride):
        for z in range(stride // 2, nz, stride):
            mag = float(magnitude[x, z])
            if not np.isfinite(mag) or mag < threshold:
                continue
            vectors.append(
                {
                    "x": round(x / max(nx - 1, 1), 5),
                    "z": round(z / max(nz - 1, 1), 5),
                    "dx": round(float(ux[x, z]), 5),
                    "dz": round(float(uz[x, z]), 5),
                    "magnitude": round(min(mag / scale, 1.0), 5),
                    "label": label,
                }
            )
    return vectors

