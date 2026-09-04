import numpy as np

from plasma_painter.data.normalization import fit_robust
from plasma_painter.features.contours import extract_contours
from plasma_painter.features.filaments import extract_filaments, periodic_distance
from plasma_painter.features.tracking import track_filaments


def test_periodic_distance_wraps():
    assert periodic_distance(0.98, 0.02) == pytest.approx(0.04)


def test_normalization_is_train_fit_and_clipped():
    stats = fit_robust([np.arange(20, dtype=float)], fit_start=0, fit_stop=8, clip_z=2)
    transformed = stats.transform(np.asarray([-1e6, stats.median, 1e6]))
    assert transformed.tolist() == [-2.0, 0.0, 2.0]
    assert stats.fit_stop == 8


def test_periodic_contours_stay_normalized():
    x = np.linspace(-1, 1, 24)[:, None]
    z = np.linspace(0, 2 * np.pi, 32, endpoint=False)[None, :]
    paths = extract_contours(x + 0.4 * np.sin(z), [0.0])
    assert paths
    assert all(0 <= value <= 1 for path in paths for point in path["points"] for value in point)


def test_filaments_join_across_periodic_edge():
    field = np.zeros((8, 10), dtype=float)
    field[3:5, [0, 9]] = 3.0
    items = extract_filaments(field, threshold=2, min_cells=4)
    assert len(items) == 1
    assert items[0]["area_cells"] == 4


def _component(x, z, area=5, peak=3, sign=1):
    return {"centroid": [x, z], "area_cells": area, "peak": peak, "sign": sign, "track_id": None}


def test_tracking_records_birth_death_merge_and_split():
    frames = [
        [_component(.4, .98), _component(.42, .06)],
        [_component(.41, .02, area=10)],
        [_component(.39, .98), _component(.43, .06)],
        [],
    ]
    tracked, events = track_filaments(frames, max_distance=.16)
    kinds = {item["type"] for group in events for item in group}
    assert {"birth", "death", "merge", "split"}.issubset(kinds)
    assert tracked[1][0]["track_id"] is not None


import pytest
