"""Gated Hungarian tracking with explicit birth/death/merge/split events."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .filaments import periodic_distance


def filament_distance(left: dict, right: dict) -> float:
    dx = float(left["centroid"][0]) - float(right["centroid"][0])
    dz = periodic_distance(float(left["centroid"][1]), float(right["centroid"][1]))
    return float(np.hypot(dx, dz))


def _cost(left: dict, right: dict) -> float:
    if int(left["sign"]) != int(right["sign"]):
        return 1e6
    displacement = filament_distance(left, right)
    area_change = abs(np.log((right["area_cells"] + 1e-6) / (left["area_cells"] + 1e-6)))
    intensity_change = abs(float(right["peak"]) - float(left["peak"])) / (
        abs(float(left["peak"])) + 1e-6
    )
    return displacement + 0.03 * area_change + 0.015 * intensity_change


def track_filaments(
    frame_components: Iterable[list[dict]], *, max_distance: float = 0.16
) -> tuple[list[list[dict]], list[list[dict]]]:
    frames = [deepcopy(frame) for frame in frame_components]
    events: list[list[dict]] = [[] for _ in frames]
    next_track = 0
    if not frames:
        return frames, events
    for component in frames[0]:
        component["track_id"] = next_track
        events[0].append({"type": "birth", "track_ids": [next_track]})
        next_track += 1
    for frame_index in range(1, len(frames)):
        previous = frames[frame_index - 1]
        current = frames[frame_index]
        if not previous:
            for component in current:
                component["track_id"] = next_track
                events[frame_index].append({"type": "birth", "track_ids": [next_track]})
                next_track += 1
            continue
        if not current:
            for component in previous:
                events[frame_index].append({"type": "death", "track_ids": [component["track_id"]]})
            continue
        matrix = np.asarray([[_cost(left, right) for right in current] for left in previous])
        rows, columns = linear_sum_assignment(matrix)
        matched_previous: set[int] = set()
        matched_current: set[int] = set()
        for row, column in zip(rows.tolist(), columns.tolist()):
            if matrix[row, column] <= max_distance:
                current[column]["track_id"] = previous[row]["track_id"]
                matched_previous.add(row)
                matched_current.add(column)
        for index, component in enumerate(previous):
            if index not in matched_previous:
                events[frame_index].append({"type": "death", "track_ids": [component["track_id"]]})
        for index, component in enumerate(current):
            if index not in matched_current:
                component["track_id"] = next_track
                events[frame_index].append({"type": "birth", "track_ids": [next_track]})
                next_track += 1
        relation_gate = 1.25 * max_distance
        for right in current:
            related = [left for left in previous if left["sign"] == right["sign"] and filament_distance(left, right) <= relation_gate]
            if len(related) >= 2:
                events[frame_index].append(
                    {
                        "type": "merge",
                        "from_track_ids": sorted({item["track_id"] for item in related}),
                        "to_track_id": right["track_id"],
                    }
                )
        for left in previous:
            related = [right for right in current if left["sign"] == right["sign"] and filament_distance(left, right) <= relation_gate]
            if len(related) >= 2:
                events[frame_index].append(
                    {
                        "type": "split",
                        "from_track_id": left["track_id"],
                        "to_track_ids": sorted({item["track_id"] for item in related}),
                    }
                )
    return frames, events

