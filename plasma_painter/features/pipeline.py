"""Compact deterministic frame and clip feature pipeline."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from plasma_painter.data.loaders import FixedPlaneData
from plasma_painter.data.normalization import RobustStats, fit_robust
from plasma_painter.data.schema import AXIS_CONVENTION, SCHEMA_VERSION

from .contours import extract_contours
from .filaments import extract_filaments
from .gradients import sample_gradient_vectors
from .tracking import track_filaments
from .transport import unavailable_transport


def encode_unit_raster(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(np.clip(values, 0.0, 1.0) * 255.0, dtype=np.uint8)
    return {
        "shape": list(array.shape),
        "dtype": "uint8",
        "encoding": "base64",
        "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def decode_unit_raster(record: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(record["data"])
    return np.frombuffer(raw, dtype=np.uint8).reshape(record["shape"]).astype(np.float32) / 255.0


def _density_fluctuation(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=-1, keepdims=True)
    scale = np.maximum(np.abs(mean), np.finfo(np.float32).eps)
    return ((values - mean) / scale).astype(np.float32)


@dataclass
class FeaturePipeline:
    config: dict[str, Any]
    stats: dict[str, RobustStats] | None = None
    filament_threshold: float | None = None

    def fit(self, data: FixedPlaneData, start: int, stop: int) -> dict[str, Any]:
        if not data.frame_start <= start < stop <= data.frame_stop:
            raise PermissionError("normalization interval is outside permitted loaded data")
        local = slice(start - data.frame_start, stop - data.frame_start)
        density = data.fields["density"][local]
        density_view = np.log1p(np.maximum(density, 0.0)) if self.config.get("density_log_transform", True) else density
        fluctuation = _density_fluctuation(density)
        values = {
            "density": density_view,
            "density_fluctuation": fluctuation,
            "potential": data.fields["potential"][local],
            "electron_temperature": data.fields["electron_temperature"][local],
            "electron_pressure": data.fields["electron_pressure"][local],
            "parallel_ion_velocity": data.fields["parallel_ion_velocity"][local],
        }
        self.stats = {
            name: fit_robust(
                [array],
                fit_start=start,
                fit_stop=stop,
                low_percentile=float(self.config["robust_low_percentile"]),
                high_percentile=float(self.config["robust_high_percentile"]),
                clip_z=float(self.config["mad_clip"]),
            )
            for name, array in values.items()
        }
        normalized_fluctuation = self.stats["density_fluctuation"].transform(fluctuation)
        self.filament_threshold = float(
            np.quantile(np.abs(normalized_fluctuation), float(self.config["filament_threshold_quantile"]))
        )
        return self.normalization_record()

    def normalization_record(self) -> dict[str, Any]:
        if self.stats is None or self.filament_threshold is None:
            raise RuntimeError("feature pipeline has not been fit")
        return {
            "fit_only_on": "art_train",
            "fields": {name: value.to_dict() for name, value in self.stats.items()},
            "filament_threshold_abs_robust_z": self.filament_threshold,
            "filament_smoothing_sigma_cells": float(self.config["filament_smoothing_sigma"]),
            "clipping_is_explicit": True,
        }

    def _geometry(self, data: FixedPlaneData, nx: int, nz: int) -> dict[str, Any]:
        plane_y = int(data.metadata["plane_y"])
        offset = 2
        radial_r = np.asarray(data.geometry["Rxy"])[offset : offset + nx, plane_y]
        radial_z = np.asarray(data.geometry["Zxy"])[offset : offset + nx, plane_y]
        shift = np.asarray(data.geometry["zShift"])[offset : offset + nx, plane_y]
        first_sol = int(round(float(data.geometry["ixseps1"]))) - offset
        wedge = 2.0 * np.pi / 5.0
        return {
            **AXIS_CONVENTION,
            "shape": [nx, nz],
            "plane_y": plane_y,
            "R_m_along_x": [round(float(item), 8) for item in radial_r],
            "Z_m_along_x": [round(float(item), 8) for item in radial_z],
            "z_shift_radians_along_x": [round(float(item), 8) for item in shift],
            "z_wedge_radians": [round(wedge * index / nz, 8) for index in range(nz)],
            "zperiod": 5,
            "periodic_axis": 1,
            "first_sol_x": first_sol,
            "separatrix_face_u": round((first_sol - 0.5) / max(nx - 1, 1), 7),
            "valid_domain": "all cells in fixed outboard-midplane plane",
        }

    def transform_frame(self, data: FixedPlaneData, frame_index: int) -> dict[str, Any]:
        if self.stats is None or self.filament_threshold is None:
            raise RuntimeError("fit must be called before transform")
        if not data.frame_start <= frame_index < data.frame_stop:
            raise PermissionError("frame is outside the loaded permitted interval")
        local = frame_index - data.frame_start
        density_raw = data.fields["density"][local]
        density_view = np.log1p(np.maximum(density_raw, 0.0)) if self.config.get("density_log_transform", True) else density_raw
        density_unit = self.stats["density"].unit_interval(density_view)
        fluctuation = _density_fluctuation(density_raw[None, ...])[0]
        fluctuation_z = self.stats["density_fluctuation"].transform(fluctuation)
        potential_z = self.stats["potential"].transform(data.fields["potential"][local])
        temperature_unit = self.stats["electron_temperature"].unit_interval(data.fields["electron_temperature"][local])
        smoothed = gaussian_filter(
            fluctuation_z,
            sigma=(float(self.config["filament_smoothing_sigma"]), float(self.config["filament_smoothing_sigma"])),
            mode=("nearest", "wrap"),
        )
        valid = np.isfinite(density_raw) & np.isfinite(potential_z)
        contours = extract_contours(smoothed, self.config["contour_levels"])
        filaments = extract_filaments(
            smoothed,
            threshold=self.filament_threshold,
            min_cells=int(self.config["filament_min_cells"]),
        )
        density_vectors = sample_gradient_vectors(
            smoothed,
            stride=int(self.config["gradient_stride"]),
            label="density_gradient_direction",
        )
        potential_vectors = sample_gradient_vectors(
            potential_z,
            stride=int(self.config["gradient_stride"]),
            label="potential_gradient_visualization_proxy_not_exb",
        )
        omega = float(data.metadata["omega_ci"])
        normalized_time = float(data.time[local])
        return {
            "schema_version": SCHEMA_VERSION,
            "source": {
                "shot": data.shot,
                "frame_index": int(frame_index),
                "synthetic": bool(data.metadata.get("synthetic", False)),
            },
            "time": {
                "normalized": normalized_time,
                "microseconds": round(normalized_time / omega * 1e6, 8) if omega else normalized_time,
            },
            "geometry": self._geometry(data, *density_raw.shape),
            "masks": {
                "valid_fraction": round(float(np.mean(valid)), 7),
                "invalid_cells": int(np.size(valid) - np.count_nonzero(valid)),
                "radial_boundary_cells": [0, density_raw.shape[0] - 1],
            },
            "rasters": {
                "density": encode_unit_raster(density_unit),
                "density_fluctuation": encode_unit_raster((fluctuation_z / (2 * self.stats["density_fluctuation"].clip_z)) + 0.5),
                "potential": encode_unit_raster((potential_z / (2 * self.stats["potential"].clip_z)) + 0.5),
                "electron_temperature": encode_unit_raster(temperature_unit),
            },
            "contours": contours,
            "vectors": {
                "density_gradient": density_vectors,
                "potential_gradient_proxy": potential_vectors,
                "exb_available": False,
                "exb_reason": "fixed-plane geometry is insufficient for a physically complete E cross B vector in the rendered plane",
            },
            "filaments": filaments,
            "events": [],
            "transport": unavailable_transport(),
            "summary": {
                "density_mean": round(float(np.mean(density_unit)), 7),
                "density_std": round(float(np.std(density_unit)), 7),
                "positive_filaments": sum(item["sign"] > 0 for item in filaments),
                "negative_filaments": sum(item["sign"] < 0 for item in filaments),
            },
        }

    def transform_clip(self, data: FixedPlaneData, start: int, stop: int, clip_id: str, split: str) -> dict[str, Any]:
        if not data.frame_start <= start < stop <= data.frame_stop:
            raise PermissionError("clip is outside the loaded permitted interval")
        frames = [self.transform_frame(data, frame) for frame in range(start, stop)]
        tracked, events = track_filaments(
            [frame["filaments"] for frame in frames],
            max_distance=float(self.config["track_max_periodic_distance"]),
        )
        for frame, components, frame_events in zip(frames, tracked, events):
            frame["filaments"] = components
            frame["events"] = frame_events
        return {
            "schema_version": SCHEMA_VERSION,
            "clip_id": clip_id,
            "split": split,
            "frame_start": start,
            "frame_stop": stop,
            "frame_count": stop - start,
            "frames": frames,
        }

