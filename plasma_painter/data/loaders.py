"""Isolated loading for fixed-plane NumPy archives and synthetic tests.

The real archive contains several chronological regions in each `.npy`
member. NumPy's normal `np.load` materializes a whole member before slicing,
so this module reads only the byte prefix authorized by the forecasting split.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.lib import format as npformat

from .splits import validate_shot


@dataclass(frozen=True)
class NpyMemberHeader:
    name: str
    shape: tuple[int, ...]
    dtype: str
    fortran_order: bool
    compression: str


def _read_header(handle: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = npformat.read_magic(handle)
    if version == (1, 0):
        return npformat.read_array_header_1_0(handle)
    if version in {(2, 0), (3, 0)}:
        return npformat.read_array_header_2_0(handle)
    raise ValueError(f"unsupported npy version {version}")


def inspect_npz_headers(path: str | Path) -> dict[str, NpyMemberHeader]:
    target = Path(path)
    with zipfile.ZipFile(target) as archive:
        result: dict[str, NpyMemberHeader] = {}
        for info in archive.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with archive.open(info) as handle:
                shape, fortran_order, dtype = _read_header(handle)
            result[info.filename[:-4]] = NpyMemberHeader(
                name=info.filename[:-4],
                shape=tuple(int(item) for item in shape),
                dtype=np.dtype(dtype).str,
                fortran_order=bool(fortran_order),
                compression=("stored" if info.compress_type == zipfile.ZIP_STORED else "compressed"),
            )
        return result


def read_prefix_member(path: str | Path, key: str, stop: int) -> np.ndarray:
    """Read frames `[0, stop)` without reading the later array payload."""

    if stop <= 0:
        raise ValueError("prefix stop must be positive")
    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("real-data isolation requires uncompressed npz members")
        with archive.open(info) as handle:
            shape, fortran_order, dtype = _read_header(handle)
            if fortran_order:
                raise ValueError("prefix loader requires C-order arrays")
            if not shape or stop > shape[0]:
                raise ValueError(f"requested prefix {stop} outside {key} shape {shape}")
            frame_values = int(np.prod(shape[1:], dtype=np.int64))
            expected_bytes = stop * frame_values * np.dtype(dtype).itemsize
            payload = handle.read(expected_bytes)
            if len(payload) != expected_bytes:
                raise EOFError(f"short read for {key}: {len(payload)} != {expected_bytes}")
    values = np.frombuffer(payload, dtype=dtype).copy()
    return values.reshape((stop, *shape[1:]))


def read_small_member(path: str | Path, key: str) -> np.ndarray:
    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        return npformat.read_array(handle, allow_pickle=False)


@dataclass
class FixedPlaneData:
    shot: str
    frame_start: int
    frame_stop: int
    fields: dict[str, np.ndarray]
    time: np.ndarray
    geometry: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def clip(self, start: int, stop: int) -> dict[str, np.ndarray]:
        if not self.frame_start <= start < stop <= self.frame_stop:
            raise PermissionError("requested clip is outside the loaded permitted interval")
        local = slice(start - self.frame_start, stop - self.frame_start)
        return {name: values[local] for name, values in self.fields.items()}


def load_fixed_plane(
    source_path: str | Path,
    geometry_path: str | Path,
    *,
    shot: str,
    field_keys: dict[str, str],
    stop: int,
    plane_y: int,
) -> FixedPlaneData:
    validate_shot(shot, str(source_path), str(geometry_path))
    source = Path(source_path)
    geometry_source = Path(geometry_path)
    # Path checks happen only after the sequestered-shot string guard above.
    if not source.is_file() or not geometry_source.is_file():
        raise FileNotFoundError("configured permitted source or geometry file is missing")
    headers = inspect_npz_headers(source)
    fields: dict[str, np.ndarray] = {}
    for semantic_name, key in field_keys.items():
        if key not in headers:
            raise KeyError(f"missing configured field {key}")
        fields[semantic_name] = read_prefix_member(source, key, stop)
    time = read_prefix_member(source, "time", stop)
    with np.load(geometry_source, allow_pickle=False) as geometry_archive:
        geometry = {
            name: np.asarray(geometry_archive[name])
            for name in (
                "Rxy",
                "Zxy",
                "Bxy",
                "Bpxy",
                "Btxy",
                "psixy",
                "zShift",
                "dx",
                "dy",
                "J",
                "g11",
                "g22",
                "g33",
                "ixseps1",
                "ixseps2",
                "jyseps1_1",
                "jyseps1_2",
                "jyseps2_1",
                "jyseps2_2",
                "psi_axis",
                "psi_bdry",
            )
        }
    metadata = {
        "plane_y": int(plane_y),
        "dt_norm": float(read_small_member(source, "dt_norm")),
        "omega_ci": float(read_small_member(source, "omega_ci")),
        "fields_available": read_small_member(source, "fields").tolist(),
        "planes_available": read_small_member(source, "planes").tolist(),
    }
    return FixedPlaneData(
        shot=str(shot),
        frame_start=0,
        frame_stop=int(stop),
        fields=fields,
        time=time,
        geometry=geometry,
        metadata=metadata,
    )


def synthetic_fixed_plane(
    *, frames: int = 16, nx: int = 32, nz: int = 40, seed: int = 7
) -> FixedPlaneData:
    if frames < 8:
        raise ValueError("synthetic smoke clips require at least eight frames")
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, nx)[:, None]
    z = np.linspace(0.0, 1.0, nz, endpoint=False)[None, :]
    density = []
    potential = []
    for frame in range(frames):
        phase = frame / frames
        periodic_distance = np.angle(np.exp(2j * np.pi * (z - 0.18 - 0.35 * phase))) / (2 * np.pi)
        blob = np.exp(-((x - 0.62 - 0.08 * np.sin(2 * np.pi * phase)) ** 2 / 0.018 + periodic_distance**2 / 0.012))
        wave = 0.16 * np.sin(2 * np.pi * (3 * z - phase)) * np.exp(-((x - 0.5) ** 2) / 0.2)
        density.append(1.0 + blob + wave + 0.005 * rng.standard_normal((nx, nz)))
        potential.append(0.2 * np.cos(2 * np.pi * (z - phase)) * (x - 0.5))
    ne = np.asarray(density, dtype=np.float32)
    phi = np.asarray(potential, dtype=np.float32)
    field_map = {
        "density": ne,
        "potential": phi,
        "electron_temperature": (0.8 + 0.15 * ne).astype(np.float32),
        "electron_pressure": (ne * (0.8 + 0.15 * ne)).astype(np.float32),
        "parallel_ion_velocity": np.gradient(phi, axis=2).astype(np.float32),
    }
    r = np.linspace(0.75, 1.15, nx + 4)[:, None] * np.ones((1, 8))
    zz = np.ones((nx + 4, 1)) * np.linspace(-0.1, 0.1, 8)[None, :]
    geometry = {
        "Rxy": r,
        "Zxy": zz,
        "Bxy": np.ones_like(r),
        "Bpxy": np.full_like(r, 0.08),
        "Btxy": np.ones_like(r),
        "psixy": np.linspace(0.9, 1.1, nx + 4)[:, None] * np.ones((1, 8)),
        "zShift": np.zeros_like(r),
        "dx": np.ones_like(r),
        "dy": np.ones_like(r),
        "J": np.ones_like(r),
        "g11": np.ones_like(r),
        "g22": np.ones_like(r),
        "g33": np.ones_like(r),
        "ixseps1": np.asarray(2 + nx // 2),
        "ixseps2": np.asarray(nx + 4),
        "jyseps1_1": np.asarray(1),
        "jyseps1_2": np.asarray(3),
        "jyseps2_1": np.asarray(3),
        "jyseps2_2": np.asarray(6),
        "psi_axis": np.asarray(0.0),
        "psi_bdry": np.asarray(1.0),
    }
    return FixedPlaneData(
        shot="synthetic",
        frame_start=0,
        frame_stop=frames,
        fields=field_map,
        time=np.arange(frames, dtype=np.float64),
        geometry=geometry,
        metadata={"plane_y": 4, "dt_norm": 1.0, "omega_ci": 1.0, "synthetic": True},
    )

