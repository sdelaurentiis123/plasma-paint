"""Build the immutable dataset and split manifest for the art pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from plasma_painter.config import artifact_root, git_state, load_config, sha256_file, write_json
from plasma_painter.data.loaders import inspect_npz_headers
from plasma_painter.data.schema import AXIS_CONVENTION, SCHEMA_VERSION
from plasma_painter.data.splits import (
    build_clips,
    parse_art_intervals,
    validate_nested_split,
    validate_shot,
)


def build_manifest(config: dict) -> dict:
    data = config["data"]
    shot = str(data["shot"])
    source = Path(data["source_path"])
    geometry = Path(data["geometry_path"])
    validate_shot(shot, str(source), str(geometry))
    if not source.is_file() or not geometry.is_file():
        raise FileNotFoundError("permitted source files are not present")
    source_digest = sha256_file(source)
    geometry_digest = sha256_file(geometry)
    if source_digest != data["source_sha256"]:
        raise ValueError("source digest differs from frozen configuration")
    if geometry_digest != data["geometry_sha256"]:
        raise ValueError("geometry digest differs from frozen configuration")
    intervals = parse_art_intervals(data["art_split"])
    validate_nested_split(intervals, data["forecasting_split"]["train"])
    clips = build_clips(
        shot,
        intervals,
        length=int(data["clip_length"]),
        stride=int(data["clip_stride"]),
        max_train_clips=int(data["max_train_clips"]),
    )
    headers = inspect_npz_headers(source)
    variables = {}
    for semantic, key in data["field_keys"].items():
        header = headers[key]
        variables[semantic] = {"archive_key": key, **asdict(header)}
    time_header = headers["time"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "config_path": config["_config_path"],
        "source": {
            "shot": shot,
            "path": str(source),
            "sha256": source_digest,
            "size_bytes": source.stat().st_size,
            "geometry_path": str(geometry),
            "geometry_sha256": geometry_digest,
            "geometry_size_bytes": geometry.stat().st_size,
            "upstream_protocol_commit": data["upstream_protocol_commit"],
            "preprocessing_script_sha256": data["preprocessing_script_sha256"],
        },
        "variables": variables,
        "time": asdict(time_header),
        "field_units": {
            "density": "simulation units; upstream conversion 1e19 m^-3",
            "potential": "simulation units; upstream conversion 50 V",
            "electron_temperature": "simulation units; upstream conversion 50 eV",
            "electron_pressure": "simulation units",
            "parallel_ion_velocity": "simulation units; parallel direction only",
        },
        "coordinates": {**AXIS_CONVENTION, **data["coordinates"], "plane_y": data["plane_y"]},
        "forecasting_split": data["forecasting_split"],
        "art_split": data["art_split"],
        "normalization_fit_interval": data["art_split"]["art_train"],
        "clips": [asdict(clip) for clip in clips],
        "clip_policy": {
            "length": data["clip_length"],
            "stride": data["clip_stride"],
            "overlap_across_roles": False,
            "cross_boundary_clips": False,
            "independence_unit": "clip_program_pair; frames within clips are not independent",
        },
        "isolation": {
            "source_prefix_stop": data["forecasting_split"]["train"][1],
            "npz_member_policy": "read byte prefix only; never materialize later frame payload",
            "external_transmission": False,
            "sequestered_shot_opened": False,
        },
        "normalization": None,
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = build_manifest(config)
    output = artifact_root(config) / "manifests" / "dataset_manifest.json"
    write_json(output, manifest)
    print(json.dumps({"manifest": str(output), "clips": len(manifest["clips"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

