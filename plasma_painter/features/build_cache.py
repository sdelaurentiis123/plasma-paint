"""Build compact feature clips from the permitted real training prefix."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from plasma_painter.config import artifact_root, load_config, write_json
from plasma_painter.data.loaders import load_fixed_plane
from plasma_painter.data.splits import build_clips, parse_art_intervals, validate_nested_split

from .pipeline import FeaturePipeline


def build_cache(config: dict) -> dict:
    data_config = config["data"]
    intervals = parse_art_intervals(data_config["art_split"])
    validate_nested_split(intervals, data_config["forecasting_split"]["train"])
    clips = build_clips(
        str(data_config["shot"]),
        intervals,
        length=int(data_config["clip_length"]),
        stride=int(data_config["clip_stride"]),
        max_train_clips=int(data_config["max_train_clips"]),
    )
    permitted_stop = int(data_config["forecasting_split"]["train"][1])
    data = load_fixed_plane(
        data_config["source_path"],
        data_config["geometry_path"],
        shot=str(data_config["shot"]),
        field_keys=data_config["field_keys"],
        stop=permitted_stop,
        plane_y=int(data_config["plane_y"]),
    )
    pipeline = FeaturePipeline(config["features"])
    train_start, train_stop = map(int, data_config["art_split"]["art_train"])
    normalization = pipeline.fit(data, train_start, train_stop)
    root = artifact_root(config)
    clip_root = root / "features" / "clips"
    index_records = []
    for clip in clips:
        record = pipeline.transform_clip(data, clip.start, clip.stop, clip.clip_id, clip.split)
        output = clip_root / f"{clip.clip_id}.json"
        write_json(output, record)
        index_records.append({**asdict(clip), "path": str(output)})
    normalization_path = root / "features" / "normalization.json"
    write_json(normalization_path, normalization)
    index = {
        "shot": data.shot,
        "permitted_source_prefix": [0, permitted_stop],
        "normalization_path": str(normalization_path),
        "clips": index_records,
    }
    index_path = root / "features" / "index.json"
    write_json(index_path, index)
    manifest_path = root / "manifests" / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["normalization"] = normalization
        write_json(manifest_path, manifest)
    return {"index": str(index_path), "normalization": str(normalization_path), "clips": len(clips)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = build_cache(load_config(args.config))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

