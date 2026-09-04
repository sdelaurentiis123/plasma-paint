"""Render content-addressed, sandboxed painter programs into a reference pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import time

import imageio.v2 as imageio
import numpy as np
import yaml

from plasma_painter.config import artifact_root, load_config, stable_hash, write_json
from plasma_painter.provenance import experiment_provenance

from .canvas_runtime import CanvasRuntime
from .canvas_runtime.runtime import RUNTIME_VERSION
from .sandbox import run_program


def render_program_clip(
    code: str,
    clip: dict[str, Any],
    config: dict[str, Any],
    style: dict[str, Any],
    *,
    seed: int,
    checkpoint: str,
    origin: str,
) -> dict[str, Any]:
    limits = config["renderer"]
    sandbox = run_program(
        code,
        clip["frames"],
        style=style,
        seed=seed,
        max_runtime_ms=int(limits["max_runtime_ms"]),
        max_operations=int(limits["max_operations"]),
        max_path_points=int(limits["max_path_points"]),
    )
    if not sandbox.valid:
        raise RuntimeError(sandbox.error or "sandbox rejected program")
    runtime = CanvasRuntime(int(limits["width"]), int(limits["height"]), style, seed)
    images = runtime.render_clip(clip["frames"], sandbox.operations_by_frame)
    program_hash = stable_hash(code)
    output = artifact_root(config) / "renders" / "programs" / RUNTIME_VERSION / program_hash / clip["clip_id"] / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, image in enumerate(images):
        path = output / f"frame-{index:03d}.png"
        image.save(path, optimize=True)
        paths.append(str(path.resolve()))
    gif = output / "clip.gif"
    imageio.mimsave(gif, [np.asarray(image) for image in images], duration=0.16, loop=0)
    trace_path = output / "operations.json"
    write_json(trace_path, sandbox.operations_by_frame)
    record = {
        "reference_id": stable_hash({"program": program_hash, "clip": clip["clip_id"], "seed": seed}),
        "program_hash": program_hash,
        "program_path": "",
        "clip_id": clip["clip_id"],
        "field_id": "Ne_y18",
        "shot": clip["frames"][0]["source"]["shot"],
        "frame_indices": [frame["source"]["frame_index"] for frame in clip["frames"]],
        "split": clip["split"],
        "style_label": style["name"],
        "human_rating": None,
        "triage_bucket": "unrated",
        "model_checkpoint": checkpoint,
        "origin": origin,
        "seed": seed,
        "renderer_runtime_version": RUNTIME_VERSION,
        "sandbox_elapsed_ms": sandbox.elapsed_ms,
        "operation_count": sum(map(len, sandbox.operations_by_frame)),
        "still": paths[0],
        "frames": paths,
        "clip": str(gif.resolve()),
        "operations": str(trace_path.resolve()),
    }
    write_json(output / "render_manifest.json", record)
    return record


def build_reference_pool(config: dict[str, Any], *, limit: int = 6, clip_count: int = 2) -> dict[str, Any]:
    started = time.perf_counter()
    root = artifact_root(config)
    candidates = json.loads((root / "programs" / "filter_report.json").read_text(encoding="utf-8"))["results"]
    candidates = [item for item in candidates if item["accepted"]][:limit]
    index = json.loads((root / "features" / "index.json").read_text(encoding="utf-8"))
    clips = [item for item in index["clips"] if item["split"] == "art_train"][:clip_count]
    style = yaml.safe_load(Path(config["renderer"]["style_config"]).read_text(encoding="utf-8"))
    records = []
    for candidate in candidates:
        code = Path(candidate["program_path"]).read_text(encoding="utf-8")
        for clip_item in clips:
            clip = json.loads(Path(clip_item["path"]).read_text(encoding="utf-8"))
            record = render_program_clip(
                code,
                clip,
                config,
                style,
                seed=int(candidate["seed"]),
                checkpoint=candidate["checkpoint"],
                origin=candidate["origin"],
            )
            record["program_path"] = candidate["program_path"]
            records.append(record)
    for bucket in ("love", "okay", "reject"):
        (root / "reference_pool" / bucket).mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "awaiting_human_triage",
        "training_eligible_bucket": "love",
        "records": records,
        "count": len(records),
    }
    manifest["provenance"] = experiment_provenance(config, wall_seconds=time.perf_counter() - started, stage="reference_pool_render")
    write_json(root / "reference_pool" / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--clip-count", type=int, default=2)
    args = parser.parse_args()
    result = build_reference_pool(load_config(args.config), limit=args.limit, clip_count=args.clip_count)
    print(json.dumps({"count": result["count"], "status": result["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
