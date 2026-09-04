"""Render the deterministic scientific-watercolor baseline on frozen clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
import yaml

from plasma_painter.config import artifact_root, load_config, stable_hash, write_json
from plasma_painter.provenance import experiment_provenance

from .compiler import validate_program
from .reference_renderers.scientific_watercolor import ScientificWatercolorRenderer


def render_clip(config: dict, clip_record: dict, *, seed: int) -> dict:
    root = artifact_root(config)
    clip_path = Path(clip_record["path"])
    clip = json.loads(clip_path.read_text(encoding="utf-8"))
    style_path = Path(config["renderer"]["style_config"])
    if not style_path.is_absolute():
        style_path = Path.cwd() / style_path
    style = yaml.safe_load(style_path.read_text(encoding="utf-8"))
    width = int(config["renderer"]["width"])
    height = int(config["renderer"]["height"])
    renderer = ScientificWatercolorRenderer(width, height, style, seed=seed)
    output_root = root / "renders" / "deterministic" / clip["clip_id"] / f"seed-{seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    frames = []
    scientific_frames = []
    operations = []
    for offset, frame in enumerate(clip["frames"]):
        image, summary = renderer.render_frame(frame, offset)
        frame_path = output_root / f"frame-{offset:03d}.png"
        image.save(frame_path, optimize=True)
        scientific = renderer.render_scientific(frame)
        scientific_path = output_root / f"scientific-{offset:03d}.png"
        scientific.save(scientific_path, optimize=True)
        frames.append(np.asarray(image))
        scientific_frames.append(np.asarray(scientific))
        operations.append(summary)
    gif_path = output_root / "clip.gif"
    scientific_gif_path = output_root / "scientific.gif"
    imageio.mimsave(gif_path, frames, duration=0.16, loop=0)
    imageio.mimsave(scientific_gif_path, scientific_frames, duration=0.16, loop=0)
    program_path = Path(config["renderer"]["baseline_program"])
    code = program_path.read_text(encoding="utf-8")
    validation = validate_program(code)
    if not validation.valid:
        raise RuntimeError("reference JavaScript painter failed validation: " + "; ".join(validation.errors))
    manifest = {
        "renderer": "deterministic_scientific_watercolor",
        "renderer_version": "0.1.0",
        "clip_id": clip["clip_id"],
        "split": clip["split"],
        "shot": clip["frames"][0]["source"]["shot"],
        "frames": [frame["source"]["frame_index"] for frame in clip["frames"]],
        "seed": seed,
        "program_path": str(program_path),
        "program_hash": stable_hash(code),
        "style": style["name"],
        "style_status": style["status"],
        "validation": validation.__dict__,
        "operations": operations,
        "still": str(output_root / "frame-000.png"),
        "clip": str(gif_path),
        "scientific_clip": str(scientific_gif_path),
    }
    write_json(output_root / "render_manifest.json", manifest)
    return manifest


def render_baselines(config: dict, splits: tuple[str, ...] = ("art_train", "art_val", "art_test")) -> dict:
    started = time.perf_counter()
    root = artifact_root(config)
    index = json.loads((root / "features" / "index.json").read_text(encoding="utf-8"))
    selected = []
    for split in splits:
        match = next(item for item in index["clips"] if item["split"] == split)
        selected.append(render_clip(config, match, seed=int(config["project"]["seed"])))
    result = {"renders": selected, "count": len(selected)}
    result["provenance"] = experiment_provenance(config, wall_seconds=time.perf_counter() - started, stage="deterministic_render")
    write_json(root / "renders" / "deterministic" / "index.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = render_baselines(load_config(args.config))
    print(json.dumps({"count": result["count"], "clips": [item["clip"] for item in result["renders"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
