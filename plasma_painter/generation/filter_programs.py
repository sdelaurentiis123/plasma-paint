"""Validate candidate programs and execute them on a short training-only clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from plasma_painter.config import artifact_root, load_config, write_json
from plasma_painter.renderer.sandbox import run_program
from plasma_painter.renderer.canvas_runtime import CanvasRuntime


def filter_candidate(code: str, frames: list[dict[str, Any]], config: dict[str, Any], style: dict[str, Any]) -> dict[str, Any]:
    profile=config['renderer'].get('profile','legacy')
    if profile=='stroke_only':
        from plasma_painter.features.stroke_samples import with_stroke_samples
        frames=[with_stroke_samples(f) for f in frames]
    result = run_program(
        code,
        frames,
        style=style,
        seed=int(config["project"]["seed"]),
        max_runtime_ms=int(config["renderer"]["max_runtime_ms"]),
        max_operations=int(config["renderer"]["max_operations"]),
        max_path_points=int(config["renderer"]["max_path_points"]),
        profile=profile,
    )
    nonempty = bool(result.operations_by_frame) and all(bool(items) for items in result.operations_by_frame)
    pixel_valid = False
    render_error = None
    if result.valid and nonempty:
        try:
            runtime = CanvasRuntime(192, 128, style, int(config['project']['seed']))
            images = runtime.render_clip(frames, result.operations_by_frame)
            blank = CanvasRuntime(192, 128, style, int(config['project']['seed']))
            paper_ops = [[op for op in ops if op['op']=='createPaper'] for ops in result.operations_by_frame]
            papers = blank.render_clip(frames, paper_ops)
            pixel_valid = len(images)==len(frames) and all(
                float(np.mean(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float)))) > 0.5
                for a,b in zip(images,papers))
        except (ValueError, TypeError, KeyError, RuntimeError, OverflowError) as error:
            render_error = str(error)
    return {
        "accepted": bool(result.valid and nonempty and pixel_valid),
        "pixel_render_valid": pixel_valid,
        "render_error": render_error,
        "sandbox_valid": result.valid,
        "nonempty_operations": nonempty,
        "elapsed_ms": result.elapsed_ms,
        "error": result.error,
        "validation": result.validation,
        "operation_counts": [len(items) for items in result.operations_by_frame],
    }


def filter_directory(config: dict[str, Any]) -> dict[str, Any]:
    root = artifact_root(config)
    candidates_dir = root / "programs" / "candidates"
    feature_index = json.loads((root / "features" / "index.json").read_text(encoding="utf-8"))
    record = next(item for item in feature_index["clips"] if item["split"] == "art_train")
    frames = json.loads(Path(record["path"]).read_text(encoding="utf-8"))["frames"][:2]
    style = yaml.safe_load(Path(config["renderer"]["style_config"]).read_text(encoding="utf-8"))
    results = []
    for metadata_path in sorted(candidates_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        code_path = Path(metadata["program_path"])
        status = filter_candidate(code_path.read_text(encoding="utf-8"), frames, config, style)
        results.append({**metadata, **status})
    accepted = sum(item["accepted"] for item in results)
    report = {
        "candidate_count": len(results),
        "accepted_count": accepted,
        "accepted_fraction": accepted / len(results) if results else 0.0,
        "training_gate_80_percent": bool(results and accepted / len(results) >= 0.80),
        "results": results,
    }
    write_json(root / "programs" / "filter_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = filter_directory(load_config(args.config))
    print(json.dumps({key: report[key] for key in ("candidate_count", "accepted_count", "accepted_fraction", "training_gate_80_percent")}, indent=2))
    return 0 if report["training_gate_80_percent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
