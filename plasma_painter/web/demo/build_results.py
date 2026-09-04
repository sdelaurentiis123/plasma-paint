"""Publish a compact, local-only view of recorded results and paired frames."""
import json
from pathlib import Path

from PIL import Image
import yaml

from plasma_painter.config import artifact_root, write_json
from plasma_painter.renderer.canvas_runtime import CanvasRuntime
from plasma_painter.renderer.sandbox import run_program


def build_results(config: dict) -> dict:
    root = artifact_root(config)
    static = Path(__file__).with_name("static")
    index = json.loads((root / "renders/deterministic/index.json").read_text())
    evaluation = json.loads((root / "evaluation/evaluation.json").read_text())
    clips = []
    code = Path(config["renderer"]["baseline_program"]).read_text()
    style = yaml.safe_load(Path(config["renderer"]["style_config"]).read_text())
    feature_index = json.loads((root / "features/index.json").read_text())
    for record in index["renders"]:
        directory = Path(record["still"]).parent
        feature_record = next(item for item in feature_index["clips"] if item["clip_id"] == record["clip_id"])
        features = json.loads(Path(feature_record["path"]).read_text())["frames"]
        limits = config["renderer"]
        sandbox = run_program(code, features, style=style, seed=record["seed"],
                              max_runtime_ms=int(limits["max_runtime_ms"]),
                              max_operations=int(limits["max_operations"]),
                              max_path_points=int(limits["max_path_points"]))
        if not sandbox.valid:
            raise RuntimeError(sandbox.error or "Baseline program failed")
        runtime = CanvasRuntime(int(limits["width"]), int(limits["height"]), style, record["seed"])
        paintings = runtime.render_clip(features, sandbox.operations_by_frame)
        frames = []
        for offset, source_index in enumerate(record["frames"]):
            pair = {"index": source_index}
            for key, prefix in (("painting", "frame"), ("scientific", "scientific")):
                name = f"results-{record['clip_id']}-{prefix}-{offset:03d}.generated.webp"
                if key == "painting":
                    paintings[offset].convert("RGB").save(static / name, "WEBP", quality=88)
                else:
                    with Image.open(directory / f"{prefix}-{offset:03d}.png") as image:
                        image.convert("RGB").save(static / name, "WEBP", quality=88)
                pair[key] = name
            frames.append(pair)
        clips.append({"id": record["clip_id"], "split": record["split"],
                      "seed": record["seed"], "frames": frames})
    baseline = evaluation["methods"]["deterministic renderer"]
    result = {
        "shot": evaluation["shot"], "clips": clips,
        "metrics": baseline["clip_level_bootstrap_95"],
        "valid_rate": baseline["valid_render_rate"],
        "median_render_ms": baseline["median_render_ms_low_resolution_evaluation"],
        "methods": [{"name": name, "status": value["status"]}
                    for name, value in evaluation["methods"].items()],
        "evaluation_commit": evaluation["git"],
        "program_hash": baseline["program_hash"],
        "evaluation_clips": evaluation["clip_count"], "seeds": evaluation["seeds"],
    }
    write_json(static / "results.generated.json", result)
    return result
