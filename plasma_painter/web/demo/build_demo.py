"""Build a local demo bundle from one permitted art-training clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from plasma_painter.config import artifact_root, load_config, stable_hash, write_json


def _compact(frame: dict) -> dict:
    return {
        "schema_version": frame["schema_version"], "source": frame["source"], "time": frame["time"],
        "geometry": {key: frame["geometry"][key] for key in ("array_axes", "periodic_axis", "separatrix_face_u", "valid_domain")},
        "rasters": frame["rasters"], "contours": frame["contours"],
        "vectors": {"density_gradient": frame["vectors"]["density_gradient"], "exb_available": False},
        "filaments": frame["filaments"], "events": frame["events"], "transport": frame["transport"], "summary": frame["summary"],
    }


def build(config: dict) -> dict:
    root = artifact_root(config)
    index = json.loads((root / "features" / "index.json").read_text(encoding="utf-8"))
    item = next(record for record in index["clips"] if record["split"] == "art_train")
    clip = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
    compact = {"clip_id": clip["clip_id"], "split": clip["split"], "seed": int(config["project"]["seed"]), "frames": [_compact(frame) for frame in clip["frames"]]}
    static = Path(__file__).with_name("static")
    static.mkdir(parents=True, exist_ok=True)
    features = static / "features.generated.json"
    features.write_text(json.dumps(compact, separators=(",", ":")) + "\n", encoding="utf-8")
    bootstrap = static / "bootstrap.generated.js"
    bootstrap.write_text("window.PLASMA_BOOTSTRAP=" + json.dumps({**compact, "frames": compact["frames"][:1]}, separators=(",", ":")) + ";\n", encoding="utf-8")
    fallback_source = root / "renders" / "deterministic" / clip["clip_id"] / f"seed-{config['project']['seed']}" / "frame-000.png"
    if not fallback_source.exists():
        raise FileNotFoundError("baseline fallback missing; run render_baseline first")
    fallback = static / "fallback.generated.webp"
    Image.open(fallback_source).convert("RGB").save(fallback, "WEBP", quality=84, method=6)
    program = Path(config["renderer"]["baseline_program"])
    painter = static / "painter.generated.js"
    painter.write_text(program.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {
        "status": "local_demo_deterministic_baseline_training_pending", "clip_id": clip["clip_id"], "split": "art_train",
        "shot": clip["frames"][0]["source"]["shot"], "frames": [frame["source"]["frame_index"] for frame in clip["frames"]],
        "seed": config["project"]["seed"], "program_hash": stable_hash(program.read_text(encoding="utf-8")),
        "feature_bytes": features.stat().st_size, "bootstrap_bytes": bootstrap.stat().st_size, "fallback_bytes": fallback.stat().st_size,
    }
    write_json(root / "web_demo" / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True); args = parser.parse_args()
    print(json.dumps(build(load_config(args.config)), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
