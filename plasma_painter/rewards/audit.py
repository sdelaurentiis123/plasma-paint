"""Audit reward variance, correlations, failure cases, and obvious hacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from plasma_painter.config import artifact_root, load_config, stable_hash, write_json
from plasma_painter.provenance import experiment_provenance
from plasma_painter.rewards.normalization import fit_normalization
from plasma_painter.training.environment import evaluate_program


ATTACKS = {
    "blank": """export function createPainter(api, styleConfig) { return { reset(seed) { api.reset(seed); }, renderFrame(frameFeatures, time, persistentState) { api.createPaper({color: styleConfig.paper || '#f2ede2', grain: 0.04}); api.scatterGrain({amount: 0.04, source: 'paper_only'}); } }; }""",
    "static": """export function createPainter(api, styleConfig) { return { reset(seed) { api.reset(seed); }, renderFrame(frameFeatures, time, persistentState) { api.createPaper({color: '#f2ede2', grain: 0.02}); api.strokePath({points: [[0.1,0.5],[0.9,0.5]], width: 0.02, opacity: 0.7, pigment: 'edge', source: 'fixed_not_data'}); } }; }""",
    "data_independent": """export function createPainter(api, styleConfig) { return { reset(seed) { api.reset(seed); }, renderFrame(frameFeatures, time, persistentState) { api.createPaper({color: '#f2ede2', grain: 0.02}); for (const z of [0.1,0.25,0.4,0.55,0.7,0.85]) { api.strokePath({points: [[0.05,z],[0.95,z]], width: 0.012, opacity: 0.55, pigment: 'positive', source: 'fixed_grid'}); } } }; }""",
    "noise_only": """export function createPainter(api, styleConfig) { return { reset(seed) { api.reset(seed); }, renderFrame(frameFeatures, time, persistentState) { api.createPaper({color: '#f2ede2', grain: 0.15}); api.scatterGrain({amount: 0.15, source: 'decorative_noise'}); } }; }""",
    "scrambled_coordinates": """export function createPainter(api, styleConfig) { return { reset(seed) { api.reset(seed); }, renderFrame(frameFeatures, time, persistentState) { api.createPaper({color: '#f2ede2', grain: 0.02}); for (const contour of frameFeatures.contours) { api.strokePath({points: contour.points.map((p) => [p[1],p[0]]), width: 0.008, opacity: 0.5, pigment: 'positive', source: 'scrambled_contour'}); } } }; }""",
    "density_only": """export function createPainter(api, styleConfig) { return { reset(seed) { api.reset(seed); }, renderFrame(frameFeatures, time, persistentState) { api.createPaper({color: '#f2ede2', grain: 0.02}); api.washRegion({raster: 'density', opacity: 0.4, bleed: 0.4, source: 'density'}); } }; }""",
}


def _contact_sheet(items: list[dict], target: Path, title: str) -> None:
    thumb_size = (240, 160)
    columns = 4
    rows = int(np.ceil(len(items) / columns))
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + 30) + 34), "#e9e5dc")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="#252822")
    for index, item in enumerate(items):
        image = Image.open(item["thumbnail"]).convert("RGB").resize(thumb_size, Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_size[0]
        y = 34 + (index // columns) * (thumb_size[1] + 30)
        sheet.paste(image, (x, y))
        draw.text((x + 5, y + thumb_size[1] + 5), f"{item['label'][:19]}  R={item['reward']:.3f}", fill="#252822")
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, optimize=True)


def run_audit(config: dict) -> dict:
    started = time.perf_counter()
    root = artifact_root(config)
    feature_index = json.loads((root / "features" / "index.json").read_text(encoding="utf-8"))
    item = next(record for record in feature_index["clips"] if record["split"] == "art_train")
    clip = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
    clip = {**clip, "frames": clip["frames"][:2], "frame_count": 2, "frame_stop": clip["frame_start"] + 2}
    candidates = [record for record in json.loads((root / "programs" / "filter_report.json").read_text(encoding="utf-8"))["results"] if record["accepted"]]
    programs = [(f"fixture-{index:02d}", Path(record["program_path"]).read_text(encoding="utf-8"), "reference_fixture") for index, record in enumerate(candidates[:24])]
    for repeat in range(3):
        for name, code in ATTACKS.items():
            programs.append((f"attack-{name}-{repeat}", code + f"\n// audit-variant-{repeat}", "known_bad"))
    # Independent ablations prevent filament and orientation signals from
    # appearing redundant merely because all gross attacks omit both.
    reference = programs[0][1]
    for repeat in range(3):
        programs.append((f"ablation-no-orientation-{repeat}", reference.replace("api.dryBrushPath", "api.strokePath") + f"\n// no-orientation-{repeat}", "known_bad"))
        programs.append((f"ablation-no-filament-dabs-{repeat}", reference.replace("api.dab", "api.poolPigment") + f"\n// no-filament-{repeat}", "known_bad"))
    records = []
    thumbnails = root / "rewards" / "thumbnails"
    for index, (label, code, kind) in enumerate(programs):
        thumbnail = thumbnails / f"{index:03d}-{stable_hash(code)[:10]}.png"
        result = evaluate_program(code, clip, config, seed=int(config["project"]["seed"]) + index, scale=0.125, capture_path=thumbnail)
        fidelity = result.get("fidelity", {})
        temporal = result.get("temporal", {})
        diversity = result.get("diversity", {})
        efficiency = result.get("efficiency", {})
        records.append({
            "label": label, "kind": kind, "program_hash": stable_hash(code), "valid": result["valid"],
            "failed": result.get("gate", {}).get("failed", []), "reward": result["aggregate"]["reward"],
            "fidelity": fidelity.get("score", 0.0), "coarse_intensity": fidelity.get("coarse_intensity", 0.0),
            "coarse_spearman_raw": fidelity.get("coarse_spearman_raw", -1.0), "contour": fidelity.get("contour", 0.0),
            "extrema": fidelity.get("extrema", 0.0), "filament": fidelity.get("filament", 0.0),
            "orientation": fidelity.get("orientation", 0.0), "temporal": temporal.get("score", 0.0),
            "diversity": diversity.get("score", 0.0), "efficiency": efficiency.get("score", 0.0),
            "thumbnail": str(thumbnail.resolve()),
        })
    calibration = [{key: record[key] for key in ("fidelity", "temporal", "diversity", "efficiency")} for record in records if record["kind"] == "reference_fixture"]
    normalization = fit_normalization(calibration)
    write_json(root / "rewards" / "normalization.json", normalization)
    names = ["fidelity", "coarse_intensity", "contour", "extrema", "filament", "orientation", "temporal", "diversity", "efficiency"]
    matrix = np.asarray([[record[name] for name in names] for record in records], dtype=float)
    correlation = np.corrcoef(matrix, rowvar=False)
    redundant = []
    for row in range(len(names)):
        for column in range(row + 1, len(names)):
            if np.isfinite(correlation[row, column]) and abs(correlation[row, column]) > 0.9:
                redundant.append({"left": names[row], "right": names[column], "correlation": float(correlation[row, column])})
    figure, axes = plt.subplots(3, 3, figsize=(11, 8))
    for axis, name in zip(axes.flat, names):
        axis.hist([record[name] for record in records], bins=12, color="#486d72", alpha=0.85); axis.set_title(name)
    figure.tight_layout(); distributions = root / "rewards" / "component_distributions.png"; figure.savefig(distributions, dpi=150); plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 8)); image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm"); axis.set_xticks(range(len(names)), names, rotation=45, ha="right"); axis.set_yticks(range(len(names)), names); figure.colorbar(image, ax=axis); figure.tight_layout(); correlations = root / "rewards" / "component_correlations.png"; figure.savefig(correlations, dpi=150); plt.close(figure)
    ordered = sorted(records, key=lambda record: record["reward"], reverse=True)
    top_path = root / "rewards" / "top20_contact_sheet.png"; low_path = root / "rewards" / "bottom20_contact_sheet.png"
    _contact_sheet(ordered[:20], top_path, "Top 20 reward-audit renders")
    _contact_sheet(list(reversed(ordered[-20:])), low_path, "Bottom 20 reward-audit renders")
    bad = [record for record in records if record["kind"] == "known_bad"]
    report = {
        "status": "pre_rl_audit_complete_pending_human_aesthetic_data", "clip_id": clip["clip_id"], "split": "art_train",
        "program_count": len(records), "valid_count": sum(record["valid"] for record in records),
        "known_bad_count": len(bad), "known_bad_rejected": sum(not record["valid"] for record in bad),
        "known_bad_rejection_fraction": float(np.mean([not record["valid"] for record in bad])),
        "component_variance": {name: float(np.var(matrix[:, index])) for index, name in enumerate(names)},
        "correlation_components": names, "correlation_matrix": correlation.tolist(), "above_abs_0_9": redundant,
        "records": records,
        "artifacts": {"distributions": str(distributions), "correlations": str(correlations), "top20": str(top_path), "bottom20": str(low_path), "normalization": str(root / "rewards" / "normalization.json")},
    }
    report["provenance"] = experiment_provenance(config, wall_seconds=time.perf_counter() - started, stage="reward_audit")
    write_json(root / "rewards" / "audit.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True); args = parser.parse_args()
    report = run_audit(load_config(args.config))
    print(json.dumps({key: report[key] for key in ("status", "program_count", "valid_count", "known_bad_count", "known_bad_rejected", "known_bad_rejection_fraction", "above_abs_0_9")}, indent=2))
    return 0 if report["known_bad_rejection_fraction"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
