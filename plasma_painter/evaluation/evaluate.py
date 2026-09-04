"""Evaluate the deterministic baseline and declare unavailable learned methods honestly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from plasma_painter.config import artifact_root, git_state, load_config, stable_hash, write_json
from plasma_painter.renderer.compiler import validate_program
from plasma_painter.provenance import experiment_provenance
from plasma_painter.training.environment import evaluate_program

from .bootstrap import bootstrap_mean_interval


METHOD_ORDER = [
    "deterministic renderer",
    "prompted base model",
    "optimized-prompt base model",
    "SFT adapter",
    "DPO adapter",
    "GRPO adapter",
]


def evaluate(config: dict) -> dict:
    started = time.perf_counter()
    root = artifact_root(config)
    index = json.loads((root / "features" / "index.json").read_text(encoding="utf-8"))
    clips = [json.loads(Path(item["path"]).read_text(encoding="utf-8")) for item in index["clips"] if item["split"] == "art_test"]
    code = Path(config["renderer"]["baseline_program"]).read_text(encoding="utf-8")
    validation = validate_program(code)
    seeds = [int(config["project"]["seed"]) + offset for offset in range(int(config["evaluation"]["seed_count"]))]
    records = []
    for clip in clips:
        for seed in seeds:
            result = evaluate_program(code, clip, config, seed=seed, scale=0.125)
            records.append({
                "clip_id": clip["clip_id"], "seed": seed, "valid": result["valid"],
                "render_ms": result.get("render_ms"), "sandbox_ms": result.get("sandbox_ms"),
                "operation_count": sum(result.get("operation_counts", [])),
                "fidelity": result.get("fidelity", {}), "temporal": result.get("temporal", {}),
                "diversity": result.get("diversity", {}), "efficiency": result.get("efficiency", {}),
                "reward": result["aggregate"]["reward"], "gate": result.get("gate"),
            })
    clip_groups = {}
    for record in records:
        clip_groups.setdefault(record["clip_id"], []).append(record)
    clip_metrics = []
    for clip_id, group in clip_groups.items():
        clip_metrics.append({
            "clip_id": clip_id,
            "valid_rate": sum(item["valid"] for item in group) / len(group),
            "fidelity": statistics.mean(item["fidelity"].get("score", 0) for item in group),
            "coarse_spearman": statistics.mean(item["fidelity"].get("coarse_spearman_raw", -1) for item in group),
            "contour": statistics.mean(item["fidelity"].get("contour", 0) for item in group),
            "extrema": statistics.mean(item["fidelity"].get("extrema", 0) for item in group),
            "filament": statistics.mean(item["fidelity"].get("filament", 0) for item in group),
            "orientation": statistics.mean(item["fidelity"].get("orientation", 0) for item in group),
            "temporal": statistics.mean(item["temporal"].get("score", 0) for item in group),
            "render_ms": statistics.mean(item["render_ms"] for item in group),
        })
    intervals = {name: bootstrap_mean_interval([item[name] for item in clip_metrics], samples=int(config["evaluation"]["bootstrap_samples"]), seed=int(config["project"]["seed"])) for name in ("valid_rate", "fidelity", "coarse_spearman", "contour", "extrema", "filament", "orientation", "temporal", "render_ms")}
    deterministic = {
        "status": "evaluated",
        "program_hash": stable_hash(code),
        "program_compilation_rate": 1.0 if validation.valid else 0.0,
        "valid_render_rate": sum(item["valid"] for item in records) / len(records),
        "runtime_failure_rate": sum(not item["valid"] for item in records) / len(records),
        "median_render_ms_low_resolution_evaluation": statistics.median(item["render_ms"] for item in records),
        "median_code_bytes": len(code.encode("utf-8")),
        "median_operation_count_per_clip": statistics.median(item["operation_count"] for item in records),
        "human_preference": None,
        "calibrated_judge_preference": None,
        "style_diversity_proxy": statistics.mean(item["diversity"].get("score", 0) for item in records),
        "clip_level_bootstrap_95": intervals,
        "records": records,
    }
    unavailable = {
        "prompted base model": "base weights not staged locally; zero external API use preserved",
        "optimized-prompt base model": "base weights not staged locally; prompt is frozen and ready",
        "SFT adapter": "entrypoint and train-only dataset ready; no GPU model update run",
        "DPO adapter": "no human chosen/rejected pairs and no SFT adapter",
        "GRPO adapter": "online LM path ready; only categorical policy/environment smoke run",
    }
    methods = {"deterministic renderer": deterministic}
    methods.update({name: {"status": "not_evaluated", "reason": reason} for name, reason in unavailable.items()})
    report = {
        "evaluation_status": "partial_pretraining_baseline",
        "shot": "85604", "split": "art_test", "clip_count": len(clips), "frames_per_clip": clips[0]["frame_count"],
        "seeds": seeds, "method_order": METHOD_ORDER, "methods": methods, "git": git_state(),
        "statistical_unit": "clip averaged over matched generation seeds; frames are not independent samples",
        "success_criteria_applied": {"pilot_pass": False, "recommendation": "REVISE", "reason": "no prompted-model, preference, or adapter comparison exists yet"},
    }
    report["provenance"] = experiment_provenance(config, wall_seconds=time.perf_counter() - started, stage="evaluation")
    write_json(root / "evaluation" / "evaluation.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True); args = parser.parse_args()
    report = evaluate(load_config(args.config)); baseline = report["methods"]["deterministic renderer"]
    print(json.dumps({"status": report["evaluation_status"], "clip_count": report["clip_count"], "seeds": report["seeds"], "valid_render_rate": baseline["valid_render_rate"], "fidelity": baseline["clip_level_bootstrap_95"]["fidelity"], "temporal": baseline["clip_level_bootstrap_95"]["temporal"], "recommendation": report["success_criteria_applied"]["recommendation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
