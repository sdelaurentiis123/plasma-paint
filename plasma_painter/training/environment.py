"""Renderer-program environment shared by evaluation and online RL."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from plasma_painter.config import artifact_root
from plasma_painter.renderer.canvas_runtime import CanvasRuntime
from plasma_painter.renderer.compiler import validate_program
from plasma_painter.renderer.sandbox import run_program
from plasma_painter.rewards.aggregate import aggregate_reward
from plasma_painter.rewards.diversity import operation_diversity
from plasma_painter.rewards.efficiency import efficiency_score
from plasma_painter.rewards.fidelity import fidelity_clip
from plasma_painter.rewards.temporal import temporal_clip
from plasma_painter.rewards.validity import validity_gate


def evaluate_program(
    code: str,
    clip: dict[str, Any],
    config: dict[str, Any],
    *,
    seed: int,
    human_aesthetic: float | None = None,
    scale: float = 0.25,
    capture_path: str | Path | None = None,
) -> dict[str, Any]:
    if config['renderer'].get('profile') == 'stroke_only':
        raise ValueError('Legacy RL fidelity metrics do not support finite marks. Use evaluation.finite_probe for diagnostics; calibrate a finite-mark fidelity gate before RL.')
    reward_config = yaml.safe_load(Path("configs/plasma_painter/rewards.yaml").read_text(encoding="utf-8"))
    style = yaml.safe_load(Path(config["renderer"]["style_config"]).read_text(encoding="utf-8"))
    validation = validate_program(code)
    sandbox = run_program(
        code,
        clip["frames"],
        style=style,
        seed=seed,
        max_runtime_ms=int(config["renderer"]["max_runtime_ms"]),
        max_operations=int(config["renderer"]["max_operations"]),
        max_path_points=int(config["renderer"]["max_path_points"]),
    )
    if not sandbox.valid:
        gate = {"valid": False, "checks": {"compiles": validation.valid, "sandbox_valid": False}, "failed": ["sandbox_valid"]}
        aggregate = aggregate_reward(
            gate=gate,
            fidelity=0,
            aesthetic=human_aesthetic,
            temporal=0,
            diversity=0,
            efficiency=0,
            weights=reward_config["weights"],
            strong_negative=reward_config["gate"]["strong_negative"],
        )
        return {"valid": False, "error": sandbox.error, "validation": sandbox.validation, "aggregate": aggregate}
    width = max(96, round(int(config["renderer"]["width"]) * scale))
    height = max(64, round(int(config["renderer"]["height"]) * scale))
    started = time.perf_counter()
    runtime = CanvasRuntime(width, height, style, seed)
    images = runtime.render_clip(clip["frames"], sandbox.operations_by_frame)
    if capture_path is not None:
        target = Path(capture_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(target, optimize=True)
    render_ms = (time.perf_counter() - started) * 1000
    fidelity = fidelity_clip(clip["frames"], images, sandbox.operations_by_frame, paper=style["paper"])
    temporal = temporal_clip(clip["frames"], images)
    diversity = operation_diversity(sandbox.operations_by_frame)
    efficiency = efficiency_score(
        code_bytes=len(code.encode("utf-8")),
        operation_count=float(np.mean([len(items) for items in sandbox.operations_by_frame])),
        runtime_ms=render_ms + float(sandbox.elapsed_ms or 0),
    )
    fidelity["static_failure"] = temporal["static_failure"]
    gate_config = reward_config["gate"]
    gate = validity_gate(
        compiles=validation.valid,
        sandbox_valid=sandbox.valid,
        fidelity=fidelity,
        minimum_nonempty_fraction=float(gate_config["minimum_nonempty_fraction"]),
        minimum_coarse_spearman=float(gate_config["minimum_coarse_spearman"]),
        minimum_extrema_recall=float(gate_config["minimum_extrema_recall"]),
        minimum_filament_correspondence=float(gate_config["minimum_filament_correspondence"]),
        minimum_orientation_agreement=float(gate_config["minimum_orientation_agreement"]),
    )
    normalization_path = artifact_root(config) / "rewards" / "normalization.json"
    normalization = None
    aggregate_components = {
        "fidelity": fidelity["score"],
        "temporal": temporal["score"],
        "diversity": diversity["score"],
        "efficiency": efficiency["score"],
    }
    if normalization_path.exists():
        from plasma_painter.rewards.normalization import apply_normalization

        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        aggregate_components = apply_normalization(aggregate_components, normalization)
    aggregate = aggregate_reward(
        gate=gate,
        fidelity=aggregate_components["fidelity"],
        aesthetic=human_aesthetic,
        temporal=aggregate_components["temporal"],
        diversity=aggregate_components["diversity"],
        efficiency=aggregate_components["efficiency"],
        weights=reward_config["weights"],
        strong_negative=reward_config["gate"]["strong_negative"],
    )
    return {
        "valid": gate["valid"],
        "error": None,
        "validation": sandbox.validation,
        "sandbox_ms": sandbox.elapsed_ms,
        "render_ms": render_ms,
        "operation_counts": [len(items) for items in sandbox.operations_by_frame],
        "fidelity": fidelity,
        "temporal": temporal,
        "diversity": diversity,
        "efficiency": efficiency,
        "gate": gate,
        "aggregate": aggregate,
        "reward_normalization": "training_reference_quantiles" if normalization is not None else "uncalibrated_unit_components",
    }
