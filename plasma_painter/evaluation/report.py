"""Write the concise preregistered pilot report from machine-readable artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasma_painter.config import artifact_root, load_config


def write_report(config: dict) -> Path:
    root = artifact_root(config)
    evaluation = json.loads((root / "evaluation" / "evaluation.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "rewards" / "audit.json").read_text(encoding="utf-8"))
    baseline = evaluation["methods"]["deterministic renderer"]
    fidelity = baseline["clip_level_bootstrap_95"]["fidelity"]
    temporal = baseline["clip_level_bootstrap_95"]["temporal"]
    text = f"""# Plasma painter pilot report

Status: **REVISE — this is a pre-training systems pilot, not an RL result.**

## Frozen protocol

- Source: TCV shot 85604, fixed outboard-midplane plane y=18.
- Evaluation: {evaluation['clip_count']} non-overlapping eight-frame clips inside art_test [384, 432), with {len(evaluation['seeds'])} matched seeds.
- Statistical unit: clip; individual frames were not treated as independent.
- The prohibited held-out shot was blocked by identifier at the loader boundary and no file for it was opened.

## Current quantitative result

The deterministic editable JavaScript renderer compiled at {baseline['program_compilation_rate']:.1%} and rendered validly at {baseline['valid_render_rate']:.1%}. Its clip-level fidelity mean was {fidelity['mean']:.3f} with a 95% bootstrap interval [{fidelity['low']:.3f}, {fidelity['high']:.3f}]. Temporal score was {temporal['mean']:.3f} [{temporal['low']:.3f}, {temporal['high']:.3f}]. Median low-resolution evaluation render time was {baseline['median_render_ms_low_resolution_evaluation']:.1f} ms per eight-frame clip.

These scores establish an implementation reference only. They do not demonstrate aesthetic improvement.

## Reward audit

The audit evaluated {audit['program_count']} programs. All {audit['known_bad_count']} blank, noise-only, static, fixed-grid, coordinate-scrambled, density-only, and independent-ablation cases failed the uncompensated gate. The high/low contact sheets were manually inspected. High scorers visibly retained the source contour bands and filament blooms; low scorers were visibly blank, noisy, static, transposed, or structure-incomplete.

Signals above an absolute correlation of 0.9 are listed in the machine audit. Filament and orientation are retained because independent ablations show that they test different failure modes; the candidate pool must be broadened before a meaningful RL run. Human aesthetics are absent, and no VLM proxy was substituted.

## Method comparison

| Method | Status |
|---|---|
| Deterministic renderer | Evaluated |
| Prompted base | Not run; local weights not staged |
| Optimized prompt | Prompt frozen; model sampling not run |
| SFT adapter | Train-only dataset and entrypoint ready; not trained |
| DPO adapter | Objective smoke passed; zero human pairs |
| GRPO adapter | 25-step categorical environment smoke passed; LM adapter not trained |

## Decision

**REVISE.** Do not advance to the controlled pilot until local base weights are staged, at least 80% of actual generated/repaired programs render, the human reference pool contains enough `love` examples, and SFT plus DPO precede the bounded online run. The preregistered success criteria remain unchanged.
"""
    target = Path("docs/plasma_painter_pilot_report.md")
    target.write_text(text, encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True); args = parser.parse_args()
    print(write_report(load_config(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
