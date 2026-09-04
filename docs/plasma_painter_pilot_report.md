# Plasma painter pilot report

Status: **REVISE — this is a pre-training systems pilot, not an RL result.**

## Frozen protocol

- Source: TCV shot 85604, fixed outboard-midplane plane y=18.
- Evaluation: 6 non-overlapping eight-frame clips inside art_test [384, 432), with 3 matched seeds.
- Statistical unit: clip; individual frames were not treated as independent.
- The prohibited held-out shot was blocked by identifier at the loader boundary and no file for it was opened.

## Current quantitative result

The deterministic editable JavaScript renderer compiled at 100.0% and rendered validly at 100.0%. Its clip-level fidelity mean was 0.681 with a 95% bootstrap interval [0.673, 0.689]. Temporal score was 0.626 [0.522, 0.730]. Median low-resolution evaluation render time was 298.9 ms per eight-frame clip.

These scores establish an implementation reference only. They do not demonstrate aesthetic improvement.

## Reward audit

The audit evaluated 48 programs. All 24 blank, noise-only, static, fixed-grid, coordinate-scrambled, density-only, and independent-ablation cases failed the uncompensated gate. The high/low contact sheets were manually inspected. High scorers visibly retained the source contour bands and filament blooms; low scorers were visibly blank, noisy, static, transposed, or structure-incomplete.

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
