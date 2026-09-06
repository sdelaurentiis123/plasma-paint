# Bounded finite-mark refinement

Rusty job **6989073**, source **9653ed6**, dedicated worktree
`/mnt/home/sdelaurentiis/ceph/plasma-paint-runs/20260904-rusty/vision-refinement-v1`.
At submission: 47 tests passed. One frozen Qwen2.5-VL-7B-Instruct, unchanged pinned
weights and public reference images. Two artists (Van Gogh and Seurat), up to four
attempts each, 2,800 new tokens per attempt, one GPU with one-hour hard wall cap.
No optimizer steps, paid APIs, deployments, shared-environment edits or raw-data access.

Corrected prompts state the exact export/reset syntax and separate explanatory
statistics from actual runtime fields. The complete working example is explicitly
a starting point, not the answer. A render is not promoted merely because it runs:
the mark operation trace must differ from the reference even after ignoring colors.
Copies are retained and fed back as rendered drafts for another attempt. This exact
copy check detects neither meaningful aesthetic novelty nor scientific fidelity;
those require subsequent inspection and measurement.

Generation uses permitted old-85604 art_train frames 0–7. All previous runs remain
unchanged. New output directory: `artifacts/plasma_painter/vision-context` within the
dedicated worktree. No guard/split boundary is crossed and 85606 remains unopened.

## Local probes prepared while the job runs

`python -m plasma_painter.evaluation.finite_probe --program PROGRAM.js --clip TRAIN_CLIP.json --output NEW_REPORT.json`

This command accepts only compact art_train clip filenames, checks every frame,
uses three seeds, and measures reproducibility, frozen-input flicker, response to
evolving input, and coarse pigment correspondence to absolute density fluctuation.
It does **not** call these exploratory measures an RL reward or a fidelity pass.
Legacy reward evaluation now explicitly rejects the finite-mark profile: the old
scorer assumes washes/dabs/gradient strokes and cannot correctly score this API.

Reference-control results on training frames 0–7 and 64–71: all three seeds valid
and reproducible, zero frozen-input change, mean absolute-fluctuation/pigment rank
correlation approximately .834 and .868 respectively (Gaussian sigma two cells).
Frames 64–71 were not included in the generation context. These numbers describe
the hand-authored control, not a model result. The reference's signed-field encoding,
filament preservation and other critical scientific criteria are not established by
that single correlation. No scientific success margin has been redefined.
