# Reconstruction before artist imitation

The previous action-format adapter regressed in full episodes. This revision replaces
scattered mechanics examples with a greedy finite-stroke reconstruction teacher.
It is an optimization algorithm, NOT a language model, RL or a learned artist.

For each real normalized fluctuation frame, an explicit signed visualization target
is constructed. Two colors are selected from a quantized approved reference image;
pigment strength is sqrt(abs(2v-1)). These are reference-derived colors, not extracted
artist technique. Grid display remains radial x / field-aligned z. No reference
objects or artist composition are substituted for plasma structures.

The teacher proposes 24 candidate finite strokes per step (six media, two lengths,
two widths) near the largest residual error, oriented along the local scalar tangent.
It retains only a stroke that decreases actual rendered RGB mean-square error.
Maximum 128 accepted strokes. All geometry and brush parameters pass the same DSL
validation as the gym. Final full-render replay must match the incremental pixels.
This spatial search is a teacher algorithm; it does not constrain the LM's placement.

Training example construction replays accepted programs in four-stroke batches,
including library images, the scientific image and the current canvas. It uses the
same painting_prompt function as inference. Episodes allow 32 four-stroke turns;
earlier comparison runs used 24, so results are not directly comparable without
matching that budget. Every example records hashes of its image inputs.

Inputs are the previously approved paired pool: shot 85604, sections y=0/18/31,
frames 0/1, inside existing art_train. Frame 0 is reconstruction training; frame 1
is reserved for this curriculum's evaluation. No raw data or research held-outs
were opened. No previous adapter is loaded: the new LoRA starts from the same base.

Provisional demonstration filter, declared before case evaluation: coarse absolute
fluctuation/pigment correlation >=.75 and at least 70% RGB MSE reduction. This is
NOT the complete scientific fidelity gate and does not enable RL. Twelve cases
(Van Gogh- and Seurat-derived color targets) pass, with correlations .795–.878;
six Manet-derived cases fail (.371–.706) and produce no SFT examples. Failed targets
and outputs remain available for inspection. No thresholds were relaxed to admit them.

The 40-step LoRA smoke run is supervised spatial reconstruction practice, not
artist-preference optimization. Three before/after action-format probes remain too
small to establish quality. Full matched episodes and nontrivial scientific checks
are still required after training. No automated aesthetic reward is used.

Commands:

```
python -m plasma_painter.training.reconstruction_data --pool artifacts/plasma_painter/paired-style-pool-v1/manifest.json --output artifacts/plasma_painter/reconstruction-curriculum-v1
sbatch scripts/rusty_reconstruction_sft.slurm
```

One GPU, one-hour hard cap, 40 steps, no sweep. Expected wall time roughly 10–25
minutes from prior runs; the hard allocation cap is the controlling bound. Actual
completion and saved weights must be checked. Tests: 77 passed, optional Verifiers
module skipped in the default Python environment. No public website changes.
