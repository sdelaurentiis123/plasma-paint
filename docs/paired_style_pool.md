# Representative artist references and plasma targets

The user authorized representative existing library images as positive aesthetic
anchors on 2026-09-07. Five images were visually inspected and selected as
style-love references: Van Gogh 28560/14586, Seurat 27992, Manet 44892/81533.
Seurat 20199 is a looser painted study and remains okay/calibration-only for the
pointillist condition. These are selected examples, not comprehensive definitions
of each artist, nor claims of individual human ratings. All original library
records include public-domain evidence and verified image hashes.

Six scientific targets use existing 85604 art_train caches, frames 0/1 in sections
y=0/18/31. Normalization and preprocessing provenance are retained. No raw dataset,
guard, upstream held-out block, frame-7 comparison input or shot 85606 was opened.
Artist references carry style roles; plasma frames carry scientific roles. There
are 18 paired tasks and no fabricated chosen/rejected preferences.

Build with:

```
python -m plasma_painter.ratings.style_pool --output artifacts/plasma_painter/paired-style-pool-v1
```

The output is non-overwriting and Git-ignored. It contains references, scientific
PNGs, compact frame features and a machine-readable manifest. All content stays
local. Select verified style references in gym_rollout using:

```
--style-pool artifacts/plasma_painter/paired-style-pool-v1/manifest.json --style van-gogh
```

Other style keys are `seurat-pointillism` and `manet`. The native Verifiers taskset
also accepts style_pool/style configuration and supplies reference images before
scientific/current-canvas images. It retains its loopback inference guard.

This creates image-level conditioning and aesthetic reference targets, not
action-level demonstrations: paintings do not contain their generating tool
trajectories. Style references must not automatically become SFT action examples,
scientific content targets, or preferred generated plasma paintings. No new weight
update or RL reward is claimed here. Full fidelity must still gate any future
aesthetic reward, and a judge must be calibrated before it controls training.
