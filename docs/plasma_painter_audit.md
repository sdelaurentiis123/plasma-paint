# Plasma painter repository audit

Audit date: 2026-09-04 (America/New_York)

## Starting state

The requested workspace was empty and was not a Git repository. A new Git
repository was initialized on branch `main`; the empty GitHub repository
`git@github-personal:sdelaurentiis123/plasma-paint.git` is configured as
`origin`. There was no local `AGENTS.md`, package manager, loader, split
manifest, visualization, website, or dependency lockfile to preserve.

The adjacent read-only forecasting repository
`/Users/stanislavdelaurentiis/tcv-diagnostics` is at commit
`43b281c6ce458c3142eca9a6a5f71bdd5cfc2ebc`. Its `AGENTS.md` and frozen
Paper-0 protocols govern the data isolation used here. No code or data in that
repository is modified by this project.

## Frozen source and split

The existing forecasting manifest
`paper0/manifests/post_ecrd_state_data_scaling_85604.json` freezes one
624-frame TCV/Hermes trajectory as:

| role | half-open frame interval |
|---|---:|
| forecasting training | `[0, 432)` |
| forecasting guard | `[432, 496)` |
| forecasting validation | `[496, 624)` |

The art experiment is nested wholly inside the forecasting training interval:

| role | half-open frame interval |
|---|---:|
| `art_train` | `[0, 288)` |
| guard | `[288, 320)` |
| `art_val` | `[320, 352)` |
| guard | `[352, 384)` |
| `art_test` | `[384, 432)` |

Clips are eight consecutive frames by default, are non-overlapping, and are
rejected if they cross any listed boundary. Normalization is fit only on
`art_train`.

## Permitted local data

The real-data source is the already extracted training/development artifact:

- path: `/Users/stanislavdelaurentiis/ben-filament/data/planes_old_85604.npz`
- SHA-256: `6c66e0c3309896832f6ef87017b943b933d01ef60fdbe0152af90965f8f85d5e`
- size: 734,842,970 bytes
- saved frame shape: `(624, 64, 81)` for each fixed-poloidal-plane field
- fields: `Ne`, `Te`, `phi`, `Pe`, `Vi`, `Ti`, `NVi`
- fixed poloidal indices: `0, 1, 16, 17, 18, 19, 30, 31`
- normalized cadence: 300; `Omega_ci = 95,788,333.03066081 s^-1`
- physical cadence: approximately 3.131905 microseconds
- extraction script SHA-256:
  `a9f0460a33b928240d6cc21ad9256c17396a452187e98b06b592bc9b680f0348`

Geometry comes from:

- path: `/Users/stanislavdelaurentiis/ben-filament/data/geom_85604.npz`
- SHA-256: `8d45aebc6fe0f45818562923fee6213806aca3339fabc2bba8002248ff25010b`
- size: 259,518 bytes
- geometry source hash recorded upstream:
  `0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8`

The painter uses the outboard-midplane row `y=18`, selected prospectively by
the upstream geometry audit as the maximum-major-radius point on the closed
separatrix. The displayed plane has logical axes `[x, z]`: `x` increases
radially; `z` is the periodic field-aligned/toroidal direction. It is not
treated as a Cartesian R-Z image. The native toroidal extent is one of five
periodic wedges (`zperiod=5`). The radial crop removes two guards on each side.
The first local scrape-off-layer cell is `x=16`, so the separatrix face lies
between local cells 15 and 16.

The real-data smoke render may read only frames in `[0, 432)`. In particular,
the implementation never opens array keys or frame slices from the two frozen
forecasting regions outside that interval.

## Available environment

- host: Apple M1 Max, 32 GPU cores, arm64
- Python: 3.10.17
- Node.js: 23.11.0; npm: 10.9.2
- disk: approximately 2.0 TiB available at audit time
- available numerical/render dependencies: NumPy, SciPy, scikit-image,
  Pillow, Flask, PyYAML, pytest, Matplotlib, imageio, h5py, PyTorch,
  Transformers
- absent training extras: PEFT, TRL, Datasets, NetCDF4 in the local Python
  environment

This machine is suitable for preprocessing, rendering, rating, reward audits,
and CPU-scale optimizer smoke tests. It is not the selected environment for a
full 7B QLoRA/GRPO pilot. No training job was launched during the audit.

## Reference audit

The supplied public references were inspected without sending any local data.
The useful transferable choices are layered translucent washes, controlled
bleed, paper texture, compact method allowlists, deterministic composition,
and pairwise taste judgments against a hand-rated reference pool. The plasma
project deliberately rejects decorative random blooms: strokes must consume
frame features. The supplied p5 sketches use p5.brush 2.1.0-beta; p5.brush is
MIT licensed. This implementation uses a smaller dependency-free Canvas2D
runtime so the browser demo is reproducible and does not load code from a CDN.

## Implementation decision

The smallest complete pilot will:

1. load only the frozen real source or explicitly labelled synthetic arrays;
2. build robust train-only statistics and compact per-frame features;
3. extract contours, signed periodic filaments, and explainable temporal
   tracks;
4. expose ten bounded painting operations through a deterministic DSL;
5. validate generated JavaScript statically and execute it in a resource-bound
   child-process sandbox;
6. render deterministic scientific-watercolor baselines and a local homepage;
7. collect append-only blind preferences;
8. expose SFT, DPO, and online grouped-policy entry points, while keeping any
   expensive model run opt-in;
9. gate all aggregate rewards on validity and fidelity.

## Audit gate

`GO` for implementation and bounded local smoke tests. Full model training is
not released by this audit; it requires installed training extras, an explicit
GPU allocation, candidate render-validity evidence, a completed reward audit,
and a human preference snapshot.

