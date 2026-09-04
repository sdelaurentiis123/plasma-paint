# Plasma painter method

## Scope and claims

This repository is an end-to-end research scaffold for learning editable
JavaScript plasma painters. The completed local result is a deterministic
scientific renderer, constrained generation environment, preference UI, reward
audit, and runnable LoRA/DPO/online-GRPO code. It is **not yet a trained painter**:
no base-model weights were staged locally and no human choices exist.

The pilot uses only TCV shot 85604 frames already inside the frozen forecasting
training block. The loader rejects the sequestered shot identifier before it
stats or opens any path. No plasma data is uploaded or sent to a model API.

## Data and coordinates

The source fields are a fixed outboard-midplane plane at poloidal index y=18.
Array axis zero is radial x, increasing outward; axis one is periodic,
field-aligned z over one of five toroidal wedges. This is not interpreted as a
Cartesian R-Z image. The separatrix face is between local radial cells 15 and
16 after two guard cells are removed.

Density is log-transformed for the wash, then mapped with robust training-only
percentiles. Density fluctuation is defined at each radial position by

$$
\widetilde{N}_e(x,z,t) = \frac{N_e(x,z,t)-\langle N_e(x,\cdot,t)\rangle_z}{\max(|\langle N_e(x,\cdot,t)\rangle_z|,\epsilon)}.
$$

Median, MAD, first percentile, and ninety-ninth percentile are fitted only on
art_train. Explicit clipping is recorded in the manifest.

Contours come from fixed robust-z levels and are stored in normalized domain
coordinates. Filaments are signed periodic connected components after a
one-cell Gaussian smoothing scale and a threshold fitted on art_train. Each
component records centroid, area, peak, elongation, orientation, and sign.
Adjacent components are linked by gated Hungarian matching with periodic z
distance. Birth, death, merge, and split events remain explicit.

Potential is genuinely available. The fixed-plane record does not contain all
geometry required to label the rendered in-plane direction as a physical
E-cross-B velocity,

$$
\mathbf{v}_{E\times B}=\frac{\mathbf{E}\times\mathbf{B}}{B^2}.
$$

Accordingly, potential and density gradient directions are labeled
visualization proxies. The source also lacks a justified local cross-field
transport diagnostic, so transport accenting is disabled instead of being
fabricated from density.

## Painter and sandbox

Generated code must export `createPainter(api, styleConfig)`, returning
`reset(seed)` and `renderFrame(frameFeatures, time, persistentState)`. The only
painting operations are:

`createPaper`, `setPalette`, `washRegion`, `strokePath`, `dryBrushPath`, `dab`,
`poolPigment`, `scatterGrain`, `fadeLayer`, and `composite`.

Coordinates and parameters have explicit bounds. Static validation rejects
network, DOM, dynamic-code, host-process, worker/timer, prototype, direct
Canvas/p5, class, and unbounded-while surfaces. Valid code runs in a separate
Node process with a V8 context timeout, a 96 MB heap, seeded randomness, frozen
API, per-operation caps, path caps, and an outer process timeout. The browser
runs only the already selected and validated painter—not the language model.

This sandbox is suitable for a local research pipeline; it is not represented
as a formally proven hostile multi-tenant isolation boundary.

## Visual mapping

- Density drives the broad wash and opacity.
- Signed density-fluctuation contours drive iron-red and blue paths.
- Gradient directions orient fine dry strokes.
- Filament centroids, areas, signs, and peaks drive blooms.
- Tracked pigment is faded and carried between frames.
- Births and merges pool pigment.
- A quiet vertical trace preserves the separatrix face.
- Seeded paper grain is the only non-structural randomness.

The source JavaScript remains editable and reusable across arbitrary feature
frames. The Python and browser runtimes consume the same bounded operation
stream.

## Reward and optimization

Validity is an uncompensated gate. Compilation, DSL use, bounded execution,
nonempty structural marks, coarse spatial correspondence, extrema,
filaments, orientation, and temporal response must pass before aesthetics are
considered. A failed gate receives a reward of negative two.

For valid programs the preregistered initial aggregate is

$$
R = 0.40R_{\mathrm{fidelity}} + 0.30R_{\mathrm{aesthetic}} + 0.15R_{\mathrm{temporal}} + 0.10R_{\mathrm{diversity}} + 0.05R_{\mathrm{efficiency}}.
$$

Continuous machine components are normalized with fifth and ninety-fifth
percentiles fitted on art-train reference variants. Human pairwise preference
is authoritative for aesthetics. In the absence of ratings a neutral value is
permitted only for infrastructure smoke tests and is labeled as such. No VLM
judge is active.

SFT uses accepted train-only programs. DPO uses only non-tie human
chosen/rejected pairs. The online path samples a group of JavaScript programs,
sandboxes and renders each on a full training clip, computes every reward
component, normalizes within the group, recomputes completion log-probability,
updates the LoRA policy, and appends source plus reward records to JSONL. DPO is
never reported as online RL.

One shared base model supports named LoRA adapters. Only `tcv-watercolor` is
released for the pilot; ink and gouache remain extension configurations until
the pilot gate passes.

## Base model and compute

The selected pilot base is `Qwen/Qwen2.5-Coder-7B-Instruct`: a 7.61B-parameter,
Apache-2.0 code model with a documented 128K family context, safely below the
prohibited 30B-plus starting point. The prompt itself is short and uses compact
feature summaries, so extended-context configuration is unnecessary. See the
[official model card](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
and [Qwen model-family release](https://qwenlm.github.io/blog/qwen2.5-coder-family/).

Preferred execution is one Perlmutter GPU under the existing NERSC allocation.
Prime Intellect is a fallback only if explicitly approved; its GPU API and
persistent disks could host the same local-only job, but no account was used,
no instance was provisioned, and no array was uploaded. See the
[Prime Intellect documentation](https://docs.primeintellect.ai/introduction).

Estimated bounded pilot use, after human data exists:

| stage | one-GPU estimate |
|---|---:|
| frozen sampling and filtering | 0.4–0.8 GPU-hours |
| 40-step LoRA SFT | 0.5–1.0 GPU-hours |
| 30-step DPO | 0.5–1.0 GPU-hours |
| 25-step online grouped RL | 2.0–4.0 GPU-hours |
| evaluation and rerenders | 0.5–0.8 GPU-hours |
| total | 3.9–7.6 GPU-hours |

The estimate stays under the eight-GPU-hour authorization but should be
revised from measured Stage-0 throughput before GRPO. Hosted pricing changes;
at the prices visible during the audit, the rough Prime hardware spend would
be about USD 8–24. NERSC is preferred and has no new external API spend.

## Commands

Prepare and inspect locally:

```bash
python -m plasma_painter.data.build_manifest --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.features.build_cache --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.renderer.render_baseline --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.generation.sample_programs --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.renderer.render_program --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.rewards.audit --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.ratings.server --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.ratings.consistency --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.training.sft --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.training.dpo --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.training.grpo --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.evaluation.evaluate --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.web.demo.server --config configs/plasma_painter/pilot.yaml
```

The first NERSC GPU run is Stage 0 only. After cloning this repository and
staging the permitted 85604 files, model, and virtual environment on NERSC:

```bash
sbatch -A "$NERSC_GPU_ACCOUNT" \
  --export=ALL,PLASMA_PAINTER_VENV="$PLASMA_PAINTER_VENV",PLASMA_PAINTER_MODEL_PATH="$PLASMA_PAINTER_MODEL_PATH",PLASMA_PAINTER_SOURCE_PATH="$PLASMA_PAINTER_SOURCE_PATH",PLASMA_PAINTER_GEOMETRY_PATH="$PLASMA_PAINTER_GEOMETRY_PATH" \
  scripts/nersc_stage0.slurm
```

NERSC documents `shared` QOS for one- or two-GPU jobs in its
[Perlmutter examples](https://docs.nersc.gov/systems/perlmutter/running-jobs/).
Do not run `--execute` SFT, DPO, or GRPO until actual model candidates pass the
eighty-percent gate and the required human preference snapshot exists.

## Reproducibility

Content hashes address every program. Dataset files, field keys, hashes,
coordinate convention, exact frames, normalization, upstream commit, seeds,
runtime version, reward version, hardware, wall time, and rating snapshot are
stored in machine-readable artifacts. Core dependency versions are frozen in
`requirements-lock.txt`. Generated data, ratings, renders, and adapters stay
outside Git.
