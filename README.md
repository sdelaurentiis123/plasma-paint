# RL-trained plasma code painter

An experimental, data-constrained pipeline in which a code model writes
editable JavaScript painters for real TCV edge-turbulence fields. Human
preference optimization may change style, while hard gates prevent appearance
from compensating for loss of plasma structure.

The current repository contains a complete deterministic and training-smoke
pipeline. It never downloads or uploads plasma data and rejects the sequestered
shot at the loader boundary. Current status is **REVISE**: the scientific
baseline works, but no language-model adapter or human-preference result is
claimed yet.

## Quick start

```bash
python -m plasma_painter.data.build_manifest --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.features.build_cache --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.renderer.render_baseline --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.generation.sample_programs --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.renderer.render_program --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.rewards.audit --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.ratings.server --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.training.sft --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.training.dpo --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.training.grpo --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.evaluation.evaluate --config configs/plasma_painter/pilot.yaml
python -m plasma_painter.web.demo.server --config configs/plasma_painter/pilot.yaml
```

Generation defaults to clearly labeled hand-authored fixtures when no local
model path is configured. Training commands default to safe dataset/objective
or renderer-environment smoke modes. A real model update requires `--execute`,
staged local weights, and the optional training dependencies documented in
[the method](docs/plasma_painter_method.md). No command calls a hosted model
API.

The rating UI is at `http://127.0.0.1:8765`; the homepage prototype is at
`http://127.0.0.1:8088`. Neither is deployed. Generated arrays, renders,
ratings, and adapters are ignored by Git.

The paired results viewer is at `http://127.0.0.1:8088/results.html`.
It provides synchronized scientific and sandboxed JavaScript painter frames,
play/pause, scrubbing, three preview clips, and recorded clip-bootstrap metrics.
The current source is the older 624-frame archive, restricted to the permitted
training prefix; the newer 1,936-frame NERSC corpus still requires access and
verification of its governing split. It has not been used in these results.

Run focused checks with:

```bash
pytest -q
```
