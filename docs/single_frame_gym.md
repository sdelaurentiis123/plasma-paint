# Single-frame painting gym

This pilot supersedes whole-program generation and temporal evaluation. Each
episode paints exactly one permitted frame; no persistent temporal state is used.
Spatial placement and choice of medium remain free. The six existing finite-stroke
media implement brush mechanics, not artist-specific painting policies.

`PaintingGym.reset()` returns a blank canvas, scientific cross-section, tool catalog,
source/geometry and budgets. `step(action)` accepts JSON data only: paint a batch,
undo the last accepted batch, or finish. Batches are atomic; rejected actions still
consume a turn and return an error. Maximum 24 turns and 256 strokes by default.
There is no eval, generated JavaScript, network or DOM in this action path.
Each stroke has bounded geometry and parameters. Layering is sequential alpha
compositing; watercolor is an approximate brush texture, not wet-fluid simulation.

The local vision driver sees artist references, the scientific image, current
canvas and previous error on each turn. It emits at most four requested strokes per
response (the environment accepts at most 16). No history-dependent animation or
frame-reusable code is required. The resulting editable JSON operation stream is
specific to that frame. The policy, rather than that stream, must generalize to
new frames. Original code-generation experiments remain preserved separately.

## Commands

```
python -m plasma_painter.training.gym_rollout --synthetic --output artifacts/plasma_painter/my-synthetic-episode
python -m plasma_painter.training.gym_rollout --section 18 --frame-offset 0 --output artifacts/plasma_painter/my-real-episode
python -m plasma_painter.training.gym_rollout --section 18 --model-path models/Qwen2.5-VL-7B-Instruct --reference /absolute/permitted/reference.png --output artifacts/plasma_painter/my-vision-episode
```

No model is loaded unless `--model-path` is provided. Default runs use a clearly
labeled, hand-authored demonstration of valid tool use; it is not a learned artist.
The model driver uses existing local weights only, with at most 24 responses of
700 tokens. No GPU run has been launched for this gym at implementation time.

Exports: painting.png, scientific.png, content-addressed program.json, full
trajectory.jsonl, manifest.json, run.json. Existing episode directories are never
overwritten. No plasma/reference data is sent to an API. Real input is restricted
to hash-verified existing training caches y=0/18/31, frames 0–7 from shot 85604.
No raw held-out data is opened. Synthetic episodes are explicitly labeled.

## Reward and training gate

The current scalar feedback is change in coarse absolute-fluctuation/pigment rank
correlation, with penalties for rejected actions and blank completion. This is
diagnostic feedback ONLY: it does not establish sign, contours, filament topology,
scientific non-inferiority, or aesthetics. `scientific_training_eligible` is always
false. Do not feed this partial reward into a meaningful RL run.

Next: collect valid observation/action demonstrations, train action-format SFT,
evaluate frozen and SFT policies in this same loop, obtain human style preferences,
complete the single-frame scientific gate, then run preference/RL updates. Temporal
criteria are deferred by explicit user instruction, not silently counted as passed.
No SFT, DPO or GRPO training is claimed by these smoke tests.
