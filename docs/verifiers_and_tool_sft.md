# Verifiers integration and first tool-use SFT

Official Verifiers source was inspected and installed at revision
25debce78aff23fca201cefe9c6f72dc65176d06 in an isolated repository-local Python 3.12
environment. Its current native API uses typed Taskset/Task and Env interactions.
The package in environments/plasma_paint_gym was created with the upstream init
scaffolder and implements that API, rather than assuming historical ToolEnv APIs.

Checks: eight permitted tasks load; multimodal interaction messages, action handling
and metric recording pass a scripted-agent test; hosted endpoints are rejected.
Two integration tests pass in the Verifiers environment. The main suite has 74
passing tests and skips the optional module when Verifiers is absent. This is not
yet an end-to-end inference-server/trainer integration test. No environment was
published. Existing Rusty Python/model environments were not changed.

The wrapper records diagnostic metrics only and marks its agent untrainable.
There is no aesthetic reward, external judge, or meaningful RL run. A future
audited single-frame gate is required before connecting training rewards.

## First actual weight-update path

`python -m plasma_painter.training.gym_sft --build --dataset <new-directory>` creates
72 accepted tool demonstrations: 48 format-training examples from frames 0/1 and
24 format-evaluation examples from frame 6, across y=0/18/31. All are inside the
existing 85604 art_train caches. Images show the scientific field and canvas before
the action. Six media and bounded normalized stroke geometry are exercised.
These hand-authored mechanics demonstrations are NOT artist-style targets or
human-preferred paintings. They are allowed to teach interface syntax without
claiming scientific or aesthetic success. No raw data or 85606 was opened.

`scripts/rusty_gym_sft.slurm` runs 32 LoRA optimizer steps on the existing local
Qwen2.5-VL-7B model (rank 8, alpha 16, q_proj/v_proj, learning rate 1e-4).
Prompt/image tokens are masked from loss, with an exact prefix check. Only adapter
parameters are optimized. Three fixed format-evaluation prompts (one per section)
are generated before/after, separate from weight updates. This tiny check measures
valid action syntax, not generalization confidence, fidelity or artist preference.

Allocation: one existing Rusty GPU, one-hour hard limit, 32 steps, no sweep or
external spending. Rough expected total is 15–40 minutes including shared-storage
model startup, to be measured rather than promised. Maximum allocated use is one
GPU-hour. Adapter saves are non-overwriting, with losses, evaluation responses,
dataset hash, seed, Git state and hardware. Scheduler logs capture failures; a job
submission must never be reported as a completed update.

Further work after a successful tool-use update: evaluate entire gym episodes with
the adapter, collect better composition demonstrations, obtain blind human style
preferences, audit the full fidelity gate, then DPO/online RL. A valid action model
alone is not a good painter.
