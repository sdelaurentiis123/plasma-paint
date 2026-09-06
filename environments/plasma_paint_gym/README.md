# plasma-paint-gym

A v1 verifiers environment, scaffolded with `init`.

## Develop

Pinned upstream revision: `25debce78aff23fca201cefe9c6f72dc65176d06`.
Install the main plasma-painter repository and this package in an isolated Python
3.11+ environment. This exports a native Taskset and SingleAgentEnv interaction.
It loads only the hash-verified permitted training caches. It does NOT publish an
environment or send data to Prime's hosted services.

```bash
uv pip install -e . -e environments/plasma_paint_gym
```

## Layout

- `plasma_paint_gym/taskset.py` — native taskset and multimodal interaction loop.

Do not use an unconfigured default eval endpoint: Verifiers defaults to hosted
inference. Our interaction rejects anything except explicit loopback HTTP inference
before sending observations and sets the agent untrainable. Use a local chat harness,
not the default code-executing bash harness. No model inference through Verifiers has
been tested yet; native typed task loading and interaction control flow were tested
with a scripted local agent. No claim of trainer/end-to-end serving compatibility.

Only diagnostic metrics are recorded; there is deliberately no reward until the
scientific gate is audited. SFT is a separate local Transformers/PEFT entrypoint:
`python -m plasma_painter.training.gym_sft`. It teaches validated action mechanics,
not aesthetic preference, and does not require optimizing an incomplete reward.
