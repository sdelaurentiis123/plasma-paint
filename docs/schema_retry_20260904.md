# Four-program schema retry

The first Rusty job, 6984999, produced 24 candidates and zero accepted programs.
It ended after 23m42s when the SFT gate correctly refused training. The prompt
lacked exact operation argument schemas and feature layouts. Generated code
guessed nonexistent fields and functions; regex checks also rejected equivalent
function-property syntax. No LM adapter was produced.

Retry job: `6985077`, source `de9ea7d`, directory
`/mnt/home/sdelaurentiis/ceph/plasma-paint-runs/20260904-rusty/retry-schema-v2`.
One Blackwell GPU, eight CPUs, 64 GB host memory, 30-minute cap, preemptible queue.
It uses a separate checkout and output directory, reusing only the prior run's
environment, public model weights, and permitted feature cache.

Changes: complete argument contracts, explicit feature schema, working reference
program, warnings for observed nonexistent fields, explicit generation attention
mask, immutable prompt files, and support for equivalent function-property syntax.
The exact export and direct API call constraints remain explicit requirements.
The reference still passes the sandbox and forbidden access remains rejected.
Local tests: 23 passed.

The task generates four schema_v2 programs and renders accepted candidates on
one training clip. There is no automatic SFT, DPO, GRPO, or larger sampling run.
This is a reference-conditioned prompt baseline and must not be confused with
the earlier zero-example prompt. Submission does not establish successful
generation or scientific fidelity; inspect output manifests before advancing.
