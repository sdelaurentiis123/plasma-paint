# Generation regression audit — 2026-09-05 local time

## Finding

The two vision runs do not establish that an image-capable model cannot write painting
code. We changed model family, prompt scaffolding and drawing API together, then
interpreted a six-completion failure as a model limitation. That conclusion was too
strong. The appropriate next step is an interface ablation, not a new architecture
or another multi-artist batch.

## Git and artifact evidence

| Revision | Change | Observed result / limitation |
| --- | --- | --- |
| `de9ea7d` | Added exact schema AND a complete working program to Qwen2.5-Coder prompts | Saved schema-v2 filter: 4/4 valid. This is example-assisted generation, not free-form invention. |
| `6cd7867`, `3a4816f`, `7eaadf2` | Added media parameters and conditional LoRA SFT | Saved media filter: 17/20 valid. Prompt still includes the complete reference; task asks for modifications. |
| `7eaadf2` / job 6985156 | Actual 40-step rank-16 SFT | Saved adapter exists. SFT record: first loss .01319, last .04272; this tiny run is not evidence of aesthetic improvement. Targets and prompts share reference code; no human preference data was used. |
| `2f0379f`, `ab1f804` | New finite-mark API and six hand-authored studies | Those displayed styles were not outputs of the adapter. Existing SFT predates this API. |
| `d2705c1` | Replaced coder with frozen Qwen2.5-VL, removed full working example, introduced image input | The new entry point never loads the saved LoRA. Job 6988641: 0/6 valid, four distinct programs. No rendered draft was available for visual revision. |
| `83ef519` | Added lifecycle checks, one-call example, chat repair | Job 6988807: 0/6 valid, five distinct programs. This was still frozen generation, not training. |

The original and retry manifests and generated programs were inspected without
editing them. Node errors reflect actual prohibited/malformed code. Nevertheless,
one retry candidate initially hit the three-second syntax-check process timeout;
its identical second copy reached runtime. We gave the model an infrastructure
failure as if it were a code defect and spent its only repair on that. The original
run also logged a syntax timeout. Cold filesystem/Node startup is plausible, but
the exact source of latency is not established.

## Input audit

The saved model chat template supports system strings, user image content and
assistant turns as used. It inserts image tokens correctly by inspection; CPU
processor preflight had verified pixels, but we did not previously save each
completion's fully serialized prompt, input shapes, image grid, token count or
termination reason. We therefore cannot retrospectively claim an end-to-end
image-conditioning audit from that preflight alone.

The saved generation config contains temperature .000001, but both vision entry
points explicitly override temperature with .7. It is **not** evidence that our
calls secretly used greedy decoding. The first run reused a seed; the second did
not. Duplicate repair output is observed, not a proven RNG defect.

The prompt said no full renderer example; the retry added a single legal mark.
Neither recent vision run got the full executable example used by the successful
coder runs. Both got field semantics in prose but no literal input sample/count.
The prior training prompt embeds a near-target working renderer. Its very low
training loss therefore cannot be equated with learning art or general DSL skill.

## Changes made now

`contract_audit.py` compares four fixed-seed, single-completion cells on the same
permitted old-85604 frames 0–7 and finite-mark API:

1. Vision model, text only, from scratch.
2. Same vision model and brief, with the two Van Gogh references and scientific image.
3. Same image condition plus the complete, explicitly hand-authored interface control.
4. Original coder model, the same text-only from-scratch prompt as cell 1.

The known working finite-mark control must pass before sampling. Both model families
remain frozen and no adapter is loaded. This is a diagnostic, not a statistically
powered model comparison. Cell 3 success alone must not be sold as original artist
learning; compare its code to the supplied example.

The runner records actual input examples/count, serialized chat, image grids, input
shapes, output token counts, token-cap flags, explicit generation arguments, raw
code, runtime results and any rendered clip. An exact-code syntax-timeout retry is
logged as infrastructure recovery, not model repair. Real runtime/geometry failures
are not relaxed. CPU `--preflight` checks actual chat/image tokenization before GPU
submission. Separate fresh output directories prevent overwriting earlier results.

Scope: two already staged 7B models loaded sequentially on one GPU, four completions
total, 2,800 new tokens each; no download, API spending, model updates or shared-env
changes. Hard cap one GPU-hour; rough expectation 15–30 minutes including shared
filesystem model loading. Same six public-reference files staged read-only, only
the two Van Gogh images consumed here. No raw shot access, held-out data or 85606.

## Remaining gaps (not hidden by valid renders)

Finite-mark acceptance checks geometry/resource bounds and nonblank pixels. It does
not establish the requested fidelity non-inferiority or aesthetic improvement.
The human preference dataset is still absent, and no language-model DPO/GRPO update
has occurred. Browser mark-runtime support and website integration are separate
unfinished work. Do not resume meaningful RL until the interface works, generated
art is inspected and reward/fidelity audits pass.

Recommendation: **REVISE** the experimental procedure. Preserve the successful
baseline as a control; test one change at a time; do not blame the model based on
the confounded batches or describe finite-style demos as trained outputs.

## Local verification before the diagnostic run

45 tests passed. Replayed all 17 previously accepted media-v3 programs on the same
two permitted training frames with today's legacy runtime: **17/17 still accepted**.
The finite-mark control also passes nonblank pixel rendering on all eight permitted
frames, with 1,041 operations per frame. These checks rule out a blanket regression
in the old valid-render path; they do not establish visual quality of new code.
