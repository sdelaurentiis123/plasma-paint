# Image-conditioned painter experiment

Original **Rusty job 6988641**, source **c61cb98**, finished in 11m10s with **0/6 valid renderers**, despite process exit 0. Its monitor was removed. The image processor preflight succeeded; this was primarily generated-code contract failure, not a missing vision input. No GPU was allocated during model download or dependency troubleshooting.

## Failure diagnosis and bounded retry

Van Gogh attempted local image loading and DOM/Canvas access, which are forbidden.
Seurat used asynchronous rendering, batched mark arguments and object-shaped points.
Manet drew paper during factory construction, discarded z, used malformed points,
and treated the encoded fluctuation zero as 0 rather than .5. Van Gogh and Manet
repairs repeated the preceding code exactly. One remote syntax-check timeout was
also logged; replay does not establish its infrastructure cause, and raising that
timeout would not fix the independently demonstrated invalid code.

Fixes: enforce per-phase VM deadlines for factory/reset as well as frame rendering;
reject asynchronous code and drawing outside renderFrame; report malformed mark
arguments directly; verify output frame count. Add a tiny executable, artist-neutral
API call example and explicitly explain that reference images are inference context,
not JavaScript-accessible files. Repairs now use assistant/user turns and a different
generation seed, and record duplicate outputs. No artist algorithm, palette, or
scientific constraint is substituted or relaxed. Filter render errors are included
in repair feedback. Raw failed programs remain unchanged.

Validation: 43 tests passed, including new initialization timeout, lifecycle,
async, malformed mark and neutral-example tests. The reference finite-mark renderer
still executes on permitted old-85604 frames 0–7: 1,041 operations on each of eight
frames. This validates interface compatibility, not model aesthetics or fidelity.

Retry scope: a fresh `vision-context-v2` worktree, the same pinned frozen model and
public reference images, three artists with two attempts each, one GPU with a
one-hour hard cap (expected roughly 10–20 minutes based on the previous run).
No optimizer updates, RL claims, new data access or external API spending.

Model: Qwen/Qwen2.5-VL-7B-Instruct, public revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. Selected as a small image-capable baseline compatible with the existing Transformers 4.56.2 / Torch 2.8 environment. Its ability to produce useful painting code remains an experimental question. This is a new frozen base, not the previous text-only LoRA adapter.

Dedicated Rusty checkout: `/mnt/home/sdelaurentiis/ceph/plasma-paint-runs/20260904-rusty/vision-context-v1`. Public model weights reside inside this checkout, not the previous run's model directory. The existing virtual environment is reused read-only. CPU processor preflight detected missing Torchvision; `torchvision==0.23.0+cu128` is installed with `--no-deps --target vision-deps` from the official PyTorch cu128 index and added to this job's PYTHONPATH. No shared environment or unrelated job is modified.

The model receives two museum images for each artist plus a scientific rendering of old-85604 training frame 0. It writes a reusable finite-mark renderer without receiving a reference program or artist-specific algorithm. Generic tools let it select media, colors, placement, number of marks and layering under the stroke-only resource and anchor constraints. It then receives its own rendered draft and is asked for one visual revision. If the draft is invalid, that second attempt instead receives the validation error; it is not falsely described as visual revision.

Six public-domain-tagged AIC images: Van Gogh 28560 / 14586; Seurat 27992 / 20199; Manet 44892 / 81533. Source URLs, museum metadata and image hashes are retained in the staged reference manifest. These are inference context, not fine-tuning targets. No Picasso/Warhol restricted images are included. No plasma arrays or images are sent to an external inference service.

Only the existing permitted old-85604 art_train clip (frames 0–7) is rendered. All fields and the scientific image use the existing training normalization; the scientific image is signed density fluctuation encoded grayscale with zero at middle gray. The generator must respond to arbitrary frame features. No new full-shot data, upstream held-out blocks or shot 85606 are accessed.

Budget: three artist contexts, two attempts each, at most six model completions of 2,800 new tokens; one Blackwell GPU and one-hour wall cap. Estimated run time 15–35 minutes, uncertain due to shared storage/model throughput; hard maximum one GPU-hour. Public weight staging used no GPU. External API spending is zero. The text-only `explore_tools` batch was not launched and is superseded for this experiment.

Outputs: `artifacts/plasma_painter/vision-context/manifest.json`, raw and extracted programs, prompts, reference IDs, validation results, source/provenance and rendered clips. Both drafts and revisions are retained regardless of quality. The self-critique is not a calibrated aesthetic judge or RL reward. There are no optimizer updates. Scientific fidelity and human preference still need evaluation; success must not be inferred from a completed job.

Local tests before staging: 34 passed. Launch script: `scripts/rusty_vision_context.slurm`. Model staging: `scripts/prepare_vision_model.py`; image selection/hash verification: `python3 -m scripts.prepare_vision_references`.
