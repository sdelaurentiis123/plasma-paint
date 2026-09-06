# Free-XY run 6989353: unsuccessful generation

Rusty completed the isolated job with exit 0 in 10m49s (approximately 0.1803
allocated GPU-hours). Process completion is not research success: all six model
completions failed execution validation; zero paintings or scientific scores exist.
The model was frozen Qwen2.5-VL-7B, not a trained style adapter or RL policy.

Evidence is retained, Git-ignored, in
`artifacts/plasma_painter/rusty-free-paint-v1/free-paint/manifest.json`, alongside
content-addressed programs and `../job-6989353.log`. No raw data, held-out shot or
other job was inspected by the completion monitor.

## Confirmed causes

1. The static validator scans comments as code. All three Van Gogh completions
   contain `// Function to calculate stroke position and path based on the field`;
   capitalized `Function` triggers the dynamic-code regex. This is our false
   positive, not an attempted dynamic-code operation.
2. Both conditions emit a named function declaration despite the body-only
   contract. Our wrapper nests it without invoking it, producing zero operations.
   Feedback reports missing paper rather than explaining this structural error.
3. Local diagnostic replay, without overwriting original outputs, removed only
   that comment and then unwrapped the returned function. Van Gogh then fails the
   minimum two-point stroke constraint. Seurat fails with `ReferenceError: x is
   not defined`. Additional visible errors include indexing gradient objects as
   arrays, invented field names and HSL strings where hexadecimal is required.
4. The three Van Gogh attempts are identical hashes; Seurat attempts 1 and 2 are
   also identical. Six completions contain only three distinct programs. Repair
   feedback did not yield a working program; duplicate attempts should be detected.

## Tools and scientific status

Van Gogh code selects watercolor; Seurat code selects ink. These are source-code
choices only: neither produced strokes on a canvas. There is no evidence of style
quality, data response, or scientific fidelity. Do not promote these to SFT targets.

## Next correction, before another allocation

Use lexical/AST-aware capability validation so comments cannot trigger code rules
while executable forbidden operations remain blocked. Validate the returned-body
structure explicitly, and give actionable repair errors (or support a precisely
specified callable function contract). Replay these saved failure cases locally,
including duplicate detection. Keep free spatial placement and model-selected media;
do not solve interface failures by supplying an artist-specific painting recipe.

The monitor submitted no further GPU job and was paused after reporting completion.

## Subsequent correction

The user requested implementation after this report. Pinned Acorn 8.15.0 now parses
comments and function boundaries without executing candidates (`npm ci --ignore-scripts`
is required alongside Python dependencies). Executable capability rules and VM limits
remain in force. A returned synchronous three-argument function is reduced to its
body by AST source offsets, without changing any painting statements. The prompt now
asks for that complete function, matching the model's natural output. Raw outputs and
normalization provenance remain recorded. Duplicate rejected programs are detected,
and stroke-shape/color errors are reported directly. Local replay of saved failures
now reaches genuine one-point-path and undefined-coordinate errors rather than the
comment false positive or silently nested function. This fixes integration, not the
quality of the original programs.
