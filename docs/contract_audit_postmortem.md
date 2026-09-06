# Why the contract diagnostic did not resolve the painter — 2026-09-05

Job 6988887 completed in 14m43s. It produced one accepted program out of four,
but that program equals the supplied reference after removing whitespace. This
is interface copying, not a new painting policy, image-based style learning or RL.

## Confirmed integration mistakes

1. The three from-scratch prompts omitted the required literal exported function
   signature. Only the example-assisted condition included it, inside the example.
   Consequently the comparison also changed whether the required entry point was
   explicitly demonstrated. The initial 1/4 acceptance figure cannot isolate image
   conditioning, model choice or useful originality.
2. `make_prompt` called a JSON object "Representative training input" and included
   `sample_count`, `value_range` and `neutral_value` alongside `stroke_samples`.
   These are explanatory metadata, not fields in the actual runtime frame. The
   image-conditioned scratch program read `frameFeatures.value_range[1]`, which
   cannot work. We taught an inaccurate input shape and then rejected its use.
3. CPU preflight checked tokenization, image grids and a hand-authored control.
   It did not check that the prompt's advertised fields exist in runtime inputs,
   or that every condition states the same mandatory exported entry point.
4. The acceptance result measured execution and nonblank pixels. It did not test
   whether the accepted output merely copied the prompt's solution. The entire
   successful program, including palette, positions, lengths and media, was copied.
5. The diagnostic performed four frozen-model completions, not optimizer updates.
   The earlier coder adapter was trained on a different API and was not loaded.
   Repeated frozen sampling cannot be described as progressing an RL training run.

## Local causal replay (original outputs preserved)

Used the already-cached, permitted old-85604 art_train frames 0–7 only. No arrays or
derived data were uploaded, and no raw shot or held-out data was opened. Variants
below existed only in memory to uncover successive failures; they were not saved
as model-generated programs, rendered for presentation, or counted as successes.

| Case | Add only missing `export` | Then correct only reset to `api.reset(seed)` |
| --- | --- | --- |
| Vision / text / scratch | Paper drawing in reset is rejected | Operation cap exceeded |
| Vision / images / scratch | Paper drawing in reset is rejected | Undefined `value_range` access |
| Coder / text / scratch | `Math.random.seed` is not a function | Mark length below minimum |

The coder's formula creates length `0.006 * abs(2 * value - 1)`. On the 4,160 actual
anchors across these eight frames, **3,598 (86.5%)** are below the .002 minimum.
Thus adding `export` alone cannot recover the current failed renderers. The image
scratch code also self-references a newly declared `const value`; further repair
would be necessary even after resolving the missing field.

## What the evidence rules out, and what it does not

All expected image tensors/grids were present. No completion reached its 2,800-token
cap; counts were 578, 526, 477 and 393. No infrastructure retry was recorded in this
run. Its failures were therefore not logged timeouts or token truncation. This
does not establish that the model used the images meaningfully.

All 17 accepted legacy media programs still passed on their original two-frame
input in the prior replay. The existing painter runtime has not stopped working
wholesale. There is also no evidence here that the general approach is impossible
or that a different model architecture is required.

## Required correction before another GPU run

- Use one authoritative tool/input contract and put the explicit exported signature
  and executable reset syntax in every generation condition.
- Serialize only genuinely available runtime fields as input examples. Label
  explanatory statistics separately and explicitly prohibit accessing them as fields.
- Test the prompt/runtime agreement, not just prompt tokenization or reference code.
- Distinguish interface-valid, copied-example, original-program and fidelity-passing
  outcomes. A copied solution must never be presented as learned creative progress.
- Teach and measure the generic finite-mark API before combining image interpretation,
  unfamiliar tool use and aesthetic invention in one untested task. Whether concise
  tool examples suffice or API SFT is needed remains to be measured.

No additional GPU job or training was launched for this postmortem. It diagnoses
the setup and preserves the evidence; it does not claim a corrected model result.
