# Free XY painting across cross-sections

The user's clarification supersedes the experimental .04-distance sample-anchor
constraint. Free-paint programs may place finite strokes anywhere in the normalized
display plane. They are not required to use a lattice, sample ID, prescribed palette,
artist-specific medium, local path shape, or fixed placement/layering algorithm.
The original scientific fidelity requirement remains: output must represent the real
field, not merely resemble plasma. Execution validity is not scientific validity.

## Model/runtime boundary

The model writes the body of a reusable renderFrame function. A trusted wrapper
provides export/reset/lifecycle only, with no painting recipe. Tools are createPaper
and paintStroke; the latter supports six media, arbitrary 2–64-point paths across
the plane, bounded width/opacity/pressure/texture, and optional stable stroke IDs.
Those IDs affect texture continuity only, not position. One stroke is finite (arc
length .0005–2 domain units); paths and brush widths are bounded but not attached
to a plasma sample. At most 1,100 strokes and 40,000 scalar query evaluations/frame.

Read-only sample(field,x,y) and gradient(field,x,y) expose normalized fields at any
chosen location. Scalar interpolation is bilinear, radial edge derivatives one-sided,
and periodic z derivative neighbors wrap. Display XY corresponds to radial x and
field-aligned z, not Cartesian R-Z; section y is a separate coordinate. Geometry,
contours and filaments are available as optional evidence, not mandatory mark sites.
Invalid-cell queries require a cell mask; the current permitted crops have no invalid
cells. Full mask/geometry fidelity remains part of the research gate, not inferred
from this prototype interface.

The bounded child runner now keeps candidate inputs and API wrapper functions in
the VM realm, disables string/Wasm code generation, and converts host-call errors
to VM errors. The null-prototype host bridge returns primitives or null-prototype
numeric gradient objects. Tests cover computed-constructor paths through API
functions, Math, inputs, styles and caught errors. Process time/memory limits and
static capability checks remain. These tests are evidence, not a formal sandbox
security certification; no untrusted external code is accepted as a deployment.

## Bounded experiment

Source input: existing old-85604 art_train section caches y=0,18,31, frames 0–7.
Normalization remains each section's recorded training-only fit. No raw shot,
upstream held-out block, guard range or 85606 is opened. The section staging index
records file hashes, original preprocessing Git information and normalization.

The model sees two public-domain-tagged artist images and the three scientific
sections. It produces a program that is run unchanged on every frame in every
section. Every attempt and error is retained. Successful rendering triggers
reproducibility, frozen-field and evolving-field probes. Coarse pigment/absolute
fluctuation rank correlation must stay within 5% of the existing control in each
section. These are partial checks; signed structure, contours, filament topology,
transport when available, calibrated aesthetics and other full pilot criteria are
not replaced by this correlation. RL cannot use the legacy reward implementation.

Two artist conditions, up to three attempts each, one existing frozen 7B vision
model, one GPU with one-hour cap. No fine-tuning is claimed by this run. The intended
training strategy remains one shared base with lightweight learned style policies,
not separately trained full artist models. Human preference and style-learning
experiments follow working, scientifically evaluated free-paint programs.

The preceding anchored JSON-tool job 6989277 was cancelled at 5m18s after the user's
clarification. Its files were retained and no other user's job was changed.
