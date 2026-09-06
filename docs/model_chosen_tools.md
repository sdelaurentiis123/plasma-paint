# Model-chosen tools, not artist presets

`renderer/paint_program.py` compiles a model-authored JSON painting DSL into editable
JavaScript using the existing finite-mark sandbox/runtime. It contains no artist
names, palettes or style branches. The model supplies each layer's medium, colors,
local stroke path, direction, pressure, texture, width, length, opacity, sample
selection and layer order. There is no complete example painting in the prompt.

The compiler owns only lifecycle, sample iteration, color interpolation, coordinate
transforms and bounded parameter-expression evaluation. Expressions may depend on
normalized signed fluctuation and its magnitude. Clamping is an explicit tool
semantic, not a hidden repair of generated JavaScript. The final operation stream
still passes the original anchor, finite path, operation-budget and pixel checks.
Model programs can therefore be rejected even after compiling successfully.

This deliberately limits expressiveness to a small first painting language: fixed
local path shapes per layer with field-dependent size/angle/pressure/opacity, six
available media, up to eight layers. It is not unrestricted painting code and does
not yet implement arbitrary pigments, simulated wet-media chemistry or temporal
filament tracking. It does remove the earlier prompt's preselected palette and
two-pass brush recipe. Adding a medium is a generic runtime extension, not an
artist-specific implementation.

`generation.tool_policy` uses the existing pinned frozen vision model and two
public-domain references per artist, plus the permitted scientific training image.
Two artists, at most three attempts each, one GPU with a one-hour hard cap. No
adapter update or RL. Raw DSL, compiled JS, chosen tools, image inputs and validation
are recorded. `scripts/rusty_tool_policy.slurm` launches this bounded test in a fresh
dedicated run directory with staged read-only dependencies/weights/reference files.

Validation before submission: 53 local tests pass; an explicitly synthetic interface
program compiles and produces nonblank finite marks on all eight permitted real
training frames. No synthetic image is represented as a model or real-data result.

Previous refinement 6989073 yielded two modified renderers after one repair each,
but both remained heavily influenced by the supplied reference. Van Gogh's frame
64–71 probe was reproducible across three seeds, with no frozen-input changes and
coarse absolute-fluctuation/pigment correlation approximately .805, versus .868 for
the control. That drop is a warning, not a scientific-success claim. These programs
are not promoted as aesthetic or fidelity successes. Original outputs are retained.
