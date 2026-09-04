# Adapter preview recovery

Rusty job 6985263 generated four adapter programs but terminated on an uncaught three-second Node syntax-check timeout. No new GPU run was needed: saved programs were downloaded and validated/rendered locally on old-85604 art_train frames 0–7, seeds 1701–1704, alongside the corresponding base programs.

Raw results: base 4/4 render; adapter 1/4 render. Three adapter programs contain a complete JavaScript function followed by a Markdown fence or explanatory prose. Separately content-addressed copies remove only those recognized suffixes, with no drawing-code edits. After this explicit wrapper repair, adapter 4/4 render. Raw outputs and failed comparisons remain preserved; repaired results are not raw generation-validity results.

Artifacts: `artifacts/plasma_painter/adapter-comparison/comparison.gif` (raw) and `artifacts/plasma_painter/adapter-comparison-wrapper-cleaned/comparison.gif` (repaired). Both include machine-readable manifests. Comparison is a training-clip preview, not held-out evaluation. Visual inspection shows no convincing improvement in expressiveness; the pale contour composition persists. Recommendation: REVISE training examples and preference collection before further optimization, not scale this recipe blindly.

Compiler now records syntax-check timeouts as invalid candidates rather than crashing the whole filtering run; timeout remains three seconds. Local tests: 30 passed. No new GPU allocation, external upload of plasma data, or access to shot 85606 occurred during recovery.
