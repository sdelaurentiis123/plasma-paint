# Medium studies — 2026-09-04

The first four valid Qwen outputs followed a deliberately conservative interface example. The schema-v2 prompt explicitly requested modest changes, so validity did not establish stylistic diversity. The browser artist studies were hand-authored, not fine-tuned outputs.

## Implemented

The bounded path DSL now supports `medium` (watercolor, bristle, graphite, charcoal, ink, pastel), `pressure` in [0.05,1], and `texture` in [0,1]. Stroke centers follow supplied paths, with brush-width-bounded transverse fibers, pressure taper and medium-specific grain. This is a lightweight mark model, not a physical pigment simulator. Existing operation and path caps remain unchanged. The default watercolor path is unchanged.

Offline rendering uses runtime `canvas-runtime-0.2.0-media`. Browser paths support the same controls but use approximate dash texture, not pixel-equivalent Pillow texture. Browser/offline visual and fidelity parity remains an evaluation blocker before these tools enter a meaningful RL run. Existing runtime limitations (including palette/compositing semantics and temporal texture) have not been solved by this change.

`media_v3` sampling uses the complete schema while removing the instruction to make only modest changes. No new sampling or GPU training job was launched in this change.

## Reproduction

```
python3 -m scripts.render_medium_study
python3 -m pytest -q
python3 -m plasma_painter.generation.sample_programs --config configs/plasma_painter/pilot.yaml --backend local --count 4 --prompt-version media_v3
```

The last command requires the configured local model path. Run in an isolated project run directory on Rusty after staging this revision; do not reuse an existing experiment's output directory.

The study uses the first cached `art_train` clip, old shot 85604 frames 0–7. It produces six stills, six GIFs, and `artifacts/plasma_painter/medium-study/comparison.png`. It explicitly checks the permitted shot and training indices. Shot 85606 was not opened. This is a hand-authored matched-tool comparison, not a trained artist result or a new fidelity result. Inspection shows that tool changes alone retain the contour-heavy composition: stroke placement and layering examples are still needed.

## Training gate

Museum references are unpaired images, not code supervision for the text-only Qwen model. They cannot be fed directly into this adapter as image-to-code examples. Use them to design and curate diverse executable stroke programs, then render those programs on permitted plasma clips. Gather human choices among those outputs; only then build SFT code examples and DPO chosen/rejected code pairs. Online RL also requires audited media-aware fidelity and an authoritative aesthetic signal. Do not use artist-image similarity as permission to invent plasma structures.

Recommendation: REVISE before training. Expand composition examples beyond contours; validate tools in both runtimes; curate preferred plasma outputs. No trained adapter or aesthetic improvement is claimed.
