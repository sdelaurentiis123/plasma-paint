# Six finite-mark studies

Rebuild command: `python3 -m scripts.render_style_atlas`.

All six studies use the same old-85604 permitted training clip, frames 0–7, fixed seed 1701, and training-normalized signed density fluctuation. Each uses 1,040 finite marks per frame and no field wash, contour tracing, bloom, or image filter. The six settings of the reusable `finite_styles.js` renderer change geometry and layering as well as color:

- Directional bristle: curved short strokes along the local visualization tangent.
- Pointillism: small separated ink stipples with amplitude-linked width.
- Angular facets: quantized directions and bent short paths.
- Tonal brushwork: broad bounded strokes with amplitude-linked length.
- Pencil: crossing fine marks with amplitude-linked width and pressure.
- Screenprint: paired flat ink bars with a discrete color mapping.

Historical artist names describe inspirations, not learned replicas. No reference artwork or trained model produced these programs. Screenprint repetition is mark-level, not duplicated plasma geometry. Facets do not reconstruct a volume. Local contrast is explicitly remapped as clip(0.5 + 2.2*(value-0.5),0,1) inside the reference code; preprocessing normalization is unchanged.

Outputs in ignored `artifacts/plasma_painter/finite-style-atlas/`: six PNGs, six animated GIFs, full per-frame operation traces, a labeled contact sheet/animation, and provenance manifest. There are 48 rendered style-frames. This is the finite-mark renderer, not the older browser-only style lab. The existing website is unchanged; artifacts can be viewed directly in the app.

Validation: 34 tests pass, including deterministic traces, finite-stroke enforcement, changed-frame response, and distinct mark geometry for all six styles. All six passed the strict profile on the eight real frames. Visual inspection shows meaningful tool differences but a regular sampling pattern, sparse stippling and limited composition; these remain construction candidates, not finished artistic policies or evidence of scientific non-inferiority. No GPU training or new remote job was launched. No shot 85606 or upstream held-out block was accessed.
