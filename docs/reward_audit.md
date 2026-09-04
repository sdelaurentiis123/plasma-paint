# Reward audit

Audit date: 2026-09-04

Status: **complete for the pre-RL machine rewards; human aesthetic calibration
is still missing.**

## Protocol

The audit used the first two frames of the first art_train clip solely for a
cheap failure-screening pass. It evaluated 48 programs: 24 valid reference
fixtures and 24 constructed failures or independent component ablations. The
fixtures are hand-authored infrastructure variants and are not represented as
language-model outputs.

The constructed cases covered blank paper, paper noise, a static line, a fixed
grid, transposed coordinates, a density-only wash, missing filament dabs, and
missing oriented strokes. All 24 failed the uncompensated validity gate. Thus a
high aesthetic value cannot rescue these cases.

## Calibration correction

An initial extrema floor fitted from one easy clip rejected the deterministic
renderer on frozen test clips. That was a calibration error, not a successful
scientific alarm. Absolute anti-hacking floors were re-fitted below the weakest
of all 16 deterministic art_train clips, while the planned method comparison
retains its stricter per-clip five-percent non-inferiority test. After this
correction the deterministic renderer passes all test clips and every
constructed hack still fails.

## Distributions and correlations

Every component has a recorded distribution and variance in
`artifacts/plasma_painter/rewards/audit.json`. The generated plots are:

- `component_distributions.png`
- `component_correlations.png`
- `top20_contact_sheet.png`
- `bottom20_contact_sheet.png`

One pair remained above an absolute correlation of 0.9: total fidelity and
filament correspondence, approximately 0.921. This is partly structural—the
filament term is a major fidelity subscore—and partly a consequence of the
small binary-ablation pool. Filament correspondence is not removed because the
independent missing-filament and missing-orientation cases demonstrate distinct
failure modes. Before a meaningful RL run, broaden the valid candidate pool
and revisit the component weight if the correlation remains above 0.9.

The earlier perfect correlation between filament and orientation disappeared
after adding independent ablations. This confirms that the two measurements
are distinguishable even though the first gross attacks omitted both.

## Manual inspection

The top-20 and bottom-20 contact sheets were inspected. Every high-scoring
image retained the elongated source contour bands, signed red/blue structure,
and localized filament blooms. Bottom-ranked images were visibly blank,
grain-only, static, fixed-grid, coordinate-transposed, density-only, or missing
a required scientific mapping. Two valid but relatively weak fixtures appeared
at the edge of the bottom sheet; they retained the mapped structures and did
not resemble a reward hack.

## Remaining gate

No human choices exist, so aesthetic reward is a labeled neutral placeholder
in optimizer smoke tests. No VLM judge was substituted. Online language-model
RL remains blocked until real generated candidates and human preferences are
available.
