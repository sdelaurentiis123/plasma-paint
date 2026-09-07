# Full-episode adapter comparison: unsuccessful transfer

Rusty job 6989566 completed in 18m10s (0.3028 allocated GPU-hours), exit 0.
Code 5532ec1. Six full 24-turn episodes: base versus saved tool-use LoRA,
matched seed 1701, shot 85604 frame 7, sections y=0/18/31, fixed permitted
Van Gogh references. Frame 7 was not used in tool SFT or its frame-6 format
evaluation. All inputs remain inside existing art_train caches. No weight
updates, RL, external uploads, raw held-out access or other-job changes occurred.

| Section | Base accepted /24 | SFT accepted /24 | Base coarse | SFT coarse |
|---|---:|---:|---:|---:|
| y=0 | 7 | 10 | 0.1239 | -0.0102 |
| y=18 | 14 | 3 | 0.2734 | 0.2245 |
| y=31 | 8 | 6 | 0.0322 | 0.0632 |

Total accepted actions: base 29/72 (40.3%), SFT 19/72 (26.4%). Mean coarse
correspondence: base .1432, SFT .0925. These are diagnostic descriptive
results, not a calibrated scientific gate or statistically reliable benchmark.
Only three frames/one seed were tested; no human preference comparison occurred.

Base retained 89 strokes; SFT retained 58. Base used watercolor, bristle,
ink and pastel; SFT also used those four media across its episodes. No distinct
learned artist style is demonstrated. Visual inspection of all six images:
base outputs are poorly organized crossing marks; adapter outputs are sparse,
especially y=18 (9 faint strokes, almost blank). Neither method reliably depicts
the dominant plasma structure. The y=18 correlation illustrates why a single
coarse score cannot substitute for completeness, nonemptiness and structure gates.

The earlier 0/3 to 2/3 format-evaluation improvement did not transfer to full
reference-conditioned gym episodes. Training used short prescribed-medium
demonstrations with two images; the episode prompt is open-ended and adds artist
references. This mismatch is a plausible contributor, not an experimentally
isolated cause. A better matched, multi-turn action curriculum should be tested
before larger training or reward optimization. Do not promote this adapter.

Artifacts: artifacts/plasma_painter/rusty-gym-compare-v1/
gym-adapter-comparison-v1 contains every painting, scientific reference, program,
trajectory and manifest. The run records dirty=true because the isolated worktree
has a local trained-adapter symlink; code commit and adapter SHA256 are recorded.

Recommendation: REVISE. No additional GPU run was launched by the monitor.
