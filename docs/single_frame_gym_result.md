# First model-driven single-frame gym episode

Rusty job 6989388, code 428a9e0, frozen Qwen2.5-VL-7B, one GPU with a
30-minute cap. Episode output completed after 24 turns. No SFT, DPO or RL
update occurred. This tests supplied tools and feedback, not trained aesthetics.

Input: existing hash-verified shot 85604 training cache, section y=18, frame 0.
References: already-staged 28560.jpg and 14586.jpg (Van Gogh). No raw data,
held-out shot, other job, or external inference service was accessed.

Local evidence: artifacts/plasma_painter/rusty-gym-v1/gym-vision-v1, with
painting.png, scientific.png, program.json, trajectory.jsonl, manifest.json,
and run.json. The separate job log is retained in the parent directory.

Results:

- 12/24 actions accepted; 12 rejected; 44 retained strokes.
- Media: watercolor 24, pastel 8, bristle 8, ink 4.
- Coarse absolute-fluctuation/pigment rank correlation: 0.2133 (diagnostic only).
- Errors: unsupported media 5; out-of-range coordinates 4; invalid JSON 2;
  invalid hexadecimal color 1. Some rejected actions also contain malformed
  point arrays, pixel coordinates, or excessive opacity; validation reports
  the first error rather than all errors in a batch.
- Visual inspection: scattered crossing lines and broad colored marks, missing
  the scientific image's dominant approximately horizontal structures. This
  is not a successful plasma representation or demonstrated artist imitation.

The supplied-tool loop now executes model actions and returns canvas observations;
the model no longer needs to invent executable rendering code. However, it still
needs action-format demonstrations/training and better spatial instruction. The
next controlled experiment should separate learning valid tool use from learning
style, with schema-constrained actions and a supervised action curriculum. Keep
scientific RL disabled until a complete single-frame fidelity gate is audited.
Do not label this trajectory a preferred/accepted SFT style target.
