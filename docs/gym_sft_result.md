# Tool-use LoRA smoke result

Rusty job 6989532 completed successfully in 10m09s on one RTX PRO 6000
Blackwell (0.1692 allocated GPU-hours). Code: 0f0d5c1. The run performed
32 actual adapter optimizer steps. All logged losses were finite; first loss
0.8811, final loss 0.5293. The measured training/evaluation section was 63.4s;
the scheduler allocation includes imports, model loading and output saving.

Three fixed format-evaluation prompts from frame 6, sections y=0/18/31, were
not used for weight updates. Valid four-stroke JSON actions improved from 0/3
to 2/3. The remaining response contained zero-length strokes. Accepted outputs
also show repeated coordinates: syntax validity is not evidence of good spatial
composition, scientific fidelity or artist imitation. Three prompts provide no
reliable generalization estimate. All prompts request watercolor on this tiny
evaluation; the other media remain untested by the before/after comparison.

The 9.7MiB adapter_model.safetensors and adapter_config.json exist under the
isolated gym-tools-sft-v1/artifacts/plasma_painter/adapters/tcv-tools-sft-v1
directory on Rusty. Weights were not downloaded by the completion monitor.
Local evidence: artifacts/plasma_painter/rusty-gym-sft-v1/run.json and
gym-sft-6989532.log. No held-out raw data, other jobs or external APIs were used.

This is supervised action-mechanics training, not aesthetic training or RL.
Recommendation: REVISE/continue the tool curriculum and evaluate complete
adapter-driven gym episodes before claiming painter improvement. Do not promote
the adapter to the website or enable the incomplete RL reward on this evidence.
