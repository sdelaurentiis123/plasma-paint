# Bounded medium-code SFT smoke

Submitted Rusty job **6985156**, source **7eaadf2**, dedicated directory `/mnt/home/sdelaurentiis/ceph/plasma-paint-runs/20260904-rusty/media-sft-v3`. Initial scheduler state: PENDING (`ReqNodeNotAvail`). No optimizer update had occurred at submission. A ten-minute read-only monitor is active for this job only.

Preflight: 28 local tests passed. Rusty did not have pytest installed; direct remote CPU checks passed baseline sandbox and nonempty pixel rendering on frames 0–1 (196 operations/frame). Chat-formatted SFT example: 3,038 tokens, 746 supervised completion tokens; 16 staged training clips. The CPU preflight initially waited on filesystem I/O and completed successfully before submission.

User authorized launching training on 2026-09-04. Dedicated Rusty run; one shared Qwen2.5-Coder-7B-Instruct base, rank-16 LoRA, 40 SFT steps, no DPO or GRPO. Job script: `scripts/rusty_media_sft.slurm`. Cap: one GPU, two hours; expected sampling about 35–50 minutes from the preceding four-program run, with training duration uncertain. No paid API spending or multi-node allocation.

Generate 20 fresh `media_v3` programs. Require 20 actual base-model candidates and >=80% accepted with nonempty pixel renders before any optimizer update. Failure stops the job. The pixel check compares output against the same seeded paper and is only a validity check, not a fidelity reward. Runtime/media parity and aesthetic/reward audits remain incomplete: this is explicitly an infrastructure SFT smoke, not a scientific pilot or preference alignment claim.

Use only previously cached old-85604 training features, nested art_train frames within 0–288. The dataset builder uses up to nine training clips (first 72 frames), paired with the baseline and up to eight accepted generated programs. No museum artwork, reference-only links, human ratings, full-shot new data, held-out blocks, or shot 85606 is consumed. Sampling/filtering uses the first training clip's frame summary and first two training frames respectively.

The SFT prompt uses the exact medium-aware contract, serialized using the model's chat template. Training rejects empty supervision and non-finite losses. Per-step loss is logged. An adapter is saved only after all 40 steps; preemption can prevent adapter completion. Final renders in this script are base-program references, not adapter-generated results. Adapter inference and comparison are still required after successful training.

The staged feature cache and base weights are reused read-only, while the feature index is rebased in the new run. Outputs must not share a prior experiment directory. No unrelated Rusty/NERSC jobs are modified. Source, configuration, dataset manifest, normalization, dependency freeze, hardware and model metadata are retained by the existing provenance path.
