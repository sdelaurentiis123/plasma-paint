"""Online grouped-relative program RL, plus a cheap categorical environment smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from plasma_painter.config import artifact_root, git_state, load_config, stable_hash, write_json
from plasma_painter.generation.prompts import build_prompt
from plasma_painter.provenance import experiment_provenance

from .adapters import hardware_record, require_local_model
from .environment import evaluate_program
from .model_runtime import load_lora_model, sequence_logprob


def _training_clips(config: dict) -> list[dict]:
    root = artifact_root(config)
    index = json.loads((root / "features" / "index.json").read_text(encoding="utf-8"))
    return [json.loads(Path(item["path"]).read_text(encoding="utf-8")) for item in index["clips"] if item["split"] == "art_train"]


def categorical_smoke(config: dict, *, steps: int) -> dict:
    """Exercise grouped advantages on real sandbox rewards without claiming LM RL."""

    root = artifact_root(config)
    candidates = [item for item in json.loads((root / "programs" / "filter_report.json").read_text(encoding="utf-8"))["results"] if item["accepted"]][:4]
    clip = _training_clips(config)[0]
    # Two adjacent real frames keep this CPU smoke cheap; --execute uses full clips.
    smoke_clip = {**clip, "frames": clip["frames"][:2], "frame_count": 2, "frame_stop": clip["frame_start"] + 2}
    measured = []
    for candidate in candidates:
        code = Path(candidate["program_path"]).read_text(encoding="utf-8")
        result = evaluate_program(code, smoke_clip, config, seed=int(candidate["seed"]), scale=0.125)
        measured.append({"program_hash": candidate["program_hash"], "program_path": candidate["program_path"], "reward": result["aggregate"]["reward"], "components": result})
    logits = torch.zeros(len(candidates), requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.08)
    generator = torch.Generator().manual_seed(int(config["project"]["seed"]))
    logs = []
    group_size = int(config["training"]["group_size"])
    for step in range(steps):
        probabilities = torch.softmax(logits, dim=0)
        choices = torch.multinomial(probabilities, group_size, replacement=True, generator=generator)
        rewards = torch.tensor([measured[index]["reward"] for index in choices.tolist()], dtype=torch.float32)
        advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
        loss = -(advantages.detach() * torch.log_softmax(logits, dim=0)[choices]).mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        logs.append({"step": step, "program_hashes": [measured[index]["program_hash"] for index in choices.tolist()], "rewards": rewards.tolist(), "advantages": advantages.tolist(), "loss": float(loss.detach())})
    return {
        "executed": False,
        "status": "real_renderer_environment_and_grouped_policy_math_smoke_not_lm_adapter",
        "steps": steps,
        "program_evaluations": measured,
        "optimizer_log": logs,
        "final_probabilities": torch.softmax(logits.detach(), dim=0).tolist(),
    }


def _extract_javascript(text: str) -> str:
    clean = text.strip()
    if "```" in clean:
        pieces = clean.split("```")
        clean = pieces[1]
        if clean.lstrip().startswith("javascript"):
            clean = clean.lstrip()[10:]
        elif clean.lstrip().startswith("js"):
            clean = clean.lstrip()[2:]
    return clean.strip()


def execute_grpo(config: dict, *, steps: int) -> dict:
    """Sample programs online, render full training clips, and update LoRA with grouped advantages."""

    require_local_model(config)
    root = artifact_root(config)
    adapter = root / "adapters" / "tcv-watercolor-dpo"
    if not adapter.exists():
        adapter = root / "adapters" / "tcv-watercolor-sft"
    if not adapter.exists():
        raise FileNotFoundError("run SFT (and preferably DPO) before online GRPO")
    model, tokenizer = load_lora_model(config, require_local_model(config), adapter_path=adapter)
    optimizer = torch.optim.AdamW((item for item in model.parameters() if item.requires_grad), lr=float(config["training"]["learning_rate"]))
    clips = _training_clips(config)[:16]
    group_size = int(config["training"]["group_size"])
    output = root / "training" / "grpo_rollouts.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = []
    started = time.perf_counter()
    for step in range(steps):
        clip = clips[step % len(clips)]
        prompt = build_prompt(clip["frames"][0], optimized=True)
        encoded = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=float(config["generation"]["temperature"]),
                top_p=float(config["generation"]["top_p"]),
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id,
            )
        completions = [_extract_javascript(tokenizer.decode(row[encoded["input_ids"].shape[1] :], skip_special_tokens=True)) for row in generated]
        results = [evaluate_program(code, clip, config, seed=int(config["project"]["seed"]) + step * group_size + index, scale=0.25) for index, code in enumerate(completions)]
        rewards = torch.tensor([item["aggregate"]["reward"] for item in results], device=next(model.parameters()).device)
        advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
        logps = torch.stack([sequence_logprob(model, tokenizer, prompt, code) for code in completions])
        with torch.no_grad():
            reference = torch.stack([sequence_logprob(model, tokenizer, prompt, code, reference=True) for code in completions])
        kl = logps - reference
        loss = -(advantages.detach() * logps).mean() + 0.01 * kl.mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        entry = {"step": step, "clip_id": clip["clip_id"], "programs": [{"program_hash": stable_hash(code), "code": code, "reward": result["aggregate"], "fidelity": result.get("fidelity"), "temporal": result.get("temporal"), "valid": result["valid"]} for code, result in zip(completions, results)], "advantages": advantages.detach().cpu().tolist(), "loss": float(loss.detach().cpu())}
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        logs.append({"step": step, "mean_reward": float(rewards.mean().cpu()), "valid_fraction": float(np.mean([item["valid"] for item in results])), "loss": float(loss.detach().cpu())})
    adapter_output = root / "adapters" / "tcv-watercolor-grpo"
    model.save_pretrained(adapter_output)
    return {"executed": True, "status": "online_lm_grpo", "steps": steps, "adapter_path": str(adapter_output), "rollout_log": str(output), "wall_seconds": time.perf_counter() - started, "steps_summary": logs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    started = time.perf_counter()
    steps = int(args.steps or config["training"]["grpo_steps"])
    result = execute_grpo(config, steps=steps) if args.execute else categorical_smoke(config, steps=steps)
    record = {**result, "stage": "grpo", "hardware": hardware_record(), "git": git_state()}
    record["provenance"] = experiment_provenance(config, wall_seconds=time.perf_counter() - started, stage="grpo")
    write_json(artifact_root(config) / "training" / "grpo_run.json", record)
    print(json.dumps({key: record[key] for key in ("stage", "status", "executed", "steps")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
