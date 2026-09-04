"""Prepare train-only SFT data and optionally run a local LoRA update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from plasma_painter.config import artifact_root, git_state, load_config, write_json
from plasma_painter.generation.prompts import build_prompt
from plasma_painter.provenance import experiment_provenance

from .adapters import adapter_plan, hardware_record, require_local_model
from .model_runtime import completion_tensors, load_lora_model


def build_sft_dataset(config: dict) -> list[dict]:
    root = artifact_root(config)
    index = json.loads((root / "features" / "index.json").read_text(encoding="utf-8"))
    train_clips = [item for item in index["clips"] if item["split"] == "art_train"]
    programs = [Path(config["renderer"]["baseline_program"])]
    filter_path = root / "programs" / "filter_report.json"
    if filter_path.exists():
        filtered = json.loads(filter_path.read_text(encoding="utf-8"))["results"]
        programs += [Path(item["program_path"]) for item in filtered if item["accepted"]][:8]
    examples = []
    for clip_item, program in zip(train_clips, programs):
        frame = json.loads(Path(clip_item["path"]).read_text(encoding="utf-8"))["frames"][0]
        examples.append(
            {
                "prompt": build_prompt(frame, media=True),
                "completion": program.read_text(encoding="utf-8"),
                "clip_id": clip_item["clip_id"],
                "split": "art_train",
                "origin": "reference_or_accepted_fixture",
            }
        )
    output = root / "training" / "sft_examples.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in examples), encoding="utf-8")
    return examples


def execute_sft(config: dict, examples: list[dict]) -> dict:
    filter_report = artifact_root(config) / "programs" / "filter_report.json"
    if not filter_report.exists():
        raise FileNotFoundError("run the generated-program filter before SFT")
    actual = [
        item
        for item in json.loads(filter_report.read_text(encoding="utf-8"))["results"]
        if item.get("origin") == "frozen_local_base_model"
    ]
    accepted = sum(item.get("accepted", False) and item.get("pixel_render_valid", False) for item in actual)
    if len(actual) < 20 or accepted / max(len(actual), 1) < 0.80:
        raise RuntimeError("SFT is gated on at least 20 actual frozen-model candidates and 80% filtered render success")
    model_path = require_local_model(config)
    torch.manual_seed(int(config['project']['seed']))
    model, tokenizer = load_lora_model(config, model_path)
    model.train()
    optimizer = torch.optim.AdamW((item for item in model.parameters() if item.requires_grad), lr=float(config["training"]["learning_rate"]))
    steps = int(config["training"]["sft_steps"])
    losses = []
    started = time.perf_counter()
    for step in range(steps):
        item = examples[step % len(examples)]
        input_ids, attention, labels = completion_tensors(tokenizer, item["prompt"], item["completion"], next(model.parameters()).device)
        loss = model(input_ids=input_ids, attention_mask=attention, labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite SFT loss; stopping without saving an adapter')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        print(f"SFT step {step + 1}/{steps}: loss={losses[-1]:.5f}", flush=True)
    output = artifact_root(config) / "adapters" / "tcv-watercolor-sft"
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    return {"executed": True, "steps": steps, "loss_first": losses[0], "loss_last": losses[-1], "adapter_path": str(output), "wall_seconds": time.perf_counter() - started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true", help="Perform the local-model LoRA update; never downloads weights.")
    args = parser.parse_args()
    config = load_config(args.config)
    started = time.perf_counter()
    examples = build_sft_dataset(config)
    result = execute_sft(config, examples) if args.execute else {
        "executed": False,
        "status": "dataset_and_entrypoint_ready_no_model_update",
        "examples": len(examples),
        "reason": "--execute not supplied and no local base-model path is configured",
    }
    record = {**result, "stage": "sft", "adapter_plan": adapter_plan(config), "hardware": hardware_record(), "git": git_state()}
    record["provenance"] = experiment_provenance(config, wall_seconds=time.perf_counter() - started, stage="sft")
    write_json(artifact_root(config) / "training" / "sft_run.json", record)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
