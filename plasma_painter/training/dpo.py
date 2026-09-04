"""Build human chosen/rejected pairs and optionally run an offline DPO adapter update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from plasma_painter.config import artifact_root, git_state, load_config, write_json
from plasma_painter.generation.prompts import build_prompt
from plasma_painter.ratings.schema import read_ratings
from plasma_painter.provenance import experiment_provenance

from .adapters import hardware_record, require_local_model
from .model_runtime import load_lora_model, sequence_logprob


def build_dpo_dataset(config: dict) -> list[dict]:
    root = artifact_root(config)
    ratings_path = Path(config["ratings"]["jsonl_path"])
    candidates = {}
    report = root / "programs" / "filter_report.json"
    if report.exists():
        candidates = {item["program_hash"]: Path(item["program_path"]).read_text(encoding="utf-8") for item in json.loads(report.read_text(encoding="utf-8"))["results"]}
    feature_index = json.loads((root / "features" / "index.json").read_text(encoding="utf-8"))
    frames = {
        item["clip_id"]: json.loads(Path(item["path"]).read_text(encoding="utf-8"))["frames"][0]
        for item in feature_index["clips"]
        if item["split"] == "art_train"
    }
    all_ratings = read_ratings(ratings_path)
    love_references = {
        item["reference_id"]
        for item in all_ratings
        if item.get("kind") == "triage" and item.get("choice") == "love"
    }
    pairs = []
    for rating in all_ratings:
        if rating.get("kind") != "pairwise" or rating.get("choice") not in {"left", "right"}:
            continue
        if len(rating.get("order", [])) != 2 or not set(rating["order"]).issubset(love_references):
            continue
        left, right = rating["left_program_hash"], rating["right_program_hash"]
        if left not in candidates or right not in candidates or rating["clip_id"] not in frames:
            continue
        chosen, rejected = (left, right) if rating["choice"] == "left" else (right, left)
        pairs.append({"prompt": build_prompt(frames[rating["clip_id"]], optimized=True), "chosen": candidates[chosen], "rejected": candidates[rejected], "clip_id": rating["clip_id"], "rating_id": rating["rating_id"]})
    path = root / "training" / "dpo_pairs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in pairs), encoding="utf-8")
    return pairs


def objective_smoke() -> dict[str, float]:
    margin = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.SGD([margin], lr=0.2)
    first = None
    for _ in range(12):
        loss = -F.logsigmoid(0.1 * margin)
        first = float(loss.detach()) if first is None else first
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return {"loss_first": float(first), "loss_last": float(loss), "learned_margin": float(margin.detach())}


def execute_dpo(config: dict, pairs: list[dict]) -> dict:
    if not pairs:
        raise RuntimeError("DPO requires at least one non-tie human chosen/rejected pair")
    require_local_model(config)
    sft_path = artifact_root(config) / "adapters" / "tcv-watercolor-sft"
    if not sft_path.exists():
        raise FileNotFoundError("run the SFT adapter before DPO")
    model, tokenizer = load_lora_model(config, require_local_model(config), adapter_path=sft_path)
    optimizer = torch.optim.AdamW((item for item in model.parameters() if item.requires_grad), lr=float(config["training"]["learning_rate"]))
    beta = 0.1
    losses = []
    started = time.perf_counter()
    for step in range(int(config["training"]["dpo_steps"])):
        item = pairs[step % len(pairs)]
        chosen = sequence_logprob(model, tokenizer, item["prompt"], item["chosen"])
        rejected = sequence_logprob(model, tokenizer, item["prompt"], item["rejected"])
        with torch.no_grad():
            ref_chosen = sequence_logprob(model, tokenizer, item["prompt"], item["chosen"], reference=True)
            ref_rejected = sequence_logprob(model, tokenizer, item["prompt"], item["rejected"], reference=True)
        loss = -F.logsigmoid(beta * ((chosen - rejected) - (ref_chosen - ref_rejected)))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    output = artifact_root(config) / "adapters" / "tcv-watercolor-dpo"
    model.save_pretrained(output)
    return {"executed": True, "steps": len(losses), "loss_first": losses[0], "loss_last": losses[-1], "adapter_path": str(output), "wall_seconds": time.perf_counter() - started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    started = time.perf_counter()
    pairs = build_dpo_dataset(config)
    if args.execute:
        result = execute_dpo(config, pairs)
    else:
        result = {"executed": False, "status": "objective_smoke_only_no_adapter_update", "human_pairs": len(pairs), "objective_smoke": objective_smoke()}
    record = {**result, "stage": "dpo", "human_preference_required": True, "hardware": hardware_record(), "git": git_state()}
    record["provenance"] = experiment_provenance(config, wall_seconds=time.perf_counter() - started, stage="dpo")
    write_json(artifact_root(config) / "training" / "dpo_run.json", record)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
