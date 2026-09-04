"""Sample local model programs or build an explicitly labeled seed-fixture pool."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator

from plasma_painter.config import artifact_root, load_config, stable_hash, write_json
from plasma_painter.provenance import experiment_provenance

from .filter_programs import filter_directory
from .prompts import build_prompt


def _strip_fence(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1]
        clean = clean.rsplit("```", 1)[0]
    return clean.strip()


def _local_samples(config: dict[str, Any], prompt: str, count: int) -> Iterator[str]:
    local_path = os.environ.get("PLASMA_PAINTER_MODEL_PATH") or config["generation"].get("local_model_path")
    if not local_path:
        raise RuntimeError("generation.local_model_path is required for local model sampling; no data or prompt is sent remotely")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import torch

    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    for index in range(count):
        torch.manual_seed(int(config["project"]["seed"]) + index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config["project"]["seed"]) + index)
        output = model.generate(
            encoded,
            do_sample=True,
            temperature=float(config["generation"]["temperature"]),
            top_p=float(config["generation"]["top_p"]),
            max_new_tokens=int(config["generation"]["max_new_tokens"]),
            num_return_sequences=1,
        )[0]
        yield _strip_fence(tokenizer.decode(output[encoded.shape[1] :], skip_special_tokens=True))


def _fixture_samples(reference: str, count: int) -> list[str]:
    """Controlled hand-authored variations for infrastructure tests, never a claimed model baseline."""

    variants = []
    contour_opacities = [("0.34", value) for value in ("0.24", "0.28", "0.32", "0.38")]
    wash_opacities = [("opacity: 0.18", f"opacity: {value}") for value in ("0.12", "0.15", "0.20")]
    persistence = [("styleConfig.persistence || 0.86", f"styleConfig.persistence || {value}") for value in ("0.78", "0.84", "0.90")]
    edits = [(None, None)] + contour_opacities + wash_opacities + persistence
    for index in range(count):
        old, new = edits[index % len(edits)]
        code = reference if old is None else reference.replace(old, new, 1)
        # Make each content-addressed fixture distinct without changing semantics.
        code = code.replace("let seedValue = 0;", f"let seedValue = 0;\n  const fixtureVariant = {index};", 1)
        variants.append(code)
    return variants


def sample_programs(config: dict[str, Any], *, backend: str = "auto", count: int | None = None, prompt_version: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    root = artifact_root(config)
    feature_index = json.loads((root / "features" / "index.json").read_text(encoding="utf-8"))
    train_record = next(item for item in feature_index["clips"] if item["split"] == "art_train")
    frame = json.loads(Path(train_record["path"]).read_text(encoding="utf-8"))["frames"][0]
    selected_prompt = prompt_version or config["generation"].get("prompt_version", "optimized_v1")
    optimized = selected_prompt == "optimized_v1"
    prompt = build_prompt(frame, optimized=optimized)
    count = int(count or config["generation"]["candidates"])
    selected = backend
    if selected == "auto":
        selected = "local" if config["generation"].get("local_model_path") else "fixture"
    if selected == "local":
        programs = _local_samples(config, prompt, count)
        origin = "frozen_local_base_model"
    elif selected == "fixture":
        reference = Path(config["renderer"]["baseline_program"]).read_text(encoding="utf-8")
        programs = _fixture_samples(reference, count)
        origin = "hand_authored_seed_fixture_not_model_output"
    else:
        raise ValueError("backend must be auto, local, or fixture")
    output = root / "programs" / "candidates"
    output.mkdir(parents=True, exist_ok=True)
    prompt_hash = stable_hash(prompt)
    records = []
    for index, code in enumerate(programs):
        program_hash = stable_hash(code)
        code_path = output / f"{program_hash}.js"
        code_path.write_text(code.rstrip() + "\n", encoding="utf-8")
        record = {
            "candidate_id": f"candidate-{index:03d}",
            "program_hash": program_hash,
            "program_path": str(code_path.resolve()),
            "origin": origin,
            "base_model": config["generation"]["base_model"] if selected == "local" else None,
            "checkpoint": config["generation"]["base_model"] if selected == "local" else "fixture",
            "prompt_version": selected_prompt,
            "prompt_hash": prompt_hash,
            "seed": int(config["project"]["seed"]) + index,
        }
        write_json(output / f"{program_hash}.json", record)
        records.append(record)
        print(f"Saved {selected_prompt} painter {index + 1}/{count}: {program_hash}", flush=True)
    prompt_path = root / "programs" / "prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    summary = {"backend": selected, "origin": origin, "count": len(records), "prompt_path": str(prompt_path), "records": records}
    summary["provenance"] = experiment_provenance(config, wall_seconds=time.perf_counter() - started, stage="program_sampling")
    write_json(root / "programs" / f"sample_manifest-{selected_prompt}.json", summary)
    write_json(root / "programs" / "sample_manifest.json", summary)
    summary["filter_report"] = filter_directory(config)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=("auto", "local", "fixture"), default="auto")
    parser.add_argument("--count", type=int)
    parser.add_argument("--prompt-version", choices=("baseline_v1", "optimized_v1"))
    args = parser.parse_args()
    result = sample_programs(load_config(args.config), backend=args.backend, count=args.count, prompt_version=args.prompt_version)
    print(json.dumps({"backend": result["backend"], "origin": result["origin"], "count": result["count"], "accepted_fraction": result["filter_report"]["accepted_fraction"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
