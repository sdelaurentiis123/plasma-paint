"""Summarize pairwise counts and repeated-pair self-consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasma_painter.config import artifact_root, load_config, write_json
from .schema import read_ratings


def _semantic_choice(item: dict) -> str:
    if item["choice"] in {"tie", "both_bad"}:
        return item["choice"]
    return item[f"{item['choice']}_program_hash"]


def summarize(config: dict) -> dict:
    path = Path(config["ratings"]["jsonl_path"])
    all_records = read_ratings(path)
    records = [item for item in all_records if item.get("kind") == "pairwise"]
    originals = {item["pair_id"]: item for item in records if not item.get("repeated")}
    repeats = [item for item in records if item.get("repeated")]
    compared = []
    for repeat in repeats:
        original = originals.get(repeat["pair_id"].removesuffix("-repeat"))
        if original:
            compared.append(_semantic_choice(original) == _semantic_choice(repeat))
    report = {
        "pairwise_judgments": len(records), "repeated_judgments": len(repeats),
        "matched_repeat_pairs": len(compared), "self_consistency": sum(compared) / len(compared) if compared else None,
        "scope": "personal_taste_alignment" if len({item["rater_id"] for item in records}) <= 1 else "multi_rater_sample",
    }
    root = artifact_root(config)
    triage = {item["reference_id"]: item["choice"] for item in all_records if item.get("kind") == "triage"}
    pool_path = root / "reference_pool" / "manifest.json"
    if pool_path.exists():
        source = json.loads(pool_path.read_text(encoding="utf-8"))["records"]
        buckets = {name: [item for item in source if triage.get(item["reference_id"]) == name] for name in ("love", "okay", "reject")}
        buckets["unrated"] = [item for item in source if item["reference_id"] not in triage]
        for name in ("love", "okay", "reject"):
            write_json(root / "reference_pool" / name / "manifest.json", {"bucket": name, "records": buckets[name]})
        write_json(root / "reference_pool" / "triage_snapshot.json", {"buckets": buckets, "training_eligible_bucket": "love"})
        report["triage_counts"] = {name: len(items) for name, items in buckets.items()}
    write_json(root / "ratings" / "summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True); args = parser.parse_args()
    print(json.dumps(summarize(load_config(args.config)), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
