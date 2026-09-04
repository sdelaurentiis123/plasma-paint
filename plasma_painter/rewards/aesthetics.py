"""Human preference is authoritative; no uncalibrated VLM reward is substituted."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json


def preference_rates(path: str | Path) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, float]] = defaultdict(lambda: {"wins": 0.0, "losses": 0.0, "ties": 0.0})
    source = Path(path)
    if not source.exists():
        return {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("kind") != "pairwise":
            continue
        left, right, choice = item["left_program_hash"], item["right_program_hash"], item["choice"]
        if choice == "left":
            counts[left]["wins"] += 1
            counts[right]["losses"] += 1
        elif choice == "right":
            counts[right]["wins"] += 1
            counts[left]["losses"] += 1
        elif choice == "tie":
            counts[left]["ties"] += 1
            counts[right]["ties"] += 1
    output = {}
    for program, record in counts.items():
        total = record["wins"] + record["losses"] + record["ties"]
        output[program] = {**record, "score": (record["wins"] + 0.5 * record["ties"]) / total if total else 0.5, "judgments": total}
    return output
