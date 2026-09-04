"""Validated append-only JSONL rating storage."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


PAIRWISE_CHOICES = frozenset({"left", "right", "tie", "both_bad"})
TRIAGE_CHOICES = frozenset({"love", "okay", "reject"})


def validate_rating(record: dict[str, Any]) -> dict[str, Any]:
    kind = record.get("kind")
    required = {"rating_id", "rater_id", "timestamp", "clip_id", "seed"}
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError("missing rating fields: " + ", ".join(missing))
    if kind == "pairwise":
        if record.get("choice") not in PAIRWISE_CHOICES:
            raise ValueError("invalid pairwise choice")
        for name in ("left_program_hash", "right_program_hash", "left_checkpoint", "right_checkpoint", "order"):
            if name not in record:
                raise ValueError(f"missing pairwise field {name}")
    elif kind == "triage":
        if record.get("choice") not in TRIAGE_CHOICES or "program_hash" not in record:
            raise ValueError("invalid triage record")
    else:
        raise ValueError("rating kind must be pairwise or triage")
    return record


def append_rating(path: str | Path, record: dict[str, Any]) -> None:
    record = dict(record)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    validate_rating(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_ratings(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
