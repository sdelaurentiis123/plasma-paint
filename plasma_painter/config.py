"""Configuration and provenance helpers shared by command-line entry points."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    # Explicit environment overrides make the frozen config portable to NERSC
    # without baking account-specific paths into Git. They never trigger I/O.
    overrides = {
        "PLASMA_PAINTER_SOURCE_PATH": ("data", "source_path"),
        "PLASMA_PAINTER_GEOMETRY_PATH": ("data", "geometry_path"),
        "PLASMA_PAINTER_MODEL_PATH": ("generation", "local_model_path"),
        "PLASMA_PAINTER_ARTIFACT_ROOT": ("project", "artifact_root"),
    }
    for environment_name, (section, key) in overrides.items():
        if value := os.environ.get(environment_name):
            config[section][key] = value
    config["_config_path"] = str(config_path)
    return config


def artifact_root(config: dict[str, Any]) -> Path:
    root = Path(config["project"]["artifact_root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    commit = run("rev-parse", "HEAD") or "UNCOMMITTED"
    return {"commit": commit, "dirty": bool(run("status", "--porcelain"))}


def write_json(path: str | Path, value: Any, *, overwrite: bool = True) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {target}")
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
