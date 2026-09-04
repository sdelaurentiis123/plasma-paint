"""Uniform experiment provenance records without touching external services."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import platform
from pathlib import Path
from typing import Any

from plasma_painter.config import artifact_root, git_state, sha256_file


def _hashed(path: Path) -> dict[str, Any] | None:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)} if path.exists() else None


def experiment_provenance(config: dict[str, Any], *, wall_seconds: float, stage: str) -> dict[str, Any]:
    root = artifact_root(config)
    ratings = Path(config["ratings"]["jsonl_path"])
    packages = {}
    for name in ("numpy", "scipy", "Pillow", "torch", "transformers", "peft", "trl"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        import torch

        hardware = {"cuda": torch.cuda.is_available(), "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}
    except ImportError:
        hardware = {"cuda": False, "cuda_devices": [], "mps": False}
    return {
        "stage": stage, "git": git_state(), "config": _hashed(Path(config["_config_path"])),
        "base_model": config["generation"]["base_model"], "tokenizer": config["generation"]["base_model"],
        "adapter_configuration": {key: config["training"][key] for key in ("style", "lora_rank", "lora_alpha", "lora_dropout", "learning_rate")},
        "dataset_manifest": _hashed(root / "manifests" / "dataset_manifest.json"),
        "feature_normalization": _hashed(root / "features" / "normalization.json"),
        "reward_normalization": _hashed(root / "rewards" / "normalization.json"),
        "exact_split": config["data"]["art_split"], "seeds": {"project": config["project"]["seed"], "evaluation_count": config["evaluation"]["seed_count"]},
        "renderer_runtime_version": "canvas-runtime-0.1.0", "reward_version": "gated-reward-0.1.0",
        "human_rating_snapshot": _hashed(ratings), "hardware": {"platform": platform.platform(), **hardware},
        "wall_seconds": float(wall_seconds), "estimated_compute": {"gpu_hours": 0.0 if not hardware["cuda"] else None, "scope": "local smoke" if not hardware["cuda"] else "GPU run"},
        "dependency_lock": _hashed(Path("requirements-lock.txt")), "packages": packages,
    }
