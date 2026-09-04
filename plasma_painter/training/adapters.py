"""Shared-base adapter plans and strict execution preflight."""

from __future__ import annotations

import platform
import os
from pathlib import Path
from typing import Any


ADAPTERS = {
    "tcv-watercolor": {"status": "pilot_target", "style": "watercolor", "reference_bucket": "love"},
    "tcv-ink": {"status": "extension_only_until_pilot_passes", "style": "ink", "reference_bucket": "love"},
    "tcv-gouache": {"status": "extension_only_until_pilot_passes", "style": "gouache", "reference_bucket": "love"},
}


def adapter_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "shared_base_model": config["generation"]["base_model"],
        "local_model_path": config["generation"].get("local_model_path"),
        "full_models_to_train": 0,
        "adapters": ADAPTERS,
        "lora": {
            "rank": config["training"]["lora_rank"],
            "alpha": config["training"]["lora_alpha"],
            "dropout": config["training"]["lora_dropout"],
            "target_modules": "all-linear",
        },
    }


def require_local_model(config: dict[str, Any]) -> Path:
    local = os.environ.get("PLASMA_PAINTER_MODEL_PATH") or config["generation"].get("local_model_path")
    if not local:
        raise RuntimeError(
            "--execute requires generation.local_model_path pointing to an already staged model; "
            "the training command never downloads weights or transmits plasma-derived data"
        )
    path = Path(local).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def hardware_record() -> dict[str, Any]:
    try:
        import torch

        cuda = torch.cuda.is_available()
        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        cuda, devices, mps = False, [], False
    return {"platform": platform.platform(), "cuda": cuda, "cuda_devices": devices, "mps": mps}
