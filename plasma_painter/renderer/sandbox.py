"""Resource-bounded child-process execution for statically valid painters."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .compiler import ProgramValidation, validate_program
from .dsl import validate_operations


@dataclass
class SandboxResult:
    valid: bool
    operations_by_frame: list[list[dict[str, Any]]]
    validation: dict[str, Any]
    error: str | None
    elapsed_ms: float | None


def run_program(
    code: str,
    frames: list[dict[str, Any]],
    *,
    style: dict[str, Any] | None = None,
    seed: int = 0,
    max_runtime_ms: int = 1500,
    max_operations: int = 1200,
    max_path_points: int = 256,
) -> SandboxResult:
    validation = validate_program(code)
    if not validation.valid:
        return SandboxResult(False, [], asdict(validation), "; ".join(validation.errors), None)
    runner = Path(__file__).with_name("sandbox_runner.mjs")
    request = {
        "code": code,
        "frames": frames,
        "style": style or {},
        "seed": int(seed),
        "maxOperations": int(max_operations),
        "maxPathPoints": int(max_path_points),
        "vmTimeoutMs": max(50, int(max_runtime_ms // max(len(frames), 1))),
    }
    try:
        result = subprocess.run(
            ["node", "--max-old-space-size=96", str(runner)],
            input=json.dumps(request, separators=(",", ":")),
            capture_output=True,
            text=True,
            check=False,
            timeout=max_runtime_ms / 1000.0 + 0.75,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(False, [], asdict(validation), "sandbox timeout", None)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "sandbox child failed").strip()[-1000:]
        return SandboxResult(False, [], asdict(validation), message, None)
    try:
        response = json.loads(result.stdout)
        operations = response["operationsByFrame"]
        for frame_operations in operations:
            validate_operations(
                frame_operations,
                max_operations=max_operations,
                max_path_points=max_path_points,
            )
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        return SandboxResult(False, [], asdict(validation), f"invalid sandbox output: {error}", None)
    return SandboxResult(True, operations, asdict(validation), None, float(response["elapsedMs"]))

