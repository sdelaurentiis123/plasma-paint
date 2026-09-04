"""Conservative static validation for generated painter JavaScript."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .dsl import ALLOWED_OPERATIONS


FORBIDDEN_PATTERNS = {
    "network": r"\b(fetch|XMLHttpRequest|WebSocket|EventSource|navigator)\b",
    "dom": r"\b(document|window|localStorage|sessionStorage|indexedDB)\b",
    "dynamic_code": r"\b(eval|Function|AsyncFunction|importScripts)\b|\bimport\s*\(",
    "host_access": r"\b(process|require|module|globalThis|Deno|Bun)\b",
    "workers_timers": r"\b(Worker|SharedWorker|setTimeout|setInterval|requestAnimationFrame)\b",
    "prototype_escape": r"\b(__proto__|prototype|constructor|Reflect|Proxy|WebAssembly|SharedArrayBuffer|Atomics)\b",
    "unbounded_loop": r"\bwhile\s*\(|\bdo\s*\{",
    "class_definition": r"\bclass\s+",
}


@dataclass
class ProgramValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)
    api_calls: list[str] = field(default_factory=list)
    code_bytes: int = 0


def validate_program(code: str, *, max_bytes: int = 24_000, node_check: bool = True) -> ProgramValidation:
    errors: list[str] = []
    encoded = code.encode("utf-8")
    if len(encoded) > max_bytes:
        errors.append(f"program exceeds {max_bytes} bytes")
    if not re.search(r"export\s+function\s+createPainter\s*\(\s*api\s*,\s*styleConfig\s*\)", code):
        errors.append("missing exact exported createPainter(api, styleConfig) function")
    if not re.search(r"\breset\s*(?::\s*function\s*)?\(\s*seed\s*\)", code):
        errors.append("missing reset(seed)")
    if not re.search(r"\brenderFrame\s*(?::\s*function\s*)?\(\s*frameFeatures\s*,\s*time\s*,\s*persistentState\s*\)", code):
        errors.append("missing renderFrame(frameFeatures, time, persistentState)")
    for label, pattern in FORBIDDEN_PATTERNS.items():
        if re.search(pattern, code):
            errors.append(f"forbidden JavaScript capability: {label}")
    api_calls = sorted(set(re.findall(r"\bapi\.([A-Za-z_$][\w$]*)\s*\(", code)))
    unknown = sorted(set(api_calls) - set(ALLOWED_OPERATIONS) - {"reset"})
    if unknown:
        errors.append(f"unknown painter API calls: {', '.join(unknown)}")
    if not set(api_calls).intersection(ALLOWED_OPERATIONS):
        errors.append("program does not use the painting DSL")
    # Direct Canvas, p5, and ambient-call surfaces are outside the DSL.
    if re.search(r"\b(getContext|createCanvas|brush\.|ctx\.|canvas\.)", code):
        errors.append("direct canvas or p5 access is forbidden")
    if node_check and not errors:
        transformed = re.sub(r"\bexport\s+function\s+createPainter", "function createPainter", code, count=1)
        with tempfile.TemporaryDirectory(prefix="plasma-painter-check-") as temporary:
            path = Path(temporary) / "candidate.js"
            path.write_text(transformed, encoding="utf-8")
            result = subprocess.run(
                ["node", "--check", str(path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        if result.returncode != 0:
            errors.append("JavaScript syntax check failed: " + result.stderr.strip().splitlines()[-1])
    return ProgramValidation(valid=not errors, errors=errors, api_calls=api_calls, code_bytes=len(encoded))
