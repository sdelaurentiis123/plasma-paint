"""Hard scientific and execution gate; appearance can never override failure."""

from __future__ import annotations

from typing import Any


def validity_gate(
    *,
    compiles: bool,
    sandbox_valid: bool,
    fidelity: dict[str, Any],
    minimum_nonempty_fraction: float,
    minimum_coarse_spearman: float,
    minimum_extrema_recall: float,
    minimum_filament_correspondence: float = 0.5,
    minimum_orientation_agreement: float = 0.25,
) -> dict[str, Any]:
    checks = {
        "compiles": bool(compiles),
        "uses_only_allowed_dsl_and_runtime_valid": bool(sandbox_valid),
        "produces_nonempty_output": (
            fidelity.get("nonempty_fraction", 0.0) >= minimum_nonempty_fraction
            and fidelity.get("structural_marks", 0.0) > 0
        ),
        "coarse_spearman_floor": fidelity.get("coarse_spearman_raw", -1.0) >= minimum_coarse_spearman,
        "extrema_recall_floor": fidelity.get("extrema", 0.0) >= minimum_extrema_recall,
        "filament_correspondence_floor": fidelity.get("filament", 0.0) >= minimum_filament_correspondence,
        "orientation_agreement_floor": fidelity.get("orientation", 0.0) >= minimum_orientation_agreement,
        "not_static_when_data_changes": not bool(fidelity.get("static_failure", False)),
    }
    return {"valid": all(checks.values()), "checks": checks, "failed": [name for name, passed in checks.items() if not passed]}
