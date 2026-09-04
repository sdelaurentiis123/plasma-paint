"""Reward aggregation with an uncompensated hard validity gate."""

from __future__ import annotations

from typing import Any


def aggregate_reward(
    *,
    gate: dict[str, Any],
    fidelity: float,
    aesthetic: float | None,
    temporal: float,
    diversity: float,
    efficiency: float,
    weights: dict[str, float],
    strong_negative: float,
) -> dict[str, Any]:
    if not gate["valid"]:
        return {"reward": float(strong_negative), "gate": gate, "aesthetic_status": "not_evaluated_due_to_gate"}
    # A neutral placeholder permits pipeline smoke tests but is explicitly not a
    # human-preference result and cannot establish the aesthetic research claim.
    aesthetic_value = 0.5 if aesthetic is None else float(aesthetic)
    components = {
        "fidelity": float(fidelity),
        "aesthetic": aesthetic_value,
        "temporal": float(temporal),
        "diversity": float(diversity),
        "efficiency": float(efficiency),
    }
    reward = sum(float(weights[name]) * value for name, value in components.items())
    return {
        "reward": float(reward),
        "gate": gate,
        "components": components,
        "aesthetic_status": "human_ratings" if aesthetic is not None else "neutral_placeholder_no_human_rating",
    }
