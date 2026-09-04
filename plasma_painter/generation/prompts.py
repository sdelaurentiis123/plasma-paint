"""Versioned prompts for frozen-model and optimized-prompt baselines."""

from __future__ import annotations

import json
from typing import Any

from plasma_painter.renderer.dsl import api_documentation


STYLE_BRIEF = """Paint TCV edge turbulence as restrained mineral watercolor on warm paper.
Density controls washes; signed density-fluctuation contours control paths; extracted
filaments control blooms; gradient vectors may orient dry marks; tracked events control
persistent pigment. Preserve the separatrix and temporal changes. Never invent particles."""


def representative_summary(frame: dict[str, Any]) -> dict[str, Any]:
    """Return semantics and counts only; never include a private raw field array."""

    return {
        "schema_version": frame["schema_version"],
        "source": {"shot": frame["source"]["shot"], "split": "training-only-example"},
        "geometry": {
            "axes": frame["geometry"]["array_axes"],
            "periodic_axis": frame["geometry"]["periodic_axis"],
            "separatrix_face_u": frame["geometry"]["separatrix_face_u"],
        },
        "available": {
            "rasters": sorted(frame["rasters"]),
            "contours": len(frame["contours"]),
            "density_gradient_vectors": len(frame["vectors"]["density_gradient"]),
            "filaments": len(frame["filaments"]),
            "event_types": sorted({item["type"] for item in frame["events"]}),
            "transport_available": frame["transport"]["available"],
            "exb_available": frame["vectors"]["exb_available"],
        },
    }


def build_prompt(frame: dict[str, Any], *, optimized: bool = False) -> str:
    rules = """
Return only JavaScript. Export exactly createPainter(api, styleConfig). The returned
object must define reset(seed) and renderFrame(frameFeatures, time, persistentState).
Use bounded for...of loops over supplied features. Do not use browser, DOM, network,
timers, dynamic code, direct canvas, p5, while loops, classes, or ambient globals.
All coordinates are normalized. Every structural mark must cite an input feature in
its source argument. The same renderer must work on arbitrary frames and clips.
""".strip()
    if optimized:
        rules += """

Validation-oriented additions: call createPaper once per frame; fade tracked pigment
before adding current filaments; map both contour signs; clamp all computed coordinates,
radii, widths, and opacities; do not emit a transport accent when transport.available
is false; avoid decorative randomness except bounded paper grain; keep under 900 calls.
"""
    return "\n\n".join(
        [
            "You write editable, scientifically constrained plasma painting programs.",
            STYLE_BRIEF,
            api_documentation(),
            "Feature summary:\n" + json.dumps(representative_summary(frame), indent=2, sort_keys=True),
            rules,
        ]
    )
