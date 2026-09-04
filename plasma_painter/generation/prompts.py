"""Versioned prompts for frozen-model and optimized-prompt baselines."""

from __future__ import annotations

import json
from pathlib import Path
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


def build_prompt(frame: dict[str, Any], *, optimized: bool = False, schema_v2: bool = False, media: bool = False) -> str:
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
    prompt = "\n\n".join(
        [
            "You write editable, scientifically constrained plasma painting programs.",
            STYLE_BRIEF,
            api_documentation(),
            "Feature summary:\n" + json.dumps(representative_summary(frame), indent=2, sort_keys=True),
            rules,
        ]
    )
    if schema_v2 or media:
        reference = (Path(__file__).parents[1] / "renderer/reference_renderers/tcv_watercolor.js").read_text()
        prompt += """

EXACT RUNTIME CONTRACT (overrides any guessed conventions):
api contains ONLY void-returning drawing functions and reset(seed). It has no
geometry, available, palette, canvas, or data properties. Never use return values
from drawing calls. Call drawing functions ONLY inside renderFrame. Use api.name
directly, not destructuring. Every drawing operation takes ONE object argument.

Operation signatures and bounds:
api.createPaper({color:'#f2ede2',grain:0.02}) // grain 0..0.2
api.setPalette({colors:['#233a4d','#2f6f73','#d6aa62']}) // 1..12 hex colors
api.washRegion({raster:'density',opacity:0.3,bleed:0.2}) // opacity 0..0.65, bleed 0..1
api.strokePath({points:contour.points,width:0.004,opacity:0.4,pigment:'positive_fluctuation',source:'density_contour'})
api.dryBrushPath({points:[[0.2,0.3],[0.21,0.32]],width:0.001,opacity:0.2,pigment:'edge',source:'density_gradient_direction'})
// paths: 2..256 normalized [x,z] points; width 0.0005..0.08; opacity 0..0.8
// The path above only demonstrates argument shape: derive actual paths from vectors.
api.dab({center:filament.centroid,radius:0.02,opacity:0.4,pigment:'positive_fluctuation',trackId:filament.track_id,source:'filament'})
api.poolPigment({center:filament.centroid,radius:0.02,opacity:0.2,source:'birth'})
// centers normalized; radii 0.001..0.2; opacity 0..0.8
api.fadeLayer({layer:'tracked-pigment',retention:0.85}) // retention 0..1
api.scatterGrain({amount:0.02,source:'paper_only',seed:0}) // amount 0..0.15
api.composite({mode:'multiply'}) // source-over, multiply, screen, soft-light only

frameFeatures is an OBJECT, with:
- contours: array of {points:[[x,z],...], sign: numeric -1 or +1, level:number}
- vectors.density_gradient: array of {x,z,dx,dz,magnitude}; directions are proxies
- filaments: array of {centroid:[x,z],area_fraction,peak,sign,track_id}
- events: array with type and track_ids; births/merges as in the working example
- rasters: object with density, density_fluctuation, potential, electron_temperature.
  These are encoded rasters; do NOT loop or decode them. Pass the raster NAME
  to washRegion and the runtime handles it.
- geometry.separatrix_face_u: normalized radial coordinate.
- transport.available: false. Do not invent transport or physical velocities.
There is NO frameFeatures.density, contour.path, filament.position, api.geometry,
or styleConfig.rasters. The summary above describes counts, not the data schema.
Named pigments: low_density, mid_density, high_density, positive_fluctuation,
negative_fluctuation, edge. Sign and source mappings must remain consistent.

WORKING COMPLETE PROGRAM. Preserve its interface and feature access. Make modest
but visible changes to wash opacity, brush widths, filament bloom strength, and
layering to create richer mineral watercolor. Keep every major mark data-driven.
Return the FULL modified program, beginning literally with:
export function createPainter(api, styleConfig) {
Do not return explanatory prose or omit export. No arbitrary new API or features.

""" + reference
    if media:
        prompt = prompt.replace(STYLE_BRIEF, "Create an intentional tool-medium study of the supplied plasma fields. Choose a coherent stroke vocabulary and preserve structure, sign, separatrix and temporal response. Gradient directions are visualization proxies, not physical flow. Never invent structures.")
        prompt = prompt.replace("Make modest\nbut visible changes to wash opacity, brush widths, filament bloom strength, and\nlayering to create richer mineral watercolor.", "Redesign the mark-making with an intentional medium: bristle painting, graphite, charcoal, ink, or pastel. Change stroke construction, widths, rhythm and layering, not only colors. Use bounded pressure and texture controls on paths; retain the data-driven geometry.")
        prompt += "\nMEDIUM STUDY: use medium, pressure, texture on strokePath/dryBrushPath. The reference is an interface example, not a composition to copy. Keep washes restrained enough that strokes remain visible. Do not claim this is a trained artist imitation.\n"
    return prompt
