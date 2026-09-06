"""Bounded painting operation schema used by generated JavaScript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math


Number = int | float


@dataclass(frozen=True)
class OperationSpec:
    name: str
    maximum_calls: int
    description: str


OPERATION_SPECS = {
    item.name: item
    for item in (
        OperationSpec("createPaper", 1, "Create deterministic paper with bounded grain."),
        OperationSpec("mark", 1100, "One finite brush or pencil mark, anchored to a supplied field sample."),
        OperationSpec("paintStroke", 1100, "A freely positioned finite stroke anywhere in the normalized cross-section; no sample anchor."),
        OperationSpec("setPalette", 2, "Select named pigment colors."),
        OperationSpec("washRegion", 96, "Lay a translucent data-linked region wash."),
        OperationSpec("strokePath", 320, "Paint a normalized contour path."),
        OperationSpec("dryBrushPath", 320, "Paint an oriented normalized proxy-vector path."),
        OperationSpec("dab", 256, "Paint a filament-linked bounded dab."),
        OperationSpec("poolPigment", 128, "Pool pigment at a birth or merge event."),
        OperationSpec("scatterGrain", 2, "Modulate paper grain; not a structural primitive."),
        OperationSpec("fadeLayer", 8, "Fade persistent data-linked pigment."),
        OperationSpec("composite", 8, "Composite bounded named layers."),
    )
}

ALLOWED_OPERATIONS = frozenset(OPERATION_SPECS)
ALLOWED_RASTERS = frozenset({"density", "density_fluctuation", "potential", "electron_temperature"})
STROKE_MEDIA = ("watercolor", "bristle", "graphite", "charcoal", "ink", "pastel")


def _color(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 7 or not value.startswith("#"):
        raise ValueError(f"{name} must be a six-digit hex color")
    try:
        int(value[1:], 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a six-digit hex color") from error
    return value


def _number(value: Any, low: float, high: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be within [{low}, {high}]")
    return result


def _point(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [x, z]")
    return [_number(value[0], 0.0, 1.0, f"{name}.x"), _number(value[1], 0.0, 1.0, f"{name}.z")]


def validate_operation(operation: dict[str, Any], *, max_path_points: int = 256) -> dict[str, Any]:
    if not isinstance(operation, dict):
        raise ValueError("operation must be an object")
    name = operation.get("op")
    if name not in ALLOWED_OPERATIONS:
        raise ValueError(f"unknown operation {name!r}")
    args = operation.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("operation args must be an object")
    if name == "createPaper":
        _number(args.get("grain", 0.04), 0.0, 0.2, "grain")
        if "color" not in args:
            raise ValueError("createPaper requires color")
        _color(args["color"], "color")
    elif name == "setPalette":
        colors = args.get("colors")
        if not isinstance(colors, list) or not 1 <= len(colors) <= 12:
            raise ValueError("palette must contain one to twelve colors")
        for index, color in enumerate(colors):
            _color(color, f"colors[{index}]")
    elif name == "washRegion":
        _number(args.get("opacity"), 0.0, 0.65, "opacity")
        _number(args.get("bleed", 0.0), 0.0, 1.0, "bleed")
        if "raster" not in args and "path" not in args:
            raise ValueError("washRegion must reference a raster or path")
        if "raster" in args and args["raster"] not in ALLOWED_RASTERS:
            raise ValueError("washRegion references an unknown raster")
        if "path" in args:
            points = args["path"]
            if not isinstance(points, list) or not 3 <= len(points) <= max_path_points:
                raise ValueError("washRegion path point count is outside the configured bound")
            for index, point in enumerate(points):
                _point(point, f"path[{index}]")
    elif name in {"strokePath", "dryBrushPath", "mark", "paintStroke"}:
        if args.get("medium", "watercolor") not in STROKE_MEDIA:
            raise ValueError("unsupported stroke medium")
        _number(args.get("pressure", 0.7), 0.05, 1.0, "pressure")
        _number(args.get("texture", 0.4), 0.0, 1.0, "texture")
        points = args.get("points")
        if not isinstance(points, list) or not 2 <= len(points) <= max_path_points:
            raise ValueError("path point count is outside the configured bound")
        for index, point in enumerate(points):
            _point(point, f"points[{index}]")
        _number(args.get("width", 0.003), 0.0005, 0.08, "width")
        _number(args.get("opacity", 0.2), 0.0, 0.8, "opacity")
        if name == "mark":
            if len(points)>8: raise ValueError("finite mark allows at most eight points")
            length=sum(math.dist(a,b) for a,b in zip(points,points[1:]))
            _number(length, .002, .06, "mark arc length")
            _number(args.get("width"), .0005, .018, "mark width")
            _color(args.get("color"), "mark color")
            if not isinstance(args.get("sample_id"), int) or isinstance(args.get("sample_id"),bool):
                raise ValueError("mark requires integer sample_id")
        if name == 'paintStroke':
            if 'medium' not in args:raise ValueError('paintStroke requires an explicit medium choice')
            if len(points)>64:raise ValueError('paintStroke allows at most 64 points')
            _number(sum(math.dist(a,b) for a,b in zip(points,points[1:])),.0005,2,'paintStroke arc length')
            _color(args.get('color'),'paintStroke color')
            if 'stroke_id' in args and (type(args['stroke_id'])!=int or not 0<=args['stroke_id']<10000000):
                raise ValueError('stroke_id must be integer 0..9999999')
    elif name in {"dab", "poolPigment"}:
        _point(args.get("center"), "center")
        _number(args.get("radius"), 0.001, 0.2, "radius")
        _number(args.get("opacity"), 0.0, 0.8, "opacity")
    elif name == "scatterGrain":
        _number(args.get("amount"), 0.0, 0.15, "amount")
    elif name == "fadeLayer":
        _number(args.get("retention"), 0.0, 1.0, "retention")
    elif name == "composite":
        mode = args.get("mode", "source-over")
        if mode not in {"source-over", "multiply", "screen", "soft-light"}:
            raise ValueError("unsupported composite mode")
    return operation


def validate_operations(
    operations: list[dict[str, Any]], *, max_operations: int = 1200, max_path_points: int = 256
) -> list[dict[str, Any]]:
    if not isinstance(operations, list) or len(operations) > max_operations:
        raise ValueError("operation list exceeds configured bound")
    counts: dict[str, int] = {}
    for operation in operations:
        validate_operation(operation, max_path_points=max_path_points)
        name = operation["op"]
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > OPERATION_SPECS[name].maximum_calls:
            raise ValueError(f"operation {name} exceeds its per-frame call cap")
    return operations


def validate_stroke_only(operations, frame):
    """Trusted host gate: generated code cannot opt out of this profile."""
    samples={s['id']:s for s in frame['stroke_samples']}
    if sum(op['op']=='createPaper' for op in operations)!=1:
        raise ValueError('stroke-only requires exactly one paper operation')
    for op in operations:
        if op['op'] not in {'createPaper','mark'}:
            raise ValueError('stroke-only forbids washes, contours, blooms and composites')
        if op['op']=='createPaper':
            if op['args'].get('grain',0)>.02: raise ValueError('stroke-only paper grain exceeds .02')
            continue
        a=op['args'];s=samples.get(a['sample_id'])
        if s is None: raise ValueError('unknown field sample anchor')
        if any(math.dist(p,[s['x'],s['z']])>.04 for p in a['points']):
            raise ValueError('mark is disconnected from its field sample')
    if not any(op['op']=='mark' for op in operations): raise ValueError('no finite marks')


def api_documentation() -> str:
    lines = ["Allowed painter API (normalized coordinates only):"]
    for spec in OPERATION_SPECS.values():
        lines.append(f"- {spec.name}: {spec.description} Maximum calls/frame: {spec.maximum_calls}.")
    lines.append("strokePath/dryBrushPath optional controls: medium in " + ", ".join(STROKE_MEDIA) + "; pressure 0.05..1; texture 0..1. These change mark construction, not field geometry. Default preserves legacy watercolor. Select tools intentionally; do not merely swap colors.")
    return "\n".join(lines)


def validate_free_paint(operations):
    if sum(op['op']=='createPaper' for op in operations)!=1:
        raise ValueError('free painting requires exactly one paper per frame')
    if any(op['op'] not in {'createPaper','paintStroke'} for op in operations):
        raise ValueError('free painting uses paper and finite paintStroke tools only')
    if any(op['op']=='createPaper' and op['args'].get('grain',0)>.02 for op in operations):
        raise ValueError('free painting paper grain exceeds .02')
    if not any(op['op']=='paintStroke' for op in operations):raise ValueError('no paint strokes')
