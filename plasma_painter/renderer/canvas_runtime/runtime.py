"""Small deterministic Canvas2D-like watercolor runtime implemented with Pillow.

The browser demo implements the same operation semantics in JavaScript.  This
Python version is used for filtering, rewards, and reproducible offline renders.
It never interprets candidate JavaScript; it only consumes validated operation
records emitted by the child-process sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from plasma_painter.features.pipeline import decode_unit_raster

RUNTIME_VERSION = "canvas-runtime-0.3.0-finite-marks"


def _rgb(value: str, fallback: str = "#30343b") -> tuple[int, int, int]:
    text = value if isinstance(value, str) and len(value.lstrip("#")) == 6 else fallback
    clean = text.lstrip("#")
    return tuple(int(clean[index : index + 2], 16) for index in (0, 2, 4))


def _point(point: list[float], width: int, height: int) -> tuple[float, float]:
    return float(point[0]) * (width - 1), float(point[1]) * (height - 1)


def _alpha_scaled(layer: Image.Image, factor: float) -> Image.Image:
    copy = layer.copy()
    copy.putalpha(copy.getchannel("A").point(lambda value: int(value * factor)))
    return copy


@dataclass
class RuntimeState:
    persistent: Image.Image | None = None
    frame_index: int = 0


class CanvasRuntime:
    """Render bounded DSL operations with deterministic seeded texture."""

    def __init__(self, width: int, height: int, style: dict[str, Any], seed: int):
        self.width = int(width)
        self.height = int(height)
        self.style = style
        self.seed = int(seed)
        self.state = RuntimeState()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.state = RuntimeState()

    def _pigment(self, name: str | None) -> tuple[int, int, int]:
        palette = self.style.get("palette", {})
        aliases = {
            "edge": "mid_density",
            "density": "mid_density",
            "positive": "positive_fluctuation",
            "negative": "negative_fluctuation",
        }
        key = aliases.get(str(name), str(name))
        return _rgb(palette.get(key, "#30343b"))

    def _wash(self, frame: dict[str, Any], args: dict[str, Any]) -> Image.Image:
        raster_name = str(args.get("raster", "density"))
        record = frame.get("rasters", {}).get(raster_name) or frame["rasters"]["density"]
        values = decode_unit_raster(record).T
        scalar = Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255), mode="L").resize(
            (self.width, self.height), Image.Resampling.BICUBIC
        )
        bleed = float(args.get("bleed", 0.25))
        scalar = scalar.filter(ImageFilter.GaussianBlur(0.5 + 3.0 * bleed))
        unit = np.asarray(scalar, dtype=np.float32) / 255.0
        palette = self.style.get("palette", {})
        low = np.asarray(_rgb(palette.get("low_density", "#233a4d")), dtype=np.float32)
        high = np.asarray(_rgb(palette.get("high_density", "#d6aa62")), dtype=np.float32)
        color = low[None, None, :] * (1.0 - unit[..., None]) + high[None, None, :] * unit[..., None]
        opacity = float(args.get("opacity", 0.15))
        alpha = np.uint8(np.clip((0.18 + 0.82 * unit) * opacity * 255, 0, 255))
        return Image.fromarray(np.dstack([np.uint8(color), alpha]), mode="RGBA")

    def _path_layer(self, args: dict[str, Any], *, dry: bool, rng: np.random.Generator) -> Image.Image:
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        points = [_point(point, self.width, self.height) for point in args["points"]]
        width = max(1, int(float(args.get("width", 0.003)) * min(self.width, self.height)))
        opacity = int(255 * float(args.get("opacity", 0.2)))
        color = _rgb(args['color']) if 'color' in args else self._pigment(args.get("pigment"))
        medium = args.get("medium", "watercolor")
        if medium != "watercolor":
            pressure = float(args.get("pressure", 0.7))
            texture = float(args.get("texture", 0.4))
            # Cross-stroke offsets remain bounded by the supplied brush width.
            # No independent particles: every mark follows the input polyline.
            fibers = {"bristle": 12, "graphite": 5, "charcoal": 7, "ink": 1, "pastel": 8}[medium]
            for fiber in range(fibers):
                offset = (fiber / max(1, fibers - 1) - 0.5) * width if fibers > 1 else 0
                shifted = []
                for i, (x, y) in enumerate(points):
                    a, b = points[max(0, i - 1)], points[min(len(points) - 1, i + 1)]
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    norm = max(1e-9, float(np.hypot(dx, dy)))
                    shifted.append((x - dy / norm * offset, y + dx / norm * offset))
                for i in range(len(shifted) - 1):
                    taper = 0.3 + 0.7 * np.sin(np.pi * (i + 0.5) / (len(shifted) - 1))
                    line_width = max(1, int(width * pressure * taper / (1 if medium == "ink" else fibers / 2)))
                    tone = tuple(min(255, max(0, c + (fiber % 3 - 1) * 12)) for c in color)
                    draw.line(shifted[i:i+2], fill=(*tone, int(opacity * pressure)), width=line_width)
            if medium in {"graphite", "charcoal", "pastel", "bristle"}:
                mask = np.asarray(layer.getchannel("A"), dtype=np.uint8)
                tooth = rng.random(mask.shape)
                dropout = texture * {"graphite": .48, "charcoal": .65, "pastel": .35, "bristle": .22}[medium]
                layer.putalpha(Image.fromarray(np.where(tooth < dropout, 0, mask).astype(np.uint8)))
            return layer.filter(ImageFilter.GaussianBlur(.4)) if medium == "charcoal" else layer
        passes = 1 if dry else 2
        for pass_index in range(passes):
            sigma = 0.25 + 0.35 * pass_index
            jittered = [(x + rng.normal(0, sigma), y + rng.normal(0, sigma)) for x, y in points]
            draw.line(jittered, fill=(*color, max(5, opacity // (pass_index + 1))), width=width)
        if dry:
            mask = np.asarray(layer.getchannel("A"), dtype=np.uint8)
            keep = rng.random(mask.shape) > 0.22
            layer.putalpha(Image.fromarray(np.where(keep, mask, 0).astype(np.uint8), mode="L"))
        return layer.filter(ImageFilter.GaussianBlur(0.25 if dry else 0.45))

    def _dab(self, args: dict[str, Any], rng: np.random.Generator, *, pool: bool) -> Image.Image:
        cx, cy = _point(args["center"], self.width, self.height)
        radius = max(1.0, float(args["radius"]) * min(self.width, self.height))
        spread = 2.7 if pool else 2.3
        left = max(0, int(cx - spread * radius))
        right = min(self.width, int(cx + spread * radius + 1))
        top = max(0, int(cy - spread * radius))
        bottom = min(self.height, int(cy + spread * radius + 1))
        yy, xx = np.mgrid[top:bottom, left:right]
        distance = np.hypot(xx - cx, yy - cy) / radius
        irregular = 1.0 + rng.normal(0.0, 0.045 if pool else 0.075, distance.shape)
        alpha = np.exp((-1.35 if pool else -1.8) * (distance * irregular) ** 2)
        if pool:
            alpha += 0.22 * np.exp(-15.0 * (distance - 0.85) ** 2)
        alpha = np.uint8(np.clip(alpha * float(args["opacity"]) * 255, 0, 255))
        color = self._pigment(args.get("pigment", "positive_fluctuation"))
        rgba = np.empty((bottom - top, right - left, 4), dtype=np.uint8)
        rgba[..., :3] = color
        rgba[..., 3] = alpha
        tile = Image.fromarray(rgba, mode="RGBA").filter(ImageFilter.GaussianBlur(max(0.3, radius * 0.035)))
        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        layer.alpha_composite(tile, (left, top))
        return layer

    def render_frame(self, frame: dict[str, Any], operations: list[dict[str, Any]]) -> Image.Image:
        source_index = int(frame.get("source", {}).get("frame_index", self.state.frame_index))
        rng = np.random.default_rng(self.seed + source_index * 7919)
        paper_color = self.style.get("paper", "#f2ede2")
        grain_amount = float(self.style.get("grain", 0.04))
        for operation in operations:
            if operation["op"] == "createPaper":
                paper_color = operation["args"].get("color", paper_color)
                grain_amount = float(operation["args"].get("grain", grain_amount))
                break
        paper = np.full((self.height, self.width, 3), _rgb(paper_color, "#f2ede2"), dtype=np.float32)
        paper += rng.normal(0.0, 255.0 * grain_amount, (self.height, self.width, 1))
        image = Image.fromarray(np.uint8(np.clip(paper, 0, 255)), mode="RGB").convert("RGBA")
        if self.state.persistent is None:
            self.state.persistent = Image.new("RGBA", image.size, (0, 0, 0, 0))
        current = Image.new("RGBA", image.size, (0, 0, 0, 0))
        for operation in operations:
            name, args = operation["op"], operation.get("args", {})
            if name == "washRegion" and "raster" in args:
                current = Image.alpha_composite(current, self._wash(frame, args))
            elif name in {"strokePath", "dryBrushPath", "mark"}:
                mark_rng = np.random.default_rng(self.seed + int(args['sample_id'])*7919) if name=='mark' else rng
                current = Image.alpha_composite(current, self._path_layer(args, dry=name == "dryBrushPath", rng=mark_rng))
            elif name == "dab":
                self.state.persistent = Image.alpha_composite(self.state.persistent, self._dab(args, rng, pool=False))
            elif name == "poolPigment":
                self.state.persistent = Image.alpha_composite(self.state.persistent, self._dab(args, rng, pool=True))
            elif name == "fadeLayer":
                self.state.persistent = _alpha_scaled(self.state.persistent, float(args.get("retention", 0.85)))
        image = Image.alpha_composite(image, current)
        image = Image.alpha_composite(image, self.state.persistent)
        self.state.frame_index += 1
        return image.convert("RGB")

    def render_clip(
        self, frames: list[dict[str, Any]], operations_by_frame: list[list[dict[str, Any]]]
    ) -> list[Image.Image]:
        self.reset(self.seed)
        return [self.render_frame(frame, operations) for frame, operations in zip(frames, operations_by_frame)]
