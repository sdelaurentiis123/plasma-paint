"""Deterministic Pillow implementation of the reference watercolor mapping."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from plasma_painter.features.pipeline import decode_unit_raster


def _hex(value: str) -> tuple[int, int, int]:
    clean = value.lstrip("#")
    return tuple(int(clean[index : index + 2], 16) for index in (0, 2, 4))


def _resize_scalar(values: np.ndarray, size: tuple[int, int], blur: float = 0.0) -> np.ndarray:
    image = Image.fromarray(np.asarray(np.clip(values, 0, 1) * 255, dtype=np.uint8), mode="L")
    image = image.resize(size, Image.Resampling.BICUBIC)
    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(image, dtype=np.float32) / 255.0


def _color_map(values: np.ndarray, low: tuple[int, int, int], middle: tuple[int, int, int], high: tuple[int, int, int]) -> np.ndarray:
    t = np.clip(values, 0.0, 1.0)[..., None]
    lower = np.asarray(low, dtype=np.float32) * (1 - 2 * t) + np.asarray(middle, dtype=np.float32) * (2 * t)
    upper = np.asarray(middle, dtype=np.float32) * (2 - 2 * t) + np.asarray(high, dtype=np.float32) * (2 * t - 1)
    return np.where(t <= 0.5, lower, upper).clip(0, 255).astype(np.uint8)


def _point(point: list[float], width: int, height: int) -> tuple[float, float]:
    return float(point[0]) * (width - 1), float(point[1]) * (height - 1)


def _bloom(size: tuple[int, int], centre: tuple[float, float], radius: float, color: tuple[int, int, int], opacity: float, rng: np.random.Generator) -> Image.Image:
    width, height = size
    x0, y0 = centre
    px_radius = float(max(2.0, radius * min(width, height)))
    left = max(0, int(x0 - 2.2 * px_radius))
    right = min(width, int(x0 + 2.2 * px_radius + 1))
    top = max(0, int(y0 - 2.2 * px_radius))
    bottom = min(height, int(y0 + 2.2 * px_radius + 1))
    yy, xx = np.mgrid[top:bottom, left:right]
    distance = np.hypot(xx - x0, yy - y0) / px_radius
    irregular = 1.0 + 0.06 * rng.standard_normal(distance.shape)
    alpha = np.exp(-1.8 * (distance * irregular) ** 2)
    ring = 0.18 * np.exp(-12.0 * (distance - 0.82) ** 2)
    alpha = np.clip((alpha + ring) * opacity * 255, 0, 255).astype(np.uint8)
    tile = np.empty((bottom - top, right - left, 4), dtype=np.uint8)
    tile[..., :3] = np.asarray(color, dtype=np.uint8)
    tile[..., 3] = alpha
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    layer.alpha_composite(Image.fromarray(tile, mode="RGBA"), (left, top))
    return layer.filter(ImageFilter.GaussianBlur(float(max(0.4, 0.05 * px_radius))))


@dataclass
class WatercolorState:
    persistent_pigment: Image.Image | None = None
    track_positions: dict[int, tuple[float, float]] = field(default_factory=dict)


class ScientificWatercolorRenderer:
    def __init__(self, width: int, height: int, style: dict[str, Any], seed: int = 1701):
        self.width = int(width)
        self.height = int(height)
        self.style = style
        self.seed = int(seed)
        self.state = WatercolorState()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.state = WatercolorState()

    def render_scientific(self, frame: dict[str, Any]) -> Image.Image:
        density = decode_unit_raster(frame["rasters"]["density"]).T
        scalar = _resize_scalar(density, (self.width, self.height), blur=0.0)
        rgb = np.repeat(np.asarray(scalar * 255, dtype=np.uint8)[..., None], 3, axis=2)
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)
        x = frame["geometry"]["separatrix_face_u"] * (self.width - 1)
        draw.line([(x, 0), (x, self.height)], fill=(205, 71, 45), width=2)
        return image

    def render_frame(self, frame: dict[str, Any], frame_offset: int = 0) -> tuple[Image.Image, dict[str, Any]]:
        rng = np.random.default_rng(self.seed + int(frame["source"]["frame_index"]) * 7919)
        paper_rgb = _hex(self.style.get("paper", "#f2ede2"))
        paper = np.full((self.height, self.width, 3), paper_rgb, dtype=np.float32)
        grain_amount = float(self.style.get("grain", 0.055))
        grain = rng.normal(0.0, 255.0 * grain_amount, (self.height, self.width, 1))
        paper = Image.fromarray(np.clip(paper + grain, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")

        density = decode_unit_raster(frame["rasters"]["density"]).T
        scalar = _resize_scalar(density, (self.width, self.height), blur=3.0)
        palette = self.style["palette"]
        pigment = _color_map(
            scalar,
            _hex(palette["low_density"]),
            _hex(palette["mid_density"]),
            _hex(palette["high_density"]),
        )
        opacity = float(self.style["opacity"]["wash"])
        alpha = np.asarray(np.clip((0.2 + 0.8 * scalar) * opacity * 255, 0, 255), dtype=np.uint8)
        wash_array = np.dstack([pigment, alpha])
        wash = Image.fromarray(wash_array, mode="RGBA").filter(ImageFilter.GaussianBlur(1.4))
        paper = Image.alpha_composite(paper, wash)

        contour_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        contour_draw = ImageDraw.Draw(contour_layer)
        contour_count = 0
        for contour in frame["contours"]:
            points = [_point(point, self.width, self.height) for point in contour["points"]]
            if len(points) < 2:
                continue
            positive = contour["sign"] > 0
            color = _hex(palette["positive_fluctuation"] if positive else palette["negative_fluctuation"])
            width = 3 if positive else 2
            base_alpha = int(255 * float(self.style["opacity"]["contour"]) * (1.0 if positive else 0.72))
            for pass_index in range(2):
                jittered = [
                    (x + rng.normal(0, 0.55 + pass_index * 0.3), y + rng.normal(0, 0.55 + pass_index * 0.3))
                    for x, y in points
                ]
                contour_draw.line(jittered, fill=(*color, max(12, base_alpha // (pass_index + 1))), width=width)
            contour_count += 1
        contour_layer = contour_layer.filter(ImageFilter.GaussianBlur(0.45))
        paper = Image.alpha_composite(paper, contour_layer)

        if self.state.persistent_pigment is None:
            self.state.persistent_pigment = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        retention = float(self.style.get("persistence", 0.86))
        faded = self.state.persistent_pigment.copy()
        faded.putalpha(faded.getchannel("A").point(lambda value: int(value * retention)))
        blooms = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        filament_count = 0
        for filament in frame["filaments"]:
            centre = _point(filament["centroid"], self.width, self.height)
            radius = min(0.12, 0.009 + 1.25 * np.sqrt(float(filament["area_fraction"])))
            color = _hex(
                palette["positive_fluctuation"] if filament["sign"] > 0 else palette["negative_fluctuation"]
            )
            alpha_value = min(0.62, 0.16 + 0.07 * abs(float(filament["peak"])))
            blooms = Image.alpha_composite(blooms, _bloom((self.width, self.height), centre, radius, color, alpha_value, rng))
            track_id = filament.get("track_id")
            if track_id is not None:
                self.state.track_positions[int(track_id)] = centre
            filament_count += 1
        event_types = [event["type"] for event in frame["events"]]
        if any(kind in {"birth", "merge"} for kind in event_types):
            blooms = blooms.filter(ImageFilter.GaussianBlur(0.7))
        self.state.persistent_pigment = Image.alpha_composite(faded, blooms)
        paper = Image.alpha_composite(paper, self.state.persistent_pigment)

        vector_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        vector_draw = ImageDraw.Draw(vector_layer)
        vectors = frame["vectors"]["density_gradient"]
        for vector in vectors:
            if vector["magnitude"] < 0.35:
                continue
            start = _point([vector["x"], vector["z"]], self.width, self.height)
            scale = 3.0 + 8.0 * vector["magnitude"]
            end = (start[0] + scale * vector["dx"], start[1] + scale * vector["dz"])
            vector_draw.line([start, end], fill=(43, 48, 45, int(18 + 34 * vector["magnitude"])), width=1)
        paper = Image.alpha_composite(paper, vector_layer)

        sep_x = frame["geometry"]["separatrix_face_u"] * (self.width - 1)
        sep_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        sep_draw = ImageDraw.Draw(sep_layer)
        sep_draw.line([(sep_x, 0), (sep_x, self.height)], fill=(61, 51, 43, 46), width=1)
        paper = Image.alpha_composite(paper, sep_layer)
        operation_summary = {
            "contour_paths": contour_count,
            "filament_dabs": filament_count,
            "gradient_strokes": sum(item["magnitude"] >= 0.35 for item in vectors),
            "event_types": event_types,
            "uses_transport_accent": False,
            "transport_reason": frame["transport"]["reason"],
        }
        return paper.convert("RGB"), operation_summary
