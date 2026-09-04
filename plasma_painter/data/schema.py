"""Machine-readable feature schema constants."""

from __future__ import annotations

SCHEMA_VERSION = "plasma-painter-features/1.0"

AXIS_CONVENTION = {
    "array_axes": ["x_radial", "z_periodic_field_aligned"],
    "x_direction": "increases_radially_outward",
    "z_direction": "logical_field_aligned_toroidal_wedge",
    "coordinates_in_paths": "normalized_domain_coordinates",
    "screenshot_pixels_used": False,
}

