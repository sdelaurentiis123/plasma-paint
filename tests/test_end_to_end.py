from pathlib import Path

import yaml

from plasma_painter.renderer.canvas_runtime import CanvasRuntime
from plasma_painter.renderer.sandbox import run_program


def test_synthetic_feature_to_program_to_pixels(config, synthetic_clip):
    assert all(frame["source"]["synthetic"] for frame in synthetic_clip["frames"])
    code = Path(config["renderer"]["baseline_program"]).read_text()
    style = yaml.safe_load(Path(config["renderer"]["style_config"]).read_text())
    sandbox = run_program(code, synthetic_clip["frames"], style=style, seed=1701)
    assert sandbox.valid, sandbox.error
    images = CanvasRuntime(192, 128, style, 1701).render_clip(synthetic_clip["frames"], sandbox.operations_by_frame)
    assert len(images) == 8
    assert all(image.getbbox() for image in images)
