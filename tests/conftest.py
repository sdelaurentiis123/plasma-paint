from pathlib import Path

import pytest

from plasma_painter.config import load_config
from plasma_painter.data.loaders import synthetic_fixed_plane
from plasma_painter.features.pipeline import FeaturePipeline


@pytest.fixture
def config():
    return load_config(Path("configs/plasma_painter/pilot.yaml"))


@pytest.fixture
def synthetic_clip(config):
    data = synthetic_fixed_plane(frames=8, nx=24, nz=32, seed=19)
    pipeline = FeaturePipeline(config["features"])
    pipeline.fit(data, 0, 8)
    return pipeline.transform_clip(data, 0, 8, "synthetic-art_train-0000-0007", "art_train")
