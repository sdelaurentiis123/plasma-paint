from datetime import datetime, timezone
import json

import numpy as np
from PIL import Image

from plasma_painter.config import stable_hash
from plasma_painter.features.pipeline import encode_unit_raster
from plasma_painter.ratings.schema import append_rating, read_ratings
from plasma_painter.ratings.server import create_app
from plasma_painter.rewards.aggregate import aggregate_reward
from plasma_painter.rewards.fidelity import fidelity_frame
from plasma_painter.rewards.validity import validity_gate


def test_known_correspondence_scores_high():
    density = np.tile(np.linspace(0, 1, 20), (12, 1))
    pixels = np.uint8(255 - density[..., None] * np.ones(3) * 180)
    frame = {"rasters": {"density": encode_unit_raster(density.T)}, "contours": [], "filaments": [], "vectors": {"density_gradient": []}, "transport": {"available": False, "reason": "synthetic"}}
    result = fidelity_frame(frame, Image.fromarray(pixels, mode="RGB"), [], paper="#ffffff")
    assert result["coarse_spearman_raw"] > .95


def test_invalid_gate_cannot_be_compensated_by_aesthetics():
    gate = {"valid": False, "failed": ["compiles"]}
    result = aggregate_reward(gate=gate, fidelity=1, aesthetic=1, temporal=1, diversity=1, efficiency=1, weights={"fidelity": .4, "aesthetic": .3, "temporal": .15, "diversity": .1, "efficiency": .05}, strong_negative=-2)
    assert result["reward"] == -2


def test_blank_output_gate_fails():
    result = validity_gate(compiles=True, sandbox_valid=True, fidelity={"nonempty_fraction": 1, "structural_marks": 0, "coarse_spearman_raw": 1, "extrema": 1, "filament": 1, "orientation": 1}, minimum_nonempty_fraction=.005, minimum_coarse_spearman=.08, minimum_extrema_recall=.1)
    assert not result["valid"]


def test_ratings_are_appended_not_overwritten(tmp_path):
    path = tmp_path / "ratings.jsonl"
    base = {"kind": "pairwise", "rater_id": "tester", "timestamp": datetime.now(timezone.utc).isoformat(), "clip_id": "synthetic-clip", "seed": [1, 2], "left_program_hash": "a", "right_program_hash": "b", "left_checkpoint": "x", "right_checkpoint": "y", "order": ["left", "right"], "choice": "left"}
    append_rating(path, {**base, "rating_id": "one"})
    append_rating(path, {**base, "rating_id": "two", "choice": "tie"})
    assert [item["rating_id"] for item in read_ratings(path)] == ["one", "two"]


def test_program_hashing_is_content_addressed():
    assert stable_hash("same") == stable_hash("same")
    assert stable_hash("same") != stable_hash("different")


def test_rating_server_triages_before_love_only_pairs(tmp_path, config):
    source = tmp_path / "painter.js"
    source.write_text("export function createPainter(){}")
    clip = tmp_path / "clip.gif"
    clip.write_bytes(b"GIF89a")
    records = []
    for index in range(3):
        records.append({"reference_id": f"ref-{index}", "program_hash": f"hash-{index}", "program_path": str(source), "clip_id": "synthetic-clip", "clip": str(clip), "seed": index, "model_checkpoint": "fixture"})
    pool = tmp_path / "reference_pool"
    pool.mkdir()
    (pool / "manifest.json").write_text(json.dumps({"records": records}))
    config["project"]["artifact_root"] = str(tmp_path)
    config["ratings"]["jsonl_path"] = str(tmp_path / "ratings.jsonl")
    app = create_app(config)
    with app.test_client() as client:
        for _ in records:
            item = client.get("/api/next").get_json()
            assert item["kind"] == "triage"
            assert client.post("/api/rate", json={"token": item["token"], "choice": "love", "comment": ""}).status_code == 200
        pair = client.get("/api/next").get_json()
        assert pair["kind"] == "pairwise"
        assert "right" in pair
