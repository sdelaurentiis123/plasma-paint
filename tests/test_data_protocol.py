from pathlib import Path

import numpy as np
import pytest

from plasma_painter.data.loaders import load_fixed_plane, read_prefix_member
from plasma_painter.data.splits import assert_no_leakage, build_clips, parse_art_intervals, validate_nested_split


SPLIT = {
    "art_train": [0, 288], "guard_train_val": [288, 320], "art_val": [320, 352],
    "guard_val_test": [352, 384], "art_test": [384, 432],
}


def test_split_and_guards_do_not_leak():
    intervals = parse_art_intervals(SPLIT)
    validate_nested_split(intervals, [0, 432])
    clips = build_clips("85604", intervals, length=8, stride=8)
    assert_no_leakage(clips, intervals)
    used = {frame for clip in clips for frame in clip.frame_indices}
    assert not used.intersection(range(288, 320))
    assert not used.intersection(range(352, 384))
    assert all(0 <= frame < 432 for frame in used)


def test_overlapping_pilot_clips_are_rejected():
    with pytest.raises(ValueError, match="overlapping"):
        build_clips("85604", parse_art_intervals(SPLIT), length=8, stride=4)


def test_prohibited_shot_rejected_before_path_access(monkeypatch):
    called = False
    original = Path.is_file

    def watched(self):
        nonlocal called
        called = True
        return original(self)

    monkeypatch.setattr(Path, "is_file", watched)
    with pytest.raises(PermissionError, match="sequestered"):
        load_fixed_plane("missing.npz", "missing-geometry.npz", shot="85606", field_keys={}, stop=8, plane_y=18)
    assert called is False


def test_prefix_member_never_returns_later_frames(tmp_path):
    path = tmp_path / "source.npz"
    values = np.arange(12 * 3, dtype=np.float32).reshape(12, 3)
    np.savez(path, field=values)
    loaded = read_prefix_member(path, "field", 5)
    assert loaded.shape == (5, 3)
    np.testing.assert_array_equal(loaded, values[:5])
