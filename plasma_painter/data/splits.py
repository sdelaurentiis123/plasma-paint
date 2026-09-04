"""Leakage-safe chronological split and clip construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


PROHIBITED_SHOT = "85606"


@dataclass(frozen=True)
class Interval:
    name: str
    start: int
    stop: int
    usable: bool

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(f"invalid half-open interval {self.name}: {self.start}:{self.stop}")

    def contains(self, start: int, stop: int) -> bool:
        return self.start <= start < stop <= self.stop


@dataclass(frozen=True)
class Clip:
    clip_id: str
    split: str
    start: int
    stop: int

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop))


def validate_shot(shot: str, *paths: str) -> None:
    values = (str(shot), *(str(path) for path in paths))
    if any(PROHIBITED_SHOT in value for value in values):
        raise PermissionError("the sequestered research shot is prohibited at the loader boundary")
    if str(shot) != "85604" and str(shot) != "synthetic":
        raise ValueError(f"pilot accepts only the development shot or labelled synthetic data: {shot}")


def parse_art_intervals(split: Mapping[str, Iterable[int]]) -> tuple[Interval, ...]:
    order = (
        ("art_train", True),
        ("guard_train_val", False),
        ("art_val", True),
        ("guard_val_test", False),
        ("art_test", True),
    )
    intervals: list[Interval] = []
    previous_stop: int | None = None
    for name, usable in order:
        raw = list(split[name])
        if len(raw) != 2:
            raise ValueError(f"{name} must have [start, stop]")
        current = Interval(name, int(raw[0]), int(raw[1]), usable)
        if previous_stop is not None and current.start != previous_stop:
            raise ValueError("art split and guard intervals must be contiguous and ordered")
        previous_stop = current.stop
        intervals.append(current)
    return tuple(intervals)


def validate_nested_split(
    art_intervals: Iterable[Interval], forecasting_train: Iterable[int]
) -> None:
    bounds = list(forecasting_train)
    if len(bounds) != 2:
        raise ValueError("forecasting train split must have two bounds")
    parent = Interval("forecasting_train", int(bounds[0]), int(bounds[1]), True)
    intervals = tuple(art_intervals)
    if not intervals:
        raise ValueError("art split cannot be empty")
    if intervals[0].start < parent.start or intervals[-1].stop > parent.stop:
        raise ValueError("art split must be wholly nested in forecasting training")


def build_clips(
    shot: str,
    intervals: Iterable[Interval],
    *,
    length: int,
    stride: int,
    max_train_clips: int | None = None,
) -> list[Clip]:
    validate_shot(shot)
    if not 8 <= length <= 16:
        raise ValueError("clip length must be between 8 and 16 frames")
    if stride < length:
        raise ValueError("overlapping clips are disabled for the pilot")
    clips: list[Clip] = []
    for interval in intervals:
        if not interval.usable:
            continue
        starts = range(interval.start, interval.stop - length + 1, stride)
        for ordinal, start in enumerate(starts):
            stop = start + length
            if not interval.contains(start, stop):
                raise AssertionError("clip crossed a split or guard boundary")
            if interval.name == "art_train" and max_train_clips is not None:
                if ordinal >= max_train_clips:
                    break
            clips.append(
                Clip(
                    clip_id=f"tcv-{shot}-{interval.name}-{start:04d}-{stop - 1:04d}",
                    split=interval.name,
                    start=start,
                    stop=stop,
                )
            )
    assert_no_leakage(clips, intervals)
    return clips


def assert_no_leakage(clips: Iterable[Clip], intervals: Iterable[Interval]) -> None:
    usable = {item.name: item for item in intervals if item.usable}
    frame_owner: dict[int, str] = {}
    for clip in clips:
        if clip.split not in usable or not usable[clip.split].contains(clip.start, clip.stop):
            raise ValueError(f"clip {clip.clip_id} is outside its declared split")
        for frame in clip.frame_indices:
            owner = frame_owner.setdefault(frame, clip.split)
            if owner != clip.split:
                raise ValueError(f"frame {frame} leaks between {owner} and {clip.split}")

