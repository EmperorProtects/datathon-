"""Валидатор data/annotations/unified.json: схема, start ≤ peak ≤ end, type_id есть в реестре."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pipeline.registry import Registry, load_registry
from training.data.common import ROOT, UNIFIED_PATH, read_json

TIME_TOL_SEC = 0.5


class Event(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str
    kind: Literal["action", "omission"]
    type_id: str
    organizer_time_sec: float = Field(ge=0)
    start_sec: float = Field(ge=0)
    peak_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    peak_frame: int = Field(ge=0)
    visibility: Literal["full", "partial", "occluded"] | None = None
    checkpoint_sec: float | None = None


class Lookalike(BaseModel):
    model_config = ConfigDict(extra="allow")

    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    what: str


class Video(BaseModel):
    model_config = ConfigDict(extra="allow")

    video_id: str
    path: str
    fps: float = Field(gt=0)
    duration_sec: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    has_violation: bool
    organizer_times_sec: list[float]
    events: list[Event]
    lookalikes: list[Lookalike] = []
    split_group: str


class Unified(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int
    videos: list[Video]


def validate(data: dict, registry: Registry, check_files: bool = True) -> list[str]:
    try:
        u = Unified.model_validate(data)
    except ValidationError as e:
        return [f"schema: {err['loc']}: {err['msg']}" for err in e.errors()]

    errors: list[str] = []
    known_types = registry.type_ids()
    skipped = {t.type_id for t in registry.skipped}
    ids: set[str] = set()
    for v in u.videos:
        p = f"{v.video_id}:"
        if v.video_id in ids:
            errors.append(f"{p} duplicate video_id")
        ids.add(v.video_id)
        if check_files and not (ROOT / v.path).is_file():
            errors.append(f"{p} file not found: {v.path}")
        if v.has_violation != bool(v.organizer_times_sec):
            errors.append(f"{p} has_violation={v.has_violation} but organizer_times_sec={v.organizer_times_sec}")
        if v.has_violation and not v.events:
            errors.append(f"{p} violation video without events")
        if not v.has_violation and v.events:
            errors.append(f"{p} no-violation video has events")
        limit = v.duration_sec + TIME_TOL_SEC
        for t in v.organizer_times_sec:
            if t > limit:
                errors.append(f"{p} organizer time {t} beyond duration {v.duration_sec:.1f}")
        for e in v.events:
            q = f"{p} {e.event_id} ({e.type_id}):"
            if not e.start_sec <= e.peak_sec <= e.end_sec:
                errors.append(f"{q} start ≤ peak ≤ end violated: {e.start_sec} / {e.peak_sec} / {e.end_sec}")
            if e.end_sec > limit:
                errors.append(f"{q} end {e.end_sec} beyond duration {v.duration_sec:.1f}")
            if e.organizer_time_sec not in v.organizer_times_sec:
                errors.append(f"{q} organizer_time_sec {e.organizer_time_sec} not among video organizer times")
            if e.type_id in skipped:
                errors.append(f"{q} type has no normative source in registry (not reportable)")
            elif e.type_id not in known_types:
                errors.append(f"{q} type_id not in registry")
            if e.kind == "omission" and e.checkpoint_sec is None:
                errors.append(f"{q} omission without checkpoint_sec")
        for lk in v.lookalikes:
            if lk.start_sec > lk.end_sec:
                errors.append(f"{p} lookalike start > end: {lk.start_sec} / {lk.end_sec}")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unified", type=Path, default=UNIFIED_PATH)
    ap.add_argument("--no-files", action="store_true", help="не проверять наличие видеофайлов")
    args = ap.parse_args()
    errors = validate(read_json(args.unified), load_registry(), check_files=not args.no_files)
    for e in errors:
        print("ERROR", e)
    print(f"{args.unified}: {'OK' if not errors else f'{len(errors)} error(s)'}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
