"""Perception: кадры с частотой analysis_fps → люди с позами и объекты, с треками."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline.ingest import iter_frames
from pipeline.types import FramePerception, VideoTimebase

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PerceptionModels:
    pose: "PoseTracker"  # noqa: F821
    objects: "ObjectTracker | None"  # noqa: F821

    @classmethod
    def load(cls, cfg: dict) -> PerceptionModels:
        from pipeline.perception.detector import ObjectTracker
        from pipeline.perception.pose import PoseTracker

        p, d, rt = cfg["pose"], cfg["detector"], cfg["perception"]
        pose = PoseTracker(str(ROOT / p["model"]), conf=p["conf"], kpt_conf=p["kpt_conf"], imgsz=p["imgsz"],
                           tracker=cfg["tracker"], device=rt["device"], half=rt["half"])
        obj_path = ROOT / d["objects_model"]
        objects = (ObjectTracker(str(obj_path), conf=d["conf"], imgsz=d["imgsz"], tracker=cfg["tracker"],
                                 device=rt["device"], half=rt["half"]) if obj_path.is_file() else None)
        return cls(pose=pose, objects=objects)

    def reset(self) -> None:
        self.pose.reset()
        if self.objects:
            self.objects.reset()


def run_perception(path: str | Path, timebase: VideoTimebase, models: PerceptionModels, cfg: dict,
                   fps: float | None = None) -> list[FramePerception]:
    fps = fps or cfg["analysis_fps"]
    batch = cfg["perception"]["batch"]
    models.reset()
    out: list[FramePerception] = []
    buf: list[tuple[int, object]] = []

    def flush() -> None:
        frames = [f for _, f in buf]
        persons = models.pose.track(frames)
        objects = models.objects.track(frames) if models.objects else [[] for _ in frames]
        for (i, _), ps, obs in zip(buf, persons, objects):
            out.append(FramePerception(frame_idx=i, t_sec=timebase.frame_to_sec(i), persons=ps, objects=obs))
        buf.clear()

    for i, rgb in iter_frames(path, timebase.sample_indices(fps)):
        buf.append((i, rgb))
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    return out
