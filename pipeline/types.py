"""Общие типы данных между модулями пайплайна."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    codec: str
    width: int
    height: int
    n_frames: int
    duration_sec: float
    avg_fps: float
    max_gap_sec: float
    dropped_frames: int  # интервалов между кадрами длиннее 1.5 × медианного


Box = tuple[float, float, float, float]  # x1, y1, x2, y2 в пикселях исходного кадра


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    box: Box
    track_id: int | None = None


@dataclass(frozen=True)
class Person:
    """Человек по YOLO-pose: рамка + 17 точек COCO (x, y, conf). Без распознавания лиц."""

    conf: float
    box: Box
    keypoints: np.ndarray  # (17, 3)
    track_id: int | None = None
    hands: tuple[tuple[float, float] | None, tuple[float, float] | None] = (None, None)  # левая, правая


@dataclass(frozen=True)
class FramePerception:
    frame_idx: int
    t_sec: float
    persons: list[Person]
    objects: list[Detection]


Side = Literal["near", "far"]  # ближняя к камере / дальняя сторона стойки
Role = Literal["cashier", "client", "collector", "other"]


@dataclass(frozen=True)
class Anchor:
    cls: str
    box: Box          # медиана по кадрам
    presence: float   # доля кадров, где объект найден


@dataclass
class Scene:
    anchors: dict[str, Anchor]
    zones: dict[str, Box]              # counter_zone, safe_zone, door_zone, staff_side, client_side, …
    staff_side: Side | None
    scene_confidence: float            # 0..1; низкая — правила, зависящие от зон, отключаются
    flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrackInfo:
    track_id: int
    role: Role
    n_frames: int
    t_first: float
    t_last: float
    side_frac: dict[str, float]        # near / far / none
    outside_room: bool = False
    reason: str = ""


@dataclass(frozen=True)
class Event:
    """Атомарное событие таймлайна: интервал + уверенность + участник и аргументы."""

    type: str
    t_start: float
    t_end: float
    conf: float
    actor: int | None = None          # track_id участника
    role: Role | None = None
    args: dict = field(default_factory=dict)  # cls, zone, from/to, …

    def to_dict(self) -> dict:
        return {"type": self.type, "t_start": round(self.t_start, 2), "t_end": round(self.t_end, 2),
                "conf": round(self.conf, 3), "actor": self.actor, "role": self.role, "args": self.args}


@dataclass
class Candidate:
    """Кандидат в нарушение от детектора; facts — наблюдаемые факты для описания."""

    type_id: str
    t_start: float
    t_peak: float
    t_end: float
    score: float
    source: str
    facts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VideoTimebase:
    """Соответствие номер кадра ↔ секунды по PTS (а не секунды × средний fps)."""

    pts_sec: np.ndarray  # время каждого декодированного кадра, от начала видео

    def __len__(self) -> int:
        return len(self.pts_sec)

    def frame_to_sec(self, frame_idx: int) -> float:
        return float(self.pts_sec[frame_idx])

    def sec_to_frame(self, t: float) -> int:
        """Ближайший по времени кадр."""
        pts = self.pts_sec
        i = int(np.searchsorted(pts, t))
        if i <= 0:
            return 0
        if i >= len(pts):
            return len(pts) - 1
        return i if pts[i] - t < t - pts[i - 1] else i - 1

    def sample_indices(self, fps: float) -> list[int]:
        """Кадры, ближайшие к равномерной сетке с шагом 1/fps (без повторов)."""
        if len(self.pts_sec) == 0:
            return []
        grid = np.arange(0.0, float(self.pts_sec[-1]) + 1e-9, 1.0 / fps)
        out: list[int] = []
        for t in grid:
            i = self.sec_to_frame(float(t))
            if not out or out[-1] != i:
                out.append(i)
        return out
