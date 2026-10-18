"""Общие типы данных между модулями пайплайна."""
from __future__ import annotations

from dataclasses import dataclass

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
