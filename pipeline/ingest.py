"""Чтение видео: метаданные, PTS каждого кадра, декодирование выбранных кадров."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import av
import numpy as np

from pipeline.errors import PipelineInputError
from pipeline.types import VideoInfo, VideoTimebase


def _open(path: str | Path) -> av.container.InputContainer:
    try:
        container = av.open(str(path))
    except (av.error.FFmpegError, OSError) as e:
        raise PipelineInputError("cannot read video file") from e
    if not container.streams.video:
        container.close()
        raise PipelineInputError("file has no video stream")
    return container


def read_timebase(path: str | Path) -> VideoTimebase:
    """PTS всех кадров (секунды от первого кадра). Декодирует поток целиком."""
    with _open(path) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        pts = [f.pts for f in c.decode(s) if f.pts is not None]
    if not pts:
        raise PipelineInputError("video has no decodable frames")
    arr = np.asarray(pts, dtype=np.float64) * tb
    arr -= arr[0]
    return VideoTimebase(pts_sec=arr)


def probe(path: str | Path, timebase: VideoTimebase | None = None) -> VideoInfo:
    timebase = timebase or read_timebase(path)
    with _open(path) as c:
        s = c.streams.video[0]
        codec, width, height = s.codec_context.name, s.codec_context.width, s.codec_context.height
    pts = timebase.pts_sec
    gaps = np.diff(pts) if len(pts) > 1 else np.array([0.0])
    median = float(np.median(gaps)) if len(pts) > 1 else 0.0
    # Длительность — до конца последнего кадра.
    duration = float(pts[-1] + median)
    if duration <= 0:
        raise PipelineInputError("video has zero duration")
    return VideoInfo(
        path=str(path),
        codec=codec,
        width=width,
        height=height,
        n_frames=len(pts),
        duration_sec=duration,
        avg_fps=len(pts) / duration,
        max_gap_sec=float(gaps.max()),
        dropped_frames=int((gaps > 1.5 * median).sum()) if median > 0 else 0,
    )


def iter_frames(path: str | Path, indices: Iterable[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Последовательно декодирует видео и отдаёт (индекс, RGB-кадр) для запрошенных индексов."""
    wanted = sorted(set(indices))
    if not wanted:
        return
    k = 0
    with _open(path) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i == wanted[k]:
                yield i, frame.to_ndarray(format="rgb24")
                k += 1
                if k == len(wanted):
                    return
