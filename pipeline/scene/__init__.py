"""Сцена и роли: якоря → зоны → роли треков; сторона персонала определяется по кассиру."""
from __future__ import annotations

from pipeline.scene.anchors import compute_anchors
from pipeline.scene.roles import assign_roles
from pipeline.scene.zones import build_scene
from pipeline.types import FramePerception, Scene, TrackInfo


def analyze_scene(frames: list[FramePerception], cfg: dict, fps: float,
                  frame_wh: tuple[int, int]) -> tuple[Scene, dict[int, TrackInfo]]:
    scene = build_scene(compute_anchors(frames, cfg), frame_wh, cfg)
    tracks, scene = assign_roles(frames, scene, cfg, fps, frame_wh)
    return scene, tracks
