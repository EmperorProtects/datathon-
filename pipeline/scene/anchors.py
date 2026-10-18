"""Устойчивые якоря сцены: медиана рамок статичных объектов за весь ролик."""
from __future__ import annotations

import numpy as np

from pipeline.types import Anchor, FramePerception


def compute_anchors(frames: list[FramePerception], cfg: dict) -> dict[str, Anchor]:
    sc = cfg["scene"]
    n = len(frames)
    if n == 0:
        return {}
    out: dict[str, Anchor] = {}
    for cls in sc["static_classes"]:
        boxes = []
        for f in frames:
            cands = [d for d in f.objects if d.cls == cls and d.conf >= sc["anchor_min_conf"]]
            if cands:
                boxes.append(max(cands, key=lambda d: d.conf).box)
        presence = len(boxes) / n
        if boxes and presence >= sc["anchor_min_presence"]:
            out[cls] = Anchor(cls=cls, box=tuple(float(v) for v in np.median(np.asarray(boxes), axis=0)),
                              presence=round(presence, 3))
    return out
