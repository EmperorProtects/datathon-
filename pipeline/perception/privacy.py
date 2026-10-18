"""Размытие голов на кадрах-доказательствах.

Область головы строится только по ключевым точкам позы (нос, глаза, уши) и плечам — без
моделей лиц, без распознавания и идентификации. Если точек головы не видно, размывается верх
рамки человека: лучше размыть лишнее, чем оставить лицо.
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline.perception.pose import HEAD_POINTS, L_SHOULDER, R_SHOULDER
from pipeline.types import Box, Person

HEAD_MARGIN = 0.6          # запас вокруг точек головы, в долях её размера
MIN_HEAD_FRAC = 0.35       # минимальный размер головы в долях ширины плеч / рамки
FALLBACK_TOP_FRAC = 0.35   # без точек головы — верхняя часть рамки человека


def head_box(p: Person, kpt_conf: float = 0.3) -> Box:
    k = p.keypoints
    x1, y1, x2, y2 = p.box
    bw = x2 - x1
    pts = k[list(HEAD_POINTS)]
    pts = pts[pts[:, 2] >= kpt_conf][:, :2]
    if len(pts) == 0:
        return x1, y1, x2, y1 + FALLBACK_TOP_FRAC * (y2 - y1)

    shoulders = k[[L_SHOULDER, R_SHOULDER]]
    shoulder_w = (abs(shoulders[0, 0] - shoulders[1, 0])
                  if (shoulders[:, 2] >= kpt_conf).all() else 0.0)
    min_size = MIN_HEAD_FRAC * max(shoulder_w, bw)
    cx, cy = pts.mean(axis=0)
    w = max(np.ptp(pts[:, 0]), min_size) * (1 + HEAD_MARGIN)
    h = max(np.ptp(pts[:, 1]), min_size) * (1 + HEAD_MARGIN)
    side = max(w, h)
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


def blur_regions(img: np.ndarray, boxes: list[Box]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        xa, ya = max(0, int(x1)), max(0, int(y1))
        xb, yb = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if xb - xa < 2 or yb - ya < 2:
            continue
        roi = out[ya:yb, xa:xb]
        k = max(15, (min(xb - xa, yb - ya) // 2) | 1)
        # сильное размытие поверх пикселизации — черты лица не восстанавливаются
        small = cv2.resize(roi, (max(1, (xb - xa) // 12), max(1, (yb - ya) // 12)), interpolation=cv2.INTER_AREA)
        roi = cv2.resize(small, (xb - xa, yb - ya), interpolation=cv2.INTER_LINEAR)
        out[ya:yb, xa:xb] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def blur_heads(img: np.ndarray, persons: list[Person], kpt_conf: float = 0.3) -> np.ndarray:
    return blur_regions(img, [head_box(p, kpt_conf) for p in persons])
