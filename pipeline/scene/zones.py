"""Зоны сцены, построенные от якорей (а не от абсолютных пикселей), и геометрия людей.

Стороны стойки: near — ближняя к камере (ниже центра стойки в кадре), far — дальняя.
Какая из них сторона персонала, решает roles.py по тому, где работает кассир.
"""
from __future__ import annotations

import numpy as np

from pipeline.perception.pose import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER
from pipeline.types import Anchor, Box, Person, Scene, Side


def expand(box: Box, frac: float, frame_wh: tuple[int, int] | None = None) -> Box:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
    out = (x1 - dx, y1 - dy, x2 + dx, y2 + dy)
    if frame_wh:
        w, h = frame_wh
        out = (max(0.0, out[0]), max(0.0, out[1]), min(float(w), out[2]), min(float(h), out[3]))
    return out


def center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: Box, b: Box) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def inside_frac(box: Box, zone: Box) -> float:
    """Доля площади box внутри zone."""
    a = area(box)
    return intersection(box, zone) / a if a > 0 else 0.0


def contains(zone: Box, pt: tuple[float, float]) -> bool:
    return zone[0] <= pt[0] <= zone[2] and zone[1] <= pt[1] <= zone[3]


def side_of(person_box: Box, counter: Box, margin_frac: float) -> Side | None:
    _, cy = center(counter)
    _, py = center(person_box)
    margin = margin_frac * (counter[3] - counter[1])
    if py > cy + margin:
        return "near"
    if py < cy - margin:
        return "far"
    return None


def body_zone(p: Person, kpt_conf: float = 0.3) -> Box:
    """Торс и бёдра по позе (куда прячут деньги), с запасом; без точек — средняя часть рамки."""
    k = p.keypoints
    pts = k[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]
    pts = pts[pts[:, 2] >= kpt_conf][:, :2]
    x1, y1, x2, y2 = p.box
    if len(pts) < 2:
        h = y2 - y1
        return x1, y1 + 0.25 * h, x2, y1 + 0.75 * h
    bx1, by1 = pts.min(axis=0)
    bx2, by2 = pts.max(axis=0)
    pad = 0.25 * max(bx2 - bx1, by2 - by1, 0.2 * (x2 - x1))
    return float(bx1 - pad), float(by1 - pad), float(bx2 + pad), float(by2 + pad)


def build_scene(anchors: dict[str, Anchor], frame_wh: tuple[int, int], cfg: dict,
                staff_side: Side | None = None) -> Scene:
    sc = cfg["scene"]
    flags: list[str] = []
    missing = [c for c in sc["required_anchors"] if c not in anchors]
    if missing:
        flags.append(f"missing_anchors:{','.join(missing)}")
    found = sum(c in anchors for c in sc["static_classes"])
    confidence = 0.0 if missing else round(found / len(sc["static_classes"]), 3)

    zones: dict[str, Box] = {}
    m = sc["zone_margin_frac"]
    for cls in ("counter", "safe", "door", "laptop", "cash_counter"):
        if cls in anchors:
            zones[f"{cls}_zone"] = expand(anchors[cls].box, m, frame_wh)
    if "counter" in anchors and staff_side:
        x1, y1, x2, y2 = anchors["counter"].box
        _, cy = center(anchors["counter"].box)
        extra = sc["side_zone_extra_width"] * (x2 - x1)
        w, h = frame_wh
        near = (max(0.0, x1 - extra), cy, min(float(w), x2 + extra), float(h))
        far = (max(0.0, x1 - extra), 0.0, min(float(w), x2 + extra), cy)
        zones["staff_side"], zones["client_side"] = (near, far) if staff_side == "near" else (far, near)
    return Scene(anchors=anchors, zones=zones, staff_side=staff_side, scene_confidence=confidence, flags=flags)


def median_person_height(boxes: list[Box]) -> float:
    return float(np.median([b[3] - b[1] for b in boxes])) if boxes else 0.0
