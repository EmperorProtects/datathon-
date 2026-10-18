"""Относительные геометрические признаки кадра: длина предплечья, кисть у объекта, зона нахождения."""
from __future__ import annotations

import math

from pipeline.perception.pose import L_ELBOW, L_WRIST, R_ELBOW, R_WRIST
from pipeline.scene.zones import body_zone, center, contains, expand, inside_frac, intersection
from pipeline.types import Box, Person, Scene


def forearm_len(p: Person, kpt_conf: float, fallback_frac: float) -> float:
    """Длина предплечья (масштаб человека); без видимых точек — доля высоты рамки."""
    k = p.keypoints
    lens = [math.dist(k[e][:2], k[w][:2]) for e, w in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST))
            if k[e][2] >= kpt_conf and k[w][2] >= kpt_conf]
    h = p.box[3] - p.box[1]
    return max(lens) if lens else fallback_frac * h


def hand_near(p: Person, box: Box, radius: float) -> bool:
    zone = (box[0] - radius, box[1] - radius, box[2] + radius, box[3] + radius)
    return any(h is not None and contains(zone, h) for h in p.hands)


def looks_outside(box: Box, scene: Scene, cfg: dict) -> bool:
    """Человек в кадре стоит за стеклом двери (в коридоре), а не в комнате."""
    door = scene.zones.get("door_zone")
    counter = scene.anchors.get("counter")
    if door is None or counter is None or inside_frac(box, door) < 0.8:
        return False
    ch = counter.box[3] - counter.box[1]
    return box[3] < counter.box[1] - cfg["roles"]["outside_feet_above_counter_frac"] * ch


def locate(box: Box, scene: Scene, persons: list[Person], bags: list[Box], cfg: dict) -> tuple[str, int | None]:
    """Где находится объект (для «деньги исчезли в …»): safe / body / bag / desk / client_side / unknown."""
    tc = cfg["timeline"]
    if "safe_zone" in scene.zones and inside_frac(box, scene.zones["safe_zone"]) >= tc["in_zone_frac"]:
        return "safe", None
    for p in persons:
        if inside_frac(box, body_zone(p)) >= tc["near_body_frac"]:
            return "body", p.track_id
    for b in bags:
        if inside_frac(box, expand(b, 0.2)) >= tc["in_zone_frac"]:
            return "bag", None
    c = center(box)
    if "counter_zone" in scene.zones and contains(scene.zones["counter_zone"], c):
        return "desk", None
    if "client_side" in scene.zones and contains(scene.zones["client_side"], c):
        return "client_side", None
    return "unknown", None


def at_workstation(p: Person, scene: Scene, margin: float) -> bool:
    zones = [expand(scene.anchors[c].box, margin) for c in ("counter", "safe") if c in scene.anchors]
    return any(intersection(p.box, z) > 0 for z in zones)
