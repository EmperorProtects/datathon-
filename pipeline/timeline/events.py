"""Атомарные события таймлайна из детекций, треков, поз и ролей.

Покадровые сигналы → интервалы (разрывы короче gap_sec склеиваются, события короче
min_event_sec отбрасываются). Состояния «сейф открыт» и «дверь открыта» по рамкам объектов
не наблюдаемы — их проверяет локальная VLM (этап 8).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from pipeline.perception.privacy import head_box
from pipeline.scene.zones import body_zone, center, contains, expand, inside_frac, intersection
from pipeline.timeline.features import at_workstation, forearm_len, hand_near, locate, looks_outside
from pipeline.types import Event, FramePerception, Scene, TrackInfo

BAG_CLASSES = ("bag", "collector_bag")
Key = tuple[str, int | None, str | None]  # (тип, actor, cls)


def mask_to_intervals(mask: np.ndarray, fps: float, min_sec: float, gap_sec: float) -> list[tuple[int, int]]:
    """Индексы [i0, i1] включительно после склейки разрывов и отбрасывания коротких."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    gap = max(1, int(round(gap_sec * fps)))
    spans, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            spans.append((start, prev))
            start = i
        prev = i
    spans.append((start, prev))
    min_len = min_sec * fps
    return [(a, b) for a, b in spans if b - a + 1 >= min_len]


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


class _Signals:
    def __init__(self, n: int):
        self.n = n
        self.mask: dict[Key, np.ndarray] = {}
        self.conf: dict[Key, np.ndarray] = {}

    def put(self, key: Key, i: int, conf: float = 1.0) -> None:
        if key not in self.mask:
            self.mask[key] = np.zeros(self.n, dtype=bool)
            self.conf[key] = np.zeros(self.n)
        self.mask[key][i] = True
        self.conf[key][i] = max(self.conf[key][i], conf)

    def get(self, key: Key) -> np.ndarray:
        return self.mask.get(key, np.zeros(self.n, dtype=bool))


def build_timeline(frames: list[FramePerception], scene: Scene, tracks: dict[int, TrackInfo],
                   cfg: dict, fps: float) -> list[Event]:
    tc, kc = cfg["timeline"], cfg["pose"]["kpt_conf"]
    n = len(frames)
    if n == 0:
        return []
    t = np.array([f.t_sec for f in frames])
    role = {tid: info.role for tid, info in tracks.items()}
    sig = _Signals(n)
    counter_zone, staff_zone = scene.zones.get("counter_zone"), scene.zones.get("staff_side")
    cash_counter_zone = scene.zones.get("cash_counter_zone")
    motion: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(n))
    prev_hands: dict[int, tuple] = {}
    holders_by_frame: list[dict[int, str]] = []
    cash_tracks: dict[int, list[tuple[int, tuple]]] = defaultdict(list)

    for i, f in enumerate(frames):
        persons = [p for p in f.persons if not looks_outside(p.box, scene, cfg)]
        r = {p.track_id: role.get(p.track_id, "other") for p in persons}
        cashiers = [p for p in persons if r[p.track_id] == "cashier"]
        clients = {p.track_id for p in persons if r[p.track_id] == "client"}

        if cashiers:
            sig.put(("cashier_present", None, None), i)
        if any(at_workstation(p, scene, tc["workstation_margin"]) for p in cashiers):
            sig.put(("cashier_at_desk", None, None), i)
        if any(p.keypoints[w][2] >= kc for p in cashiers for w in (9, 10)):
            sig.put(("cashier_hands_visible", None, None), i)
        if clients:
            sig.put(("client_at_window", None, None), i)
        if len(clients) >= 2:
            sig.put(("multiple_clients", None, None), i)
        for p in persons:
            if r[p.track_id] == "collector":
                sig.put(("collector_present", p.track_id, None), i)
            elif r[p.track_id] == "other" and staff_zone and contains(staff_zone, center(p.box)):
                sig.put(("outsider_in_staff_zone", p.track_id, None), i)

        dets = [d for d in f.objects if d.conf >= tc["object_conf"]]
        bags = [d.box for d in dets if d.cls in BAG_CLASSES]
        for d in dets:
            if d.cls in tc["desk_objects"] and counter_zone and contains(counter_zone, center(d.box)):
                sig.put(("object_on_desk", None, d.cls), i, d.conf)
            if d.cls == "cup":
                for p in persons:
                    if intersection(expand(d.box, 0.3), head_box(p, kc)) > 0:
                        sig.put(("drinking", p.track_id, None), i, d.conf)

        holders: dict[int, str] = {}
        for d in dets:
            if d.cls not in tc["cash_classes"]:
                continue
            if d.track_id is not None:
                cash_tracks[d.track_id].append((i, d.box))
            for p in persons:
                radius = tc["hand_radius_forearm"] * forearm_len(p, kc, tc["forearm_fallback_frac"])
                if hand_near(p, d.box, radius):
                    holders[p.track_id] = r[p.track_id]
                    sig.put(("hold_cash", p.track_id, None), i, d.conf)
                    if inside_frac(d.box, body_zone(p, kc)) >= tc["near_body_frac"] and not (
                            counter_zone and contains(counter_zone, center(d.box))):
                        sig.put(("cash_near_body", p.track_id, None), i, d.conf)
            if any(inside_frac(d.box, expand(b, 0.2)) >= tc["in_zone_frac"] for b in bags):
                sig.put(("cash_near_bag", None, None), i, d.conf)
            if "safe_zone" in scene.zones and inside_frac(d.box, scene.zones["safe_zone"]) >= tc["in_zone_frac"]:
                sig.put(("cash_in_safe", None, None), i, d.conf)
            if cash_counter_zone and inside_frac(d.box, cash_counter_zone) >= tc["in_zone_frac"]:
                sig.put(("cash_at_counter_machine", None, None), i, d.conf)
        holders_by_frame.append(holders)
        if len(set(holders.values())) >= 2:
            sig.put(("cash_handover", None, None), i)

        for p in cashiers:
            fl = forearm_len(p, kc, tc["forearm_fallback_frac"])
            prev = prev_hands.get(p.track_id)
            if prev is not None:
                d = [np.hypot(a[0] - b[0], a[1] - b[1]) / fl for a, b in zip(p.hands, prev) if a and b]
                if d:
                    motion[p.track_id][i] = float(np.mean(d))
            prev_hands[p.track_id] = p.hands

    # --- производные сигналы ---
    at_desk = sig.get(("cashier_at_desk", None, None))
    for key in [k for k in sig.mask if k[0] == "object_on_desk"]:
        m = sig.mask[key] & ~at_desk
        if m.any():
            sig.mask[("object_unattended", None, key[2])] = m
            sig.conf[("object_unattended", None, key[2])] = sig.conf[key] * m
    if any(c in scene.anchors for c in ("counter", "safe")):  # без якорей «на месте» не определить
        sig.mask[("cashier_left_zone", None, None)] = ~at_desk
        sig.conf[("cashier_left_zone", None, None)] = (~at_desk).astype(float)

    w = max(1, int(round(tc["recount_window_sec"] * fps)))
    machine = sig.get(("cash_at_counter_machine", None, None))
    for tid, mo in motion.items():
        busy = _rolling_mean(mo, w) >= tc["recount_motion"]
        m = (sig.get(("hold_cash", tid, None)) & busy) | (machine & sig.get(("hold_cash", tid, None)))
        if m.any():
            sig.mask[("recount_visible", tid, None)] = m
            sig.conf[("recount_visible", tid, None)] = m.astype(float)

    events: list[Event] = []
    for key, m in sig.mask.items():
        etype, actor, cls = key
        for a, b in mask_to_intervals(m, fps, tc["min_event_sec"], tc["gap_sec"]):
            c = sig.conf[key][a:b + 1]
            conf = float(c[c > 0].mean()) if (c > 0).any() else 0.0
            args = {"cls": cls} if cls else {}
            if etype == "cash_handover":
                args.update(_handover_direction(holders_by_frame, a, b, fps))
            events.append(Event(etype, float(t[a]), float(t[b]), round(conf, 3), actor,
                                role.get(actor) if actor is not None else None, args))

    events += _cash_disappeared(frames, cash_tracks, scene, role, cfg, fps, t)
    events += _presence_edges(events, float(t[0]), float(t[-1]))
    return sorted(events, key=lambda e: (e.t_start, e.type))


def _handover_direction(holders_by_frame: list[dict[int, str]], a: int, b: int, fps: float) -> dict:
    """Кто держал деньги раньше — тот передаёт."""
    roles_in = {r for h in holders_by_frame[a:b + 1] for r in h.values()}
    look = max(1, int(1.5 * fps))
    first_seen: dict[str, int] = {}
    for i in range(max(0, a - look), b + 1):
        for r in holders_by_frame[i].values():
            first_seen.setdefault(r, i)
    order = sorted((r for r in roles_in if r in first_seen), key=lambda r: first_seen[r])
    return {"from_role": order[0], "to_role": order[-1]} if len(order) >= 2 else {}


def _cash_disappeared(frames, cash_tracks, scene, role, cfg, fps, t) -> list[Event]:
    """Трек денег закончился, и в том же месте деньги не появились снова."""
    tc = cfg["timeline"]
    n = len(frames)
    reappear = int(round(tc["disappear_reappear_sec"] * fps))
    out = []
    for tid, obs in cash_tracks.items():
        last_i, last_box = obs[-1]
        if len(obs) < max(2, tc["min_event_sec"] * fps) or last_i >= n - 1 - reappear:
            continue
        later = [d.box for f in frames[last_i + 1:last_i + 1 + reappear] for d in f.objects
                 if d.cls in tc["cash_classes"] and intersection(d.box, expand(last_box, 0.3)) > 0]
        if later:
            continue
        f = frames[last_i]
        persons = [p for p in f.persons if not looks_outside(p.box, scene, cfg)]
        bags = [d.box for d in f.objects if d.cls in BAG_CLASSES]
        zone, actor = locate(last_box, scene, persons, bags, cfg)
        out.append(Event("cash_disappeared", float(t[last_i]), float(t[last_i]), 1.0, actor,
                         role.get(actor) if actor is not None else None, {"last_zone": zone, "cash_track": tid}))
    return out


def _presence_edges(events: list[Event], t_first: float, t_last: float) -> list[Event]:
    """Точечные события входа/выхода из интервалов присутствия. На границах видео интервал
    обрезан записью — входа или выхода там не было, событие не создаётся."""
    pairs = {"client_at_window": ("client_arrived", "client_left_window"),
             "collector_present": ("collector_entered", "collector_left"),
             "cashier_left_zone": ("cashier_left", "cashier_returned")}
    out = []
    for e in events:
        if e.type in pairs:
            start, end = pairs[e.type]
            if e.t_start > t_first:
                out.append(Event(start, e.t_start, e.t_start, e.conf, e.actor, e.role, e.args))
            if e.t_end < t_last:
                out.append(Event(end, e.t_end, e.t_end, e.conf, e.actor, e.role, e.args))
    return out
