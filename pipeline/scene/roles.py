"""Роли людей только по геометрии и траекториям (без распознавания лиц).

1. Люди за стеклом двери (почти всё время в зоне двери и мельче людей в комнате) — other.
2. Короткие треки — other (шум).
3. Кассир — самый долгий трек в комнате; сторона стойки, где он проводит больше времени, —
   сторона персонала. Другие треки на стороне персонала, почти не пересекающиеся с кассиром
   по времени, — фрагменты кассира (трекер рвёт треки при окклюзиях).
4. Инкассатор — сумка инкассации у рук или корпуса в нескольких кадрах.
5. На стороне клиента — клиент; одновременно с кассиром на стороне персонала — other (посторонний).
"""
from __future__ import annotations

from collections import defaultdict

from pipeline.scene.zones import body_zone, build_scene, expand, inside_frac, intersection, side_of
from pipeline.types import FramePerception, Person, Scene, TrackInfo

COLLECTOR_OBJECTS = ("collector_bag",)


def _overlap(a: list[float], b: list[float]) -> float:
    """Пересечение интервалов [t_first, t_last]."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _holds(p: Person, box) -> bool:
    if intersection(body_zone(p), box) > 0:
        return True
    return any(h is not None and box[0] <= h[0] <= box[2] and box[1] <= h[1] <= box[3]
               for h in p.hands)


def assign_roles(frames: list[FramePerception], scene: Scene, cfg: dict, fps: float,
                 frame_wh: tuple[int, int]) -> tuple[dict[int, TrackInfo], Scene]:
    rc, sc = cfg["roles"], cfg["scene"]
    counter = scene.anchors.get("counter")
    door = scene.zones.get("door_zone")

    tracks: dict[int, list[tuple[float, Person, FramePerception]]] = defaultdict(list)
    for f in frames:
        for p in f.persons:
            if p.track_id is not None:
                tracks[p.track_id].append((f.t_sec, p, f))
    if not tracks:
        return {}, scene

    heights = sorted(p.box[3] - p.box[1] for obs in tracks.values() for _, p, _ in obs)
    ref_height = heights[int(0.75 * (len(heights) - 1))]  # люди в комнате — крупные

    stats = {}
    for tid, obs in tracks.items():
        sides = defaultdict(int)
        in_door = bag_frames = 0
        for _, p, f in obs:
            s = side_of(p.box, counter.box, sc["side_margin_frac"]) if counter else None
            sides[s or "none"] += 1
            if door and inside_frac(p.box, door) >= 0.8:
                in_door += 1
            if any(d.cls in COLLECTOR_OBJECTS and _holds(p, expand(d.box, 0.2)) for d in f.objects):
                bag_frames += 1
        n = len(obs)
        mean_h = sum(p.box[3] - p.box[1] for _, p, _ in obs) / n
        stats[tid] = {
            "n": n, "t": [obs[0][0], obs[-1][0]],
            "side_frac": {k: round(sides[k] / n, 3) for k in ("near", "far", "none")},
            "outside": door is not None and in_door / n >= rc["outside_door_frac"]
                       and mean_h < rc["outside_height_frac"] * ref_height,
            "bag_frames": bag_frames,
        }

    def info(tid, role, reason, outside=False):
        s = stats[tid]
        return TrackInfo(track_id=tid, role=role, n_frames=s["n"], t_first=s["t"][0], t_last=s["t"][1],
                         side_frac=s["side_frac"], outside_room=outside, reason=reason)

    result: dict[int, TrackInfo] = {}
    room = []
    for tid, s in stats.items():
        if s["outside"]:
            result[tid] = info(tid, "other", "за стеклом двери", outside=True)
        elif s["n"] / fps < rc["min_track_sec"]:
            result[tid] = info(tid, "other", "короткий трек")
        else:
            room.append(tid)
    if not room:
        return result, scene

    main = max(room, key=lambda t: stats[t]["n"])
    staff = scene.staff_side
    if staff is None and counter:
        sf = stats[main]["side_frac"]
        staff = "near" if sf["near"] >= sf["far"] else "far"
        scene = build_scene(scene.anchors, frame_wh, cfg, staff_side=staff)
    client = {"near": "far", "far": "near"}.get(staff)

    result[main] = info(main, "cashier", "самый долгий трек в комнате")
    cashier_spans = [stats[main]["t"]]
    for tid in sorted(room, key=lambda t: -stats[t]["n"]):
        if tid == main:
            continue
        s = stats[tid]
        life = max(s["t"][1] - s["t"][0], 1.0 / fps)
        overlap = max(_overlap(s["t"], span) for span in cashier_spans) / life
        on_staff = staff is not None and s["side_frac"][staff] >= rc["staff_side_frac"]
        on_client = client is not None and s["side_frac"][client] >= rc["client_side_frac"]
        if s["bag_frames"] >= rc["collector_bag_min_frames"]:
            result[tid] = info(tid, "collector", f"сумка инкассации в {s['bag_frames']} кадрах")
        elif on_staff and overlap <= rc["cashier_overlap_tol"]:
            result[tid] = info(tid, "cashier", "фрагмент трека кассира")
            cashier_spans.append(s["t"])
        elif on_client:
            result[tid] = info(tid, "client", "сторона клиента")
        elif on_staff:
            result[tid] = info(tid, "other", "посторонний на стороне персонала")
        elif counter is None:
            result[tid] = info(tid, "client", "стойка не найдена: не кассир → клиент (низкая уверенность)")
        else:
            result[tid] = info(tid, "other", "сторона не определена")
    return result, scene

