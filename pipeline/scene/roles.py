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


def _iou(a, b) -> float:
    inter = intersection(a, b)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _coverage(spans: list[list[float]]) -> float:
    """Длина объединения интервалов."""
    total, end = 0.0, float("-inf")
    for a, b in sorted(spans):
        if b > end:
            total += b - max(a, end)
            end = b
    return total


def _staff_side_by_coverage(room: list[int], stats: dict, side_frac: float) -> str | None:
    """Сторона, треки на которой покрывают большую часть видео (кассир на месте всё время,
    даже если трекер рвёт его трек); при равенстве — по суммарному числу кадров."""
    score = {}
    for side in ("near", "far"):
        tids = [t for t in room if stats[t]["side_frac"][side] >= side_frac]
        score[side] = (round(_coverage([stats[t]["t"] for t in tids]), 1), sum(stats[t]["n"] for t in tids))
    if score["near"] == score["far"] == (0.0, 0):
        return None
    return max(score, key=score.get)


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
        bottom = sorted(p.box[3] for _, p, _ in obs)[n // 2]
        # За стеклом: почти всё время в зоне двери и ноги заметно выше верхнего края стойки
        # (стоит далеко, в коридоре). Без стойки — грубый признак «заметно мельче людей в комнате»
        # (со стойкой не используется: клиент за дальним краем стола мелкий из-за перспективы).
        if counter:
            cy1, ch = counter.box[1], counter.box[3] - counter.box[1]
            far_away = bottom < cy1 - rc["outside_feet_above_counter_frac"] * ch
        else:
            far_away = mean_h < rc["outside_height_frac"] * ref_height
        stats[tid] = {
            "n": n, "t": [obs[0][0], obs[-1][0]],
            "side_frac": {k: round(sides[k] / n, 3) for k in ("near", "far", "none")},
            "outside": door is not None and in_door / n >= rc["outside_door_frac"] and far_away,
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

    staff = scene.staff_side
    if staff is None and counter:
        staff = _staff_side_by_coverage(room, stats, rc["staff_side_frac"])
        if staff:
            scene = build_scene(scene.anchors, frame_wh, cfg, staff_side=staff)
    client = {"near": "far", "far": "near"}.get(staff)

    on_staff_side = [t for t in room if staff and stats[t]["side_frac"][staff] >= rc["staff_side_frac"]]
    main = max(on_staff_side or room, key=lambda t: stats[t]["n"])
    result[main] = info(main, "cashier", "самый долгий трек на стороне персонала" if on_staff_side
                        else "самый долгий трек в комнате (сторона не определена)")
    cashier_spans = [stats[main]["t"]]
    cashier_boxes = {f.frame_idx: p.box for _, p, f in tracks[main]}
    for tid in sorted(room, key=lambda t: -stats[t]["n"]):
        if tid == main:
            continue
        s = stats[tid]
        life = max(s["t"][1] - s["t"][0], 1.0 / fps)
        overlap = max(_overlap(s["t"], span) for span in cashier_spans) / life
        on_staff = staff is not None and s["side_frac"][staff] >= rc["staff_side_frac"]
        on_client = client is not None and s["side_frac"][client] >= rc["client_side_frac"]
        shared = [(_iou(p.box, cashier_boxes[f.frame_idx]), intersection(p.box, cashier_boxes[f.frame_idx]) > 0)
                  for _, p, f in tracks[tid] if f.frame_idx in cashier_boxes]
        if shared and sorted(i for i, _ in shared)[len(shared) // 2] >= rc["duplicate_iou"]:
            result[tid] = info(tid, "cashier", "дубль рамки кассира")
        elif (on_staff and shared and s["n"] / fps <= rc["cashier_part_max_sec"]
              and sum(touch for _, touch in shared) >= 0.8 * len(shared)):
            result[tid] = info(tid, "cashier", "короткая частичная рамка на кассире")
        elif s["bag_frames"] >= rc["collector_bag_min_frames"]:
            result[tid] = info(tid, "collector", f"сумка инкассации в {s['bag_frames']} кадрах")
        elif on_staff and overlap <= rc["cashier_overlap_tol"]:
            result[tid] = info(tid, "cashier", "фрагмент трека кассира")
            cashier_spans.append(s["t"])
            cashier_boxes.update({f.frame_idx: p.box for _, p, f in tracks[tid]})
        elif on_client:
            result[tid] = info(tid, "client", "сторона клиента")
        elif on_staff:
            result[tid] = info(tid, "other", "посторонний на стороне персонала")
        elif counter is None:
            result[tid] = info(tid, "client", "стойка не найдена: не кассир → клиент (низкая уверенность)")
        else:
            result[tid] = info(tid, "other", "сторона не определена")
    return result, scene

