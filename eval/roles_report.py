"""Отчёт этапа 4 → eval/reports/roles.html: якоря, сторона персонала, роли треков, кадры с подписями.

  python -m eval.roles_report                    # якоря из детектора (кэш data/perception/full)
  python -m eval.roles_report --manual-anchors   # стойка/сейф/дверь из ручных зон (только для разработки)

Ручные зоны используются только для проверки автоопределения (IoU якорей) или как временная
подмена, пока нет детектора объектов. Лица на кадрах размыты.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from pipeline.ingest import iter_frames
from pipeline.perception.privacy import blur_heads
from pipeline.scene import analyze_scene
from pipeline.scene.anchors import compute_anchors
from pipeline.scene.roles import assign_roles
from pipeline.scene.zones import area, build_scene, intersection
from pipeline.types import Anchor
from training.cache_perception import CACHE_DIR, load_cached
from training.data.common import REPORTS_DIR, ROOT, UNIFIED_PATH, read_json

ROLE_COLORS = {"cashier": (40, 200, 90), "client": (60, 140, 255), "collector": (255, 160, 0), "other": (200, 200, 200)}
MANUAL_LABELS = {"counter": "counter", "safe": "safe", "door": "door"}


def manual_anchors(video: dict) -> dict[str, Anchor]:
    out = {}
    for z in video.get("zones", []):
        if z["label"] in MANUAL_LABELS and z.get("bbox_xyxy"):
            out[MANUAL_LABELS[z["label"]]] = Anchor(cls=z["label"], box=tuple(map(float, z["bbox_xyxy"])), presence=1.0)
    return out


def manual_staff_side(video: dict) -> str | None:
    zones = {z["label"]: z.get("bbox_xyxy") for z in video.get("zones", [])}
    cs, cl = zones.get("cashier_side"), zones.get("client_side")
    if not cs or not cl:
        return None
    return "near" if (cs[1] + cs[3]) > (cl[1] + cl[3]) else "far"


def iou(a, b) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def sheet(path: Path, frames, tracks, n: int = 5, save_to: Path | None = None) -> str:
    with_people = [f for f in frames if f.persons] or frames
    picks = [with_people[int(i)] for i in np.linspace(0, len(with_people) - 1, n)]
    by_idx = {f.frame_idx: f for f in picks}
    tiles = []
    for i, rgb in iter_frames(path, by_idx):
        f = by_idx[i]
        img = blur_heads(rgb, f.persons)
        for p in f.persons:
            role = tracks[p.track_id].role if p.track_id in tracks else "other"
            c = ROLE_COLORS[role]
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            cv2.putText(img, f"{role[:4]} #{p.track_id}", (x1 + 2, max(12, y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 2)
        cv2.putText(img, f"{f.t_sec:.1f}s", (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (232, 424)))
    img = Image.fromarray(np.concatenate(tiles, axis=1))
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_to, quality=80)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manual-anchors", action="store_true")
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "roles.html")
    ap.add_argument("--json-out", type=Path, default=REPORTS_DIR / "roles.json")
    args = ap.parse_args()

    with open(ROOT / "configs" / "pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tag = "full" if (CACHE_DIR / "full").is_dir() and not args.manual_anchors else "pose"
    sections, summary = [], {}
    for v in read_json(UNIFIED_PATH)["videos"]:
        vid = v["video_id"]
        if args.videos and vid not in args.videos:
            continue
        frames = load_cached(vid, tag)
        wh = (v["width"], v["height"])
        fps = cfg["analysis_fps"]
        manual = manual_anchors(v)
        if args.manual_anchors:
            scene = build_scene(manual, wh, cfg)
            tracks, scene = assign_roles(frames, scene, cfg, fps, wh)
            anchor_note = "якоря: ручные зоны (подмена для разработки)"
        else:
            scene, tracks = analyze_scene(frames, cfg, fps, wh)
            detected = compute_anchors(frames, cfg)
            ious = {c: round(iou(detected[c].box, manual[c].box), 2) for c in manual if c in detected}
            anchor_note = f"IoU найденных якорей с ручными зонами: {ious or '—'}"
        expected = manual_staff_side(v)
        roles = {tid: t.role for tid, t in tracks.items()}
        summary[vid] = {"staff_side": scene.staff_side, "manual_staff_side": expected,
                        "scene_confidence": scene.scene_confidence, "flags": scene.flags,
                        "tracks": {str(t.track_id): {"role": t.role, "n": t.n_frames, "t": [round(t.t_first, 1), round(t.t_last, 1)],
                                                    "side_frac": t.side_frac, "reason": t.reason} for t in tracks.values()}}
        side_ok = "" if expected is None else ("✓" if expected == scene.staff_side else "✗ (ручная: " + expected + ")")
        rows = "".join(
            f"<tr><td>{t.track_id}</td><td style='color:rgb{ROLE_COLORS[t.role]}'><b>{t.role}</b></td><td>{t.n_frames}</td>"
            f"<td>{t.t_first:.1f}–{t.t_last:.1f}</td><td>{t.side_frac}</td><td>{html.escape(t.reason)}</td></tr>"
            for t in sorted(tracks.values(), key=lambda t: -t.n_frames))
        sections.append(
            f"<h2>{vid}</h2><p>Сторона персонала: <b>{scene.staff_side}</b> {side_ok}; scene_confidence={scene.scene_confidence}; "
            f"{html.escape(anchor_note)}; флаги: {scene.flags or '—'}</p>"
            f"<p class=s>{html.escape(v.get('summary_ru', ''))}</p>"
            f"<table><tr><th>Трек</th><th>Роль</th><th>Кадров</th><th>Время, с</th><th>Стороны</th><th>Почему</th></tr>{rows}</table>"
            f"<img src='data:image/jpeg;base64,{sheet(ROOT / v['path'], frames, tracks, save_to=args.out.parent / 'roles' / f'{vid}.jpg')}'>")
        print(vid, scene.staff_side, side_ok, {r: list(roles.values()).count(r) for r in set(roles.values())})

    ok = [s for s in summary.values() if s["manual_staff_side"]]
    agree = sum(s["staff_side"] == s["manual_staff_side"] for s in ok)
    head = f"<p>Сторона персонала совпала с ручной разметкой в {agree}/{len(ok)} видео. Кэш perception: {tag}.</p>"
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>CashGuard — роли</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1c2129}}table{{border-collapse:collapse;margin:6px 0}}
th,td{{border:1px solid #d0d5dc;padding:3px 7px;text-align:left;font-size:13px}}th{{background:#f2f4f7}}img{{max-width:100%}}
.s{{color:#5f6b7a;max-width:1100px}}</style></head><body><h1>Сцена и роли (этап 4)</h1>{head}{''.join(sections)}</body></html>"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"staff side agreement {agree}/{len(ok)}; wrote {args.out}")


if __name__ == "__main__":
    main()
