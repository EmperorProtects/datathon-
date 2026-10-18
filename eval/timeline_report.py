"""Отчёт этапа 5 → eval/reports/timeline.html (+ timeline/{video}.json и .png).

Приёмка: для нарушений-действий ключевые события (registry.timeline_key_events) есть в окне
±window_sec от peak_sec. Для бездействий — на интервале [start, end] ± window_sec (справочно, этап 7).
"""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import yaml

from pipeline.registry import load_registry
from pipeline.scene import analyze_scene
from pipeline.timeline import build_timeline, save_timeline
from pipeline.types import Event
from training.cache_perception import CACHE_DIR, load_cached
from training.data.common import REPORTS_DIR, ROOT, UNIFIED_PATH, read_json

WINDOW_SEC = 2.0


def matches(e: Event, spec: str) -> bool:
    etype, _, cls = spec.partition(":")
    return e.type == etype and (not cls or e.args.get("cls") == cls)


def check(events: list[Event], requirements: list[str], lo: float, hi: float) -> tuple[bool, list[str]]:
    near = [e for e in events if e.t_end >= lo and e.t_start <= hi]
    detail, ok = [], True
    for req in requirements:
        hit = any(matches(e, alt) for alt in req.split("|") for e in near)
        ok &= hit
        detail.append(("✓ " if hit else "✗ ") + req)
    return ok, detail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="full", help="кэш perception: full | oof | pose")
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "timeline.html")
    args = ap.parse_args()

    with open(ROOT / "configs" / "pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    reg = load_registry()
    keys = {t.type_id: t.timeline_key_events for t in reg.types + reg.skipped}
    fps = cfg["analysis_fps"]
    out_dir = args.out.parent / "timeline"
    rows, stats = [], {"action": [0, 0], "omission": [0, 0]}

    for v in read_json(UNIFIED_PATH)["videos"]:
        vid = v["video_id"]
        if args.videos and vid not in args.videos:
            continue
        if not (CACHE_DIR / args.tag / f"{vid}.pkl").is_file():
            continue
        frames = load_cached(vid, args.tag)
        scene, tracks = analyze_scene(frames, cfg, fps, (v["width"], v["height"]))
        events = build_timeline(frames, scene, tracks, cfg, fps)
        marks = [(e["peak_sec"], e["type_id"]) for e in v["events"]]
        save_timeline(events, v["duration_sec"], out_dir / f"{vid}.json", out_dir / f"{vid}.png", marks)

        for gt in v["events"]:
            req = keys.get(gt["type_id"], [])
            if not req:
                continue
            if gt["kind"] == "action":
                lo, hi = gt["peak_sec"] - WINDOW_SEC, gt["peak_sec"] + WINDOW_SEC
            else:
                lo, hi = gt["start_sec"] - WINDOW_SEC, gt["end_sec"] + WINDOW_SEC
            ok, detail = check(events, req, lo, hi)
            stats[gt["kind"]][0] += ok
            stats[gt["kind"]][1] += 1
            rows.append((vid, gt, ok, detail, lo, hi))
        print(vid, f"{len(events)} events, scene_confidence={scene.scene_confidence}")

    def rate(k):
        hit, tot = stats[k]
        return f"{hit}/{tot} ({100 * hit / tot:.0f}%)" if tot else "—"

    body = []
    for vid, gt, ok, detail, lo, hi in rows:
        png = out_dir / f"{vid}.png"
        img = base64.b64encode(png.read_bytes()).decode("ascii") if png.is_file() else ""
        body.append(
            f"<h3 class={'ok' if ok else 'bad'}>{vid}: {gt['type_id']} ({gt['kind']}), окно {lo:.1f}–{hi:.1f} с — "
            f"{'есть' if ok else 'НЕТ'}</h3><p>{' · '.join(html.escape(d) for d in detail)}</p>"
            f"<details><summary>таймлайн</summary><img src='data:image/png;base64,{img}'></details>")
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>CashGuard — таймлайн</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1c2129}}h3{{font-size:15px;margin:18px 0 2px}}
.ok{{color:#1a7f37}}.bad{{color:#c93c37}}img{{max-width:100%}}</style></head><body>
<h1>Таймлайн событий (этап 5)</h1><p>Кэш perception: <b>{args.tag}</b>. Окно ±{WINDOW_SEC} с.</p>
<p>Действия: ключевые события найдены в <b>{rate('action')}</b> (приёмка ≥ 80%).<br>
Бездействия (справочно, этап 7): {rate('omission')}.</p>{''.join(body)}</body></html>"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"actions {rate('action')}, omissions {rate('omission')}; wrote {args.out}")


if __name__ == "__main__":
    main()
