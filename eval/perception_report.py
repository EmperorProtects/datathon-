"""Отчёт этапа 3 → eval/reports/perception.html: mAP50 по фолдам, скорость perception, треки на 3 видео.

Скорость меряется на полном проходе run_perception (декодирование + pose + объекты + трекинг)
при analysis_fps из configs/pipeline.yaml. Лица на визуализации размыты.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from pipeline.ingest import iter_frames, read_timebase
from pipeline.perception import PerceptionModels, run_perception
from pipeline.perception.privacy import blur_heads
from training.data.common import REPORTS_DIR, ROOT, UNIFIED_PATH, read_json

CV_SUMMARY = ROOT / "runs" / "detector" / "cv_summary.json"
SHEET_FRAMES = 8


def _color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return 200, 200, 200
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw(rgb: np.ndarray, fp) -> np.ndarray:
    img = blur_heads(rgb, fp.persons)
    for p in fp.persons:
        x1, y1, x2, y2 = map(int, p.box)
        c = _color(p.track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, f"P{p.track_id}", (x1 + 2, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        for hand in p.hands:
            if hand:
                cv2.circle(img, tuple(map(int, hand)), 5, c, -1)
    for d in fp.objects:
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 210, 0), 1)
        cv2.putText(img, f"{d.cls}#{d.track_id}", (x1 + 2, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 210, 0), 1)
    cv2.putText(img, f"{fp.t_sec:.1f}s", (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def contact_sheet(path: Path, frames, n: int = SHEET_FRAMES) -> str:
    picks = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, n)]
    by_idx = {fp.frame_idx: fp for fp in picks}
    tiles = [draw(rgb, by_idx[i]) for i, rgb in iter_frames(path, by_idx)]
    tiles = [cv2.resize(t, (232, 424)) for t in tiles]
    sheet = np.concatenate(tiles, axis=1)
    buf = io.BytesIO()
    Image.fromarray(sheet).save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def track_stats(frames) -> dict:
    person_ids = {p.track_id for f in frames for p in f.persons if p.track_id is not None}
    obj = {}
    for f in frames:
        for d in f.objects:
            obj.setdefault(d.cls, set()).add(d.track_id)
    return {"person_tracks": len(person_ids),
            "persons_per_frame": round(float(np.mean([len(f.persons) for f in frames])), 2),
            "object_tracks": {k: len(v) for k, v in sorted(obj.items())}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", nargs="*", default=["X-4", "X-15", "X-30"])
    ap.add_argument("--bench", type=int, default=6, help="число видео для замера скорости (самые длинные)")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "perception.html")
    args = ap.parse_args()

    with open(ROOT / "configs" / "pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    videos = {v["video_id"]: v for v in read_json(UNIFIED_PATH)["videos"]}
    models = PerceptionModels.load(cfg)
    if models.objects is None:
        raise SystemExit(f"objects model not found: {cfg['detector']['objects_model']}")

    # прогрев
    first = next(iter(videos.values()))
    run_perception(ROOT / first["path"], read_timebase(ROOT / first["path"]), models, cfg)

    bench_rows, total_frames, total_sec = [], 0, 0.0
    for v in sorted(videos.values(), key=lambda v: -v["duration_sec"])[: args.bench]:
        path = ROOT / v["path"]
        t0 = time.perf_counter()
        tb = read_timebase(path)
        t1 = time.perf_counter()
        frames = run_perception(path, tb, models, cfg)
        t2 = time.perf_counter()
        total_frames += len(frames)
        total_sec += t2 - t1
        bench_rows.append([v["video_id"], f"{v['duration_sec']:.1f}", len(frames), f"{t1 - t0:.1f}",
                           f"{t2 - t1:.1f}", f"{len(frames) / (t2 - t1):.1f}",
                           f"{(t2 - t0) * 145 / v['duration_sec']:.0f}"])
    speed = total_frames / total_sec

    sheets = []
    for vid in args.videos:
        v = videos[vid]
        path = ROOT / v["path"]
        frames = run_perception(path, read_timebase(path), models, cfg)
        stats = track_stats(frames)
        sheets.append(f"<h3>{vid} ({v['duration_sec']:.0f} с)</h3><p><code>{html.escape(json.dumps(stats, ensure_ascii=False))}</code></p>"
                      f"<img src='data:image/jpeg;base64,{contact_sheet(path, frames)}'>")

    cv_html = "<p>Нет runs/detector/cv_summary.json — кросс-валидация не запускалась.</p>"
    if CV_SUMMARY.is_file():
        cv = read_json(CV_SUMMARY)
        rows = "".join(f"<tr><td>{f['fold']}</td><td>{', '.join(f['val_groups'])}</td><td>{f['n_train']}/{f['n_val']}</td>"
                       f"<td>{f['map50']:.3f}</td><td>{f['map50_95']:.3f}</td></tr>" for f in cv["folds"])
        cls_rows = "".join(f"<tr><td>{c}</td><td>{s['mean']:.3f}</td><td>{s['folds_present']}</td></tr>"
                           for c, s in sorted(cv["per_class_mean_map50"].items(), key=lambda kv: -kv[1]["mean"]))
        cv_html = (f"<p>Средний mAP50 = <b>{cv['mean_map50']:.3f}</b>, mAP50-95 = {cv['mean_map50_95']:.3f}</p>"
                   f"<table><tr><th>Фолд</th><th>Валидационные семейства</th><th>Кадров train/val</th><th>mAP50</th><th>mAP50-95</th></tr>{rows}</table>"
                   f"<h3>По классам (среднее по фолдам, где класс есть в валидации)</h3>"
                   f"<table><tr><th>Класс</th><th>mAP50</th><th>Фолдов</th></tr>{cls_rows}</table>")

    bench_html = ("<table><tr><th>Видео</th><th>Длит., с</th><th>Кадров анализа</th><th>PTS, с</th><th>Perception, с</th>"
                  "<th>Кадров/с</th><th>Оценка для 145 с, с</th></tr>"
                  + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in bench_rows) + "</table>")
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>CashGuard — perception</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1c2129}}table{{border-collapse:collapse;margin:8px 0 16px}}
th,td{{border:1px solid #d0d5dc;padding:4px 8px;text-align:left}}th{{background:#f2f4f7}}img{{max-width:100%}}</style></head><body>
<h1>Perception (этап 3)</h1>
<h2>Детектор объектов: GroupKFold по визуальным семействам</h2>{cv_html}
<h2>Скорость</h2><p>analysis_fps = {cfg['analysis_fps']}, pose imgsz = {cfg['pose']['imgsz']}, objects imgsz = {cfg['detector']['imgsz']},
batch = {cfg['perception']['batch']}, FP16 = {cfg['perception']['half']}. Средняя скорость: <b>{speed:.1f} кадров/с</b>
(декодирование + pose + объекты + трекинг).</p>{bench_html}
<h2>Треки (головы размыты)</h2>{''.join(sheets)}
</body></html>"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"perception speed {speed:.1f} frames/s; wrote {args.out}")


if __name__ == "__main__":
    main()
