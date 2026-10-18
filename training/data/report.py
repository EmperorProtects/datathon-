"""HTML-отчёт по датасету → eval/reports/dataset.html.

Содержит: сводку, распределение типов и нормативов, таблицу видео, сравнение таймкода
организаторов с границами событий (конвенция для D4), кадры в моменты таймкодов организаторов.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from pipeline.ingest import iter_frames, read_timebase
from pipeline.registry import load_registry
from training.data.common import PROBE_PATH, REPORTS_DIR, ROOT, UNIFIED_PATH, read_json
from training.data.validate import validate

THUMB_W = 180


def _thumb(path: Path, times: list[float]) -> dict[float, str]:
    tb = read_timebase(path)
    idx = {tb.sec_to_frame(t): t for t in times}
    out = {}
    for i, rgb in iter_frames(path, idx):
        img = Image.fromarray(rgb)
        img.thumbnail((THUMB_W, THUMB_W * 2))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        out[idx[i]] = base64.b64encode(buf.getvalue()).decode("ascii")
    return out


def _stats(values: list[float]) -> str:
    if not values:
        return "—"
    return (f"медиана {statistics.median(values):+.1f} с, |Δ|≤1 с: {sum(abs(v) <= 1 for v in values)}/{len(values)}, "
            f"диапазон [{min(values):+.1f}; {max(values):+.1f}]")


def _table(headers: list[str], rows: list[list]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def build(unified: dict, probe: dict, thumbs: bool = True) -> str:
    reg = load_registry()
    names = {t.type_id: t.name_ru for t in reg.types + reg.skipped}
    videos = unified["videos"]
    errors = validate(unified, reg)
    e = html.escape

    n_viol = sum(v["has_violation"] for v in videos)
    events = [(v, ev) for v in videos for ev in v["events"]]
    durations = [v["duration_sec"] for v in videos]
    summary = _table(["Показатель", "Значение"], [
        ["Записей в xlsx организаторов", len(probe)],
        ["Уникальных видео (после объединения побайтовых дублей)", len(videos)],
        ["Дубли", ", ".join(f"{v['video_id']} = {', '.join(v['duplicates'])}" for v in videos if v["duplicates"]) or "—"],
        ["С нарушением / без", f"{n_viol} / {len(videos) - n_viol}"],
        ["Таймкодов организаторов", sum(len(v["organizer_times_sec"]) for v in videos)],
        ["Событий (командная разметка)", f"{len(events)}: действий {sum(ev['kind'] == 'action' for _, ev in events)}, "
                                         f"бездействий {sum(ev['kind'] == 'omission' for _, ev in events)}"],
        ["Событий на видео с нарушением", ", ".join(f"{k}: {c}" for k, c in sorted(
            Counter(len(v["events"]) for v in videos if v["has_violation"]).items()))],
        ["Длительность, с", f"сумма {sum(durations) / 60:.1f} мин; мин {min(durations):.1f}, медиана "
                            f"{statistics.median(durations):.1f}, макс {max(durations):.1f}"],
        ["Разрешение / кодек", ", ".join(f"{k}: {c}" for k, c in Counter(
            f"{p['width']}×{p['height']} {p['codec']}" for p in probe.values()).items())],
        ["Средний fps", f"{min(p['avg_fps'] for p in probe.values()):.2f}–{max(p['avg_fps'] for p in probe.values()):.2f}"],
        ["Пропуски кадров (интервал > 1.5× медианы)", f"в {sum(p['dropped_frames'] > 0 for p in probe.values())} видео, "
                                                     f"макс. интервал {max(p['max_gap_sec'] for p in probe.values()) * 1000:.0f} мс"],
        ["Валидатор unified.json", "OK" if not errors else "<br>".join(e(x) for x in errors)],
    ])

    by_type = defaultdict(list)
    for v, ev in events:
        by_type[ev["type_id"]].append(v["video_id"])
    type_rows = []
    for tid, vids in sorted(by_type.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        try:
            src = reg.source(tid)
            norm = f"{e(src.act)}, {e(src.clause)}"
        except KeyError:
            norm = "<b class=bad>нет норматива — тип не сообщается</b>"
        kind = next(ev["kind"] for _, ev in events if ev["type_id"] == tid)
        type_rows.append([f"<code>{tid}</code>", e(names.get(tid, "?")), kind, len(vids), ", ".join(vids), norm])
    types_html = _table(["type_id", "Название", "Вид", "Событий", "Видео", "Основной норматив"], type_rows)

    act_counter = Counter()
    for v, ev in events:
        try:
            s = reg.source(ev["type_id"])
            act_counter[(s.act, s.clause)] += 1
        except KeyError:
            act_counter[("—", "нет норматива")] += 1
    acts_html = _table(["Акт", "Пункт", "Событий"],
                       [[e(a), e(c), n] for (a, c), n in act_counter.most_common()])

    conv_rows, deltas = [], defaultdict(list)
    for v, ev in events:
        org = ev["organizer_time_sec"]
        row = {"start": org - ev["start_sec"], "peak": org - ev["peak_sec"]}
        if ev.get("checkpoint_sec") is not None:
            row["checkpoint"] = org - ev["checkpoint_sec"]
        for k, d in row.items():
            deltas[(ev["kind"], k)].append(d)
        conv_rows.append([v["video_id"], f"<code>{ev['type_id']}</code>", ev["kind"], f"{org:.1f}",
                          f"{ev['start_sec']:.1f}", f"{ev['peak_sec']:.1f}", f"{ev['end_sec']:.1f}",
                          "—" if ev.get("checkpoint_sec") is None else f"{ev['checkpoint_sec']:.1f}"])
    conv_summary = _table(["Вид", "Δ = t_org − …", "Статистика"], [
        [kind, ref, _stats(deltas[(kind, ref)])]
        for kind in ("action", "omission") for ref in ("start", "peak", "checkpoint") if deltas[(kind, ref)]
    ])
    conv_html = conv_summary + _table(["Видео", "Тип", "Вид", "t_org", "start", "peak", "end", "checkpoint"], conv_rows)

    video_rows = []
    for v in videos:
        p = probe[v["video_id"]]
        cells = [
            v["video_id"] + (f"<br><small>= {', '.join(v['duplicates'])}</small>" if v["duplicates"] else ""),
            f"{v['duration_sec']:.1f}", f"{p['avg_fps']:.2f}", p["dropped_frames"],
            "<b class=bad>нарушение</b>" if v["has_violation"] else "без нарушения",
            ", ".join(f"{t:.0f}" for t in v["organizer_times_sec"]) or "—",
            "<br>".join(f"<code>{ev['type_id']}</code>" for ev in v["events"]) or "—",
            len(v["lookalikes"]),
        ]
        if thumbs:
            times = v["organizer_times_sec"] or [v["duration_sec"] / 2]
            imgs = _thumb(ROOT / v["path"], times)
            cells.append(" ".join(
                f"<figure><img src='data:image/jpeg;base64,{imgs[t]}'><figcaption>{t:.1f} с</figcaption></figure>"
                for t in times if t in imgs))
        video_rows.append(cells)
    headers = ["Видео", "Длит., с", "fps", "Пропуски", "Метка", "t_org, с", "События", "Lookalikes"]
    videos_html = _table(headers + (["Кадр в t_org (для видео без нарушения — середина)"] if thumbs else []), video_rows)

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>CashGuard — датасет</title>
<style>
body {{ font: 14px/1.45 system-ui, sans-serif; margin: 24px; color: #1c2129; }}
h1 {{ font-size: 22px; }} h2 {{ margin-top: 32px; font-size: 18px; }}
table {{ border-collapse: collapse; margin: 8px 0 16px; }}
th, td {{ border: 1px solid #d0d5dc; padding: 4px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f2f4f7; }} code {{ font-size: 12px; }} .bad {{ color: #c93c37; }}
figure {{ display: inline-block; margin: 0 6px 0 0; }} figure img {{ width: {THUMB_W}px; display: block; }}
figcaption {{ font-size: 12px; color: #5f6b7a; }} .note {{ color: #5f6b7a; max-width: 900px; }}
</style></head><body>
<h1>Датасет CashGuard</h1>
<p class="note">Метки и таймкоды — из open_labels.xlsx организаторов. Типы событий, границы и нормативы —
командная разметка с выборочной визуальной проверкой ассистентом, не подтверждённая экспертом банка;
тексты нормативов не сверены с официальной редакцией.</p>
<h2>Сводка</h2>{summary}
<h2>Типы нарушений</h2>{types_html}
<h2>Распределение по актам и пунктам</h2>{acts_html}
<h2>Таймкод организаторов относительно границ события</h2>
<p class="note">Нужно для выбора конвенции таймкода (omission_timecode в configs/pipeline.yaml):
к какой точке события ближе таймкод организаторов.</p>{conv_html}
<h2>Видео</h2>{videos_html}
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unified", type=Path, default=UNIFIED_PATH)
    ap.add_argument("--probe", type=Path, default=PROBE_PATH)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "dataset.html")
    ap.add_argument("--no-thumbs", action="store_true")
    args = ap.parse_args()
    page = build(read_json(args.unified), read_json(args.probe), thumbs=not args.no_thumbs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
