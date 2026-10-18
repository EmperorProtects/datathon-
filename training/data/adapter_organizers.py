"""open_labels.xlsx (организаторы) + forte_annotations_v5/unified.json (команда) → data/annotations/unified.json.

Источник истины для вердикта и таймкодов — xlsx организаторов. Из командной разметки берутся
события (тип, границы, видимость, ссылки на нормативы), lookalikes и зоны.

Побайтовые дубли видео (по sha256) объединяются в одну запись (решение команды): вердикт
должен совпадать, таймкоды организаторов объединяются, событиям одного типа из разных
копий соответствует одна запись. Каждому событию назначается ближайший к его пику таймкод
организаторов.

Метаданные (fps, длительность, размер) и peak_frame пересчитываются по PTS из probe.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.ingest import read_timebase
from training.data.common import (
    FORTE_DIR,
    LABELS_XLSX,
    PROBE_PATH,
    ROOT,
    UNIFIED_PATH,
    read_json,
    read_labels,
    video_sort_key,
    write_json,
)

EVENT_EXTRA_FIELDS = (
    "rule_refs", "conditions_to_verify", "must_not_infer", "visual_support", "training_eligibility",
    "boundary_uncertainty_sec", "temporal_extent_censored_at_video_end", "review_status", "annotation_source",
)
VIDEO_EXTRA_FIELDS = ("zones", "zones_frame_time_sec", "summary_ru", "proposed_split_group")


def _nearest(values: list[float], t: float) -> float:
    return min(values, key=lambda v: abs(v - t))


def _convert_event(e: dict, video_id: str, n: int, org_times: list[float], pts) -> dict:
    peak = e["peak_sec"]
    out = {
        "event_id": f"{video_id}_e{n}",
        "kind": e["kind"],
        "type_id": e["type_id"],
        "organizer_time_sec": _nearest(org_times, peak if peak is not None else e["organizer_time_sec"]),
        "organizer_act": None,
        "organizer_clause": None,
        "start_sec": e["start_sec"],
        "peak_sec": peak,
        "end_sec": e["end_sec"],
        "peak_frame": pts.sec_to_frame(peak) if peak is not None else None,
        "actor_role": e.get("actor_role"),
        "visibility": e.get("visibility"),
        "episode": e.get("episode"),
        "expected_step": e.get("expected_step"),
        "checkpoint_sec": e.get("checkpoint_sec"),
        "note": e.get("note") or "",
        "source_event_id": e["event_id"],
    }
    out.update({k: e[k] for k in EVENT_EXTRA_FIELDS if k in e})
    return out


def build(labels, forte: dict, probe: dict, timebases: dict) -> dict:
    forte_by_id = {v["video_id"]: v for v in forte["videos"]}
    label_by_id = {r.video_id: r for r in labels}
    missing = set(label_by_id) ^ set(probe)
    if missing:
        raise ValueError(f"xlsx rows and video files differ: {sorted(missing)}")

    groups: dict[str, list[str]] = defaultdict(list)
    for vid, info in probe.items():
        groups[info["sha256"]].append(vid)

    videos = []
    for members in groups.values():
        members.sort(key=video_sort_key)
        canon = members[0]
        flags = {label_by_id[m].has_violation for m in members}
        if len(flags) != 1:
            raise ValueError(f"duplicates {members} have conflicting organizer labels")
        has_violation = flags.pop()
        org_times = sorted({label_by_id[m].time_sec for m in members if label_by_id[m].time_sec is not None})

        # События: одна запись на тип; при дублях берём событие из канонической копии.
        events_by_type: dict[str, dict] = {}
        for m in members:
            for e in forte_by_id.get(m, {}).get("events", []):
                events_by_type.setdefault(e["type_id"], e)
        pts = timebases[canon]
        events = [
            _convert_event(e, canon, i, org_times, pts)
            for i, e in enumerate(sorted(events_by_type.values(), key=lambda e: e["peak_sec"] or 0), start=1)
        ]

        lookalikes, seen = [], set()
        for m in members:
            for lk in forte_by_id.get(m, {}).get("lookalikes", []):
                key = (lk["start_sec"], lk["end_sec"], lk["what"])
                if key not in seen:
                    seen.add(key)
                    lookalikes.append({k: lk[k] for k in ("start_sec", "end_sec", "what", "related_type_ids",
                                                           "why_not_violation") if k in lk})

        info, src = probe[canon], forte_by_id.get(canon, {})
        video = {
            "video_id": canon,
            "path": info["path"],
            "fps": round(info["avg_fps"], 4),
            "duration_sec": round(info["duration_sec"], 4),
            "width": info["width"],
            "height": info["height"],
            "n_frames": info["n_frames"],
            "sha256": info["sha256"],
            "has_violation": has_violation,
            "organizer_times_sec": org_times,
            "organizer_rows": [
                {"video_id": m, "excel_row": label_by_id[m].excel_row, "raw_name": label_by_id[m].raw_name,
                 "raw_time": label_by_id[m].raw_time, "raw_label": label_by_id[m].raw_label}
                for m in members
            ],
            "duplicates": members[1:],
            "events": events,
            "lookalikes": lookalikes,
            "zones_frame": src.get("zones_frame"),
            "split_group": canon,
        }
        video.update({k: src[k] for k in VIDEO_EXTRA_FIELDS if k in src})
        videos.append(video)

    videos.sort(key=lambda v: video_sort_key(v["video_id"]))
    return {
        "schema_version": 1,
        "generated_by": "training/data/adapter_organizers.py",
        "generated_on": dt.date.today().isoformat(),
        "sources": {"organizer_labels": "data/raw/open/open_labels.xlsx",
                    "team_annotations": f"forte_annotations_v5/unified.json (schema {forte.get('schema_version')})"},
        "conventions": {
            "timestamps": "seconds from first decoded frame (PTS)",
            "frame_index_base": 0,
            "organizer_time": "point timecode from xlsx; event.organizer_time_sec = nearest to peak_sec",
            "events": "team visual candidates (AI-assisted sampled review), not organizer ground truth",
        },
        "videos": videos,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS_XLSX)
    ap.add_argument("--forte", type=Path, default=FORTE_DIR / "unified.json")
    ap.add_argument("--probe", type=Path, default=PROBE_PATH)
    ap.add_argument("--out", type=Path, default=UNIFIED_PATH)
    args = ap.parse_args()

    probe = read_json(args.probe)
    timebases = {vid: read_timebase(ROOT / info["path"]) for vid, info in probe.items()}
    unified = build(read_labels(args.labels), read_json(args.forte), probe, timebases)
    write_json(args.out, unified)
    vs = unified["videos"]
    print(f"wrote {args.out}: {len(vs)} videos ({sum(v['has_violation'] for v in vs)} with violation), "
          f"{sum(len(v['events']) for v in vs)} events, duplicates merged: "
          f"{[(v['video_id'], v['duplicates']) for v in vs if v['duplicates']]}")


if __name__ == "__main__":
    main()
