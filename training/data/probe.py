"""Метаданные всех видео по PTS: fps, пропуски кадров (VFR), длительность, разрешение, кодек, sha256.

Выход: data/annotations/probe.json (используется адаптером и отчётом).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from pipeline.ingest import probe, read_timebase
from training.data.common import PROBE_PATH, VIDEOS_DIR, find_videos, rel_path, sha256, write_json


def run(videos_dir: Path = VIDEOS_DIR) -> dict:
    out = {}
    for vid, path in find_videos(videos_dir).items():
        tb = read_timebase(path)
        info = asdict(probe(path, tb))
        info.update(video_id=vid, file_name=path.name, sha256=sha256(path), size_bytes=path.stat().st_size)
        info["path"] = rel_path(path)
        out[vid] = info
        print(f"{vid:6s} {info['width']}x{info['height']} {info['codec']} {info['avg_fps']:.2f}fps "
              f"{info['duration_sec']:6.1f}s frames={info['n_frames']} dropped={info['dropped_frames']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", type=Path, default=VIDEOS_DIR)
    ap.add_argument("--out", type=Path, default=PROBE_PATH)
    args = ap.parse_args()
    write_json(args.out, run(args.videos))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
