"""Кадры видео по PTS → data/frames/{video_id}/{frame_idx:06d}.jpg + index.json (кадр → секунда).

По умолчанию 10 кадров/с (кадр, ближайший к сетке с шагом 0.1 с); --full-fps — все кадры.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from pipeline.ingest import iter_frames, read_timebase
from training.data.common import ROOT, UNIFIED_PATH, read_json, write_json

FRAMES_DIR = ROOT / "data" / "frames"


def extract(video_path: Path, out_dir: Path, fps: float | None, quality: int = 90) -> int:
    tb = read_timebase(video_path)
    indices = list(range(len(tb))) if fps is None else tb.sample_indices(fps)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for i, rgb in iter_frames(video_path, indices):
        name = f"{i:06d}.jpg"
        Image.fromarray(rgb).save(out_dir / name, quality=quality)
        index.append({"frame_idx": i, "pts_sec": round(tb.frame_to_sec(i), 6), "file": name})
    write_json(out_dir / "index.json", {"fps": fps, "n_source_frames": len(tb), "frames": index})
    return len(index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unified", type=Path, default=UNIFIED_PATH)
    ap.add_argument("--out", type=Path, default=FRAMES_DIR)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--full-fps", action="store_true")
    ap.add_argument("--videos", nargs="*", help="только эти video_id")
    args = ap.parse_args()
    fps = None if args.full_fps else args.fps
    sub = "full" if fps is None else f"{fps:g}fps"
    for v in read_json(args.unified)["videos"]:
        if args.videos and v["video_id"] not in args.videos:
            continue
        n = extract(ROOT / v["path"], args.out / sub / v["video_id"], fps)
        print(f"{v['video_id']}: {n} frames")


if __name__ == "__main__":
    main()
