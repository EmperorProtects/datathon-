"""Кэш perception для train-видео → data/perception/{tag}/{video_id}.pkl (list[FramePerception]).

Нужен для разработки сцены, ролей и таймлайна без повторного прогона моделей.
Если weights/objects_best.pt ещё нет, сохраняются только люди и позы (tag=pose).
"""
from __future__ import annotations

import argparse
import pickle
import time

import yaml

from pipeline.ingest import read_timebase
from pipeline.perception import PerceptionModels, run_perception
from training.data.common import ROOT, UNIFIED_PATH, read_json

CACHE_DIR = ROOT / "data" / "perception"


def load_cached(video_id: str, tag: str = "full"):
    with open(CACHE_DIR / tag / f"{video_id}.pkl", "rb") as f:
        return pickle.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--batch", type=int, help="переопределить perception.batch (меньше — меньше VRAM)")
    ap.add_argument("--objects-model", help="другие веса объектов, напр. runs/detector/fold0/weights/best.pt")
    args = ap.parse_args()

    with open(ROOT / "configs" / "pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.batch:
        cfg["perception"]["batch"] = args.batch
    if args.objects_model:
        cfg["detector"]["objects_model"] = args.objects_model
    models = PerceptionModels.load(cfg)
    tag = "full" if models.objects else "pose"
    out_dir = CACHE_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in read_json(UNIFIED_PATH)["videos"]:
        if args.videos and v["video_id"] not in args.videos:
            continue
        path = ROOT / v["path"]
        t0 = time.perf_counter()
        frames = run_perception(path, read_timebase(path), models, cfg)
        with open(out_dir / f"{v['video_id']}.pkl", "wb") as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
        dt = time.perf_counter() - t0
        print(f"{v['video_id']}: {len(frames)} frames in {dt:.1f}s ({len(frames) / dt:.1f} fps) → {tag}")


if __name__ == "__main__":
    main()
