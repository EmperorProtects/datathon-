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
    ap.add_argument("--oof", action="store_true",
                    help="out-of-fold: каждое видео — моделью фолда, где его семейство было в валидации (tag=oof)")
    args = ap.parse_args()
    if args.oof:
        return run_oof(args)

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


def run_oof(args) -> None:
    from training.train_detector import RUNS_DIR, frames_by_group, load_config, make_folds

    with open(ROOT / "configs" / "pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.batch:
        cfg["perception"]["batch"] = args.batch
    with open(ROOT / "data" / "annotations" / "forte_v5" / "forte_annotations_v5" / "objects" / "split_manifest.json",
              encoding="utf-8") as f:
        import json
        family = json.load(f)["groups"]
    folds = make_folds(frames_by_group(), load_config()["folds"])
    fold_of = {g: k for k, gs in enumerate(folds) for g in gs}
    out_dir = CACHE_DIR / "oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = [v for v in read_json(UNIFIED_PATH)["videos"] if not args.videos or v["video_id"] in args.videos]
    for k in range(len(folds)):
        weights = RUNS_DIR / f"fold{k}" / "weights" / "best.pt"
        todo = [v for v in videos if fold_of[family[v["video_id"]]] == k
                and not (out_dir / f"{v['video_id']}.pkl").is_file()]
        if not todo or not (RUNS_DIR / f"fold{k}" / "metrics.json").is_file():  # фолд дообучен
            continue
        cfg["detector"]["objects_model"] = str(weights.relative_to(ROOT))
        models = PerceptionModels.load(cfg)
        for v in todo:
            path = ROOT / v["path"]
            frames = run_perception(path, read_timebase(path), models, cfg)
            with open(out_dir / f"{v['video_id']}.pkl", "wb") as f:
                pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"{v['video_id']}: fold{k}, {len(frames)} frames → oof")


if __name__ == "__main__":
    main()
