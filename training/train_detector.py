"""Дообучение YOLO11 на объектах кассы (командная разметка forte_annotations_v5/objects).

  python -m training.train_detector --fold 0        # обучить фолд k (GroupKFold по визуальным семействам)
  python -m training.train_detector --fold all      # все фолды подряд + сводка mAP50
  python -m training.train_detector --final         # финальная модель на всех кадрах → weights/objects_best.pt

Сплит — по видео: все кадры видео (и всё его визуальное семейство) попадают в одну часть.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from training.data.common import FORTE_DIR, ROOT, normalize_video_id, write_json

OBJECTS_DIR = FORTE_DIR / "objects"
WORK_DIR = ROOT / "data" / "yolo"
RUNS_DIR = ROOT / "runs" / "detector"
CONFIG = ROOT / "configs" / "train_detector.yaml"


def load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_names() -> list[str]:
    with open(OBJECTS_DIR / "dataset.yaml", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    return [names[i] for i in sorted(names)]


def frames_by_group() -> dict[str, list[Path]]:
    """Кадры, сгруппированные по визуальному семейству; дубли видео — в группе оригинала."""
    with open(OBJECTS_DIR / "split_manifest.json", encoding="utf-8") as f:
        groups = json.load(f)["groups"]
    out: dict[str, list[Path]] = defaultdict(list)
    for img in sorted((OBJECTS_DIR / "images").glob("*.jpg")):
        vid = normalize_video_id(img.stem.rsplit("_f", 1)[0])
        out[groups[vid]].append(img.resolve())
    return dict(out)


def make_folds(groups: dict[str, list[Path]], k: int) -> list[set[str]]:
    """Жадный GroupKFold: семейства по убыванию размера — в фолд с наименьшим числом кадров."""
    folds: list[set[str]] = [set() for _ in range(k)]
    sizes = [0] * k
    for g in sorted(groups, key=lambda g: (-len(groups[g]), g)):
        i = sizes.index(min(sizes))
        folds[i].add(g)
        sizes[i] += len(groups[g])
    return folds


def write_dataset(name: str, train: list[Path], val: list[Path]) -> Path:
    d = WORK_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.txt").write_text("\n".join(p.as_posix() for p in train) + "\n", encoding="utf-8")
    (d / "val.txt").write_text("\n".join(p.as_posix() for p in val) + "\n", encoding="utf-8")
    cfg = {"path": d.as_posix(), "train": "train.txt", "val": "val.txt",
           "names": dict(enumerate(class_names()))}
    with open(d / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return d / "dataset.yaml"


def train(dataset_yaml: Path, run_name: str, cfg: dict):
    from ultralytics import YOLO

    last = RUNS_DIR / run_name / "weights" / "last.pt"
    if last.is_file() and not (RUNS_DIR / run_name / "metrics.json").is_file():
        print(f"resuming {run_name} from {last}")
        return YOLO(str(last)).train(resume=True)
    model = YOLO(str(ROOT / cfg["base_model"]))
    return model.train(data=str(dataset_yaml), project=str(RUNS_DIR), name=run_name, exist_ok=True,
                       device=0, **cfg["train"])


def per_class_map50(results, names: list[str]) -> dict[str, float]:
    return {names[c]: round(float(results.box.ap50[i]), 4) for i, c in enumerate(results.box.ap_class_index)}


def run_fold(k: int, cfg: dict) -> dict:
    groups = frames_by_group()
    folds = make_folds(groups, cfg["folds"])
    val_groups = folds[k]
    train_imgs = [p for g, ps in groups.items() if g not in val_groups for p in ps]
    val_imgs = [p for g in sorted(val_groups) for p in groups[g]]
    ds = write_dataset(f"fold{k}", train_imgs, val_imgs)
    res = train(ds, f"fold{k}", cfg)
    out = {"fold": k, "val_groups": sorted(val_groups), "n_train": len(train_imgs), "n_val": len(val_imgs),
           "map50": round(float(res.box.map50), 4), "map50_95": round(float(res.box.map), 4),
           "per_class_map50": per_class_map50(res, class_names())}
    write_json(RUNS_DIR / f"fold{k}" / "metrics.json", out)
    return out


def summarize(results: list[dict]) -> dict:
    per_class: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for c, v in r["per_class_map50"].items():
            per_class[c].append(v)
    summary = {
        "folds": results,
        "mean_map50": round(sum(r["map50"] for r in results) / len(results), 4),
        "mean_map50_95": round(sum(r["map50_95"] for r in results) / len(results), 4),
        "per_class_mean_map50": {c: {"mean": round(sum(v) / len(v), 4), "folds_present": len(v)}
                                 for c, v in sorted(per_class.items())},
    }
    write_json(RUNS_DIR / "cv_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", help="номер фолда или all")
    g.add_argument("--final", action="store_true")
    args = ap.parse_args()
    cfg = load_config()

    if args.final:
        imgs = [p for ps in frames_by_group().values() for p in ps]
        # val = train: ранняя остановка выключена, метрика только для контроля сходимости
        ds = write_dataset("final", imgs, imgs)
        cfg["train"]["patience"] = 0
        train(ds, "final", cfg)
        best = RUNS_DIR / "final" / "weights" / "best.pt"
        shutil.copy2(best, ROOT / "weights" / "objects_best.pt")
        print(f"copied {best} → weights/objects_best.pt")
        return

    ks = range(cfg["folds"]) if args.fold == "all" else [int(args.fold)]
    results = []
    for k in ks:
        r = run_fold(k, cfg)
        results.append(r)
        print(f"fold {k}: mAP50={r['map50']} mAP50-95={r['map50_95']} val={r['val_groups']}")
    if args.fold == "all":
        s = summarize(results)
        print(f"CV mean mAP50={s['mean_map50']}")


if __name__ == "__main__":
    main()
