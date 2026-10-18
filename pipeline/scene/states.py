"""Визуальные состояния статичных объектов без нейросетей: экран ноутбука светится,
вид сейфа/двери изменился относительно эталона (кадр, когда кассир на месте).

Сравнение — в рамках якорей, найденных детектором, поэтому не зависит от ракурса.
"""
from __future__ import annotations

import numpy as np

from pipeline.ingest import iter_frames
from pipeline.scene.zones import expand
from pipeline.types import Box, Scene, VideoTimebase


def _crop(rgb: np.ndarray, box: Box) -> np.ndarray:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
    return rgb[y1:y2, x1:x2].astype(np.float32)


def _screen_brightness(crop: np.ndarray) -> float:
    """Доля ярких пикселей в рамке ноутбука (светящийся экран)."""
    if crop.size == 0:
        return 0.0
    gray = crop.mean(axis=2)
    return float((gray > 170).mean())


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or a.shape != b.shape:
        return 0.0
    return float(np.abs(a - b).mean() / 255.0)


def measure_states(path: str, timebase: VideoTimebase, scene: Scene, ref_t: float,
                   times: list[float]) -> dict[float, dict[str, float]]:
    """Для каждого момента из times: яркость экрана ноутбука и изменение сейфа/двери/стола от эталона."""
    boxes = {c: expand(scene.anchors[c].box, 0.05) for c in ("laptop", "safe", "door", "counter")
             if c in scene.anchors}
    if not boxes:
        return {}
    want = {timebase.sec_to_frame(ref_t): "ref"}
    for t in times:
        want.setdefault(timebase.sec_to_frame(t), t)
    crops: dict[object, dict[str, np.ndarray]] = {}
    for i, rgb in iter_frames(path, want):
        crops[want[i]] = {c: _crop(rgb, b) for c, b in boxes.items()}
    ref = crops.get("ref", {})
    out = {}
    for t in times:
        key = want.get(timebase.sec_to_frame(t))
        c = crops.get(key)
        if c is None:
            continue
        out[t] = {
            "laptop_bright": _screen_brightness(c["laptop"]) if "laptop" in c else 0.0,
            **{f"{k}_change": _diff(c[k], ref[k]) for k in ("safe", "door", "counter") if k in c and k in ref},
        }
    return out
