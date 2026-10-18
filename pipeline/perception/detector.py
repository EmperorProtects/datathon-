"""Объекты кассы: дообученный YOLO11 (weights/objects_best.pt) + ByteTrack."""
from __future__ import annotations

import numpy as np

from pipeline.types import Detection


def detections_from_result(result) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.int().cpu().numpy()
    ids = boxes.id.int().cpu().numpy() if boxes.id is not None else [None] * len(xyxy)
    return [Detection(cls=names[int(c)], conf=float(p), box=tuple(map(float, b)),
                      track_id=None if t is None else int(t))
            for b, p, c, t in zip(xyxy, confs, clss, ids)]


class ObjectTracker:
    def __init__(self, model_path: str, conf: float, imgsz: int, tracker: str, device: int | str = 0,
                 half: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf, self.imgsz, self.tracker, self.device, self.half = conf, imgsz, tracker, device, half

    @property
    def class_names(self) -> list[str]:
        return list(self.model.names.values())

    def track(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        bgr = [f[..., ::-1] for f in frames]
        results = self.model.track(bgr, persist=True, tracker=self.tracker, conf=self.conf, imgsz=self.imgsz,
                                   device=self.device, half=self.half, verbose=False)
        return [detections_from_result(r) for r in results]

    def reset(self) -> None:
        for t in getattr(self.model.predictor, "trackers", None) or []:
            t.reset()
