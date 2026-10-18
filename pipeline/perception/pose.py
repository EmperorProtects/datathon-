"""Люди и позы: Ultralytics YOLO11-pose + ByteTrack.

Pose-модель сама выдаёт рамки людей, поэтому отдельный детектор людей не нужен (экономит
проход по кадрам). Роли определяются позже только по геометрии и траекториям.
"""
from __future__ import annotations

import numpy as np

from pipeline.types import Person

# Индексы точек COCO-17
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP = 11, 12
HEAD_POINTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)

HAND_EXTRAPOLATION = 0.3  # кисть = запястье + 0.3 × (запястье − локоть)


def hand_point(kpts: np.ndarray, elbow: int, wrist: int, min_conf: float) -> tuple[float, float] | None:
    (ex, ey, ec), (wx, wy, wc) = kpts[elbow], kpts[wrist]
    if wc < min_conf:
        return None
    if ec < min_conf:
        return float(wx), float(wy)
    return float(wx + HAND_EXTRAPOLATION * (wx - ex)), float(wy + HAND_EXTRAPOLATION * (wy - ey))


def persons_from_result(result, kpt_conf: float) -> list[Person]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or result.keypoints is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    ids = boxes.id.int().cpu().numpy() if boxes.id is not None else [None] * len(xyxy)
    kpts = result.keypoints.data.cpu().numpy()  # (n, 17, 3)
    out = []
    for box, conf, tid, k in zip(xyxy, confs, ids, kpts):
        hands = (hand_point(k, L_ELBOW, L_WRIST, kpt_conf), hand_point(k, R_ELBOW, R_WRIST, kpt_conf))
        out.append(Person(conf=float(conf), box=tuple(map(float, box)), keypoints=k,
                          track_id=None if tid is None else int(tid), hands=hands))
    return out


class PoseTracker:
    def __init__(self, model_path: str, conf: float, kpt_conf: float, imgsz: int, tracker: str,
                 device: int | str = 0, half: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf, self.kpt_conf, self.imgsz, self.tracker = conf, kpt_conf, imgsz, tracker
        self.device, self.half = device, half

    def track(self, frames: list[np.ndarray]) -> list[list[Person]]:
        """Кадры (RGB) строго по порядку времени; трекер сохраняет состояние между вызовами."""
        bgr = [f[..., ::-1] for f in frames]
        results = self.model.track(bgr, persist=True, tracker=self.tracker, conf=self.conf, imgsz=self.imgsz,
                                   device=self.device, half=self.half, verbose=False, classes=[0])
        return [persons_from_result(r, self.kpt_conf) for r in results]

    def reset(self) -> None:
        for t in getattr(self.model.predictor, "trackers", None) or []:
            t.reset()
