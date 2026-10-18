"""Оркестратор пайплайна.

Этап 1: заглушка — через MOCK_DELAY_SEC секунд возвращает валидный mock-ответ.
"""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

from pipeline.errors import PipelineInputError


def warmup() -> None:
    """Загрузка и прогрев моделей (этап 3+)."""


def _mock_frame(text: str) -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 360), (60, 64, 72))
    draw = ImageDraw.Draw(img)
    draw.rectangle((180, 120, 460, 260), outline=(230, 80, 60), width=4)
    draw.text((20, 20), text, fill=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def analyze(video_path: str) -> dict:
    path = Path(video_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise PipelineInputError("video file is missing or empty")

    time.sleep(float(os.environ.get("MOCK_DELAY_SEC", "5")))
    t = 12.4
    return {
        "verdict": "violation",
        "violations": [
            {
                "type": "MOCK: тестовое нарушение",
                "timecode_sec": t,
                "description": (
                    "Заглушка пайплайна (этап 1). Реальный анализ не выполнялся. "
                    "Норматив: <будет взят из rules/registry.yaml>."
                ),
                "evidence_frames": [
                    {"timestamp_sec": t - 1.0, "image_base64": _mock_frame(f"MOCK t={t - 1.0:.1f}s")},
                    {"timestamp_sec": t, "image_base64": _mock_frame(f"MOCK t={t:.1f}s")},
                ],
                "rule": "MOCK_RULE",
                "confidence": 0.0,
            }
        ],
    }
