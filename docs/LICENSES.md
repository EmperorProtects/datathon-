# Лицензии

Обновляется на каждом этапе при добавлении модели или библиотеки.

## Модели

| Модель | Назначение | Лицензия | Этап |
|---|---|---|---|
| YOLO11m (`yolo11m.pt`) | детекция людей, трекинг | AGPL-3.0 | 3 |
| YOLO11m-pose (`yolo11m-pose.pt`) | позы (17 точек COCO) | AGPL-3.0 | 3 |
| YOLO-World v2 s (`yolov8s-worldv2.pt`) | пре-разметка train-кадров | AGPL-3.0 | 3 |
| YOLOE-11m-seg (`yoloe-11m-seg.pt`) + MobileCLIP (`mobileclip_blt.ts`) | пре-разметка train-кадров | AGPL-3.0 (MobileCLIP — Apple, см. репозиторий ml-mobileclip) | 3 |
| Qwen2.5-VL-3B-Instruct-AWQ | локальная VLM (D3, верификация) | **Qwen Research License** — только некоммерческое использование | 8 |
| SigLIP base (`google/siglip-base-patch16-224`) | признаки для D2 | Apache-2.0 | 9 |

**О Qwen2.5-VL-3B.** Веса открыты и запускаются локально, но лицензия не OSI-совместимая:
разрешено только некоммерческое использование (исследования и оценка). Проект некоммерческий (датасон),
поэтому модель выбрана осознанно: из моделей, которые влезают в 6 ГБ VRAM, у неё лучшее качество.
Замена с Apache-2.0 без изменения кода — Qwen2-VL-2B-Instruct (путь задаётся в `configs/models.yaml`).

## Библиотеки (этап 1)

| Библиотека | Лицензия |
|---|---|
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| python-multipart | Apache-2.0 |
| Pydantic | MIT |
| redis-py | MIT |
| RQ | BSD-2-Clause |
| PyYAML | MIT |
| Pillow | MIT-CMU (HPND) |
| Redis 7.2 (сервер, образ; 7.4+ уже не BSD) | BSD-3-Clause |
| nginx | BSD-2-Clause |
