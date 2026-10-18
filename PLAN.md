# PLAN.md — AI-сервис выявления нарушений кассовой дисциплины (Datathon 26, ForteBank)

> Документ для Claude Code. Реализуй проект **поэтапно**, строго в порядке этапов.
> После каждого этапа: запусти тесты, проверь критерии приёмки, сделай коммит `stage-N: ...`.
> Если чего-то не хватает (формат JSON организаторов, GPU, веса) — остановись и спроси, не выдумывай.

---

## 0. Контекст и жёсткие ограничения

**Задача.** Веб-сервис + API: пользователь загружает `.mov` (≈29 МБ, 10–145 с, ≥29 fps, без звука, камера сверху над банковской кассой). Сервис возвращает вердикт `violation | no_violation` и список нарушений: тип, таймкод (сек), описание, 1–3 кадра-доказательства (JPEG base64), правило и нормативный источник (акт РК / НБРК, пункт).

**Жёсткие требования ТЗ (нарушение = дисквалификация или потеря баллов):**
1. **Полная локальность инференса.** При обработке видео запрещены облачные AI API (LLM, VLM, распознавание). Все модели внутри контейнера. После загрузки весов — работа без интернета (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, у воркера нет сети).
2. **Запрещено распознавание и идентификация лиц.** Роли определяются только геометрией и траекторией. На кадрах-доказательствах лица размываются.
3. **Только визуальный анализ.** Звук не используется.
4. **Таймаут 5 минут** на видео (внутренний бюджет 270 с, далее graceful degradation).
5. **Пропуск нарушения недопустим, ложных срабатываний — минимум.** Вердикт оценивается по видео целиком.
6. **Не утверждать нарушение, если для вывода нужны невидимые данные.** Каждое нарушение подкреплено видимым кадром.
7. **Нормативные ссылки — только реальные**, из реестра `rules/registry.yaml`. Никаких сгенерированных пунктов.
8. **Код и модели — open source.** Стек: Ultralytics YOLO (AGPL-3.0 — учесть в README, раздел «Лицензии»), остальное — Apache/MIT/BSD. Лицензию каждой модели фиксировать в `docs/LICENSES.md`.
9. **Тестовые видео жюри не сохранять** после обработки дольше, чем нужно для ответа (TTL 1 час), не логировать кадры.

**Данные:** 42 train-видео, суммарно ≈23 мин, JSON организаторов: видео → время нарушения → акт и норматив. Типы нарушений включают **действия** (что-то сделано не так) и **бездействие** (обязательный шаг не выполнен).

**Железо (предположение, уточнить):** 1 GPU ≥24 ГБ VRAM (L4 / A10 / 4090), 32 ГБ RAM, Ubuntu, Docker + NVIDIA Container Toolkit.

---

## 1. Архитектура (кратко)

```
.mov ─► [0] Ingest ─► [1] Perception (YOLO) ─► [2] Scene & Roles ─► [3] Event Timeline
                                                                       │
     ┌──────────────────┬──────────────────────┬───────────────────────┼─────────────────┐
     ▼                  ▼                      ▼                       ▼                 │
 [D1] Rules       [D2] Chunk head        [D3] VLM scan          [D4] Procedure tracker   │
 (сигнатуры       (SigLIP + поза         (окна 5 с, шаг 3 с,     (эпизоды, обязательные   │
  действий)        → MLP + kNN)           локальная VLM)          шаги → пропуски)        │
     └──────────────────┴──────────┬───────────┴───────────────────────┘                 │
                                   ▼                                                     │
                  [4] Fusion: накопление уверенности + временной NMS                     │
                                   ▼                                                     │
                  [5] Evidence gate + VLM-верификация  ◄──── rules/registry.yaml ────────┘
                                   ▼
                  [6] Thresholds (LOVO) ─► [7] Evidence frames + описание + норматив ─► JSON
```

Принципы:
- **D1, D3, D4** — основные детекторы; **D2** — только усилитель скора (сам кандидата не поднимает).
- Признаки **относительные** (зоны по якорям, расстояния в долях роста/предплечья), чтобы переживать смену ракурса.
- Описание строится из **фактов таймлайна**; VLM только формулирует и не может противоречить фактам.

---

## 2. Структура репозитория

```
cashguard/
├── PLAN.md
├── README.md
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.worker
├── .env.example                 # API_KEY, пути, пороги
├── configs/
│   ├── pipeline.yaml            # fps, окна, пороги, веса фьюжена, бюджет времени
│   └── models.yaml              # пути к весам, устройства, precision
├── rules/
│   ├── registry.yaml            # типы нарушений ↔ сигнатуры ↔ нормативы
│   └── procedures.yaml          # эпизоды и обязательные шаги (для D4)
├── app/
│   ├── api/                     # FastAPI: роуты, схемы, auth
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── auth.py
│   │   └── jobs.py
│   ├── worker/
│   │   └── tasks.py             # RQ-задача analyze_video(job_id)
│   ├── web/                     # UI (статический, без сборки)
│   │   ├── index.html
│   │   ├── app.js
│   │   └── style.css
│   └── storage.py               # файлы задач, TTL-очистка
├── pipeline/
│   ├── types.py                 # dataclasses: Frame, Detection, Track, Event, Candidate, Violation
│   ├── ingest.py
│   ├── perception/
│   │   ├── detector.py          # Ultralytics YOLO (люди + объекты) + tracking
│   │   ├── pose.py              # Ultralytics YOLO-pose
│   │   └── privacy.py           # размытие голов по ключевым точкам позы
│   ├── scene/
│   │   ├── anchors.py           # стойка, ящик, сейф, дверь, лоток
│   │   ├── zones.py
│   │   └── roles.py
│   ├── timeline/
│   │   ├── events.py            # атомарные события
│   │   └── features.py          # относительные признаки
│   ├── detectors/
│   │   ├── d1_rules.py
│   │   ├── d2_chunk_head.py
│   │   ├── d3_vlm_scan.py
│   │   └── d4_procedure.py
│   ├── vlm/
│   │   ├── client.py            # локальный VLM (vLLM или transformers)
│   │   └── prompts.py
│   ├── fusion.py
│   ├── gate.py
│   ├── evidence.py              # выбор кадров, боксы, blur, JPEG→base64
│   ├── describe.py              # описания из фактов
│   └── run.py                   # оркестратор + бюджет времени
├── training/
│   ├── data/
│   │   ├── adapter_organizers.py   # JSON организаторов → unified.json
│   │   ├── probe.py                # ffprobe, статистика
│   │   ├── extract_frames.py
│   │   ├── prelabel_yoloworld.py   # авто-пре-разметка объектов
│   │   └── make_chunks.py          # чанки + KCAS + GroupKFold
│   ├── train_detector.py
│   ├── train_chunk_head.py
│   └── build_prototypes.py
├── eval/
│   ├── lovo.py                  # leave-one-video-out прогон всего пайплайна
│   ├── pseudo_views.py          # синтетическая смена ракурса
│   ├── metrics.py
│   └── report.py                # HTML-отчёт с кадрами
├── data/                        # не в git
│   ├── raw/                     # видео + JSON организаторов
│   ├── annotations/             # unified.json, боксы YOLO, зоны
│   └── frames/
├── weights/                     # не в git, монтируются в образ
├── docs/
│   ├── ARCHITECTURE.md
│   ├── NORMATIVE.md
│   ├── VIOLATIONS.md
│   ├── LIMITATIONS.md
│   └── LICENSES.md
└── tests/
```

---

## 3. Форматы данных

### 3.1 Unified-разметка (`data/annotations/unified.json`)
Единый внутренний формат. Адаптер из JSON организаторов пишется на этапе 2 **после осмотра реального файла**.

```json
{
  "videos": [{
    "video_id": "video_017",
    "path": "data/raw/video_017.mov",
    "fps": 29.97, "duration_sec": 44.9, "width": 1920, "height": 1080,
    "has_violation": true,
    "events": [{
      "event_id": "video_017_e1",
      "kind": "action",                    // action | omission
      "type_id": "CASH_TO_PERSONAL",       // из rules/registry.yaml
      "organizer_time_sec": 44.0,          // как у организаторов
      "organizer_act": "...", "organizer_clause": "...",
      "start_sec": 42.1, "peak_sec": 44.2, "end_sec": 45.0,
      "peak_frame": 1325,
      "actor_role": "cashier",
      "visibility": "full",                // full | partial | occluded
      // только для omission:
      "episode": {"type": "cash_withdrawal", "start_sec": 12.4, "end_sec": 31.0},
      "expected_step": "recount_visible",
      "checkpoint_sec": 29.8,
      "note": ""
    }],
    "lookalikes": [{"start_sec": 10.0, "end_sec": 12.5, "what": "кладёт купюры в ящик"}],
    "zones_frame": 0,
    "split_group": "video_017"
  }]
}
```

### 3.2 Реестр правил (`rules/registry.yaml`)
```yaml
- type_id: CASH_TO_PERSONAL
  name_ru: "Перемещение наличных в личные вещи/одежду"
  kind: action
  class: executional
  signature:                     # для D1
    sequence: [hold_cash, cash_near_body_or_bag, cash_disappeared]
    forbid_between: [cash_in_drawer]
    max_span_sec: 6
    actor_role: cashier
  observable_if: {key_object_visible_ratio: 0.5}
  source:
    act: "<точное название акта из JSON организаторов>"
    clause: "<пункт>"
    url: "<adilet.zan.kz/...>"
  description_template: "В {t:.1f} с кассир удерживает купюры, затем они перемещаются к {zone} и исчезают из вида, не попав в денежный ящик."
  vlm_question: "Перемещает ли кассир наличные в карман, одежду или личную сумку?"
```
Нормативы берутся **только** из JSON организаторов и проверенных актов. Поле `source` обязательно; правило без него не загружается (валидатор).

### 3.3 Процедуры (`rules/procedures.yaml`) — для D4 (бездействие)
```yaml
- episode: cash_withdrawal
  start_trigger: client_at_window
  end_trigger: client_left_window
  steps:
    - id: recount_visible
      must_occur_before: cash_handover_to_client
      omission_type_id: NO_RECOUNT_BEFORE_HANDOVER
      observable_if: {cashier_hands_visible_ratio: 0.7}
```
Конкретные эпизоды и шаги заполняются после анализа датасета (этап 2).

### 3.4 Ответ API (строго по ТЗ, раздел 10)
```json
{
  "job_id": "3f2a9c1e",
  "status": "done",
  "verdict": "violation",
  "violations": [{
    "type": "Название типа нарушения",
    "timecode_sec": 44.2,
    "description": "Что видно на видео и почему это может быть нарушением. Норматив: <акт>, п. <пункт>.",
    "evidence_frames": [{"timestamp_sec": 44.2, "image_base64": "/9j/4AAQ..."}]
  }]
}
```
- Обязательные поля — только эти. Дополнительные поля (`rule`, `confidence`, `source`) допускаются **внутри** элемента нарушения как расширение, но не ломают схему; сделать флаг `STRICT_SCHEMA=true`, который их отключает.
- `verdict = "no_violation"` ⇔ `violations = []`.
- `evidence_frames`: 1–3 элемента, JPEG, base64 **без** префикса `data:`.
- Статусы: `processing | done | failed`. При `failed` — поле `error`.

---

## 4. Этапы разработки

### Этап 1. Каркас сервиса с заглушкой пайплайна
**Задачи:**
- FastAPI: `POST /api/v1/analyze` (multipart, поле `file`, только `.mov`/`video/quicktime`, лимит 200 МБ) → `{"job_id": ...}`; `GET /api/v1/jobs/{job_id}`.
- Auth: заголовок `X-API-Key`, сравнение через `hmac.compare_digest`, иначе 401. Ключ из `.env`.
- Очередь: Redis + RQ, отдельный воркер с GPU. Одна задача на GPU одновременно.
- Хранилище задач: `storage/jobs/{job_id}/` (видео, `result.json`, `status.json`), TTL-очистка 1 ч.
- Заглушка `pipeline.run.analyze(path) -> dict` возвращает валидный mock-ответ через 5 с.
- UI (`/`): загрузка файла, прогресс-статус (polling), вердикт, карточки нарушений (тип, таймкод, описание, норматив, кадры), таймлайн с маркерами. Ключ API в UI — через отдельный серверный роут без ключа для пользователей UI (UI ходит на `/ui/analyze`, который проксирует внутрь), чтобы ключ не светился в браузере.
- `docker-compose.yml`: `api`, `worker`, `redis`, `nginx` (TLS, `client_max_body_size 200m`, таймауты).
- `/healthz`, `/readyz` (модели загружены).

**Приёмка:** `curl` из ТЗ работает; без ключа — 401; UI показывает mock-результат; pydantic-схема ответа покрыта тестом.

### Этап 2. Данные: адаптер, статистика, кадры
**Задачи:**
- `probe.py`: ffprobe всех видео → таблица (fps, VFR?, длительность, разрешение, кодек).
- Осмотреть JSON организаторов и написать `adapter_organizers.py` → `unified.json` (заполнить `organizer_*`, `has_violation`, черновые `peak_sec = organizer_time_sec`).
- Отчёт по датасету: число видео с/без нарушений, нарушений на видео, распределение по актам/пунктам, длительности.
- Сгруппировать нарушения по (акт, пункт) → черновой `registry.yaml` с `type_id` (названия типов — заполняет человек).
- `extract_frames.py`: кадры по PTS, 10 fps (+ карта кадр→секунда), полный fps по запросу.
- Таймкоды везде в секундах по PTS; утилита `sec_to_frame / frame_to_sec`.

**Приёмка:** `unified.json` проходит валидатор (схема + `start ≤ peak ≤ end` + `type_id` есть в реестре); отчёт по датасету в `eval/reports/dataset.html`.

### Этап 3. Perception (Ultralytics YOLO)
**Задачи:**
- `detector.py`: YOLO11 (`yolo11m.pt` или `l`) для `person` + кастомная модель объектов (классы: `cash`, `cash_bundle`, `cash_drawer`, `safe`, `bag`, `collector_bag`, `cassette`, `phone`, `counter`, `tray`, `door`). Трекинг — встроенный `model.track(..., tracker="bytetrack.yaml", persist=True)`.
- `prelabel_yoloworld.py`: пре-разметка объектов через YOLO-World / YOLOE (open-vocabulary, Ultralytics) на кадрах train → экспорт в формате YOLO для проверки человеком (CVAT/Label Studio).
- `train_detector.py`: дообучение YOLO11 на проверенных кадрах. Аугментации обязательно: `degrees=15, perspective=0.0008, scale=0.5, translate=0.2, fliplr=0.5, mosaic=1.0, hsv_*`. Сплит по видео.
- `pose.py`: YOLO11-pose (17 COCO-точек: запястья, локти, плечи, голова). Кисть = экстраполяция локоть→запястье на 0.3 длины предплечья.
- `privacy.py`: блюр головы по точкам nose/eyes/ears (bbox головы с запасом), без face-моделей и без идентификации.
- Анализ на `analysis_fps` (по умолчанию 10, в конфиге), батчинг, FP16.

**Приёмка:** mAP50 по объектам на LOVO-фолдах в отчёте; визуализация треков на 3 видео; скорость ≥ 60 кадров/с на целевом GPU (иначе снизить fps/размер).

### Этап 4. Scene & Roles
**Задачи:**
- `anchors.py`: устойчивые якоря по медиане детекций за ролик (стойка, ящик, сейф, дверь, лоток). Если якорь не найден — флаг `scene_confidence`.
- `zones.py`: зоны **от якорей**: `drawer_zone`, `counter_zone`, `staff_side`, `client_side`, `door_zone`, `body_zone(track)` (торс/бёдра по позе), `bag_zone(bag)`.
- `roles.py`: `cashier` (трек, >60% времени на `staff_side`, длительное присутствие), `client` (на `client_side`), `collector` (вход через дверь в `staff_side` + `collector_bag`/`cassette`), `other`.
- Ручные полигоны зон (из разметки) — только для проверки автоопределения.

**Приёмка:** доля верно определённых ролей на train ≥ 90% (по ручной проверке/разметке); при отсутствии якорей сцена помечается и пайплайн не падает.

### Этап 5. Event Timeline
**Задачи:** из детекций/треков/поз — атомарные события с интервалами и уверенностью:
`hold_cash(actor)`, `cash_in_drawer`, `cash_near_body`, `cash_near_bag`, `cash_disappeared(last_zone)`, `cash_handover(from,to)`, `drawer_open`, `drawer_open_unattended`, `cashier_left_zone`, `cashier_returned`, `client_at_window`, `client_left_window`, `collector_entered/left`, `outsider_in_staff_zone`, `recount_visible` (повторяющееся движение кистей над купюрами), `cashier_hands_visible`.
- Признаки — относительные: расстояния в долях длины предплечья, нормированные на размер человека.
- Сглаживание: события короче `min_event_sec` отбрасываются, разрывы короче `gap_sec` склеиваются.
- Отладка: `timeline.json` + PNG-диаграмма таймлайна.

**Приёмка:** на роликах с нарушениями-действиями ключевые события сигнатуры присутствуют в окне ±2 с от `peak_sec` в ≥ 80% случаев.

### Этап 6. D1 — Rule engine (действия)
**Задачи:**
- Загрузка и валидация `registry.yaml`.
- Матчинг сигнатур как последовательностей событий с ограничениями (`sequence`, `forbid_between`, `max_span_sec`, `actor_role`).
- Проверка `observable_if`: если ключевой объект/руки не видны достаточно — кандидат не создаётся.
- Выход: `Candidate(type_id, t_start, t_peak, t_end, score, source="D1", facts=[...])`.

**Приёмка:** юнит-тесты на синтетических таймлайнах; recall по действиям на train ≥ 0.8 при precision ≥ 0.6 (до фьюжена).

### Этап 7. D4 — Procedure tracker (бездействие)
**Задачи:**
- Нарезка на эпизоды по `procedures.yaml` (`start_trigger`/`end_trigger`).
- Для каждого обязательного шага: если наступил `must_occur_before` (контрольная точка) или конец эпизода, а шага не было → кандидат пропуска.
- Проверка наблюдаемости окна ожидания (`observable_if`); не наблюдаемо → не утверждать.
- Таймкод = контрольная точка (конвенцию сверить с таймкодами организаторов на этапе 2, сделать настраиваемой: `omission_timecode: checkpoint | window_start | window_mid`).
- Факты для описания: границы окна, доля видимости рук, что именно не наблюдалось.

**Приёмка:** на всех train-нарушениях типа «бездействие» кандидат есть, таймкод в пределах допуска от организаторского.

### Этап 8. D3 — локальная VLM: скан и верификация
**Модель:** Qwen2.5-VL-7B-Instruct (проверить лицензию, зафиксировать в LICENSES.md), AWQ/INT4 или FP16 в зависимости от VRAM. Сервинг — vLLM в контейнере воркера (или transformers, если vLLM не поднимется). Строго офлайн.
**Задачи:**
- **Скан:** окна 5 с, шаг 3 с, 6 кадров на окно (с боксами людей/объектов и подписанными ролями). Промпт: каталог типов из реестра (`name_ru` + `vlm_question`) + «отвечай только по видимому». Ответ — JSON по схеме (guided decoding / JSON-schema в vLLM): `{"violation_observed": bool, "type_id": str|null, "confidence": 0-1, "what_is_visible": str}`. `type_id` ∉ реестра → `"UNKNOWN_DEVIATION"`.
- **Эпизодная проверка (для D4):** 12–16 кадров по эпизоду, вопрос «выполнен ли шаг X между t1 и t2? да / нет / не видно».
- **Верификация кандидатов:** кадры полного fps вокруг пика, вопрос конкретного правила.
- **Формулировка описания:** на вход — факты таймлайна, на выход — 1–2 предложения; если формулировка противоречит фактам — используется шаблон.
- Кэш по (video_hash, window) на время задачи.

**Приёмка:** ответы всегда валидный JSON; время скана ролика 145 с ≤ 70 с; на train VLM-скан находит ≥ 85% нарушений (до фьюжена).

### Этап 9. D2 — голова на чанках (усилитель)
**Задачи:**
- `make_chunks.py`: окна 3 с, шаг 0.5 с. Метки: позитив — содержит `peak_frame` или покрывает ≥50% интервала события; частичное перекрытие 1–50% — игнор в loss; hard negatives — окна до события (KCAS: бимодальное семплирование вокруг события, σ≈1.5 с, δ≈2 с) + `lookalikes`. Сплит `GroupKFold` по видео.
- Признаки: SigLIP (`google/siglip-base-patch16-224`, Apache-2.0) по 6 кадрам окна → mean+max pooling, + относительные позовые/событийные признаки окна.
- Аугментации: одна случайная гомография/кроп/масштаб на всё окно (псевдо-ракурс).
- Модель: логистическая регрессия или MLP (1 скрытый слой) с весами классов. Тип — kNN к прототипам эпизодов из train.
- В фьюжене D2 только повышает скор существующих кандидатов.

**Приёмка:** PR-AUC на LOVO в отчёте; влияние D2 на итоговые метрики измерено (ablation on/off).

### Этап 10. Fusion, gate, thresholds, evidence
**Задачи:**
- `fusion.py`: объединение кандидатов D1/D3/D4 (+буст D2), накопление уверенности по соседним окнам с затуханием, временной NMS по типу (`nms_window_sec`).
- `gate.py`: кандидат проходит, если
  (a) согласны ≥2 детектора из {D1, D3, D4}, **или**
  (b) один детектор со `score ≥ τ_high` и ключевой объект виден, **или**
  (c) `τ_low ≤ score < τ_high` и VLM-верификация = да.
  Для бездействия дополнительно: окно наблюдаемо.
- Пороги подбираются `eval/lovo.py` с целевой функцией: **0 пропусков на уровне видео**, затем минимум ложных срабатываний.
- `scene_confidence` низкий → правила, зависящие от зон, отключаются, решение за D3 по всему ролику.
- `evidence.py`: пиковый кадр (максимум скора внутри интервала, полный fps) + до/после; боксы участников с ролями; блюр голов; JPEG q=85, длинная сторона ≤1280; base64 без префикса.
- `describe.py`: «что видно» + «почему это может указывать на нарушение» + «Норматив: <акт>, п. <пункт>». Без юридических выводов, формулировка «признаки возможного нарушения».

**Приёмка:** JSON проходит схему ТЗ; ни одного нарушения без кадра и без `source`.

### Этап 11. Оценка
**Задачи:**
- `lovo.py`: полный пайплайн leave-one-video-out (модели с обучением — переобучаются без тестового видео).
- `metrics.py` (зеркало критериев жюри): accuracy вердикта, **пропуски на уровне видео**, ложные срабатывания, ошибка таймкода |t_pred − t_org| (медиана, доля ≤1 с), точность типа, доля ответов с корректным нормативом.
- `pseudo_views.py`: для каждого тестового видео 4 варианта — перспектива ±25°, кроп 70%, поворот 12°, сдвиг центра — и те же метрики отдельно.
- `report.py`: HTML-отчёт с кадрами по каждой ошибке.

**Приёмка:** отчёт строится одной командой `make eval`.

### Этап 12. Производительность, надёжность, деплой
**Задачи:**
- Оркестратор с бюджетом: ingest+perception ≤ 60 с, D1/D2/D4 ≤ 10 с, D3 ≤ 90 с, верификация ≤ 40 с. При превышении: снизить fps анализа → сократить D3 до окон вокруг кандидатов → вернуть результат по D1/D4 (с пометкой в логах). Жёсткий стоп 270 с.
- Прогрев моделей при старте воркера; `/readyz` = true только после прогрева.
- Обработка ошибок: битый файл, не `.mov`, нулевая длительность → `failed` с понятным `error`.
- Веса копируются в образ на этапе сборки; воркер без сети (`network_mode` / отдельная internal-сеть с Redis).
- Публичный URL: nginx + TLS (Let's Encrypt), healthcheck, автоперезапуск.
- Нагрузочный тест: 5 видео подряд и 3 параллельно — все ≤ 5 мин (очередь учитывается).

**Приёмка:** `scripts/smoke.sh` прогоняет 3 видео через публичный URL с ключом и валидирует ответы.

### Этап 13. Документация для сдачи
- `README.md`: запуск, использование, API-ключ, примеры `curl`.
- `docs/ARCHITECTURE.md`: схема, модули, модели, локальность.
- `docs/NORMATIVE.md`: найденные акты, пункты, связь «источник → нарушение → что видно на видео».
- `docs/VIOLATIONS.md`: перечень распознаваемых типов с описанием и примерами кадров.
- `docs/LIMITATIONS.md`: ограничения (один ракурс в train, малый датасет, окклюзии, когда сервис отвечает «не утверждаем») и известные ошибки из отчёта LOVO.
- `docs/LICENSES.md`: лицензии всех моделей и библиотек (Ultralytics — AGPL-3.0).
- Указать: облачные сервисы использовались только для пре-разметки train-данных; инференс полностью локальный.

---

## 5. Конфиг по умолчанию (`configs/pipeline.yaml`)
```yaml
analysis_fps: 10
evidence_full_fps: true
detector: {person_model: yolo11m.pt, objects_model: weights/objects_best.pt, conf: 0.25, imgsz: 960}
pose: {model: yolo11m-pose.pt, conf: 0.3}
tracker: bytetrack.yaml
timeline: {min_event_sec: 0.3, gap_sec: 0.5}
d2: {window_sec: 3.0, stride_sec: 0.5, frames_per_window: 6}
d3: {window_sec: 5.0, stride_sec: 3.0, frames_per_window: 6, max_windows: 50}
fusion: {decay: 0.25, nms_window_sec: 3.0, w: {d1: 1.0, d3: 1.0, d4: 1.0, d2_boost: 0.2}}
gate: {tau_low: 0.35, tau_high: 0.75, min_agree: 2}
omission_timecode: checkpoint
budget_sec: {total: 270, perception: 60, d3: 90, verify: 40}
strict_schema: false
```

## 6. Правила работы для Claude Code
- Один этап — одна ветка/серия коммитов; не начинать следующий этап, пока не пройдена приёмка текущего.
- Все пороги и параметры — в конфиге, не в коде.
- Типы между модулями — только через `pipeline/types.py`.
- Никаких обращений в сеть из `pipeline/` и `app/worker/` (тест-линтер: grep на `requests`, `httpx`, `openai`, `google.generativeai` в этих каталогах).
- Нормативы не генерировать; при отсутствии `source` — ошибка валидации.
- Любая неоднозначность в данных организаторов — вопрос пользователю, а не догадка.
