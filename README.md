# CashGuard — выявление нарушений кассовой дисциплины по видео

Веб-сервис и API (Datathon 26, ForteBank). Принимает `.mov`-видео с камеры над кассой и возвращает
вердикт `violation | no_violation` со списком нарушений: тип, таймкод, описание, кадры-доказательства, норматив.

> Статус: **этап 1** — каркас сервиса, пайплайн-заглушка возвращает mock-ответ. План — `PLAN.md`.

## Принципы

- Инференс полностью локальный: облачные AI API не используются, у воркера нет выхода в сеть.
- Лица не распознаются и не идентифицируются; на кадрах-доказательствах головы размываются.
- Видео удаляется сразу после обработки, результат — через 1 час.

## Запуск

```bash
cp .env.example .env          # задать API_KEY и DOMAIN
# TLS-сертификаты: ./certs/fullchain.pem, ./certs/privkey.pem (Let's Encrypt)
docker compose up -d --build
```

Нужны Docker и NVIDIA Container Toolkit (для воркера).

Получение сертификата Let's Encrypt (nginx уже запущен и отвечает на 80-м порту):

```bash
docker run --rm -v datathon_certbot-www:/var/www/certbot -v "$PWD/certs-le:/etc/letsencrypt" \
  certbot/certbot certonly --webroot -w /var/www/certbot -d "$DOMAIN" --agree-tos -m you@example.com -n
cp certs-le/live/$DOMAIN/{fullchain,privkey}.pem certs/ && docker compose restart nginx
```

## API

Все запросы к `/api/v1/*` требуют заголовок `X-API-Key`.

```bash
curl -X POST https://$DOMAIN/api/v1/analyze \
  -H "X-API-Key: $API_KEY" \
  -F "file=@video.mov"
# {"job_id": "3f2a9c1e5b7d4a60"}

curl https://$DOMAIN/api/v1/jobs/3f2a9c1e5b7d4a60 -H "X-API-Key: $API_KEY"
```

Ответ:

```json
{
  "job_id": "3f2a9c1e5b7d4a60",
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

- `status`: `processing | done | failed`; при `failed` — поле `error`.
- `verdict = "no_violation"` ⇔ `violations = []`.
- `evidence_frames`: 1–3 JPEG в base64 без префикса `data:`.
- Дополнительные поля нарушения (`rule`, `confidence`, `source`) отключаются через `STRICT_SCHEMA=true`.
- Ограничения загрузки: только `.mov`, до 200 МБ.

Служебные эндпоинты: `GET /healthz` (процесс жив), `GET /readyz` (Redis, воркер и модели готовы).

Веб-интерфейс — `https://$DOMAIN/`. Он ходит на `/ui/*` без ключа, поэтому ключ не попадает в браузер.

## Разработка

```bash
pip install -r requirements-dev.txt
make test
```

## Лицензии

Ultralytics YOLO распространяется под **AGPL-3.0**; сервис, использующий его по сети, должен
публиковать исходный код под совместимой лицензией. Полный перечень — `docs/LICENSES.md`.
