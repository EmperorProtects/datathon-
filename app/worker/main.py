"""Точка входа воркера: прогрев моделей, затем RQ SimpleWorker (без fork — модели остаются в памяти/GPU).

Один процесс воркера = одна задача на GPU одновременно.
"""
from __future__ import annotations

import logging
import os
import time

from redis import Redis
from rq import SimpleWorker
from rq.timeouts import TimerDeathPenalty

from app.config import get_settings
from app.worker.readiness import READY_KEY


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    conn = Redis.from_url(settings.redis_url)

    from pipeline.run import warmup

    conn.delete(READY_KEY)
    warmup()
    conn.set(READY_KEY, str(time.time()))
    worker = SimpleWorker([settings.queue_name], connection=conn)
    if os.name == "nt":
        # На Windows нет SIGALRM — таймаут задачи через таймер (локальная разработка без Docker).
        worker.death_penalty_class = TimerDeathPenalty
    try:
        worker.work(with_scheduler=False)
    finally:
        conn.delete(READY_KEY)


if __name__ == "__main__":
    main()
