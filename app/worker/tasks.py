"""RQ-задача анализа видео. Выполняется в воркере без доступа в сеть."""
from __future__ import annotations

import logging
import time

from app import storage
from app.api.schemas import AnalysisResult

log = logging.getLogger(__name__)


def analyze_video(job_id: str) -> None:
    from pipeline.run import analyze

    started = time.monotonic()
    storage.set_status(job_id, "processing", started_at=time.time())
    try:
        raw = analyze(str(storage.video_path(job_id)))
        result = AnalysisResult.model_validate(raw)
        storage.write_result(job_id, result.model_dump(exclude_none=True))
        storage.set_status(job_id, "done", elapsed_sec=round(time.monotonic() - started, 2))
        log.info("job %s done in %.1fs, verdict=%s", job_id, time.monotonic() - started, result.verdict)
    except Exception as e:
        log.exception("job %s failed", job_id)
        storage.set_status(job_id, "failed", error=_public_error(e))
        raise
    finally:
        # Видео жюри не храним дольше, чем нужно для ответа.
        storage.delete_video(job_id)


def on_failure(job, connection, type, value, traceback) -> None:  # noqa: A002 (сигнатура RQ)
    """Срабатывает и при таймауте RQ, когда analyze_video не успела записать статус."""
    job_id = job.id
    try:
        st = storage.get_status(job_id) or {}
        if st.get("status") != "failed":
            storage.set_status(job_id, "failed", error=_public_error(value))
        storage.delete_video(job_id)
    except Exception:
        log.exception("on_failure handler for %s failed", job_id)


def _public_error(e: BaseException | None) -> str:
    from rq.timeouts import JobTimeoutException

    from pipeline.errors import PipelineInputError

    if isinstance(e, JobTimeoutException):
        return "processing timeout exceeded"
    if isinstance(e, PipelineInputError):
        return str(e)
    return "internal processing error"
