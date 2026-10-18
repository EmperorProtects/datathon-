"""Приём видео, постановка в очередь RQ, чтение статуса задачи."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile, status
from redis import Redis
from rq import Callback, Queue

from app import storage
from app.api.schemas import JobResponse
from app.config import get_settings

log = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"video/quicktime", "application/octet-stream", ""}
# Первые атомы QuickTime/ISO-BMFF контейнера (байты 4..8 файла).
MOV_ATOMS = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}
CHUNK = 1024 * 1024


@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(get_settings().redis_url)


def get_queue() -> Queue:
    return Queue(get_settings().queue_name, connection=get_redis())


def check_content_length(request: Request) -> None:
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > get_settings().max_upload_bytes + CHUNK:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file too large")


def _validate_upload_meta(file: UploadFile) -> None:
    name = (file.filename or "").lower()
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    if not name.endswith(".mov") or ctype not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "only .mov (video/quicktime) files are accepted")


async def _save_upload(file: UploadFile, dest: Path) -> None:
    limit = get_settings().max_upload_bytes
    size = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(CHUNK):
            if size == 0 and (len(chunk) < 8 or chunk[4:8] not in MOV_ATOMS):
                raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "file is not a QuickTime .mov container")
            size += len(chunk)
            if size > limit:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file too large")
            out.write(chunk)
    if size == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")


async def submit(file: UploadFile, queue: Queue) -> str:
    _validate_upload_meta(file)
    job_id = storage.new_job_id()
    storage.create_job(job_id)
    try:
        await _save_upload(file, storage.video_path(job_id))
    except BaseException:
        storage.delete_job(job_id)
        raise
    settings = get_settings()
    queue.enqueue(
        "app.worker.tasks.analyze_video",
        job_id,
        job_id=job_id,
        job_timeout=settings.job_timeout_sec,
        result_ttl=settings.job_ttl_sec,
        failure_ttl=settings.job_ttl_sec,
        on_failure=Callback("app.worker.tasks.on_failure"),
    )
    log.info("job %s enqueued", job_id)
    return job_id


def load(job_id: str) -> JobResponse:
    if not storage.is_valid_job_id(job_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    st = storage.get_status(job_id)
    if st is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    if st["status"] == "done":
        result = storage.read_result(job_id)
        if result is None:
            return JobResponse(job_id=job_id, status="processing")
        return JobResponse(job_id=job_id, status="done", verdict=result["verdict"], violations=result["violations"])
    if st["status"] == "failed":
        return JobResponse(job_id=job_id, status="failed", error=st.get("error") or "processing failed")
    return JobResponse(job_id=job_id, status="processing")
