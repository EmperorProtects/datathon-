"""Файловое хранилище задач: storage/jobs/{job_id}/{input.mov,status.json,result.json}.

Видео удаляется сразу после обработки; каталог задачи — по истечении TTL (по умолчанию 1 ч).
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
from pathlib import Path

from app.config import get_settings

JOB_ID_RE = re.compile(r"^[0-9a-f]{16}$")
VIDEO_NAME = "input.mov"


def new_job_id() -> str:
    return secrets.token_hex(8)


def is_valid_job_id(job_id: str) -> bool:
    return bool(JOB_ID_RE.match(job_id))


def jobs_root() -> Path:
    root = get_settings().storage_dir / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_dir(job_id: str) -> Path:
    if not is_valid_job_id(job_id):
        raise ValueError(f"invalid job_id: {job_id!r}")
    return jobs_root() / job_id


def video_path(job_id: str) -> Path:
    return job_dir(job_id) / VIDEO_NAME


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def create_job(job_id: str) -> Path:
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=False)
    set_status(job_id, "processing")
    return d


def set_status(job_id: str, status: str, **extra) -> None:
    path = job_dir(job_id) / "status.json"
    current = _read_json(path) or {"created_at": time.time()}
    current.update({"status": status, "updated_at": time.time(), **extra})
    _write_json_atomic(path, current)


def get_status(job_id: str) -> dict | None:
    return _read_json(job_dir(job_id) / "status.json")


def write_result(job_id: str, result: dict) -> None:
    _write_json_atomic(job_dir(job_id) / "result.json", result)


def read_result(job_id: str) -> dict | None:
    return _read_json(job_dir(job_id) / "result.json")


def delete_video(job_id: str) -> None:
    video_path(job_id).unlink(missing_ok=True)


def delete_job(job_id: str) -> None:
    shutil.rmtree(job_dir(job_id), ignore_errors=True)


def cleanup_expired(ttl_sec: int | None = None, now: float | None = None) -> int:
    """Удаляет каталоги задач старше TTL. Возвращает число удалённых."""
    ttl = get_settings().job_ttl_sec if ttl_sec is None else ttl_sec
    now = time.time() if now is None else now
    removed = 0
    for d in jobs_root().iterdir():
        if not d.is_dir() or not is_valid_job_id(d.name):
            continue
        status = _read_json(d / "status.json") or {}
        created = status.get("created_at", d.stat().st_mtime)
        if now - created > ttl:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed
