"""Настройки сервиса из переменных окружения (.env) с дефолтами из configs/pipeline.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CONFIG = ROOT / "configs" / "pipeline.yaml"


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache
def pipeline_config() -> dict:
    with open(PIPELINE_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class Settings:
    api_key: str
    redis_url: str
    queue_name: str
    storage_dir: Path
    job_ttl_sec: int
    max_upload_mb: int
    job_timeout_sec: int
    strict_schema: bool
    mock_delay_sec: float

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    env = os.environ
    strict = env.get("STRICT_SCHEMA")
    return Settings(
        api_key=env.get("API_KEY", ""),
        redis_url=env.get("REDIS_URL", "redis://localhost:6379/0"),
        queue_name=env.get("QUEUE_NAME", "cashguard"),
        storage_dir=Path(env.get("STORAGE_DIR", str(ROOT / "storage"))),
        job_ttl_sec=int(env.get("JOB_TTL_SEC", "3600")),
        max_upload_mb=int(env.get("MAX_UPLOAD_MB", "200")),
        job_timeout_sec=int(env.get("JOB_TIMEOUT_SEC", "300")),
        strict_schema=_bool(strict) if strict else bool(pipeline_config().get("strict_schema", False)),
        mock_delay_sec=float(env.get("MOCK_DELAY_SEC", "5")),
    )
