import fakeredis
import pytest
from fastapi.testclient import TestClient
from rq import Queue

API_KEY = "test-key-123"
# Минимальный заголовок QuickTime: атом ftyp.
MOV_BYTES = b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  " + b"\x00" * 1024


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MOCK_DELAY_SEC", "0")
    monkeypatch.delenv("STRICT_SCHEMA", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def redis_conn():
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def client(redis_conn):
    from app.api import jobs
    from app.api.main import app

    app.dependency_overrides[jobs.get_queue] = lambda: Queue("cashguard", connection=redis_conn, is_async=False)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def auth():
    return {"X-API-Key": API_KEY}
