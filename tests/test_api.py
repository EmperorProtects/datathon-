import os
import time

from app import storage
from app.api.schemas import JobResponse
from tests.conftest import MOV_BYTES


def upload(client, path="/api/v1/analyze", headers=None, name="clip.mov", data=MOV_BYTES, ctype="video/quicktime"):
    return client.post(path, headers=headers or {}, files={"file": (name, data, ctype)})


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_not_ready_without_worker(client):
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_analyze_requires_key(client):
    assert upload(client).status_code == 401
    assert upload(client, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/jobs/" + "a" * 16).status_code == 401


def test_rejects_non_mov(client, auth):
    assert upload(client, headers=auth, name="clip.mp4").status_code == 415
    assert upload(client, headers=auth, ctype="image/png").status_code == 415
    assert upload(client, headers=auth, data=b"not a video at all").status_code == 415
    assert upload(client, headers=auth, data=b"").status_code in (400, 415)
    assert list(storage.jobs_root().iterdir()) == []


def test_curl_default_content_type_accepted(client, auth):
    assert upload(client, headers=auth, ctype="application/octet-stream").status_code == 200


def test_full_flow(client, auth):
    r = upload(client, headers=auth)
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert storage.is_valid_job_id(job_id)

    r = client.get(f"/api/v1/jobs/{job_id}", headers=auth)
    assert r.status_code == 200
    body = r.json()
    JobResponse.model_validate(body)
    assert body["status"] == "done"
    assert body["verdict"] == "violation"
    assert 1 <= len(body["violations"][0]["evidence_frames"]) <= 3
    # видео удалено после обработки
    assert not storage.video_path(job_id).exists()


def test_strict_schema_env(client, auth, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("STRICT_SCHEMA", "true")
    get_settings.cache_clear()
    job_id = upload(client, headers=auth).json()["job_id"]
    viol = client.get(f"/api/v1/jobs/{job_id}", headers=auth).json()["violations"][0]
    assert set(viol) == {"type", "timecode_sec", "description", "evidence_frames"}


def test_ui_proxy_without_key(client):
    r = upload(client, path="/ui/analyze")
    assert r.status_code == 200
    body = client.get(f"/ui/jobs/{r.json()['job_id']}").json()
    assert body["status"] == "done"


def test_unknown_and_malformed_job_ids(client, auth):
    assert client.get("/api/v1/jobs/" + "0" * 16, headers=auth).status_code == 404
    assert client.get("/api/v1/jobs/..%2F..%2Fetc", headers=auth).status_code == 404


def test_failed_job_reports_error(client, auth, monkeypatch):
    import pipeline.run
    from pipeline.errors import PipelineInputError

    def boom(_):
        raise PipelineInputError("video has zero duration")

    monkeypatch.setattr(pipeline.run, "analyze", boom)
    job_id = upload(client, headers=auth).json()["job_id"]
    body = client.get(f"/api/v1/jobs/{job_id}", headers=auth).json()
    assert body == {"job_id": job_id, "status": "failed", "error": "video has zero duration"}
    assert not storage.video_path(job_id).exists()


def test_internal_errors_are_not_leaked(client, auth, monkeypatch):
    import pipeline.run

    def boom(_):
        raise RuntimeError("/secret/path/model.pt not found")

    monkeypatch.setattr(pipeline.run, "analyze", boom)
    job_id = upload(client, headers=auth).json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}", headers=auth).json()["error"] == "internal processing error"


def test_upload_size_limit(client, auth, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    big = MOV_BYTES + b"\x00" * (3 * 1024 * 1024)
    assert upload(client, headers=auth, data=big).status_code == 413
    assert list(storage.jobs_root().iterdir()) == []


def test_ttl_cleanup():
    old, fresh = storage.new_job_id(), storage.new_job_id()
    storage.create_job(old)
    storage.create_job(fresh)
    status_path = storage.job_dir(old) / "status.json"
    storage.set_status(old, "done")
    # состарим задачу
    import json
    st = json.loads(status_path.read_text(encoding="utf-8"))
    st["created_at"] = time.time() - 7200
    status_path.write_text(json.dumps(st), encoding="utf-8")

    assert storage.cleanup_expired(ttl_sec=3600) == 1
    assert not os.path.exists(storage.job_dir(old))
    assert os.path.exists(storage.job_dir(fresh))
