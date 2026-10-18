"""FastAPI: публичное API (/api/v1, с X-API-Key), UI-прокси (/ui, без ключа в браузере), healthchecks."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from rq import Queue, Worker

from app import storage
from app.api import jobs
from app.api.auth import require_api_key
from app.api.schemas import AnalyzeResponse, to_public
from app.config import get_settings
from app.worker.readiness import READY_KEY

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
CLEANUP_INTERVAL_SEC = 300


async def _cleanup_loop() -> None:
    while True:
        try:
            removed = await asyncio.to_thread(storage.cleanup_expired)
            if removed:
                log.info("cleanup: removed %d expired jobs", removed)
        except Exception:
            log.exception("cleanup failed")
        await asyncio.sleep(CLEANUP_INTERVAL_SEC)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()


app = FastAPI(title="CashGuard", version="0.1.0", lifespan=lifespan)


def _job_json(job_id: str) -> JSONResponse:
    return JSONResponse(to_public(jobs.load(job_id), strict=get_settings().strict_schema))


# --- публичное API ---------------------------------------------------------

@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(require_api_key), Depends(jobs.check_content_length)],
)
async def analyze(file: UploadFile = File(...), queue: Queue = Depends(jobs.get_queue)) -> AnalyzeResponse:
    return AnalyzeResponse(job_id=await jobs.submit(file, queue))


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> JSONResponse:
    return _job_json(job_id)


# --- UI-прокси: те же операции, ключ не передаётся в браузер ---------------

@app.post("/ui/analyze", response_model=AnalyzeResponse, dependencies=[Depends(jobs.check_content_length)])
async def ui_analyze(file: UploadFile = File(...), queue: Queue = Depends(jobs.get_queue)) -> AnalyzeResponse:
    return AnalyzeResponse(job_id=await jobs.submit(file, queue))


@app.get("/ui/jobs/{job_id}")
def ui_get_job(job_id: str) -> JSONResponse:
    return _job_json(job_id)


# --- health ----------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(queue: Queue = Depends(jobs.get_queue)) -> JSONResponse:
    checks = {"redis": False, "worker": False, "models": False}
    try:
        conn = queue.connection
        checks["redis"] = bool(conn.ping())
        checks["worker"] = Worker.count(queue=queue) > 0
        checks["models"] = bool(conn.exists(READY_KEY))
    except Exception:
        log.exception("readiness check failed")
    ready = all(checks.values())
    return JSONResponse({"ready": ready, **checks}, status_code=200 if ready else 503)


# --- статический UI --------------------------------------------------------

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
