#!/usr/bin/env python3
"""
Document Ingestion Service — OCI TIER (Port 8074)
Zero Azure dependency. Gemini 2.5 Flash for PDF understanding, embedding, grouping.

Pipeline:
  1. PDF → Markdown via Gemini 2.5 Flash Document Understanding
  2. Figures described via OCI Gemini 2.5 Pro
  3. LLM page grouping via OCI Gemini 2.5 Flash
  4. Dense vectors: OCI Cohere Embed v4.0 (1536-dim)
  5. → Oracle 26ai AI Vector Search (HNSW INMEMORY + Oracle Text)

Endpoints:
    POST /document              — submit job → returns job_id
    GET  /job/{job_id}          — poll job status
    GET  /jobs                  — list recent jobs
    GET  /document/{doc_id}     — chunk count in Oracle
    GET  /health                — service status
"""

import os
import sys
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
load_dotenv()

# Ensure local modules are importable
SERVICE_DIR = Path(__file__).parent.resolve()
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

# Pipeline modules (chunk_types, chunk_extractors) loaded via ingest_pipeline.py
import oracle_vectordb
import qdrant_utils
import ingest_pipeline
from ingest_pipeline import process_pdf_oci, ingest_md_oci

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TIER       = "oci"
PDF_ROOT   = Path(os.getenv("PDF_ROOT", "/home/admincsp/multimodal-rag/azadea/data")).resolve()
MD_OUT_DIR = Path(os.getenv("MD_OUT_DIR", "/home/admincsp/multimodal-rag/azadea/md_out_data_oci")).resolve()
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "/home/admincsp/multimodal-rag/azadea/images_oci")).resolve()
PORT       = int(os.getenv("SERVICE_PORT", "8074"))
JOB_TTL_HOURS = 24
MAX_JOBS_LIST = 100

# S3
S3_SERVICE_URL = os.getenv("S3_SERVICE_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY  = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY  = os.getenv("S3_SECRET_KEY")
S3_ENABLED     = all([S3_SERVICE_URL, S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_KEY])

# BrainShift Companion file-download API (fetch source PDFs by fileId instead of S3)
COMPANION_API_URL   = os.getenv("COMPANION_API_URL", "https://companion-api.azadeans.com").rstrip("/")
COMPANION_API_TOKEN = os.getenv("COMPANION_API_TOKEN")  # Bearer token for the download API

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("ingestion_oci")


# ===========================================================================
# JOB STATE
# ===========================================================================

class JobStatus(str, Enum):
    PENDING    = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED  = "COMPLETED"
    FAILED     = "FAILED"


@dataclass
class JobState:
    job_id: str
    status: JobStatus
    operation: str
    filename: str
    doc_id: str
    tier: str = TIER
    s3_url: Optional[str] = None
    file_id: Optional[str] = None
    step: str = ""
    progress: str = ""
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    chunks_created: int = 0
    chunks_deleted: int = 0
    error: Optional[str] = None
    recovery: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(self).items()}


_jobs: Dict[str, JobState] = {}
_doc_locks: Dict[str, asyncio.Lock] = {}
_background_tasks: set = set()


def _get_doc_lock(doc_id: str) -> asyncio.Lock:
    if doc_id not in _doc_locks:
        _doc_locks[doc_id] = asyncio.Lock()
    return _doc_locks[doc_id]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _create_job(operation, filename, doc_id, s3_url=None, file_id=None):
    job = JobState(
        job_id=uuid.uuid4().hex[:12], status=JobStatus.PENDING,
        operation=operation, filename=filename, doc_id=doc_id, s3_url=s3_url,
        file_id=file_id,
        step="queued", progress="Job submitted", created_at=_now_iso(),
    )
    _jobs[job.job_id] = job
    return job


def _cleanup_expired_jobs():
    try:
        now = time.time()
        expired = [jid for jid, j in _jobs.items()
                   if (now - datetime.fromisoformat(j.created_at).timestamp()) > JOB_TTL_HOURS * 3600]
        for jid in expired:
            del _jobs[jid]
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


# ===========================================================================
# CLIENTS
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting OCI ingestion service on port {PORT}...")

    loop = asyncio.get_running_loop()
    # Live store is Qdrant. Oracle is optional/fallback — don't let it block startup.
    try:
        total = count_doc_in_qdrant  # ensure helper imports resolve
        coll = ingest_pipeline.OCI_QDRANT_COLLECTION
        n = ingest_pipeline._qdrant_client.count(coll, exact=True).count
        logger.info(f"OCI tier ready. Qdrant collection '{coll}' ({n} chunks).")
    except Exception as e:
        logger.warning(f"Qdrant health check failed at startup: {e}")
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="Document Ingestion — OCI Tier",
    description=(
        "Gemini 2.5 Flash Document Understanding + Cohere Embed v4.0. Zero Azure. "
        "Zero Azure LLM/embedding dependency."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class DocumentRequest(BaseModel):
    filename: str
    operation: str
    s3_url: Optional[str] = None
    file_id: Optional[str] = None   # BrainShift Companion file id — fetched via the download API


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ===========================================================================
# HELPERS
# ===========================================================================

def find_pdf(filename: str) -> Optional[Path]:
    stem = Path(filename).stem
    if any(c in stem for c in ("*", "?", "[", "]")):
        return None
    try:
        matches = list(PDF_ROOT.rglob(f"{stem}.pdf"))
        if matches:
            return matches[0]
        lower = stem.lower()
        matches = [p for p in PDF_ROOT.rglob("*.pdf") if p.stem.lower() == lower]
        return matches[0] if matches else None
    except OSError:
        return None


def _get_s3_client():
    import boto3
    return boto3.client("s3", endpoint_url=S3_SERVICE_URL,
                        aws_access_key_id=S3_ACCESS_KEY,
                        aws_secret_access_key=S3_SECRET_KEY, region_name="eu-frankfurt-1")


def download_from_s3(s3_url: str, filename: str) -> Path:
    s3 = _get_s3_client()
    parsed = urlparse(s3_url)
    path = unquote(parsed.path)
    if "/b/" in path and "/o/" in path:
        object_key = path.split("/o/", 1)[1]
    else:
        key = path.lstrip("/")
        object_key = key[len(S3_BUCKET_NAME) + 1:] if S3_BUCKET_NAME and key.startswith(f"{S3_BUCKET_NAME}/") else key
    local_path = PDF_ROOT / object_key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(S3_BUCKET_NAME, object_key, str(local_path))
    return local_path


def download_from_companion(file_id: str, filename: str) -> Path:
    """Fetch a source PDF from the BrainShift Companion download API by file id,
    instead of S3. GET {COMPANION_API_URL}/api/Files/Download?id={file_id}."""
    import requests
    headers = {}
    if COMPANION_API_TOKEN:
        headers["Authorization"] = f"Bearer {COMPANION_API_TOKEN}"
    url = f"{COMPANION_API_URL}/api/Files/Download"
    resp = requests.get(url, params={"id": file_id}, headers=headers, timeout=120, stream=True)
    resp.raise_for_status()
    local_path = PDF_ROOT / Path(filename).name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    return local_path


# Live store is Qdrant (moved off Oracle 26ai). Delete/count operate on the
# same collection the OCI agent reads + ingest_pipeline writes.
def delete_doc_from_qdrant(doc_id: str) -> int:
    return qdrant_utils.delete_by_doc(
        ingest_pipeline._qdrant_client, ingest_pipeline.OCI_QDRANT_COLLECTION, doc_id
    )


def count_doc_in_qdrant(doc_id: str) -> int:
    return qdrant_utils.count_by_doc(
        ingest_pipeline._qdrant_client, ingest_pipeline.OCI_QDRANT_COLLECTION, doc_id
    )


def remove_md_files(doc_id: str):
    for suffix in [".md", "_figures.json"]:
        path = MD_OUT_DIR / f"{doc_id}{suffix}"
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def run_pdf_to_md(pdf_path: Path) -> Path:
    """Gemini 2.5 Flash PDF→MD (text + tables + image descriptions in one pass)."""
    md_path, _ = process_pdf_oci(pdf_path, MD_OUT_DIR)
    return md_path


def run_md_to_oracle(md_path: Path) -> int:
    """Chunk extraction + OCI LLM grouping + OCI embed → Oracle 26ai."""
    chunks = ingest_md_oci(md_path)
    if chunks == 0:
        raise RuntimeError(f"Ingestion returned 0 chunks for '{md_path.name}'")
    return chunks


# ===========================================================================
# BACKGROUND JOB
# ===========================================================================

async def _execute_job(job: JobState, pdf_path: Optional[Path]):
    loop = asyncio.get_running_loop()
    async with _get_doc_lock(job.doc_id):
        try:
            job.status = JobStatus.PROCESSING
            job.started_at = _now_iso()

            if job.operation == "delete":
                job.step = "deleting"
                job.progress = "Removing chunks..."
                deleted = await loop.run_in_executor(None, delete_doc_from_qdrant, job.doc_id)
                job.chunks_deleted = deleted
                await loop.run_in_executor(None, remove_md_files, job.doc_id)
                job.status = JobStatus.COMPLETED
                job.completed_at = _now_iso()
                job.progress = f"Deleted {deleted} chunks."

            elif job.operation in ("add", "update"):
                # Fetch source PDF. Prefer the Companion download API (by file_id);
                # fall back to S3 url; else the PDF must already be on disk.
                if job.file_id and pdf_path is None:
                    job.step = "downloading"
                    job.progress = "Downloading from Companion file API..."
                    try:
                        pdf_path = await loop.run_in_executor(None, download_from_companion, job.file_id, job.filename)
                    except Exception as e:
                        job.status = JobStatus.FAILED
                        job.completed_at = _now_iso()
                        job.error = f"Companion download failed: {e}"
                        return
                elif job.s3_url and pdf_path is None:
                    job.step = "downloading"
                    job.progress = "Downloading from S3..."
                    try:
                        pdf_path = await loop.run_in_executor(None, download_from_s3, job.s3_url, job.filename)
                    except Exception as e:
                        job.status = JobStatus.FAILED
                        job.completed_at = _now_iso()
                        job.error = f"S3 download failed: {e}"
                        return

                if job.operation == "update":
                    job.step = "deleting"
                    job.progress = "Removing existing chunks..."
                    deleted = await loop.run_in_executor(None, delete_doc_from_qdrant, job.doc_id)
                    job.chunks_deleted = deleted
                    await loop.run_in_executor(None, remove_md_files, job.doc_id)
                elif job.operation == "add":
                    job.step = "validating"
                    existing = await loop.run_in_executor(None, count_doc_in_qdrant, job.doc_id)
                    if existing > 0:
                        job.status = JobStatus.FAILED
                        job.completed_at = _now_iso()
                        job.error = f"Already indexed with {existing} chunks. Use operation='update'."
                        return

                # PDF → MD (Docling + Gemini vision)
                job.step = "pdf_to_md"
                job.progress = "Gemini 2.5 Flash PDF → Markdown (parallel page batches)..."
                try:
                    md_path = await loop.run_in_executor(None, run_pdf_to_md, pdf_path)
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.completed_at = _now_iso()
                    job.error = f"PDF→MD failed: {e}"
                    job.recovery = f"Retry with operation='{job.operation}'."
                    return

                # MD → Oracle (LLM grouping + OCI embed)
                job.step = "md_to_qdrant"
                job.progress = "Gemini Flash page grouping → Cohere Embed v4.0 → Qdrant..."
                try:
                    chunks = await loop.run_in_executor(None, run_md_to_oracle, md_path)
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.completed_at = _now_iso()
                    job.error = f"MD→Oracle failed: {e}"
                    job.recovery = "MD created. Retry with operation='add'."
                    return

                job.chunks_created = chunks
                job.status = JobStatus.COMPLETED
                job.completed_at = _now_iso()
                op = "Updated" if job.operation == "update" else "Ingested"
                job.progress = f"{op} {chunks} OCI-processed chunks (Gemini Flash + Cohere Embed v4.0)."

        except Exception as e:
            job.status = JobStatus.FAILED
            job.completed_at = _now_iso()
            job.error = str(e)
            logger.exception(f"Job {job.job_id} failed: {e}")


# ===========================================================================
# ENDPOINTS
# ===========================================================================

@app.post("/document", response_model=JobSubmitResponse)
async def submit_document_job(request: DocumentRequest):
    operation = request.operation.lower().strip()
    filename = request.filename.strip()
    s3_url = (request.s3_url or "").strip() or None
    file_id = (request.file_id or "").strip() or None

    if operation not in ("add", "update", "delete"):
        raise HTTPException(status_code=400, detail=f"Invalid operation '{operation}'.")

    doc_id = Path(filename).stem
    pdf_path: Optional[Path] = None

    if operation in ("add", "update"):
        if file_id:
            # Fetched from the Companion download API at job time. Token optional
            # (may be unauthenticated on the internal network).
            pass
        elif s3_url:
            if not S3_ENABLED:
                raise HTTPException(status_code=400, detail="S3 not configured.")
        else:
            pdf_path = find_pdf(filename)
            if pdf_path is None:
                raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found under {PDF_ROOT}.")

    _cleanup_expired_jobs()
    job = _create_job(operation, filename, doc_id, s3_url=s3_url, file_id=file_id)
    task = asyncio.create_task(_execute_job(job, pdf_path))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return JobSubmitResponse(job_id=job.job_id, status=job.status.value,
                             message=f"Poll GET /job/{job.job_id} for status.")


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.to_dict()


@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, doc_id: Optional[str] = None, limit: int = MAX_JOBS_LIST):
    jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    if status:
        jobs = [j for j in jobs if j.status.value == status.upper()]
    if doc_id:
        jobs = [j for j in jobs if j.doc_id == doc_id]
    return [j.to_dict() for j in jobs[:limit]]


@app.get("/document/{doc_id}")
def get_document_info(doc_id: str):
    try:
        count = count_doc_in_qdrant(doc_id)
        return {"doc_id": doc_id, "backend": "qdrant",
                "collection": ingest_pipeline.OCI_QDRANT_COLLECTION, "tier": TIER,
                "total_chunks": count, "indexed": count > 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    try:
        coll = ingest_pipeline.OCI_QDRANT_COLLECTION
        total = ingest_pipeline._qdrant_client.count(coll, exact=True).count
        return {"status": "ok", "tier": TIER, "backend": "qdrant", "collection": coll,
                "total_chunks": total,
                "active_jobs": sum(1 for j in _jobs.values() if j.status in (JobStatus.PENDING, JobStatus.PROCESSING)),
                "pipeline": "Gemini 2.5 Flash DU + Cohere Embed v4.0 + Qdrant (dense+sparse)"}
    except Exception as e:
        return {"status": "degraded", "tier": TIER, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info", workers=1, timeout_keep_alive=300)
