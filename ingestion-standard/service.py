#!/usr/bin/env python3
"""
Document Ingestion Service — STANDARD TIER (Port 8072)
AWS Textract-style async job pattern: submit → poll → get result.

Pipeline:
  1. PDF → Markdown via Azure Document Intelligence (OCR + high-res + figures)
  2. Figures described via GPT-4 Vision → appended to markdown
  3. Typed semantic chunking (embedding-similarity, NO LLM page grouping)
     Creates 7 chunk types: image_description, ocr_detail, control,
     definition, table_summary, page_context (per-page), doc_summary

Adds GPT-4V figure extraction over Basic tier.
No GPT-4o page grouping (that's Premium).
"""

import os
import sys
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPELINE_DIR = Path(os.getenv("PIPELINE_DIR", "/home/admincsp/multimodal-rag/azadea")).resolve()
PDF_ROOT     = Path(os.getenv("PDF_ROOT", str(PIPELINE_DIR / "data"))).resolve()
MD_OUT_DIR   = Path(os.getenv("MD_OUT_DIR", str(PIPELINE_DIR / "md_out_data_standard"))).resolve()

os.environ.setdefault("MULTIMODAL_IMAGES_DIR", str(PIPELINE_DIR / "images"))

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from qdrant_client import QdrantClient
from qdrant_client import models as qm
import azure_doc_intelligence_qdrant as qdrant_ingest
from multimodal_extractor import get_aoai_client
from ingest_data_folder_multimodal import (
    process_single_pdf_multimodal,
    ingest_single_md_typed,
)
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TIER           = "standard"
COLLECTION     = os.getenv("QDRANT_COLLECTION", "docs_standard_azadea")
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
PORT           = int(os.getenv("SERVICE_PORT", "8072"))
JOB_TTL_HOURS  = 24
MAX_JOBS_LIST  = 100

# S3
S3_SERVICE_URL = os.getenv("S3_SERVICE_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY  = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY  = os.getenv("S3_SECRET_KEY")
S3_ENABLED     = all([S3_SERVICE_URL, S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_KEY])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("ingestion_standard")


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


def _create_job(operation: str, filename: str, doc_id: str,
                 s3_url: Optional[str] = None) -> JobState:
    job = JobState(
        job_id=uuid.uuid4().hex[:12],
        status=JobStatus.PENDING,
        operation=operation,
        filename=filename,
        doc_id=doc_id,
        s3_url=s3_url,
        step="queued",
        progress="Job submitted, waiting to start",
        created_at=_now_iso(),
    )
    _jobs[job.job_id] = job
    return job


def _cleanup_expired_jobs() -> None:
    try:
        now = time.time()
        expired = [
            jid for jid, j in _jobs.items()
            if (now - datetime.fromisoformat(j.created_at).timestamp()) > JOB_TTL_HOURS * 3600
        ]
        for jid in expired:
            del _jobs[jid]
        active_doc_ids = {j.doc_id for j in _jobs.values()}
        stale_locks = [d for d in _doc_locks if d not in active_doc_ids]
        for d in stale_locks:
            del _doc_locks[d]
    except Exception as e:
        logger.error(f"Job cleanup failed (non-fatal): {e}")


# ===========================================================================
# CLIENTS
# ===========================================================================

_doc_client: Optional[DocumentIntelligenceClient] = None
_aoai_client = None
_qdrant_client: Optional[QdrantClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _doc_client, _aoai_client, _qdrant_client

    logger.info(f"Starting STANDARD ingestion service on port {PORT}...")

    di_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    di_key      = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    if not di_endpoint or not di_key:
        raise RuntimeError("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and _KEY must be set")

    _doc_client = DocumentIntelligenceClient(
        endpoint=di_endpoint, credential=AzureKeyCredential(di_key)
    )
    _aoai_client = get_aoai_client()
    _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

    loop = asyncio.get_running_loop()
    dim = await loop.run_in_executor(None, qdrant_ingest.infer_embedding_dim)
    await loop.run_in_executor(None, qdrant_ingest.ensure_collection, _qdrant_client, COLLECTION, dim)

    logger.info(f"STANDARD tier ready. Collection '{COLLECTION}'.")
    yield
    logger.info("Shutdown.")


# ===========================================================================
# APP
# ===========================================================================

app = FastAPI(
    title="Document Ingestion — Standard Tier",
    description=(
        "Azure DI OCR + GPT-4V figure extraction → typed semantic chunking → Qdrant. "
        "7 chunk types. No LLM page grouping."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class DocumentRequest(BaseModel):
    filename: str
    operation: str
    s3_url: Optional[str] = None


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ===========================================================================
# PIPELINE HELPERS
# ===========================================================================

def _qdrant() -> QdrantClient:
    if _qdrant_client is None:
        raise RuntimeError("QdrantClient not initialized")
    return _qdrant_client


def find_pdf(filename: str) -> Optional[Path]:
    stem = Path(filename).stem
    if any(c in stem for c in ('*', '?', '[', ']')):
        return None
    target = f"{stem}.pdf"
    try:
        matches = list(PDF_ROOT.rglob(target))
        if matches:
            return matches[0]
        lower_stem = stem.lower()
        matches = [p for p in PDF_ROOT.rglob("*.pdf") if p.stem.lower() == lower_stem]
        return matches[0] if matches else None
    except OSError:
        return None


def _get_s3_client():
    import boto3
    return boto3.client(
        "s3", endpoint_url=S3_SERVICE_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="eu-frankfurt-1",
    )


def parse_s3_object_key(s3_url: str) -> str:
    parsed = urlparse(s3_url)
    path = unquote(parsed.path)
    if "/b/" in path and "/o/" in path:
        return path.split("/o/", 1)[1]
    key = path.lstrip("/")
    if S3_BUCKET_NAME and key.startswith(f"{S3_BUCKET_NAME}/"):
        key = key[len(S3_BUCKET_NAME) + 1:]
    return key


def download_from_s3(s3_url: str, filename: str) -> Path:
    s3 = _get_s3_client()
    object_key = parse_s3_object_key(s3_url)
    local_path = PDF_ROOT / object_key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading s3://{S3_BUCKET_NAME}/{object_key} → {local_path}")
    s3.download_file(S3_BUCKET_NAME, object_key, str(local_path))
    if not local_path.exists():
        raise RuntimeError(f"S3 download completed but file not found at {local_path}")
    logger.info(f"Downloaded {local_path.stat().st_size:,} bytes")
    return local_path


def delete_doc_from_qdrant(client: QdrantClient, doc_id: str) -> int:
    count = client.count(
        collection_name=COLLECTION,
        count_filter=qm.Filter(must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
        ]),
        exact=True,
    ).count
    if count > 0:
        client.delete(
            collection_name=COLLECTION,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[
                    qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
                ])
            ),
        )
    return count


def remove_md_files(doc_id: str) -> None:
    for suffix in [".md", "_figures.json"]:
        path = MD_OUT_DIR / f"{doc_id}{suffix}"
        if path.exists():
            try:
                path.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove {path.name}: {e}")


def run_pdf_to_md(pdf_path: Path) -> Path:
    """Standard: Azure DI OCR + GPT-4V figure extraction."""
    MD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = process_single_pdf_multimodal(
        (pdf_path, MD_OUT_DIR, _doc_client, _aoai_client, 1, 1)
    )
    if result_path is None:
        raise RuntimeError(f"Multimodal PDF conversion failed for '{pdf_path.name}'")
    return result_path


def run_md_to_qdrant(md_path: Path) -> int:
    """Standard: typed semantic chunking (per-page, no LLM grouping)."""
    chunks = ingest_single_md_typed(
        (md_path, _qdrant(), COLLECTION, 1, 1)
    )
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
                job.progress = f"Removing all chunks for '{job.doc_id}'..."
                deleted = await loop.run_in_executor(None, delete_doc_from_qdrant, _qdrant(), job.doc_id)
                job.chunks_deleted = deleted
                await loop.run_in_executor(None, remove_md_files, job.doc_id)
                job.status = JobStatus.COMPLETED
                job.completed_at = _now_iso()
                job.progress = f"Deleted {deleted} chunks."

            elif job.operation in ("add", "update"):
                # S3 download
                if job.s3_url and pdf_path is None:
                    job.step = "downloading"
                    job.progress = "Downloading PDF from S3..."
                    try:
                        pdf_path = await loop.run_in_executor(None, download_from_s3, job.s3_url, job.filename)
                    except Exception as e:
                        job.status = JobStatus.FAILED
                        job.completed_at = _now_iso()
                        job.error = f"S3 download failed: {e}"
                        job.recovery = "Check S3 URL and credentials, then retry."
                        return

                if job.operation == "update":
                    job.step = "deleting"
                    job.progress = "Removing existing chunks..."
                    deleted = await loop.run_in_executor(None, delete_doc_from_qdrant, _qdrant(), job.doc_id)
                    job.chunks_deleted = deleted
                    await loop.run_in_executor(None, remove_md_files, job.doc_id)

                elif job.operation == "add":
                    job.step = "validating"
                    job.progress = "Checking if document is already indexed..."
                    existing = await loop.run_in_executor(
                        None,
                        lambda: _qdrant().count(
                            collection_name=COLLECTION,
                            count_filter=qm.Filter(must=[
                                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=job.doc_id))
                            ]),
                            exact=True,
                        ).count
                    )
                    if existing > 0:
                        job.status = JobStatus.FAILED
                        job.completed_at = _now_iso()
                        job.error = f"Document already indexed with {existing} chunks. Use operation='update'."
                        return

                # PDF → MD (multimodal with figure extraction)
                job.step = "pdf_to_md"
                job.progress = "Converting PDF → Markdown via Azure DI + GPT-4V figure extraction..."
                try:
                    md_path = await loop.run_in_executor(None, run_pdf_to_md, pdf_path)
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.completed_at = _now_iso()
                    job.error = f"PDF→MD failed: {e}"
                    job.recovery = f"Retry with operation='{job.operation}'."
                    return

                # MD → Qdrant (typed semantic chunking)
                job.step = "md_to_qdrant"
                job.progress = "Typed semantic chunking → embedding → Qdrant hybrid upsert..."
                try:
                    chunks = await loop.run_in_executor(None, run_md_to_qdrant, md_path)
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.completed_at = _now_iso()
                    job.error = f"MD→Qdrant failed: {e}"
                    job.recovery = "MD file created. Retry with operation='add'."
                    return

                job.chunks_created = chunks
                job.status = JobStatus.COMPLETED
                job.completed_at = _now_iso()
                op_label = "Updated" if job.operation == "update" else "Ingested"
                job.progress = (
                    f"{op_label} {chunks} typed chunks "
                    f"(image_description, ocr_detail, control, definition, "
                    f"table_summary, page_context, doc_summary)."
                )

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
    filename  = request.filename.strip()
    s3_url    = (request.s3_url or "").strip() or None

    if operation not in ("add", "update", "delete"):
        raise HTTPException(status_code=400, detail=f"Invalid operation '{operation}'.")

    doc_id = Path(filename).stem
    pdf_path: Optional[Path] = None

    if operation in ("add", "update"):
        if s3_url:
            if not S3_ENABLED:
                raise HTTPException(status_code=400, detail="S3 not configured.")
        else:
            pdf_path = find_pdf(filename)
            if pdf_path is None:
                raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found under {PDF_ROOT}.")

    _cleanup_expired_jobs()
    job = _create_job(operation, filename, doc_id, s3_url=s3_url)
    task = asyncio.create_task(_execute_job(job, pdf_path))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return JobSubmitResponse(
        job_id=job.job_id, status=job.status.value,
        message=f"Job submitted. Poll GET /job/{job.job_id} for status.",
    )


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
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
        client = _qdrant()
        count_result = client.count(
            collection_name=COLLECTION,
            count_filter=qm.Filter(must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
            ]),
            exact=True,
        )
        scroll_result, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=qm.Filter(must=[
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
            ]),
            limit=500,
            with_payload=qm.PayloadSelectorInclude(include=["chunk_type"]),
            with_vectors=False,
        )
        type_counts: dict = {}
        for pt in scroll_result:
            ct = (pt.payload or {}).get("chunk_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1

        return {
            "doc_id": doc_id,
            "collection": COLLECTION,
            "tier": TIER,
            "total_chunks": count_result.count,
            "chunk_types": type_counts,
            "indexed": count_result.count > 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    try:
        client = _qdrant()
        info = client.get_collection(COLLECTION)
        return {
            "status": "ok",
            "tier": TIER,
            "collection": COLLECTION,
            "total_points": info.points_count,
            "active_jobs": sum(1 for j in _jobs.values() if j.status in (JobStatus.PENDING, JobStatus.PROCESSING)),
        }
    except Exception as e:
        return {"status": "degraded", "tier": TIER, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info", workers=1, timeout_keep_alive=300)
