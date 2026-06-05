#!/usr/bin/env python3
"""
Document Ingestion Service — Port 8070
AWS Textract-style async job pattern: submit → poll → get result.

Pipeline:
  1. PDF → Markdown via Azure Document Intelligence (OCR + high-res + figures)
  2. Figures described via GPT-4 Vision → appended to markdown
  3. Markdown → Qdrant via LLM-guided page grouping (GPT-4o groups pages by topic)
     Creates 7 typed chunk types: image_description, ocr_detail, control,
     definition, table_summary, page_context (LLM-grouped), doc_summary

Endpoints:
    POST /document              — submit job → returns job_id instantly
    GET  /job/{job_id}          — poll job status + progress
    GET  /jobs                  — list recent jobs
    GET  /document/{doc_id}     — chunk count + type breakdown in Qdrant
    GET  /health                — Qdrant status

Required env vars:
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, AZURE_DOCUMENT_INTELLIGENCE_KEY
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
    QDRANT_URL (default: http://localhost:6333)
    QDRANT_API_KEY (optional)

Optional env vars:
    AZURE_OPENAI_EMBED_DEPLOYMENT (default: text-embedding-3-large)
    AZURE_OPENAI_CHAT_DEPLOYMENT  (default: gpt-4o)
    AOAI_VISION_DEPLOYMENT        (default: gpt-4.1)
    MULTIMODAL_IMAGES_DIR         (default: AZADEA_DIR/images)
    S3_SERVICE_URL, S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_KEY
        (OCI S3-compatible storage — enables s3_url in requests)
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
# Absolute paths
# ---------------------------------------------------------------------------
AZADEA_DIR = Path("/home/admincsp/multimodal-rag/azadea").resolve()
PDF_ROOT   = AZADEA_DIR / "data"
MD_OUT_DIR = AZADEA_DIR / "md_out_data_multimodal"

os.environ.setdefault("MULTIMODAL_IMAGES_DIR", str(AZADEA_DIR / "images"))

# ---------------------------------------------------------------------------
# sys.path — AZADEA_DIR first so pipeline modules resolve their local imports
# ---------------------------------------------------------------------------
if str(AZADEA_DIR) not in sys.path:
    sys.path.insert(0, str(AZADEA_DIR))

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from qdrant_client import QdrantClient                                    # noqa: E402
from qdrant_client import models as qm                                    # noqa: E402
import azure_doc_intelligence_qdrant as qdrant_ingest                     # noqa: E402
from multimodal_extractor import get_aoai_client                          # noqa: E402
from ingest_data_folder_multimodal import process_single_pdf_multimodal   # noqa: E402
from llm_semantic_chunker import (                                        # noqa: E402
    ingest_single_md_llm,
    COLLECTION_NAME as LLM_COLLECTION,
)
from azure.core.credentials import AzureKeyCredential                     # noqa: E402
from azure.ai.documentintelligence import DocumentIntelligenceClient      # noqa: E402

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException                                # noqa: E402
from fastapi.middleware.cors import CORSMiddleware                        # noqa: E402
from pydantic import BaseModel                                            # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COLLECTION     = LLM_COLLECTION  # "docs_llm_chunked_azadea"
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
JOB_TTL_HOURS  = 24              # auto-expire jobs older than this
MAX_JOBS_LIST  = 100             # max jobs returned by GET /jobs

# OCI S3-compatible storage
S3_SERVICE_URL = os.getenv("S3_SERVICE_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY  = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY  = os.getenv("S3_SECRET_KEY")
S3_ENABLED     = all([S3_SERVICE_URL, S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_KEY])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("doc_ingestion")


# ===========================================================================
# JOB STATE MODEL
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
    operation: str          # "add" | "update" | "delete"
    filename: str
    doc_id: str
    s3_url: Optional[str] = None  # if provided, PDF was downloaded from S3
    step: str = ""          # current step: "downloading", "validating", "pdf_to_md", "md_to_qdrant", "deleting"
    progress: str = ""      # human-readable progress message
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    chunks_created: int = 0
    chunks_deleted: int = 0
    error: Optional[str] = None
    recovery: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(self).items()}


# In-memory job store
_jobs: Dict[str, JobState] = {}

# Per-doc concurrency lock — prevents concurrent ops on same document
_doc_locks: Dict[str, asyncio.Lock] = {}

# Strong references to background tasks so GC cannot collect them mid-run
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
    """Remove jobs older than JOB_TTL_HOURS to prevent memory leak."""
    try:
        now = time.time()
        expired = [
            jid for jid, j in _jobs.items()
            if (now - datetime.fromisoformat(j.created_at).timestamp()) > JOB_TTL_HOURS * 3600
        ]
        for jid in expired:
            del _jobs[jid]
        # Also evict doc locks for expired docs to prevent unbounded growth
        active_doc_ids = {j.doc_id for j in _jobs.values()}
        stale_locks = [d for d in _doc_locks if d not in active_doc_ids]
        for d in stale_locks:
            del _doc_locks[d]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired jobs, {len(stale_locks)} stale locks")
    except Exception as e:
        logger.error(f"Job cleanup failed (non-fatal): {e}")


# ===========================================================================
# SHARED CLIENT STATE (initialized at startup via lifespan)
# ===========================================================================

_doc_client: Optional[DocumentIntelligenceClient] = None
_aoai_client = None
_qdrant_client: Optional[QdrantClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _doc_client, _aoai_client, _qdrant_client

    logger.info("Starting ingestion service — initializing clients...")

    di_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    di_key      = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    if not di_endpoint or not di_key:
        raise RuntimeError(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
            "AZURE_DOCUMENT_INTELLIGENCE_KEY must be set"
        )
    _doc_client = DocumentIntelligenceClient(
        endpoint=di_endpoint, credential=AzureKeyCredential(di_key)
    )
    _aoai_client = get_aoai_client()
    _qdrant_client = QdrantClient(
        url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False
    )

    loop = asyncio.get_running_loop()
    dim = await loop.run_in_executor(None, qdrant_ingest.infer_embedding_dim)
    await loop.run_in_executor(
        None, qdrant_ingest.ensure_collection,
        _qdrant_client, COLLECTION, dim
    )

    logger.info(f"Clients initialized. Collection '{COLLECTION}' ready.")
    yield
    logger.info("Shutdown.")


# ===========================================================================
# APP
# ===========================================================================

app = FastAPI(
    title="Document Ingestion Service",
    description=(
        "Async document ingestion with job polling (AWS Textract pattern). "
        "Submit a job, get a job_id, poll for status."
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# REQUEST / RESPONSE MODELS
# ===========================================================================

class DocumentRequest(BaseModel):
    filename: str                    # e.g. "ABS - DMD - 008 - Shipment E-Invoice - G - 1.pdf"
    operation: str                   # "add" | "update" | "delete"
    s3_url: Optional[str] = None     # OCI S3 URL — if provided, PDF is downloaded from S3


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ===========================================================================
# PIPELINE HELPERS (blocking — run in executor)
# ===========================================================================

def _qdrant() -> QdrantClient:
    if _qdrant_client is None:
        raise RuntimeError("QdrantClient not initialized")
    return _qdrant_client


def find_pdf(filename: str) -> Optional[Path]:
    stem = Path(filename).stem
    # Reject glob metacharacters to prevent rglob injection
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
    except OSError as e:
        logger.error(f"Failed to search PDF directory {PDF_ROOT}: {e}")
        return None


def _get_s3_client():
    """Create a boto3 S3 client for OCI-compatible object storage."""
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=S3_SERVICE_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="eu-frankfurt-1",
    )


def parse_s3_object_key(s3_url: str) -> str:
    """
    Extract the object key from an OCI Object Storage URL.

    Supports:
      - S3-compat: https://<namespace>.compat.objectstorage.<region>.oraclecloud.com/<bucket>/<key>
      - Native OCI: https://objectstorage.<region>.oraclecloud.com/n/<ns>/b/<bucket>/o/<key>
    """
    parsed = urlparse(s3_url)
    path = unquote(parsed.path)

    # Native OCI format: /n/<namespace>/b/<bucket>/o/<object_key>
    if "/b/" in path and "/o/" in path:
        return path.split("/o/", 1)[1]

    # S3-compat format: /<bucket>/<key> or just /<key>
    # Strip leading slash and bucket name if present
    key = path.lstrip("/")
    if key.startswith(f"{S3_BUCKET_NAME}/"):
        key = key[len(S3_BUCKET_NAME) + 1:]
    return key


def download_from_s3(s3_url: str, filename: str) -> Path:
    """
    Download a PDF from OCI S3-compatible storage to PDF_ROOT.

    If s3_url is a full OCI URL, extracts the object key from it.
    Preserves the subfolder structure from the S3 key under PDF_ROOT.
    """
    s3 = _get_s3_client()
    object_key = parse_s3_object_key(s3_url)

    # Determine local destination — preserve S3 folder structure
    local_path = PDF_ROOT / object_key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading s3://{S3_BUCKET_NAME}/{object_key} → {local_path}")
    s3.download_file(S3_BUCKET_NAME, object_key, str(local_path))

    if not local_path.exists():
        raise RuntimeError(f"S3 download completed but file not found at {local_path}")

    logger.info(f"Downloaded {local_path.stat().st_size:,} bytes → {local_path.name}")
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
        logger.info(f"Deleted {count} Qdrant points for doc_id='{doc_id}'")
    return count


def remove_md_files(doc_id: str) -> None:
    for suffix in [".md", "_figures.json"]:
        path = MD_OUT_DIR / f"{doc_id}{suffix}"
        if path.exists():
            try:
                path.unlink()
                logger.info(f"Removed: {path.name}")
            except OSError as e:
                logger.warning(f"Failed to remove {path.name}: {e}")


def run_pdf_to_md(pdf_path: Path) -> Path:
    MD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = process_single_pdf_multimodal(
        (pdf_path, MD_OUT_DIR, _doc_client, _aoai_client, 1, 1)
    )
    if result_path is None:
        raise RuntimeError(
            f"Azure Document Intelligence conversion failed for '{pdf_path.name}'."
        )
    return result_path


def run_md_to_qdrant(md_path: Path) -> int:
    chunks = ingest_single_md_llm((md_path, _qdrant(), COLLECTION, 1, 1, False))
    if chunks == 0:
        raise RuntimeError(
            f"Ingestion pipeline returned 0 chunks for '{md_path.name}'. "
            "Check logs for LLM grouping, embedding, or Qdrant upsert errors."
        )
    return chunks


# ===========================================================================
# BACKGROUND JOB EXECUTION
# ===========================================================================

async def _execute_job(job: JobState, pdf_path: Optional[Path]):
    """
    Run the full pipeline for a job in the background.
    Updates job.status/step/progress at each stage.
    """
    loop = asyncio.get_running_loop()

    async with _get_doc_lock(job.doc_id):
        try:
            job.status = JobStatus.PROCESSING
            job.started_at = _now_iso()

            if job.operation == "delete":
                # --- DELETE ---
                job.step = "deleting"
                job.progress = f"Removing all chunks for '{job.doc_id}' from Qdrant..."
                deleted = await loop.run_in_executor(
                    None, delete_doc_from_qdrant, _qdrant(), job.doc_id
                )
                job.chunks_deleted = deleted

                job.step = "cleanup"
                job.progress = "Removing MD and figures files..."
                await loop.run_in_executor(None, remove_md_files, job.doc_id)

                job.status = JobStatus.COMPLETED
                job.completed_at = _now_iso()
                job.progress = f"Deleted {deleted} chunks for '{job.doc_id}'."

            elif job.operation in ("add", "update"):
                # --- S3 DOWNLOAD (if needed) ---
                if job.s3_url and pdf_path is None:
                    job.step = "downloading"
                    job.progress = f"Downloading PDF from S3 storage..."
                    try:
                        pdf_path = await loop.run_in_executor(
                            None, download_from_s3, job.s3_url, job.filename
                        )
                    except Exception as e:
                        job.status = JobStatus.FAILED
                        job.completed_at = _now_iso()
                        job.error = f"S3 download failed: {e}"
                        job.recovery = "Check S3 URL and credentials, then retry."
                        logger.error(f"[{job.operation}] S3 download failed for '{job.filename}': {e}")
                        return

                if job.operation == "add":
                    # --- ADD ---
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
                        job.error = (
                            f"Document '{job.doc_id}' already indexed with {existing} chunks. "
                            "Use operation='update' to re-index."
                        )
                        return

                    # Step: PDF → MD
                    job.step = "pdf_to_md"
                    job.progress = (
                        "Converting PDF to Markdown via Azure Document Intelligence + GPT-4 Vision "
                        "(OCR, figures extraction)..."
                    )
                    try:
                        md_path = await loop.run_in_executor(None, run_pdf_to_md, pdf_path)
                    except Exception as e:
                        job.status = JobStatus.FAILED
                        job.completed_at = _now_iso()
                        job.error = f"Add failed during PDF→MD: {e}"
                        job.recovery = "Fix the issue and retry with operation='add'."
                        logger.error(f"[add] PDF→MD failed for '{job.doc_id}': {e}")
                        return

                    # Step: MD → Qdrant
                    job.step = "md_to_qdrant"
                    job.progress = (
                        "Running LLM page grouping (GPT-4o) → embedding → Qdrant hybrid upsert..."
                    )
                    try:
                        chunks = await loop.run_in_executor(None, run_md_to_qdrant, md_path)
                    except Exception as e:
                        job.status = JobStatus.FAILED
                        job.completed_at = _now_iso()
                        job.error = f"Add failed during MD→Qdrant: {e}"
                        job.recovery = (
                            "MD file was created but Qdrant ingestion failed. "
                            "Retry with operation='add' (after deleting partial data if any)."
                        )
                        logger.error(f"[add] MD→Qdrant failed for '{job.doc_id}': {e}")
                        return

                    job.chunks_created = chunks
                    job.status = JobStatus.COMPLETED
                    job.completed_at = _now_iso()
                    job.progress = (
                        f"Successfully ingested {chunks} chunks "
                        f"(image_description, ocr_detail, control, definition, "
                        f"table_summary, page_context, doc_summary)."
                    )

                else:
                    # --- UPDATE ---
                    # Step: Delete existing
                    job.step = "deleting"
                    job.progress = f"Removing existing chunks for '{job.doc_id}' from Qdrant..."
                    deleted = await loop.run_in_executor(
                        None, delete_doc_from_qdrant, _qdrant(), job.doc_id
                    )
                    job.chunks_deleted = deleted

                    job.step = "cleanup"
                    job.progress = "Removing old MD and figures files..."
                    await loop.run_in_executor(None, remove_md_files, job.doc_id)

                    # Step: Re-create from PDF
                    job.step = "pdf_to_md"
                    job.progress = (
                        "Re-converting PDF to Markdown via Azure Document Intelligence + GPT-4 Vision..."
                    )
                    try:
                        md_path = await loop.run_in_executor(None, run_pdf_to_md, pdf_path)
                    except Exception as e:
                        job.status = JobStatus.FAILED
                        job.completed_at = _now_iso()
                        job.error = f"Update failed during PDF→MD: {e}"
                        job.recovery = (
                            f"Old {deleted} chunks were already deleted. "
                            "Use operation='add' to restore the document."
                        )
                        logger.error(f"[update] PDF→MD failed for '{job.doc_id}': {e}")
                        return

                    # Step: Ingest
                    job.step = "md_to_qdrant"
                    job.progress = (
                        "Running LLM page grouping (GPT-4o) → embedding → Qdrant hybrid upsert..."
                    )
                    try:
                        chunks = await loop.run_in_executor(None, run_md_to_qdrant, md_path)
                    except Exception as e:
                        job.status = JobStatus.FAILED
                        job.completed_at = _now_iso()
                        job.error = f"Update failed during MD→Qdrant: {e}"
                        job.recovery = (
                            f"Old {deleted} chunks were deleted. MD file was re-created. "
                            "Use operation='add' to re-ingest."
                        )
                        logger.error(f"[update] MD→Qdrant failed for '{job.doc_id}': {e}")
                        return

                    job.chunks_created = chunks
                    job.status = JobStatus.COMPLETED
                    job.completed_at = _now_iso()
                    job.progress = (
                        f"Updated: removed {deleted} old chunks, created {chunks} new chunks."
                    )

        except Exception as e:
            job.status = JobStatus.FAILED
            job.completed_at = _now_iso()
            job.error = str(e)
            logger.exception(f"Job {job.job_id} ({job.operation}) failed: {e}")


# ===========================================================================
# ENDPOINTS
# ===========================================================================

@app.post("/document", response_model=JobSubmitResponse)
async def submit_document_job(request: DocumentRequest):
    """
    Submit a document ingestion job. Returns immediately with a job_id.
    Poll GET /job/{job_id} for status and result.

    - **add**: PDF → Markdown (multimodal) → LLM-chunk → Qdrant
    - **update**: Delete existing → re-run full add pipeline
    - **delete**: Remove all chunks + MD files
    """
    operation = request.operation.lower().strip()
    filename  = request.filename.strip()
    s3_url    = (request.s3_url or "").strip() or None

    if operation not in ("add", "update", "delete"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid operation '{operation}'. Must be: add | update | delete",
        )

    doc_id = Path(filename).stem

    # For add/update: resolve PDF source
    pdf_path: Optional[Path] = None
    if operation in ("add", "update"):
        if s3_url:
            # S3 mode — validate credentials are configured; download happens in background
            if not S3_ENABLED:
                raise HTTPException(
                    status_code=400,
                    detail="S3 storage not configured. Set S3_SERVICE_URL, S3_BUCKET_NAME, "
                           "S3_ACCESS_KEY, S3_SECRET_KEY in environment.",
                )
            # pdf_path stays None — will be resolved during job execution after download
        else:
            # Local mode — PDF must already exist
            pdf_path = find_pdf(filename)
            if pdf_path is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF '{filename}' not found under {PDF_ROOT}. "
                           "Provide s3_url to download from S3 storage.",
                )

    # Periodic cleanup of expired jobs
    _cleanup_expired_jobs()

    # Create job and start background execution
    job = _create_job(operation, filename, doc_id, s3_url=s3_url)
    task = asyncio.create_task(_execute_job(job, pdf_path))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    logger.info(f"Job {job.job_id} submitted: {operation} '{filename}'")
    return JobSubmitResponse(
        job_id=job.job_id,
        status=job.status.value,
        message=f"Job submitted. Poll GET /job/{job.job_id} for status.",
    )


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Poll job status. Returns full job state including step, progress, and result.

    Statuses: PENDING → PROCESSING → COMPLETED / FAILED
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.to_dict()


@app.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    doc_id: Optional[str] = None,
    limit: int = MAX_JOBS_LIST,
):
    """
    List recent jobs, newest first.
    Optional filters: ?status=PROCESSING&doc_id=ABS...&limit=10
    """
    jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)

    if status:
        status_upper = status.upper()
        jobs = [j for j in jobs if j.status.value == status_upper]

    if doc_id:
        jobs = [j for j in jobs if j.doc_id == doc_id]

    return [j.to_dict() for j in jobs[:limit]]


@app.get("/document/{doc_id}")
def get_document_info(doc_id: str):
    """
    Check Qdrant indexing status for a document.
    Returns chunk count and per-type breakdown.

    Note: sync def — FastAPI runs it in a threadpool, avoiding event-loop blocking.
    """
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

        md_file      = MD_OUT_DIR / f"{doc_id}.md"
        figures_file = MD_OUT_DIR / f"{doc_id}_figures.json"

        return {
            "doc_id":       doc_id,
            "collection":   COLLECTION,
            "total_chunks": count_result.count,
            "chunk_types":  type_counts,
            "indexed":      count_result.count > 0,
            "md_file":      str(md_file) if md_file.exists() else None,
            "figures_file": str(figures_file) if figures_file.exists() else None,
        }

    except Exception as e:
        logger.error(f"Failed to get document info for '{doc_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document info for '{doc_id}'.")


@app.get("/health")
def health():
    """
    Check service + Qdrant connectivity.

    Note: sync def — FastAPI runs it in a threadpool, avoiding event-loop blocking.
    """
    try:
        client = _qdrant()
        info   = client.get_collection(COLLECTION)
        return {
            "status":       "ok",
            "collection":   COLLECTION,
            "total_points": info.points_count,
            "active_jobs":  sum(1 for j in _jobs.values() if j.status in (JobStatus.PENDING, JobStatus.PROCESSING)),
            "total_jobs":   len(_jobs),
            "pdf_root":     str(PDF_ROOT),
            "md_out_dir":   str(MD_OUT_DIR),
        }
    except Exception as e:
        logger.error(f"Health check failed — Qdrant unreachable: {e}")
        return {"status": "degraded", "error": str(e)}


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    # workers=1 mandatory — pipeline modules hold module-level singletons.
    # timeout_keep_alive=300 — background jobs run long but HTTP responses are instant.
    uvicorn.run(
        app, host="0.0.0.0", port=8070,
        log_level="info", workers=1, timeout_keep_alive=300,
    )
