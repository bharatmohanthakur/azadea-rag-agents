# Document Ingestion Service — API Usage Guide

**Base URL:** `http://localhost:8070`
**Pattern:** AWS Textract-style async jobs (submit → poll → get result)

---

## Quick Start

```bash
# 1. Start the service
python3 document_ingestion_service.py

# 2. Add a document
curl -X POST http://localhost:8070/document \
  -H "Content-Type: application/json" \
  -d '{"filename": "My Policy.pdf", "operation": "add"}'

# Response: {"job_id": "a1b2c3d4e5f6", "status": "PENDING", "message": "..."}

# 3. Poll until complete
curl http://localhost:8070/job/a1b2c3d4e5f6

# 4. Verify in Qdrant
curl http://localhost:8070/document/My%20Policy
```

---

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | Azure Document Intelligence endpoint URL |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | Azure Document Intelligence API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | *(none)* | Qdrant API key (if auth enabled) |
| `AZURE_OPENAI_EMBED_DEPLOYMENT` | `text-embedding-3-large` | Embedding model deployment name |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | `gpt-4o` | Chat model for LLM page grouping |
| `AOAI_VISION_DEPLOYMENT` | `gpt-4.1` | Vision model for figure description |
| `MULTIMODAL_IMAGES_DIR` | `<AZADEA_DIR>/images` | Directory for extracted figure images |

---

## Endpoints

### POST /document — Submit a Job

Submit an ingestion job. Returns immediately with a `job_id`.

**Request:**
```json
{
  "filename": "ABS - DMD - 008 - Shipment E-Invoice - G - 1.pdf",
  "operation": "add"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | PDF filename (must exist under the PDF data directory) |
| `operation` | string | One of: `add`, `update`, `delete` |

**Operations:**

| Operation | What it does |
|-----------|-------------|
| `add` | PDF → Markdown (multimodal OCR + figures) → LLM page grouping → embed → Qdrant. Fails if document is already indexed. |
| `update` | Deletes all existing chunks from Qdrant → removes old MD files → re-runs full `add` pipeline from PDF. |
| `delete` | Removes all chunks from Qdrant + deletes MD and figures files. No PDF needed. |

**Response (200):**
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "PENDING",
  "message": "Job submitted. Poll GET /job/a1b2c3d4e5f6 for status."
}
```

**Errors:**

| Status | Reason |
|--------|--------|
| 400 | Invalid operation (not add/update/delete) |
| 404 | PDF file not found (for add/update operations) |

**Examples:**

```bash
# Add a new document
curl -X POST http://localhost:8070/document \
  -H "Content-Type: application/json" \
  -d '{"filename": "HRD - LVE - 001 - Annual Leave - P - 5.pdf", "operation": "add"}'

# Update an existing document (re-ingest from PDF)
curl -X POST http://localhost:8070/document \
  -H "Content-Type: application/json" \
  -d '{"filename": "HRD - LVE - 001 - Annual Leave - P - 5.pdf", "operation": "update"}'

# Delete a document from Qdrant
curl -X POST http://localhost:8070/document \
  -H "Content-Type: application/json" \
  -d '{"filename": "HRD - LVE - 001 - Annual Leave - P - 5.pdf", "operation": "delete"}'
```

---

### GET /job/{job_id} — Poll Job Status

Poll for job progress. Call repeatedly until `status` is `COMPLETED` or `FAILED`.

**Response (200):**
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "PROCESSING",
  "operation": "update",
  "filename": "FNB - QCO - 015 - Food Allergies - B - 4.pdf",
  "doc_id": "FNB - QCO - 015 - Food Allergies - B - 4",
  "step": "pdf_to_md",
  "progress": "Re-converting PDF to Markdown via Azure Document Intelligence + GPT-4 Vision...",
  "created_at": "2026-03-31T05:23:12.105629+00:00",
  "started_at": "2026-03-31T05:23:12.106268+00:00",
  "completed_at": null,
  "chunks_created": 0,
  "chunks_deleted": 3,
  "error": null,
  "recovery": null
}
```

**Job statuses:**

```
PENDING → PROCESSING → COMPLETED
                     → FAILED
```

| Status | Meaning |
|--------|---------|
| `PENDING` | Job queued, waiting to start |
| `PROCESSING` | Pipeline is running — check `step` and `progress` for details |
| `COMPLETED` | Job finished successfully — check `chunks_created` / `chunks_deleted` |
| `FAILED` | Job failed — check `error` for details, `recovery` for next steps |

**Processing steps (in order):**

| Step | Shown during |
|------|-------------|
| `validating` | Checking if document is already indexed (add only) |
| `deleting` | Removing existing chunks from Qdrant (update/delete) |
| `cleanup` | Removing old MD and figures files (update/delete) |
| `pdf_to_md` | Azure Document Intelligence OCR + GPT-4 Vision figure extraction |
| `md_to_qdrant` | LLM page grouping (GPT-4o) → embedding → Qdrant upsert |

**On failure — the `recovery` field:**

When a job fails mid-pipeline (especially during `update`), the `recovery` field tells you what state the document is in and what to do next:

```json
{
  "status": "FAILED",
  "error": "Update failed during PDF→MD: Azure timeout",
  "recovery": "Old 78 chunks were already deleted. Use operation='add' to restore the document."
}
```

**Errors:**

| Status | Reason |
|--------|--------|
| 404 | Job ID not found (expired after 24 hours or never existed) |

**Example polling loop (bash):**

```bash
JOB_ID="a1b2c3d4e5f6"
while true; do
  RESULT=$(curl -s http://localhost:8070/job/$JOB_ID)
  STATUS=$(echo $RESULT | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  echo "Status: $STATUS"
  if [ "$STATUS" = "COMPLETED" ] || [ "$STATUS" = "FAILED" ]; then
    echo $RESULT | python3 -m json.tool
    break
  fi
  sleep 3
done
```

**Example polling loop (Python):**

```python
import requests, time

job_id = "a1b2c3d4e5f6"
while True:
    resp = requests.get(f"http://localhost:8070/job/{job_id}").json()
    print(f"[{resp['step']}] {resp['progress']}")
    if resp["status"] in ("COMPLETED", "FAILED"):
        break
    time.sleep(3)

if resp["status"] == "COMPLETED":
    print(f"Done! Created {resp['chunks_created']} chunks.")
else:
    print(f"Error: {resp['error']}")
    if resp.get("recovery"):
        print(f"Recovery: {resp['recovery']}")
```

---

### GET /jobs — List Jobs

List recent jobs, newest first. Optional filters.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | string | *(all)* | Filter by status: `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED` |
| `doc_id` | string | *(all)* | Filter by document ID (PDF stem) |
| `limit` | int | 100 | Max jobs to return |

**Examples:**

```bash
# All jobs
curl http://localhost:8070/jobs

# Only running jobs
curl "http://localhost:8070/jobs?status=PROCESSING"

# Jobs for a specific document
curl "http://localhost:8070/jobs?doc_id=HRD%20-%20LVE%20-%20001%20-%20Annual%20Leave%20-%20P%20-%205"

# Last 5 failed jobs
curl "http://localhost:8070/jobs?status=FAILED&limit=5"
```

**Response (200):**
```json
[
  {
    "job_id": "a1b2c3d4e5f6",
    "status": "COMPLETED",
    "operation": "add",
    "filename": "...",
    "doc_id": "...",
    ...
  }
]
```

---

### GET /document/{doc_id} — Document Info

Check Qdrant indexing status for a document. The `doc_id` is the PDF filename without `.pdf` extension.

**Example:**
```bash
# URL-encode spaces as %20
curl "http://localhost:8070/document/FNB%20-%20QCO%20-%20015%20-%20Food%20Allergies%20-%20B%20-%204"
```

**Response (200):**
```json
{
  "doc_id": "FNB - QCO - 015 - Food Allergies - B - 4",
  "collection": "docs_llm_chunked_azadea",
  "total_chunks": 3,
  "chunk_types": {
    "page_context": 2,
    "doc_summary": 1
  },
  "indexed": true,
  "md_file": "/home/admincsp/multimodal-rag/azadea/md_out_data_multimodal/FNB - QCO - 015 - Food Allergies - B - 4.md",
  "figures_file": null
}
```

**Chunk types you may see:**

| Type | Description |
|------|-------------|
| `image_description` | GPT-4 Vision descriptions of figures/diagrams in the PDF |
| `ocr_detail` | OCR-extracted text from figure tags |
| `control` | Control points extracted from the document |
| `definition` | Definitions and notes |
| `table_summary` | Table summaries (LLM-generated text for search + raw markdown in `full_table` for answers) |
| `page_context` | LLM-grouped page content (GPT-4o decides which pages belong together topically) |
| `doc_summary` | Document-level summary with metadata |

---

### GET /health — Health Check

Check service status and Qdrant connectivity.

**Example:**
```bash
curl http://localhost:8070/health
```

**Response (200 — healthy):**
```json
{
  "status": "ok",
  "collection": "docs_llm_chunked_azadea",
  "total_points": 10194,
  "active_jobs": 0,
  "total_jobs": 1,
  "pdf_root": "/home/admincsp/multimodal-rag/azadea/data",
  "md_out_dir": "/home/admincsp/multimodal-rag/azadea/md_out_data_multimodal"
}
```

**Response (200 — degraded):**
```json
{
  "status": "degraded",
  "error": "Connection refused"
}
```

---

## Pipeline Architecture

```
PDF file
  │
  ├─ Step 1: Azure Document Intelligence (prebuilt-layout)
  │    ├─ OCR with HIGH_RESOLUTION
  │    ├─ MARKDOWN output format
  │    └─ FIGURES extraction
  │
  ├─ Step 2: GPT-4 Vision (figure description)
  │    ├─ Each figure → image bytes → GPT-4V description
  │    ├─ Descriptions appended to markdown
  │    └─ Figure metadata saved as _figures.json
  │
  └─ Step 3: LLM Semantic Chunking (GPT-4o page grouping)
       │
       ├─ 1. image_description — from _figures.json
       ├─ 2. ocr_detail — OCR text from <figure> tags
       ├─ 3. control — regex-extracted control points
       ├─ 4. definition — regex-extracted definitions/notes
       ├─ 5. table_summary — LLM summary (for search) + full markdown table (for answers)
       ├─ 6. page_context — LLM groups pages by topic (GPT-4o)
       ├─ 7. doc_summary — document-level summary
       │
       └─ Hybrid embed + upsert to Qdrant
            ├─ Dense: Azure text-embedding-3-large
            ├─ Sparse: BM25 via fastembed
            └─ Server-side RRF fusion
```

---

## Concurrency and Constraints

- **workers=1** — Required because pipeline modules use module-level singletons (BM25 model, Azure clients).
- **Per-document locking** — Concurrent operations on the same document are serialized. Different documents can process in parallel.
- **Job TTL** — Jobs expire after 24 hours and are cleaned up automatically.
- **Background execution** — Jobs run as asyncio background tasks. The HTTP response returns immediately.

---

## Common Workflows

### Bulk re-index all documents

```bash
# List all PDFs
find /home/admincsp/multimodal-rag/azadea/data -name "*.pdf" | while read pdf; do
  FILENAME=$(basename "$pdf")
  echo "Submitting: $FILENAME"
  curl -s -X POST http://localhost:8070/document \
    -H "Content-Type: application/json" \
    -d "{\"filename\": \"$FILENAME\", \"operation\": \"update\"}" | python3 -c "import json,sys; j=json.load(sys.stdin); print(f'  job_id: {j[\"job_id\"]}')"
  sleep 1  # avoid overwhelming the service
done
```

### Check for failed jobs

```bash
curl -s "http://localhost:8070/jobs?status=FAILED" | python3 -c "
import json, sys
jobs = json.load(sys.stdin)
for j in jobs:
    print(f'{j[\"filename\"]}')
    print(f'  Error: {j[\"error\"]}')
    if j.get('recovery'):
        print(f'  Recovery: {j[\"recovery\"]}')
    print()
"
```

### Verify a document has all expected chunk types

```bash
DOC_ID="ABS - DMD - 008 - Shipment E-Invoice - G - 1"
curl -s "http://localhost:8070/document/$(python3 -c "import urllib.parse; print(urllib.parse.quote('$DOC_ID'))")" | python3 -c "
import json, sys
doc = json.load(sys.stdin)
print(f'Document: {doc[\"doc_id\"]}')
print(f'Total chunks: {doc[\"total_chunks\"]}')
print(f'Chunk types:')
for ct, count in sorted(doc['chunk_types'].items()):
    print(f'  {ct}: {count}')
if not doc['indexed']:
    print('WARNING: Document is not indexed!')
"
```
