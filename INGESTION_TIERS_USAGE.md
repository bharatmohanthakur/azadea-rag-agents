# Ingestion Tiers — Usage Guide

Four ingestion services run in parallel, each producing chunks at a different quality/cost level.

## Service overview

| Tier | Port | Backend | Collection / Table | What it does |
|---|---|---|---|---|
| **BASIC** | 8071 | Qdrant | `docs_basic_azadea` | Azure DI **OCR only** → semantic chunks |
| **STANDARD** | 8072 | Qdrant | `docs_standard_azadea` | OCR + **GPT-4V** image descriptions → 7 typed chunks |
| **PREMIUM** | 8083 | Qdrant | `docs_premium_azadea` | OCR + GPT-4V + **GPT-4o page grouping** → LLM-grouped typed chunks |
| **OCI** | 8074 | Oracle 26ai | `rag_chunks` | Gemini 2.5 Flash + Cohere v4.0 + Oracle hybrid search |

Production read corpus (DO NOT WRITE TO): `docs_llm_chunked_azadea` (10,216 points, frozen, served by `rag-azure-7867`).

---

## REST contract — identical on all 4 tiers

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/document` | Submit a doc → returns `job_id` |
| `GET`  | `/job/{job_id}` | Poll job state |
| `GET`  | `/jobs?status=PROCESSING&doc_id=<id>&limit=50` | List recent jobs |
| `GET`  | `/document/{doc_id}` | Chunk count + chunk-type breakdown for a doc |
| `GET`  | `/health` | Liveness + collection size + active jobs |

### POST /document — request body

```json
{
  "filename": "<pdf basename>",
  "operation": "add" | "update" | "delete",
  "s3_url": "<optional OCI Object Storage URL>"
}
```

If `s3_url` is omitted, the file must already exist somewhere under `PDF_ROOT`
(`/home/admincsp/multimodal-rag/azadea/data/`). Filename matching is by stem (case-insensitive fallback).

### Operation semantics

| Operation | Behavior |
|---|---|
| `add` | Ingest fresh. **Fails** with `Already indexed with N chunks` if the doc_id already has chunks. Use `update` instead. |
| `update` | Delete existing chunks for this doc_id → re-ingest. Idempotent re-runs. Safe for content edits. |
| `delete` | Remove all chunks for the doc_id from the collection. No re-ingestion. |

### POST /document — response

```json
{
  "job_id": "a4d35483f5e7",
  "status": "PENDING",
  "message": "Poll GET /job/a4d35483f5e7 for status."
}
```

### GET /job/{job_id} — response (in flight)

```json
{
  "job_id": "a4d35483f5e7",
  "status": "PROCESSING",
  "operation": "add",
  "filename": "policy_X.pdf",
  "doc_id": "policy_X",
  "tier": "premium",
  "step": "pdf_to_md",
  "progress": "Azure DI OCR + figures...",
  "created_at": "2026-05-02T12:00:00+00:00",
  "started_at": "2026-05-02T12:00:01+00:00",
  "completed_at": null,
  "chunks_created": 0,
  "chunks_deleted": 0,
  "error": null
}
```

Status transitions: `PENDING` → `PROCESSING` → `COMPLETED` | `FAILED`.

### GET /document/{doc_id} — response

```json
{
  "doc_id": "policy_X",
  "collection": "docs_premium_azadea",
  "tier": "premium",
  "total_chunks": 47,
  "chunk_types": {
    "page_context": 22,
    "table_summary": 8,
    "image_description": 12,
    "control": 3,
    "definition": 1,
    "doc_summary": 1
  },
  "indexed": true
}
```

OCI tier returns the same shape but with `backend: "oracle_26ai"` and `table: "rag_chunks"`.

---

## Latency expectations per tier

Rough end-to-end ingestion time per PDF (10-page document):

| Tier | Cold | Warm | Cost driver |
|---|---|---|---|
| Basic | 10–20 s | 8–15 s | Azure DI OCR only |
| Standard | 30–90 s | 25–60 s | + N × GPT-4V calls (1 per figure) |
| Premium | 60–180 s | 50–120 s | + GPT-4o page grouping (1 call per doc) |
| OCI | 30–60 s | 20–40 s | Gemini 2.5 Flash one-shot |

50+ page docs at Premium can run 3+ minutes — that's why the API is async-job, not synchronous.

---

## End-to-end recipe

```bash
# 1. Submit (Basic = port 8071)
JOB=$(curl -s -X POST http://localhost:8071/document \
  -H "Content-Type: application/json" \
  -d '{"filename":"ACC - NONM - 014 - Fixed Asset Management - P - 1.pdf","operation":"add"}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')

echo "Submitted: $JOB"

# 2. Poll until terminal
while true; do
  STATE=$(curl -s http://localhost:8071/job/$JOB \
    | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["status"],d.get("step",""))')
  echo "  $STATE"
  [[ "$STATE" == COMPLETED* || "$STATE" == FAILED* ]] && break
  sleep 5
done

# 3. Final state
curl -s http://localhost:8071/job/$JOB | python3 -m json.tool

# 4. Verify chunks landed
curl -s "http://localhost:8071/document/ACC - NONM - 014 - Fixed Asset Management - P - 1" \
  | python3 -m json.tool
```

To target a different tier, change the port in all 3 URLs:
- 8071 → BASIC
- 8072 → STANDARD
- 8083 → PREMIUM
- 8074 → OCI

---

## Operations cheatsheet

### Re-ingest a doc you already submitted (e.g., after fixing a broken PDF)
```bash
curl -X POST http://localhost:8083/document \
  -H "Content-Type: application/json" \
  -d '{"filename":"<pdf>","operation":"update"}'
```

### Delete a doc from the corpus
```bash
curl -X POST http://localhost:8071/document \
  -H "Content-Type: application/json" \
  -d '{"filename":"<pdf>","operation":"delete"}'
```

### List recent jobs (any tier)
```bash
curl -s 'http://localhost:8071/jobs?limit=20' | python3 -m json.tool
curl -s 'http://localhost:8071/jobs?status=FAILED' | python3 -m json.tool
```

### Liveness check on all 4 tiers
```bash
for entry in BASIC:8071 STANDARD:8072 PREMIUM:8083 OCI:8074; do
  port=${entry#*:}
  printf "%-9s :%s -> " "${entry%:*}" "$port"
  curl -s -m 3 http://localhost:$port/health | python3 -m json.tool
done
```

---

## Reading the data back (Q&A side)

The ingestion services are **write-only** — they don't answer questions. Each Qdrant collection
needs a paired RAG reader to be queryable.

| Collection | Currently read by |
|---|---|
| `docs_llm_chunked_azadea` | **`rag-azure-7867`** (production reader, frozen corpus) |
| `docs_basic_azadea` | (no reader — collection exists but unused at query time) |
| `docs_standard_azadea` | (no reader) |
| `docs_premium_azadea` | (no reader) |
| Oracle `rag_chunks` | **`rag-oci-7874`** |

To query a tier you just ingested into, either:
1. Point a new RAG process at the matching collection: `QDRANT_COLLECTION=docs_basic_azadea python3 rag_server_llm_chunked.py`
2. Or reconfigure `rag-azure-7867`'s collection env and restart it (will lose access to the production
   corpus until reverted)

---

## Process management (currently NOT systemd-managed)

The 3 tier services were launched as background `nohup` processes. They survive shell exit but
**not server reboot**. Logs go to:

| Tier | Log file |
|---|---|
| Basic | `/tmp/tier_logs/basic.log` |
| Standard | `/tmp/tier_logs/standard.log` |
| Premium | `/tmp/tier_logs/premium.log` |
| OCI | systemd journal (`journalctl -u ingestion-oci-8074`) |

### Find a tier's PID
```bash
sudo lsof -i :8071     # for Basic
sudo lsof -i :8072     # for Standard
sudo lsof -i :8083     # for Premium
```

### Stop a tier
```bash
kill <PID>
# or
pkill -f "ingestion-basic"
```

### Restart a tier
```bash
cd /home/admincsp/graphiti_fixed_test/ingestion-basic    # or -standard, -premium
SERVICE_PORT=8071 nohup /home/admincsp/multimodal-rag/azadea/.venv/bin/python3 service.py \
  > /tmp/tier_logs/basic.log 2>&1 &
```

For Premium, also pass `QDRANT_COLLECTION=docs_premium_azadea` (now also the in-source default,
but explicit is safer).

---

## Choosing a tier — decision matrix

| Use case | Recommended tier |
|---|---|
| Text-heavy policy docs, no diagrams | Basic |
| HR onboarding materials with screenshots/charts | Standard |
| Multi-page workflows where context spans pages (SOPs, checklists) | Premium |
| Cost-sensitive deployment that wants OCI-native stack (no Azure dep) | OCI |
| Benchmarking quality differences across approaches | All four — same source PDF, 4 collections |

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `PDF '<x>' not found under <PDF_ROOT>` | Filename doesn't exist locally and no `s3_url` | Place PDF under `/home/admincsp/multimodal-rag/azadea/data/` or pass `s3_url` |
| `Already indexed with N chunks` | Used `add` on a doc that's already in the collection | Use `operation: "update"` instead |
| `S3 not configured.` | Passed `s3_url` but env vars missing | Verify `S3_*` keys in `/home/admincsp/graphiti_fixed_test/.env` |
| `PDF→MD failed` | Azure DI quota / network / malformed PDF | Check `error` field in job result; re-submit with same operation |
| Job stays `PROCESSING` forever | Service crashed mid-run, in-memory job state lost | Restart the tier; jobs are not persisted across restarts |
