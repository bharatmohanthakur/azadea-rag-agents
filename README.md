# Azadea RAG Agents

Production tool-calling RAG agents that answer Azadea Group employees' questions
about internal policies and procedures (HR, Finance, IT, Operations, Stock, F&B,
and more). The assistant persona is **Dea**.

Two parallel serving tiers run the same agent design over the same knowledge base,
differing only in LLM provider and a few capabilities:

| Tier  | Port | Model                              | Provider   | Streaming |
|-------|------|------------------------------------|------------|-----------|
| Azure | 7867 | `anthropic/claude-sonnet-4.6`      | OpenRouter | real token-by-token |
| OCI   | 7874 | `google.gemini-2.5-pro`            | OCI GenAI  | simulated (Gemini buffers) |

Both read from a shared **Qdrant** vector store and **Redis** conversation store.

---

## Architecture

```
          ┌─────────────── BrainShift Companion (UI / platform) ───────────────┐
          │   sends: User Context block + Request                              │
          └───────────────┬───────────────────────────┬───────────────────────┘
                          │                           │
                  POST /query/stream           POST /query/stream
                          ▼                           ▼
              ┌───────────────────┐         ┌───────────────────┐
              │  Azure tier :7867 │         │  OCI tier :7874   │
              │  Claude Sonnet 4.6│         │  Gemini 2.5 Pro   │
              │  (OpenRouter)     │         │  (OCI GenAI)      │
              └─────────┬─────────┘         └─────────┬─────────┘
                        │   tool-calling agent loop   │
                        ▼                             ▼
        ┌───────────────────────────┐   ┌───────────────────────────┐
        │ Qdrant docs_llm_chunked_  │   │ Qdrant docs_oci_ingested_ │
        │ azadea (Azure embeddings) │   │ azadea (OCI Cohere v4)    │
        └───────────────────────────┘   └───────────────────────────┘
                        └──────────── Redis (conversation + profile) ┘
```

### The agent loop
Each turn the model may call tools, then synthesises a grounded answer:

- `get_document_knowledge(query)` — Qdrant **hybrid** search (dense + sparse BM25,
  RRF fusion) with **neighbor table-summary expansion** (pulls a doc's full tables
  when any chunk of it is retrieved).
- `get_history(limit)` — recent conversation messages.
- `get_user_profile()` / `save_user_profile(attributes)` — per-user attributes
  (role, country, brand, …) baked into the system prompt for personalisation.
- `get_clarification(...)` — optional second-opinion specialist, gated by
  `AUTO_CLARIFY` (default **off**: the main model handles clarification inline).

History and the user profile are injected into the **system prompt** every turn,
so the model never has to call a tool just to know who the user is.

---

## Components

| Service                  | Port | Role                                              |
|--------------------------|------|---------------------------------------------------|
| `rag-azure-7867`         | 7867 | Azure tier chatbot (Claude Sonnet 4.6 + streaming)|
| `rag-oci-7874`           | 7874 | OCI tier chatbot (Gemini 2.5 Pro)                 |
| `ingestion-oci-8074`     | 8074 | OCI ingestion → Qdrant (`docs_oci_ingested_azadea`)|
| `ingestion-azure-8070`   | 8070 | Azure ingestion → Qdrant (`docs_llm_chunked_azadea`)|
| Qdrant                   | 6333 | vector store (dense + sparse)                     |
| Redis                    | 6379 | conversation + user profile store                 |

---

## Endpoints

### Chatbot (`:7867`, `:7874`)
- `POST /query` — synchronous; returns the final answer.
- `POST /query/stream` — **SSE** stream. Events: `status` → `source_found` →
  `progress` → `token` → `done` (`error` on failure). This is what production uses.
- `GET /health`

Request body: `{ "query": "<User Context block + Request>", "user_id": "<id>" }`

### Ingestion (`:8074` OCI)
`POST /document`:
```json
{
  "filename": "FDR - STR - 002 - Merchandise Exchange and Refund - B - 36.pdf",
  "operation": "add | update | delete",
  "file_id":  "<BrainShift Companion file id>",   // optional — fetch via download API
  "s3_url":   "<s3 url>"                           // optional — fetch from S3
}
```
Source resolution order: `file_id` → `s3_url` → local PDF on disk. Returns a
`job_id`; poll `GET /job/{job_id}`. Pipeline: PDF → Markdown (Gemini vision) →
typed chunking → OCI Cohere Embed v4 → Qdrant (dense + sparse).

---

## Retrieval notes

- **Tables are embedded with their values.** Table chunks embed the structural
  summary **plus the full markdown table** (brand/country/period rows), so a query
  like "Zara refund period" matches the actual `Zara | All Countries | 30 Days`
  row. (Embedding the summary alone left table values unsearchable.)
- **Hybrid + neighbor expansion** is what surfaces the right policy: dense finds
  semantically-similar prose, sparse matches exact terms, and neighbor expansion
  attaches a retrieved document's tables to the context.

---

## Configuration

All secrets come from the environment (a gitignored `.env`) — **never** hardcoded.
Required keys (see `ingestion-oci/.env.example`):

```
# LLM providers
OPENROUTER_API_KEY=...
OPENROUTER_MODEL_FAST=anthropic/claude-sonnet-4.6   # Azure tier model
OCI_CHAT_MODEL=google.gemini-2.5-pro                # OCI tier model
AZURE_OPENAI_API_KEY=...                            # Azure embeddings

# OCI identity (for OCI GenAI + Cohere embeddings)
OCI_USER=...  OCI_FINGERPRINT=...  OCI_TENANCY=...
OCI_COMPARTMENT_ID=...  OCI_KEY_FILE=/path/to/oci_api_key.pem

# Oracle 26ai (legacy/fallback store)
ORACLE_DB_USER=...  ORACLE_DB_PASSWORD=...  ORACLE_WALLET_PASSWORD=...

# Object storage (source PDFs)
S3_SERVICE_URL=...  S3_BUCKET_NAME=...  S3_ACCESS_KEY=...  S3_SECRET_KEY=...

# Behaviour flags
REAL_STREAM=1        # Azure: real token streaming on /query/stream
AUTO_CLARIFY=0       # off: main model handles clarification inline
OCI_REASONING_EFFORT=LOW
```

Services are managed by systemd (`rag-azure-7867`, `rag-oci-7874`,
`ingestion-oci-8074`, `ingestion-azure-8070`).

---

## Repository layout

```
rag_server_azure_tools.py     # Azure tier agent (FastAPI, port 7867)
agent_tools_azure.py          # Azure tool implementations + Qdrant retrieval
ingestion-oci/
  rag_server_oci_tools.py     # OCI tier agent (port 7874)
  agent_tools.py              # OCI tool implementations + retrieval
  oci_clients.py              # OCI GenAI client factories (env-only config)
  oci_chat.py                 # OCI chat / tool-calling / streaming wrappers
  ingest_pipeline.py          # PDF→MD→chunk→embed→Qdrant ingestion
  service.py                  # ingestion HTTP service (port 8074)
  qdrant_utils.py             # hybrid search, sparse vectors, upsert helpers
```

> Internal Azadea project. Credentials are environment-only; do not commit `.env`,
> `*.pem`, the Oracle wallet, or OCI config files.
