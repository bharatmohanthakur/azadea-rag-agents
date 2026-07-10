"""
Tool definitions and dispatch — OCI-RETRIEVAL variant for the native-Anthropic
agent (rag_server_oci_anthropic, port 7875).

Copy of agent_tools_azure_rerank.py with ONLY the retrieval plumbing swapped to
the OCI tier:
  - Qdrant collection: docs_oci_ingested_azadea (OCI-ingested corpus)
  - Dense query embeddings: OCI Cohere Embed v4.0 (1536-dim) via embed_query_oci
  - Sparse: the OCI tier's fastembed BM25 builder (matches ingestion)
  - list_documents: filters by doc_id prefix (the OCI payload has no `domain` field)

Everything else is untouched: the OpenRouter cross-encoder reranker
(cohere/rerank-4-fast), neighbor table expansion, tool schemas, dispatch, and
the Redis conv_manager. Does NOT touch any running service.
"""

import hashlib
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

# OCI identity must be in the environment before importing oci_pipeline
# (oci_clients raises at import when OCI_USER etc. are missing). The secrets
# live in the root .env.
load_dotenv(Path(__file__).parent / ".env")

# Reuse existing singletons (these are already initialized in rag_server_gemini)
from qdrant_client import QdrantClient, models as qm
from rag_server_gemini import conv_manager, openrouter_client

# OCI retrieval stack lives under ingestion-oci/ — appended (not prepended) to
# sys.path so those modules can't shadow root-level ones.
sys.path.append(str(Path(__file__).parent / "ingestion-oci"))
from oci_pipeline import embed_query_oci
import qdrant_utils as oci_qdrant_utils

# Model used by the get_clarification specialist tool. Pro tier (more
# capable than the Flash model used in the main agent loop) — only fires
# when the agent suspects retrieval ambiguity, so cost stays bounded.
CLARIFICATION_MODEL = os.getenv("CLARIFICATION_MODEL", "google/gemini-2.5-pro")

# ── Reranker config ─────────────────────────────────────────────────────────
# After hybrid search + neighbor expansion, score every candidate chunk against
# the query with a cross-encoder and keep the top RERANK_TOP_N. RERANK=0 turns
# it off (falls back to the plain hybrid result, identical to the base agent).
RERANK = os.getenv("RERANK", "1") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cohere/rerank-4-fast")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "8"))
RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "8"))
# Neighbor-expanded tables are pulled precisely because they hold answer values
# yet score low on prose similarity. Keep them out of the cut by default so the
# reranker can shrink the prose chunks without discarding answer tables.
RERANK_PROTECT_NEIGHBORS = os.getenv("RERANK_PROTECT_NEIGHBORS", "1") == "1"

logger = logging.getLogger("agent_tools_oci_claude")


def _rerank_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reorder/trim hybrid-search candidates by query relevance via OpenRouter's
    rerank endpoint. The document text given to the reranker is the chunk's
    full_table when present (table answers live there), else its text. On any
    failure we log and return the original list unchanged — retrieval must never
    break because the reranker hiccuped."""
    if not RERANK or len(chunks) <= 1:
        return chunks

    # Optionally hold neighbor-expanded tables aside so they survive the cut.
    if RERANK_PROTECT_NEIGHBORS:
        protected = [c for c in chunks if c.get("from_neighbor_expansion")]
        candidates = [c for c in chunks if not c.get("from_neighbor_expansion")]
    else:
        protected, candidates = [], list(chunks)
    if len(candidates) <= 1:
        return chunks

    docs = [(c.get("full_table") or c.get("text") or "") for c in candidates]
    try:
        resp = requests.post(
            f"{str(openrouter_client.base_url).rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {openrouter_client.api_key}",
                     "Content-Type": "application/json"},
            json={"model": RERANK_MODEL, "query": query, "documents": docs,
                  "top_n": min(RERANK_TOP_N, len(docs))},
            timeout=RERANK_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as e:
        logger.warning(f"rerank failed ({e}); using hybrid order")
        return chunks

    if not results:
        return chunks

    reranked = []
    for r in results:
        idx = r.get("index")
        if idx is None or idx >= len(candidates):
            continue
        c = dict(candidates[idx])
        c["rerank_score"] = round(r.get("relevance_score", 0.0), 4)
        reranked.append(c)

    kept = len(reranked)
    logger.info(f"rerank: {len(candidates)} prose candidates → kept {kept} "
                f"(+{len(protected)} protected neighbor tables)")
    # Reranked prose first (relevance order), then any protected neighbor tables.
    return reranked + protected

# Local Qdrant client + the OCI-ingested collection (Cohere v4 vectors, 1536-dim)
QDRANT_LOCAL_URL = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
COLLECTION_NAME_V2 = os.getenv("OCI_QDRANT_COLLECTION", "docs_oci_ingested_azadea")
qdrant_client = QdrantClient(url=QDRANT_LOCAL_URL, check_compatibility=False)


# Embedding cache disabled — every query embeds fresh. OCI Cohere Embed v4.0
# with SEARCH_QUERY input type (must match the collection's ingestion model).
def _cached_embed_dense(text: str) -> List[float]:
    return embed_query_oci(text)


# ── Source-file extension recovery ────────────────────────────────────────────
# Chunks store source_file as the internal markdown name ("X.md"). The frontend
# wants the ORIGINAL document name ("X.pdf" / "X.docx"). Resolve the real
# extension from disk (majority PDF, ~50 DOCX). Cached per name.
from functools import lru_cache

_SRC_ROOT = Path(os.getenv("PDF_ROOT", "/home/admincsp/multimodal-rag/azadea/data"))
_SRC_EXTS = (".pdf", ".docx", ".txt")


def _norm_src_stem(s: str) -> str:
    return " ".join(str(s).replace("–", "-").replace("—", "-").split()).casefold()


@lru_cache(maxsize=1)
def _source_ext_index() -> Dict[str, str]:
    """One-time index {normalized-stem: real-filename} over the source corpus."""
    idx: Dict[str, str] = {}
    try:
        for p in _SRC_ROOT.rglob("*"):
            if p.suffix.lower() in _SRC_EXTS:
                idx.setdefault(_norm_src_stem(p.stem), p.name)
    except Exception as e:
        logger.warning(f"source-ext index build failed: {e}")
    return idx


@lru_cache(maxsize=8192)
def _real_source_name(source_file: str) -> str:
    """Map a stored source_file to the original document name with its true
    extension. 'X.md' → 'X.pdf'/'X.docx'; non-.md names pass through unchanged.
    Falls back to '<stem>.pdf' when no source is found on disk (majority PDF)."""
    if not source_file:
        return source_file
    ext = Path(source_file).suffix.lower()
    if ext and ext != ".md":
        return source_file
    stem = Path(source_file).stem
    return _source_ext_index().get(_norm_src_stem(stem), f"{stem}.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas (OpenAI SDK format — used by openrouter_client.chat.completions.create)
# ─────────────────────────────────────────────────────────────────────────────

TOOLS_OPENAI: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_history",
            "description": (
                "Fetch the recent conversation history (user and assistant messages) "
                "for the current user. Use this when the user's current message contains "
                "pronouns or ellipsis or otherwise depends on prior turns to be interpreted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of most recent messages to return. Defaults to 10.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_knowledge",
            "description": (
                "Search the Azadea corporate knowledge base for relevant policy, procedure, "
                "or SOP documents. Use this for any documented company process, policy, "
                "or SOP question. The query must be a STANDALONE, self-contained phrase — "
                "resolve any pronouns or ellipsis from prior turns first (call get_history "
                "if needed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A clear, standalone search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of document chunks to retrieve (default 7).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": (
                "Retrieve stored profile attributes for the current user (e.g. role, country, "
                "brand, department, employment_type). Returns an empty object if nothing is "
                "stored yet. Call this before answering any policy question whose answer may "
                "depend on the user's role, country, brand, or similar attribute — so you can "
                "personalise the response without asking the user again."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_profile",
            "description": (
                "Store profile attributes the user has just disclosed about themselves so they "
                "are remembered across turns. Pass only attributes the user has explicitly "
                "shared in this conversation (e.g. their role, country, brand, department, "
                "employment type). Existing attributes are merged — pass only what is new or "
                "changed. Do not infer or guess attributes the user did not state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "attributes": {
                        "type": "object",
                        "description": "Map of attribute name → value (e.g. {\"role\": \"shop_manager\", \"country\": \"KSA\"}).",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["attributes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "List ALL policy/procedure documents in a category — a complete catalog, "
                "not a search. Use this when the user asks to 'list / share / give me all "
                "the X policies' or 'all the documents/links related to X' (e.g. all IT "
                "policies, all HR policies). Returns every document name in that category. "
                "For a specific question about what a policy SAYS, use get_document_knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The policy area, e.g. 'IT', 'HR', 'Finance', 'Sales', 'F&B', 'Stock'.",
                    },
                },
                "required": ["category"],
            },
        },
    },
    # NOTE: get_clarification tool exists in _DISPATCH but is temporarily
    # NOT registered in TOOLS_OPENAI — Flash appeared to choke (empty
    # response) when the 5-tool schema was exposed. Re-enable when the
    # cause is diagnosed; the implementation is verified working via
    # direct-call smoke test.
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool execution
# ─────────────────────────────────────────────────────────────────────────────

def _tool_get_history(args: Dict[str, Any], user_id: str) -> str:
    limit = int(args.get("limit") or 10)
    msgs = conv_manager.get_history(user_id, limit=limit)
    out = []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")
        # Strip the upstream User Context wrapper if present so the agent sees the
        # actual user-typed text (not the wrapping metadata)
        if "Request:" in content:
            content = content.split("Request:", 1)[-1].lstrip("\r\n").strip()
        out.append({"role": role, "content": content})
    return json.dumps(out, ensure_ascii=False)


def _hybrid_points(query: str, limit: int):
    """One dense+sparse RRF hybrid search for `query`. Returns a (possibly empty)
    list of scored points; raises on embedding/search failure."""
    dense_q = _cached_embed_dense(query)
    sparse_q = oci_qdrant_utils.build_sparse_query_vector(query)
    result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME_V2,
        prefetch=[
            qm.Prefetch(query=dense_q, using="dense", limit=20),
            qm.Prefetch(query=sparse_q, using="sparse", limit=20),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=limit,
    )
    return list(result.points) if result and result.points else []


def _tool_get_document_knowledge(args: Dict[str, Any], user_id: str) -> str:
    """Dense + sparse hybrid via Qdrant + neighbor table expansion + full_table
    content. No content truncation.

    Rewrite-robust retrieval: the agent composes its own search query, and a
    rewrite can lose recall (e.g. appending 'policy' pulls policy-titled docs and
    drops the doc that actually defines the user's term). So when the server
    passes the user's ORIGINAL question (`_user_query`, injected server-side —
    not part of the model-facing schema), we ALSO search with the user's words
    and merge the candidate pools (recall widening only). The reranker anchor
    stays the MODEL's query — on follow-ups the raw user text is a context-free
    fragment, while the model's rewrite carries the resolved context — so the
    cross-encoder keeps what's relevant and cuts the widening noise. Generic by
    design: no term/keyword special-casing; final context size unchanged."""
    query = (args.get("query") or "").strip()
    user_query = (args.get("_user_query") or "").strip()
    top_k = int(args.get("top_k") or 7)
    if not query:
        return json.dumps({"error": "query is required"})

    try:
        points = _hybrid_points(query, top_k + 3)
    except Exception as e:
        logger.warning(f"retrieval failed for model query: {e}")
        return json.dumps({"error": f"retrieval failed: {e}"})

    # Second pass with the user's own words when they differ from the model's
    # rewrite — merged by point id, so candidates only widen, never duplicate.
    if user_query and user_query.casefold() != query.casefold():
        try:
            extra = _hybrid_points(user_query, top_k + 3)
            seen_ids = {p.id for p in points}
            added = [p for p in extra if p.id not in seen_ids]
            if added:
                logger.info(f"user-query pass added {len(added)} candidate(s) "
                            f"beyond the model's rewrite")
            points = points + added
        except Exception as e:
            logger.warning(f"user-query retrieval pass failed (non-fatal): {e}")

    if not points:
        return json.dumps({"results": [], "note": "no relevant documents found in the knowledge base"})

    # Cap the pool: both passes contribute up to top_k each; the reranker
    # downstream still keeps only RERANK_TOP_N, so context size is unchanged.
    pool_cap = top_k * 2
    chunks = []
    doc_ids_seen = set()
    table_keys_seen = set()

    for p in points[:pool_cap]:
        pl = p.payload or {}
        chunk_type = pl.get("chunk_type", "")
        doc_id = pl.get("doc_id", "")
        full_table = pl.get("full_table", "") or ""
        text = pl.get("text", "") or ""
        source_file = pl.get("source_file", "")
        page = pl.get("page")
        if doc_id:
            doc_ids_seen.add(doc_id)

        # Prepend an explicit source header so the LLM sees the filename right
        # at the start of the chunk's content — makes citation natural and
        # avoids cases where the LLM skims past the structured `source` field
        # in the surrounding JSON metadata.
        header = f"[Source: {_real_source_name(source_file)} | Page: {page}]\n" if source_file else ""

        entry = {
            "source": _real_source_name(source_file),
            "doc_id": doc_id,
            "score": round(p.score or 0.0, 3),
            "chunk_type": chunk_type,
            "page": page,
        }
        if chunk_type == "table_summary" and full_table:
            entry["full_table"] = header + full_table   # complete, no truncation
            entry["text"] = header + text                # complete summary
            table_keys_seen.add(full_table)
        else:
            entry["text"] = header + text                # complete chunk text, no truncation
        chunks.append(entry)

    # Neighbor table expansion — fetch additional table_summary chunks for each retrieved doc
    for doc_id in list(doc_ids_seen):
        try:
            scroll_result, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME_V2,
                scroll_filter=qm.Filter(must=[
                    qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
                    qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value="table_summary")),
                ]),
                limit=10,
                with_payload=qm.PayloadSelectorInclude(
                    include=["text", "full_table", "source_file", "chunk_type", "page"]
                ),
                with_vectors=False,
            )
        except Exception:
            scroll_result = []

        for tp in (scroll_result or []):
            tpl = tp.payload or {}
            ft = tpl.get("full_table", "") or ""
            if not ft or ft in table_keys_seen:
                continue
            # Note: we intentionally omit the `score` field for neighbor-pulled
            # chunks. Earlier we sent `score: None`, but that signalled
            # "unverified" to Gemini and it down-weighted these chunks during
            # synthesis — even when the neighbor table contained the specific
            # answer (e.g. the 20%/35% Decathlon rates). Removing the field
            # entirely makes neighbor chunks indistinguishable from retrieved
            # ones in terms of trust signal.
            n_src = tpl.get("source_file", "")
            n_page = tpl.get("page")
            n_header = f"[Source: {_real_source_name(n_src)} | Page: {n_page}]\n" if n_src else ""
            chunks.append({
                "source": _real_source_name(n_src),
                "doc_id": doc_id,
                "chunk_type": "table_summary",
                "full_table": n_header + ft,            # complete, no truncation
                "text": n_header + (tpl.get("text") or ""),
                "page": n_page,
                "from_neighbor_expansion": True,
            })
            table_keys_seen.add(ft)

    # Rerank the combined candidate set (hybrid hits + neighbor tables) and keep
    # only the most relevant, trimming context before it reaches the LLM.
    # Rerank against the MODEL's query — on follow-ups the raw user text can be
    # a context-free fragment ("how do I apply for it?"), while the model's
    # rewrite carries the resolved context. The user-query pass above is used
    # ONLY to widen the candidate pool; the cross-encoder then keeps whatever is
    # genuinely relevant (a literal-term chunk like "Paid Off" scores high even
    # against the rewrite) and cuts the noise the extra pass brought in.
    chunks = _rerank_chunks(query, chunks)

    return json.dumps({"results": chunks}, ensure_ascii=False)


def _tool_get_user_profile(args: Dict[str, Any], user_id: str) -> str:
    try:
        profile = conv_manager.get_user_profile(user_id) or {}
        return json.dumps(profile, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"get_user_profile failed: {e}")
        return json.dumps({})


def _tool_save_user_profile(args: Dict[str, Any], user_id: str) -> str:
    attrs = args.get("attributes") or {}
    if not isinstance(attrs, dict) or not attrs:
        return json.dumps({"saved": False, "reason": "no attributes provided"})
    # coerce all values to strings — Redis hash stores strings
    cleaned = {str(k): str(v) for k, v in attrs.items() if v is not None and str(v).strip()}
    if not cleaned:
        return json.dumps({"saved": False, "reason": "no non-empty attributes"})
    try:
        conv_manager.update_user_profile(user_id, cleaned)
        return json.dumps({"saved": True, "attributes": cleaned})
    except Exception as e:
        logger.warning(f"save_user_profile failed: {e}")
        return json.dumps({"saved": False, "reason": f"error: {e}"})


_CLARIFICATION_SYSTEM_PROMPT = """You are a query-disambiguation analyst for the
Dea knowledge assistant. A faster model (Flash) has just retrieved
some policy documents in response to a user's question. Your job is to decide
whether those documents are sufficient to answer the user confidently, or
whether one or more clarifying questions should be asked first.

Output STRICT JSON with three fields, no other text:
{
  "needs_clarification": true | false,
  "rationale": "one short sentence explaining your decision",
  "questions": ["question 1", "question 2", ...]
}

Rules:
- Set needs_clarification=true only when the retrieved documents reveal that
  the correct answer would change meaningfully depending on attributes the user
  has NOT provided (and are NOT in their stored profile), or when the documents
  contain multiple distinct policy variants that could each apply.
- If needs_clarification=false, set questions=[] (empty array).
- If true, list 1–4 short, targeted questions that would resolve the ambiguity.
- Do NOT ask for context the documents do not actually require.
- Profile already provided (use as authoritative): pre-existing attributes.
"""


def _tool_get_clarification(args: Dict[str, Any], user_id: str) -> str:
    """Call Pro to decide whether clarifying questions are needed AFTER retrieval.
    Receives the original query + the retrieved chunks JSON; consults the user's
    stored profile so it doesn't ask for attributes the system already has."""
    query = (args.get("query") or "").strip()
    chunks_json = args.get("retrieved_chunks_json") or ""
    if not query:
        return json.dumps({"needs_clarification": False, "rationale": "no query supplied", "questions": []})

    try:
        profile = conv_manager.get_user_profile(user_id) or {}
    except Exception:
        profile = {}

    user_block = (
        f"User question: {query}\n\n"
        f"Stored user profile (already known — do NOT ask for these): {json.dumps(profile, ensure_ascii=False)}\n\n"
        f"Retrieved documents (JSON, full chunks):\n{chunks_json}"
    )

    try:
        resp = openrouter_client.chat.completions.create(
            model=CLARIFICATION_MODEL,
            messages=[
                {"role": "system", "content": _CLARIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_block},
            ],
            temperature=0.2,
            # Gemini 2.5 Pro is a thinking model. Cap thinking at ~20% of
            # max_tokens via reasoning.effort=low so the bulk of the budget
            # is available for the JSON output. This also cuts latency
            # roughly in half compared to default thinking depth.
            extra_body={"reasoning": {"effort": "low"}},
            max_tokens=4000,
            timeout=60.0,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        # Strip code fences if model wrapped output
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
        parsed = json.loads(content)
        # Guarantee schema
        out = {
            "needs_clarification": bool(parsed.get("needs_clarification", False)),
            "rationale": str(parsed.get("rationale", ""))[:500],
            "questions": [str(q) for q in (parsed.get("questions") or []) if str(q).strip()][:4],
        }
        logger.info(f"get_clarification: needs={out['needs_clarification']}, n_questions={len(out['questions'])}")
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        logger.exception(f"get_clarification failed: {e}")
        # Fail-open: don't block the agent; let it answer with what it has
        return json.dumps({
            "needs_clarification": False,
            "rationale": f"clarification check failed ({type(e).__name__}); proceeding without it",
            "questions": [],
        })


_DOMAIN_ALIASES = {
    "it": "ITD", "information technology": "ITD", "itd": "ITD",
    "hr": "HRD", "human resources": "HRD", "hrd": "HRD",
    "finance": "ACC", "accounting": "ACC", "acc": "ACC", "accounts": "ACC",
    "sales": "SALES",
    "f&b": "FNB", "food and beverage": "FNB", "food": "FNB", "fnb": "FNB",
    "stock": "SMD", "stock management": "SMD", "smd": "SMD",
    "operations": "OPS", "ops": "OPS",
    "audit": "AUD", "inventory": "INV", "logistics": "LOX",
    "absher": "ABS", "abs": "ABS", "fdr": "FDR",
}


def _tool_list_documents(args: Dict[str, Any], user_id: str) -> str:
    """Complete catalog of documents in a category (metadata listing, not search).

    The OCI collection's payload has NO `domain` field (unlike the Azure one),
    so instead of a server-side filter we scroll the whole collection once and
    match the document-code prefix of doc_id/source_file (e.g. 'HRD', 'ITD') —
    the corpus naming convention puts the domain code first."""
    cat = (args.get("category") or args.get("domain") or "").strip().lower()
    code = _DOMAIN_ALIASES.get(cat, cat.upper())
    seen, offset = set(), None
    try:
        while True:
            res, offset = qdrant_client.scroll(
                COLLECTION_NAME_V2,
                limit=1000, offset=offset,
                with_payload=["source_file", "doc_id"], with_vectors=False,
            )
            for p in res:
                pl = p.payload or {}
                sf = pl.get("source_file", "") or pl.get("doc_id", "")
                if not sf:
                    continue
                name = sf[:-3] if sf.endswith(".md") else sf
                # match the leading domain code (before the first separator)
                head = name.replace("–", "-").split("-", 1)[0].strip().upper()
                if head == code:
                    seen.add(name)
            if offset is None:
                break
    except Exception as e:
        return json.dumps({"category": code, "count": 0, "documents": [], "error": str(e)})
    docs = sorted(seen)
    return json.dumps({"category": code, "count": len(docs), "documents": docs})


_DISPATCH = {
    "get_history": _tool_get_history,
    "get_document_knowledge": _tool_get_document_knowledge,
    "get_user_profile": _tool_get_user_profile,
    "save_user_profile": _tool_save_user_profile,
    "get_clarification": _tool_get_clarification,
    "list_documents": _tool_list_documents,
}


def execute_tool(name: str, arguments_json: str, user_id: str) -> str:
    """Run a tool by name. Returns a string safe to feed back as a tool message content."""
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid arguments JSON: {e}"})

    handler = _DISPATCH.get(name)
    if handler is None:
        return json.dumps({"error": f"unknown tool '{name}'"})

    try:
        return handler(args, user_id)
    except Exception as e:
        logger.exception(f"tool {name} crashed: {e}")
        return json.dumps({"error": f"tool {name} crashed: {e}"})


def get_tool_names() -> List[str]:
    return list(_DISPATCH.keys())
