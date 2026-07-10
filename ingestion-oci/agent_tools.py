"""
Tool definitions and dispatch for the OCI tool-calling RAG agent.

Four tools:
  - get_history(limit)            → returns recent conversation messages
  - get_document_knowledge(query) → Qdrant dense+sparse hybrid (RRF fusion)
  - get_user_profile()            → returns stored user attributes
  - save_user_profile(attributes) → persists user-disclosed attributes

Retrieval architecture (mirrors the Azure agent):
  - Qdrant collection `docs_oci_ingested_azadea` (11.5k chunks, dense+sparse vectors)
  - Dense vectors: OCI Cohere Embed v4.0 (1536-dim) — embed_query_oci
  - Sparse vectors: fastembed Qdrant/bm25 — qdrant_utils.build_sparse_query_vector
  - Server-side RRF fusion via Prefetch + FusionQuery
  - Neighbor table_summary expansion via scroll filter on doc_id+chunk_type

The Oracle 26ai store remains available as a fallback (oracle_vectordb still
imports cleanly) but is no longer used by the agent's retrieval path.
"""

import json
import logging
import os
from typing import Any, Dict, List

from oci.generative_ai_inference.models import FunctionDefinition
from qdrant_client import QdrantClient, models as qm

from oci_pipeline import embed_query_oci
from qdrant_utils import build_sparse_query_vector
from oci_chat import oci_chat_json

# Model used by the get_clarification specialist tool. Pro tier — only
# fires when the agent suspects retrieval ambiguity, so cost stays bounded.
CLARIFICATION_MODEL = os.getenv("OCI_CLARIFICATION_MODEL", "google.gemini-2.5-pro")
from conversation_manager import get_conversation_manager

logger = logging.getLogger("agent_tools")

# ─────────────────────────────────────────────────────────────────────────────
# Qdrant client + collection
# ─────────────────────────────────────────────────────────────────────────────

QDRANT_LOCAL_URL = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("OCI_QDRANT_COLLECTION", "docs_oci_ingested_azadea")
qdrant_client = QdrantClient(url=QDRANT_LOCAL_URL, check_compatibility=False)


# Embedding cache disabled — every query embeds fresh.
def _cached_embed_query(text: str) -> List[float]:
    return embed_query_oci(text)


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas (OCI native FunctionDefinition format)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS: List[FunctionDefinition] = [
    FunctionDefinition(
        type="FUNCTION",
        name="get_history",
        description=(
            "Fetch the recent conversation history (user and assistant messages) "
            "for the current user. Use this when the user's current message contains "
            "pronouns or ellipsis or otherwise depends on prior turns to be interpreted."
        ),
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of most recent messages to return. Defaults to 10.",
                }
            },
            "required": [],
        },
    ),
    FunctionDefinition(
        type="FUNCTION",
        name="get_document_knowledge",
        description=(
            "Search the Azadea corporate knowledge base for relevant policy, procedure, "
            "or SOP documents. Use this for any documented company process, policy, "
            "or SOP question. The query must be a STANDALONE, self-contained phrase — "
            "resolve any pronouns or ellipsis from prior turns first (call get_history "
            "if needed)."
        ),
        parameters={
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
    ),
    FunctionDefinition(
        type="FUNCTION",
        name="get_user_profile",
        description=(
            "Retrieve stored profile attributes for the current user (e.g. role, country, "
            "brand, department, employment_type). Returns an empty object if nothing is "
            "stored yet. Call this before answering any policy question whose answer may "
            "depend on the user's role, country, brand, or similar attribute — so you can "
            "personalise the response without asking the user again."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    FunctionDefinition(
        type="FUNCTION",
        name="save_user_profile",
        description=(
            "Store profile attributes the user has just disclosed about themselves so they "
            "are remembered across turns. Pass only attributes the user has explicitly "
            "shared in this conversation (e.g. their role, country, brand, department, "
            "employment type). Existing attributes are merged — pass only what is new or "
            "changed. Do not infer or guess attributes the user did not state."
        ),
        parameters={
            "type": "object",
            "properties": {
                "attributes": {
                    "type": "object",
                    "description": "Map of attribute name → value.",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["attributes"],
        },
    ),
    FunctionDefinition(
        type="FUNCTION",
        name="list_documents",
        description=(
            "List ALL policy/procedure documents in a category — a complete catalog, "
            "not a search. Use this when the user asks to 'list / share / give me all "
            "the X policies' or 'all the documents/links related to X' (e.g. all IT "
            "policies, all HR policies). Returns every document name in that category. "
            "For a specific question about what a policy SAYS, use get_document_knowledge "
            "instead."
        ),
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The policy area, e.g. 'IT', 'HR', 'Finance', 'Sales', 'F&B', 'Stock'.",
                }
            },
            "required": ["category"],
        },
    ),
    # NOTE: get_clarification exists in _DISPATCH but is NOT registered in
    # TOOL_DEFINITIONS — the agent loop invokes it automatically server-side
    # after get_document_knowledge instead of exposing it to Flash. Flash
    # chokes (empty replies) when the 5-tool schema is exposed; keeping the
    # surface at 4 tools and doing the clarification consult invisibly.
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool execution
# ─────────────────────────────────────────────────────────────────────────────

def _tool_get_history(args: Dict[str, Any], user_id: str) -> str:
    limit = int(args.get("limit") or 10)
    msgs = get_conversation_manager().get_history(user_id, limit=limit)
    out = []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")
        # Strip the upstream User Context wrapper if present so the agent sees
        # just the actual user turn text (not the metadata block).
        if "Request:" in content:
            content = content.split("Request:", 1)[-1].lstrip("\r\n").strip()
        out.append({"role": role, "content": content})
    return json.dumps(out, ensure_ascii=False)


def _tool_get_document_knowledge(args: Dict[str, Any], user_id: str) -> str:
    """Qdrant dense+sparse hybrid via Prefetch+RRF + neighbor table_summary
    expansion. Mirrors the Azure agent's retrieval. Full chunk text + full_table
    content returned with no truncation."""
    query = (args.get("query") or "").strip()
    top_k = int(args.get("top_k") or 7)
    if not query:
        return json.dumps({"error": "query is required"})

    try:
        dense_q = _cached_embed_query(query)
        sparse_q = build_sparse_query_vector(query)
    except Exception as e:
        logger.warning(f"embedding failed: {e}")
        return json.dumps({"error": f"embedding failed: {e}"})

    # Hybrid search: dense + sparse with RRF fusion (same as Azure agent)
    try:
        result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qm.Prefetch(query=dense_q, using="dense", limit=20),
                qm.Prefetch(query=sparse_q, using="sparse", limit=20),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k + 3,
        )
    except Exception as e:
        logger.warning(f"Qdrant search failed: {e}")
        return json.dumps({"error": f"retrieval failed: {e}"})

    if not result or not result.points:
        return json.dumps({"results": [], "note": "no relevant documents found in the knowledge base"})

    chunks = []
    doc_ids_seen = set()
    table_keys_seen = set()

    for p in result.points[:top_k]:
        pl = p.payload or {}
        chunk_type = pl.get("chunk_type", "")
        doc_id = pl.get("doc_id", "")
        full_table = pl.get("full_table", "") or ""
        text = pl.get("text", "") or ""
        source_file = pl.get("source_file", "")
        page = pl.get("page")
        if doc_id:
            doc_ids_seen.add(doc_id)

        # Prepend an explicit source header so the LLM sees the filename at the
        # start of each chunk — makes citation natural and avoids the LLM
        # skimming past the structured `source` JSON metadata.
        header = f"[Source: {source_file} | Page: {page}]\n" if source_file else ""

        entry = {
            "source": source_file,
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
                collection_name=COLLECTION_NAME,
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
            n_header = f"[Source: {n_src} | Page: {n_page}]\n" if n_src else ""
            chunks.append({
                "source": n_src,
                "doc_id": doc_id,
                "chunk_type": "table_summary",
                "full_table": n_header + ft,            # complete, no truncation
                "text": n_header + (tpl.get("text") or ""),
                "page": n_page,
                "from_neighbor_expansion": True,
            })
            table_keys_seen.add(ft)

    return json.dumps({"results": chunks}, ensure_ascii=False)


def _tool_get_user_profile(args: Dict[str, Any], user_id: str) -> str:
    try:
        profile = get_conversation_manager().get_user_profile(user_id) or {}
        return json.dumps(profile, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"get_user_profile failed: {e}")
        return json.dumps({})


def _tool_save_user_profile(args: Dict[str, Any], user_id: str) -> str:
    attrs = args.get("attributes") or {}
    if not isinstance(attrs, dict) or not attrs:
        return json.dumps({"saved": False, "reason": "no attributes provided"})
    cleaned = {str(k): str(v) for k, v in attrs.items() if v is not None and str(v).strip()}
    if not cleaned:
        return json.dumps({"saved": False, "reason": "no non-empty attributes"})
    try:
        get_conversation_manager().update_user_profile(user_id, cleaned)
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

_CLARIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_clarification": {"type": "boolean"},
        "rationale": {"type": "string"},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["needs_clarification", "rationale", "questions"],
}


def _tool_get_clarification(args: Dict[str, Any], user_id: str) -> str:
    """Call OCI Gemini Pro to decide whether clarifying questions are needed
    after retrieval. Receives the original query + the retrieved chunks JSON;
    consults the user's stored profile so it doesn't ask for attributes the
    system already has."""
    query = (args.get("query") or "").strip()
    chunks_json = args.get("retrieved_chunks_json") or ""
    if not query:
        return json.dumps({"needs_clarification": False, "rationale": "no query supplied", "questions": []})

    try:
        profile = get_conversation_manager().get_user_profile(user_id) or {}
    except Exception:
        profile = {}

    user_block = (
        f"User question: {query}\n\n"
        f"Stored user profile (already known — do NOT ask for these): {json.dumps(profile, ensure_ascii=False)}\n\n"
        f"Retrieved documents (JSON, full chunks):\n{chunks_json}"
    )

    try:
        # OCI Gemini 2.5 Pro is a thinking model; give it enough headroom for
        # reasoning + the final JSON object.
        parsed = oci_chat_json(
            messages=[
                ("system", _CLARIFICATION_SYSTEM_PROMPT),
                ("user", user_block),
            ],
            schema=_CLARIFICATION_SCHEMA,
            model=CLARIFICATION_MODEL,
            temperature=0.2,
            max_tokens=4000,
        )
        out = {
            "needs_clarification": bool(parsed.get("needs_clarification", False)),
            "rationale": str(parsed.get("rationale", ""))[:500],
            "questions": [str(q) for q in (parsed.get("questions") or []) if str(q).strip()][:4],
        }
        logger.info(f"get_clarification: needs={out['needs_clarification']}, n_questions={len(out['questions'])}")
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        logger.exception(f"get_clarification failed: {e}")
        return json.dumps({
            "needs_clarification": False,
            "rationale": f"clarification check failed ({type(e).__name__}); proceeding without it",
            "questions": [],
        })


# Friendly category names → the `domain` code stored on each chunk.
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
    """Return ALL document names in a category (a metadata listing, not a search).
    Used for 'list/share all the X policies' style requests."""
    cat = (args.get("category") or args.get("domain") or "").strip().lower()
    code = _DOMAIN_ALIASES.get(cat, cat.upper())
    seen = set()
    offset = None
    try:
        while True:
            res, offset = qdrant_client.scroll(
                COLLECTION_NAME,
                scroll_filter=qm.Filter(must=[
                    qm.FieldCondition(key="domain", match=qm.MatchValue(value=code))
                ]),
                limit=1000, offset=offset,
                with_payload=["source_file"], with_vectors=False,
            )
            for p in res:
                sf = (p.payload or {}).get("source_file", "")
                if sf:
                    seen.add(sf[:-3] if sf.endswith(".md") else sf)
            if offset is None:
                break
    except Exception as e:
        logger.warning(f"list_documents failed: {e}")
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
    """Run a tool by name. arguments_json is the raw JSON string from FunctionCall.arguments.
    Returns a string (JSON or plain) safe to feed back as a ToolMessage content."""
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid arguments JSON: {e}"})

    handler = _DISPATCH.get(name)
    if handler is None:
        return json.dumps({"error": f"unknown tool '{name}'"})

    try:
        result = handler(args, user_id)
        return result
    except Exception as e:
        logger.exception(f"tool {name} crashed: {e}")
        return json.dumps({"error": f"tool {name} crashed: {e}"})


def get_tool_names() -> List[str]:
    return list(_DISPATCH.keys())
