"""
Tool definitions and dispatch for the AZURE-tier tool-calling RAG agent.

Two tools (mirror of OCI agent's tools, but Azure-backed):
  - get_history(limit)            → recent conversation messages from Redis
  - get_document_knowledge(query) → Qdrant hybrid (dense + sparse) search

Reuses existing Azure infrastructure via imports — does NOT touch any running service:
  - qdrant_client + COLLECTION_NAME_V2 (docs_llm_chunked_azadea)
  - rag_impl.embed_dense_azure (Azure OpenAI text-embedding-3-large)
  - rag_impl.build_sparse_query_vector (BM25 sparse)
  - conv_manager (Redis-backed conversation history)
"""

import hashlib
import json
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List

# Reuse existing singletons (these are already initialized in rag_server_gemini)
from qdrant_client import QdrantClient, models as qm
from rag_server_gemini import conv_manager, openrouter_client
import azure_doc_intelligence_qdrant as rag_impl   # same alias used by rag_server_llm_chunked

# Model used by the get_clarification specialist tool. Pro tier (more
# capable than the Flash model used in the main agent loop) — only fires
# when the agent suspects retrieval ambiguity, so cost stays bounded.
CLARIFICATION_MODEL = os.getenv("CLARIFICATION_MODEL", "google/gemini-2.5-pro")

logger = logging.getLogger("agent_tools_azure")

# Local Qdrant client + collection — same defaults as the running pipeline service
QDRANT_LOCAL_URL = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
COLLECTION_NAME_V2 = os.getenv("QDRANT_COLLECTION", "docs_llm_chunked_azadea")
qdrant_client = QdrantClient(url=QDRANT_LOCAL_URL, check_compatibility=False)


# Embedding cache disabled — every query embeds fresh.
def _cached_embed_dense(text: str) -> List[float]:
    return rag_impl.embed_dense_azure([text])[0]


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


def _tool_get_document_knowledge(args: Dict[str, Any], user_id: str) -> str:
    """Mirror rag_server_llm_chunked's retrieve_fast: dense + sparse hybrid via Qdrant
    + neighbor table expansion + full_table content. No content truncation."""
    query = (args.get("query") or "").strip()
    top_k = int(args.get("top_k") or 7)
    if not query:
        return json.dumps({"error": "query is required"})

    try:
        dense_q = _cached_embed_dense(query)
        sparse_q = rag_impl.build_sparse_query_vector(query)
    except Exception as e:
        logger.warning(f"embedding failed: {e}")
        return json.dumps({"error": f"embedding failed: {e}"})

    # Hybrid search: dense + sparse with RRF fusion (same as pipeline service)
    try:
        result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME_V2,
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

        # Prepend an explicit source header so the LLM sees the filename right
        # at the start of the chunk's content — makes citation natural and
        # avoids cases where the LLM skims past the structured `source` field
        # in the surrounding JSON metadata.
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


_DISPATCH = {
    "get_history": _tool_get_history,
    "get_document_knowledge": _tool_get_document_knowledge,
    "get_user_profile": _tool_get_user_profile,
    "save_user_profile": _tool_save_user_profile,
    "get_clarification": _tool_get_clarification,
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
