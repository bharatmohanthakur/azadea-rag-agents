#!/usr/bin/env python3
"""
OCI RAG API Server — Port 7874
Full OCI replacement of rag_server_llm_chunked.py.

Zero Azure, zero OpenRouter. All LLM calls go through OCI GenAI.

Pipeline:
  1. Classification: rule-based (0ms) → OCI Gemini 2.5 Flash (UnifiedClassifier)
  2. Query embedding: OCI Cohere Embed v4.0 (SEARCH_QUERY)
  3. Hybrid search: Qdrant (dense + sparse RRF) on docs_oci_ingested_azadea
  4. Answer generation: OCI Gemini 2.5 Flash native SDK (streaming + non-streaming)

Endpoints:
  POST /query         — JSON response
  POST /query/stream  — SSE streaming (token-by-token)
  GET  /health        — service status
"""

import os
import sys
import json
import asyncio
import logging
import uuid
import hashlib
import time as time_module
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Ensure local modules importable
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

# Native OCI — no LangChain, no OpenAI SDK
from oci_clients import OCI_CHAT_MODEL
from oci_chat import oci_chat, oci_chat_async, oci_stream_async
from oci_openai_adapter import OciAsOpenAI
from oci_pipeline import embed_query_oci
import oracle_vectordb

# Independent modules (no rag_server_gemini import)
from conversation_manager import get_conversation_manager
from clarification_tracker import ClarificationTracker
from unified_classifier import UnifiedClassifier
from llm_classifier import init_llm_classifier
from topic_change_classifier import TopicChangeClassifier, get_or_init_topic_change_classifier

from rag_utils import (
    format_gfm_to_html,
    log_request,
    count_tokens,
    get_user_history,
    rewrite_query_with_history,
    logger,
)

# QdrantClient no longer needed — using Oracle 26ai

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

# =====================================================================
# CONFIGURATION
# =====================================================================
PORT = int(os.getenv("RAG_PORT", "7874"))

# ---------------------------------------------------------------------------
# All singletons use lazy init — safe for multi-worker (gunicorn fork).
# Each worker gets its own instances after fork, no shared mutable state.
# ---------------------------------------------------------------------------
_conv_manager = None
_clarification_tracker = None
_oci_client = None
_unified_classifier: Optional[UnifiedClassifier] = None
_EMBED_CACHE_MAX = 500
_embed_cache: OrderedDict = OrderedDict()


def _get_conv_manager():
    global _conv_manager
    if _conv_manager is None:
        _conv_manager = get_conversation_manager()
    return _conv_manager


def _get_clarification_tracker():
    global _clarification_tracker
    if _clarification_tracker is None:
        _clarification_tracker = ClarificationTracker(_get_conv_manager())
    return _clarification_tracker


def _get_oci_client():
    global _oci_client
    if _oci_client is None:
        _oci_client = OciAsOpenAI()
        init_llm_classifier(_oci_client, OCI_CHAT_MODEL, cache_enabled=False)
    return _oci_client


def get_or_init_unified_classifier() -> UnifiedClassifier:
    global _unified_classifier
    if _unified_classifier is None:
        _unified_classifier = UnifiedClassifier(
            llm_client=_get_oci_client(),
            deployment_name=OCI_CHAT_MODEL,
            cache_enabled=False,
            cache_ttl_seconds=300,
        )
        logger.info(f"Unified classifier initialized (OCI {OCI_CHAT_MODEL}, cache DISABLED)")
    return _unified_classifier


def _get_topic_change_classifier() -> TopicChangeClassifier:
    return get_or_init_topic_change_classifier(_get_oci_client(), OCI_CHAT_MODEL)


def _topic_change_check(history, query_text: str) -> Optional["TopicChangeResult"]:
    """Run topic-change/ambiguity gate. Returns the result only if either flag is true
    AND a non-empty clarification was produced. None otherwise (continue to
    UnifiedClassifier). Skipped on first turn."""
    recent_user_queries = [
        m.get("content", "") for m in history if m.get("role") == "user"
    ][-7:]
    if not recent_user_queries:
        return None
    result = _get_topic_change_classifier().classify(
        recent_user_queries=recent_user_queries,
        current_user_query=query_text,
    )
    if (result.topic_changed or result.is_ambiguous) and result.suggested_question.strip():
        return result
    return None


def _cached_embed_query(text: str) -> List[float]:
    """Cache OCI Cohere embedding results."""
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in _embed_cache:
        _embed_cache.move_to_end(key)
        return _embed_cache[key]
    vec = embed_query_oci(text)
    _embed_cache[key] = vec
    if len(_embed_cache) > _EMBED_CACHE_MAX:
        _embed_cache.popitem(last=False)
    return vec


# Classification is fully delegated to UnifiedClassifier (LLM). Greetings,
# thanks, casual, FORMAT, and PROFILE_UPDATE routing are all decided by the model.


# =====================================================================
# Source → S3 URL mapping
# =====================================================================
# Local PDF tree and S3 bucket layout have diverged (some folders renamed),
# so we cannot derive object_key from local path. Instead we build a
# basename->object_key index by listing the bucket once at first use.
import urllib.parse
S3_SERVICE_URL = (os.getenv("S3_SERVICE_URL") or "").rstrip("/")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME") or ""
S3_ACCESS_KEY  = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY  = os.getenv("S3_SECRET_KEY")
S3_REGION      = os.getenv("S3_REGION", "eu-frankfurt-1")

S3_PRESIGN_EXPIRES_SEC = int(os.getenv("S3_PRESIGN_EXPIRES_SEC", "3600"))  # 1 hour default

_s3_index: Optional[Dict[str, str]] = None   # basename(.pdf) -> object_key
_s3_client = None                            # lazy boto3 client (signing only, no listing)


def _get_s3_client():
    """Lazy boto3 client. Reused across all presign calls — no per-call cost."""
    global _s3_client
    if _s3_client is None:
        if not (S3_SERVICE_URL and S3_BUCKET_NAME and S3_ACCESS_KEY and S3_SECRET_KEY):
            return None
        import boto3
        _s3_client = boto3.client(
            "s3", endpoint_url=S3_SERVICE_URL,
            aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION,
        )
    return _s3_client


def _build_s3_index() -> Dict[str, str]:
    """One-shot listing of all .pdf objects in the bucket; map basename -> full key."""
    s3 = _get_s3_client()
    if s3 is None:
        logger.warning("S3 env vars incomplete; s3_url field will be omitted from responses")
        return {}
    idx: Dict[str, str] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET_NAME):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                idx[Path(key).name] = key
    logger.info(f"S3 index built: {len(idx)} PDFs in bucket {S3_BUCKET_NAME}")
    return idx


def _get_s3_index() -> Dict[str, str]:
    global _s3_index
    if _s3_index is None:
        try:
            _s3_index = _build_s3_index()
        except Exception as e:
            logger.error(f"Failed to build S3 index: {e}")
            _s3_index = {}
    return _s3_index


def build_s3_url(source_file: str) -> Optional[str]:
    """Generate a presigned URL for the .pdf matching this source_file.
    URL is browser-clickable for S3_PRESIGN_EXPIRES_SEC seconds.
    Returns None if the PDF is not in the bucket or S3 is misconfigured."""
    pdf_basename = Path(source_file).stem + ".pdf"
    object_key = _get_s3_index().get(pdf_basename)
    if not object_key:
        return None
    s3 = _get_s3_client()
    if s3 is None:
        return None
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": object_key},
        ExpiresIn=S3_PRESIGN_EXPIRES_SEC,
    )


def _cached_s3_url(source_file: str) -> Optional[str]:
    """No caching — presigned URLs expire, so each query gets a fresh signature.
    The S3 index lookup is O(1) and HMAC-SHA256 is microseconds. Total cost ~1-3ms."""
    try:
        return build_s3_url(source_file)
    except Exception as e:
        logger.warning(f"build_s3_url failed for {source_file!r}: {e}")
        return None


# =====================================================================
# Retrieval
# =====================================================================
async def retrieve_fast(query: str, user_id: str, top_k: int = 7) -> Dict[str, Any]:
    """Hybrid retrieval: Oracle vector search + Oracle Text keyword search."""
    sources = []
    try:
        loop = asyncio.get_event_loop()

        # 1. Embed query (OCI Cohere)
        t0 = time_module.time()
        dense_q = await loop.run_in_executor(None, _cached_embed_query, query)
        embed_time = time_module.time() - t0

        # 2. Oracle hybrid search (vector + keyword in one call)
        t0 = time_module.time()
        results = await loop.run_in_executor(
            None, oracle_vectordb.hybrid_search, dense_q, query, top_k
        )
        search_time = time_module.time() - t0

        if not results:
            return {"context": f"No documents found for: {query}", "sources": [], "images": []}

        table_parts = []
        text_parts = []
        doc_ids_seen = set()
        table_texts_seen = set()

        for r in results[:top_k]:
            src_file = r.get("source_file", "unknown")
            chunk_type = r.get("chunk_type", "text")
            doc_id = r.get("doc_id", "unknown")
            text = r.get("text", "")
            full_table = r.get("full_table", "")
            doc_ids_seen.add(doc_id)

            _entry = {
                "id": r.get("chunk_id", ""), "score": round(r.get("score", 0.5), 4),
                "source": src_file, "text_snippet": text[:200], "chunk_type": chunk_type,
            }
            _s3 = _cached_s3_url(src_file)
            if _s3:
                _entry["s3_url"] = _s3
            sources.append(_entry)

            if chunk_type == "table_summary" and full_table:
                if full_table not in table_texts_seen:
                    table_parts.append(f"[{src_file}] Table:\n{full_table}")
                    table_texts_seen.add(full_table)
            elif text:
                text_parts.append(f"[{src_file}] {text}")

        # 3. Neighbor table expansion (Oracle SQL)
        for doc_id in doc_ids_seen:
            try:
                table_chunks = await loop.run_in_executor(
                    None, oracle_vectordb.scroll_tables, doc_id
                )
                for tp in table_chunks:
                    ft = tp.get("full_table", "")
                    if ft and ft not in table_texts_seen:
                        table_parts.append(f"[{tp.get('source_file', '')}] Table:\n{ft}")
                        table_texts_seen.add(ft)
            except Exception:
                pass

        context = "\n\n".join(table_parts + text_parts)
        logger.info(f"RETRIEVAL: embed={embed_time:.3f}s, search={search_time:.3f}s, ctx={len(context)} chars")
        return {"context": context, "sources": sources, "images": []}

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {"context": f"Error: {str(e)}", "sources": [], "images": []}


# =====================================================================
# System Prompts
# =====================================================================
_SYSTEM_PROMPT_SIMPLE = """You are Dea — the internal knowledge assistant for Azadea Group employees.
Your knowledge base covers Azadea policies, procedures, and SOPs across many domains:
HR, Operations, Finance & Accounting, IT, Stock Management, F&B, Marketing, BCP, and compliance.
You are NOT an HR-only bot. Match each question to the right domain naturally.

TONE — apply to EVERY response:
- Be warm, friendly, and helpful — like a supportive colleague, not a corporate manual.
- Use natural, conversational phrasing. Avoid stiff/legalistic language ("hereby", "as per", "kindly note").
- Stay friendly even when the answer is "I don't have that information" or when correcting a misunderstanding.
- Light positive openers are welcome ("Sure!", "Of course —", "Happy to help with that.") but DON'T repeat your name in every reply.
- Keep it concise. Friendly does NOT mean wordy or padded with filler.

Answer using ONLY the provided context.
Rules:
- Use the user's profile (role / country / brand / department) when it's relevant to the answer
- Include EXACT numbers, dates, percentages from the context
- Use bullet points or short numbered lists when listing multiple items, steps, or conditions
- The user's question may contain typos or synonyms. Match their intent to the context terminology
  (e.g. "dress code" = "uniform", "leave" = "vacation/annual leave", "fire" = "termination",
        "pay" = "salary/compensation", "DCR" = "Daily Cash Report")
- If the context does NOT contain information to answer the question, respond warmly and CRISPLY:
  acknowledge you don't have that info. THEN look at the [source_file] tags in the context provided —
  if any are even loosely related to the user's question, name 1-2 of them in plain language as adjacent
  topics you CAN cover. If nothing in the context is related at all, just invite the user to ask
  something else. Keep the whole reply under 2 sentences. Examples:
    - context has "HRD - GEN - 001 - Annual Leave" but user asked about taxes →
      "I don't have info on that, but I can help with leave policies or other HR topics — what would you like to know?"
    - context has only generic admin docs but user asked about weather →
      "I don't have info on that — what else can I help you with?"
  Do NOT hardcode a fixed list of domains. Do NOT volunteer unrelated facts from the context.
- Respond in the SAME LANGUAGE as the user's question. If the question is in Arabic, respond in Arabic.
- Cite source documents naturally
- Do NOT say "consult HR" as a generic fallback — only mention HR if the question is actually HR-related.
  For non-HR questions, suggest the relevant team (e.g. Finance, IT, Operations) or just answer based
  on the documents."""

_SYSTEM_PROMPT_CLARIFY = """You are Dea — the internal knowledge assistant for Azadea Group employees.
Your knowledge base covers HR, Operations, Finance, IT, Stock Management, F&B, Marketing, and BCP.
You are NOT an HR-only bot. Use the user's stored profile (role/country/brand/department) when relevant.

TONE — apply to EVERY response (whether you ANSWER or CLARIFY):
- Be warm, friendly, and helpful — like a supportive colleague, not a corporate manual.
- Use natural, conversational phrasing. Avoid stiff/legalistic language.
- When asking a clarification, frame it gently ("Just to make sure I give you the right info —", "Could you tell me which …").
- Keep it concise. Friendly does NOT mean wordy.

Your job is to decide: can you answer this question directly, or do you need more details from the user?

STEP 0 — SCOPE CHECK: If the question is clearly OUTSIDE the Azadea knowledge base (e.g. weather, sports,
  general trivia, driving, current events), start with "ANSWER:" and respond warmly + crisply: acknowledge
  you don't have that info. THEN look at the [source_file] tags in the context — if 1-2 are even loosely
  related, name those topics in plain language as alternatives you CAN cover; if nothing in the context
  matches the user's intent, just invite them to ask something else. Keep it under 2 sentences. Examples:
    - "ANSWER: I don't have info on weather, but I can help with leave policies or onboarding —
       what would you like to know?"
    - "ANSWER: I don't have info on that — what else can I help you with?"
  Do NOT hardcode a fixed list of domains. Do NOT proceed to STEP 1 in this case.

STEP 1: Read the context chunks carefully.
STEP 2: Check if the answer depends on a detail the user did NOT provide, such as:
  - Country of employment (if policy varies by country)
  - Employee type (shop/back office/part-time)
  - Department or brand
  - Specific situation details

STEP 3: Decide:
  A) If the answer is the SAME regardless of country/role/type → start with "ANSWER:" and provide the answer.
  B) If the answer VARIES by country, employee type, or role → you MUST start with "CLARIFY:" and ask a follow-up. Do NOT list all variations — narrow it down first.
  C) If the context CHUNKS don't actually address the question (retrieval missed) → start with "ANSWER:" and
     use the same dynamic-from-context redirect from STEP 0: surface 1-2 topics actually present in the
     [source_file] tags as alternatives. Do NOT make up an answer from unrelated chunks.

Rules:
- PREFER asking clarification over dumping long answers. If you see numbers/rules that differ by country or position in the context, ALWAYS clarify first.
- The follow-up must list 2-3 specific options found in the context (e.g. "I found leave policies for Lebanon, UAE, and Jordan. Which country are you asking about?")
- Only answer directly (ANSWER:) when the policy is universal/identical across all countries and roles, OR the user already specified their country/role.
- Include EXACT numbers, dates, percentages from the context when answering
- Respond in the SAME LANGUAGE as the user's question
- Maximum 3 follow-up options — pick the most relevant from the context
- You MUST start your response with either "ANSWER:" or "CLARIFY:" — no exceptions"""


# =====================================================================
# Helper
# =====================================================================
def _extract_original_query_from_history(history: List[Dict[str, Any]]) -> str:
    if not history or len(history) < 2:
        return ""
    for i in range(len(history) - 1, 0, -1):
        if history[i].get("role") == "assistant":
            if i > 0 and history[i - 1].get("role") == "user":
                return history[i - 1].get("content", "")
    return ""


async def _generate_answer_or_clarify(query: str, context: str, user_profile: Optional[Dict[str, str]] = None) -> str:
    system_prompt = _SYSTEM_PROMPT_CLARIFY
    if user_profile:
        profile_lines = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in user_profile.items())
        system_prompt = f"{system_prompt}\n\nUSER PROFILE (apply only when relevant):\n{profile_lines}"
    messages = [
        ("system", system_prompt),
        ("user", f"Question: {query}\n\nContext:\n{context}"),
    ]
    text = await oci_chat_async(messages)
    if text.startswith("ANSWER:"):
        return text[len("ANSWER:"):].strip()
    if text.startswith("CLARIFY:"):
        return text
    return text


async def _generate_profile_ack(query: str, attrs: Dict[str, Any], existing_profile: Dict[str, str]) -> str:
    """
    Generate a varied, natural acknowledgement when the user discloses profile info.
    Used as a soft fallback if UnifiedClassifier didn't produce conversational_response.
    Stays brief (1-2 sentences) and asks what the user wants to know next.
    """
    new_attrs_str = ", ".join(f"{k}={v}" for k, v in (attrs or {}).items()) or "(none)"
    full_profile_str = ", ".join(f"{k}={v}" for k, v in (existing_profile or {}).items()) or "(empty)"
    messages = [
        ("system", "You are Dea, a friendly Azadea Group assistant. The user just shared a fact about themselves. "
                   "Acknowledge it warmly in ONE sentence and invite them to ask a question. "
                   "Vary your wording — never start with the same phrase twice. Do not echo the entire profile. "
                   "No prefixes like ANSWER: or CLARIFY:. Plain text, under 30 words."),
        ("user", f"User said: \"{query}\"\nNew attributes: {new_attrs_str}\nFull stored profile: {full_profile_str}\nReply:"),
    ]
    text = await oci_chat_async(messages)
    return (text or "").strip() or "Thanks for letting me know — what would you like help with?"


# =====================================================================
# APP
# =====================================================================
app = FastAPI(title="OCI RAG API Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"


class QueryResponse(BaseModel):
    response: str
    metadata: Dict[str, Any] = {}


# =====================================================================
# POST /query (non-streaming)
# =====================================================================
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    request_id = str(uuid.uuid4())[:8]
    start_time = time_module.time()
    timings = {}

    try:
        query_text = request.query.strip()
        user_id = request.user_id or "default_user"
        log_request(request_id, "OCI_QUERY_START", {"query": query_text})

        history = get_user_history(_get_conv_manager(), user_id, use_summarization=False)
        active_clarification = _get_clarification_tracker().get_active_session(user_id)
        previous_response = ""
        if history:
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    previous_response = msg.get("content", "")
                    break

        # ── Topic-change / ambiguity gate ─────────────────────────────────
        # Skip if a clarification session is already active (existing handling
        # owns multi-turn flow) or chain-depth limit reached.
        if not active_clarification and _get_clarification_tracker().get_chain_depth(user_id) < 3:
            tc = await asyncio.to_thread(_topic_change_check, history, query_text)
            if tc is not None:
                logger.info(f"[{request_id}] TOPIC_GATE_FIRED changed={tc.topic_changed} ambiguous={tc.is_ambiguous} conf={tc.confidence:.2f}")
                _get_clarification_tracker().create_session(
                    user_id, query_text, [tc.suggested_question], "", []
                )
                _get_conv_manager().add_message(user_id, "user", query_text, {"request_id": request_id})
                _get_conv_manager().add_message(user_id, "assistant", tc.suggested_question, {"request_id": request_id, "query_type": "intent_clarification"})
                return QueryResponse(
                    response=format_gfm_to_html(tc.suggested_question),
                    metadata={"request_id": request_id, "route": "INTENT_CLARIFY", "topic_changed": tc.topic_changed, "is_ambiguous": tc.is_ambiguous, "confidence": tc.confidence, "elapsed_sec": round(time_module.time() - start_time, 3)},
                )

        # Classification — LLM-only via UnifiedClassifier
        t0 = time_module.time()
        classifier = get_or_init_unified_classifier()
        clarification_question = ""
        original_query = ""
        if active_clarification:
            clarification_question = getattr(active_clarification, 'questions', [''])[0] if active_clarification else ""
            original_query = getattr(active_clarification, 'original_query', "") if active_clarification else ""

        # Load stored profile so the classifier sees it BEFORE deciding the route
        user_profile = _get_conv_manager().get_user_profile(user_id)

        classification = await asyncio.to_thread(
            classifier.classify, query=query_text, conversation_history=history,
            previous_response=previous_response, active_clarification=bool(active_clarification),
            clarification_question=clarification_question, original_query=original_query,
            user_profile=user_profile or None,
        )
        route = "CONVERSATIONAL" if classification.is_conversational else (classification.rag_route or "SIMPLE")
        fast_response = classification.conversational_response if classification.is_conversational else None
        timings["1_classify"] = time_module.time() - t0
        if classification.is_clarification_answer and route != "CLARIFICATION_ANSWER":
            route = "CLARIFICATION_ANSWER"

        # CONVERSATIONAL
        if route == "CONVERSATIONAL":
            response_text = fast_response or "Hi! I'm Dea. How can I help you with Azadea policies, operations, or procedures?"
            _get_conv_manager().add_message(user_id, "user", query_text, {"request_id": request_id})
            _get_conv_manager().add_message(user_id, "assistant", response_text, {"request_id": request_id, "query_type": "conversational"})
            return QueryResponse(response=response_text, metadata={"request_id": request_id, "route": "CONVERSATIONAL", "elapsed_sec": round(time_module.time() - start_time, 3)})

        # PROFILE_UPDATE
        if route == "PROFILE_UPDATE":
            attrs = getattr(classification, "profile_attributes", None) if classification else None
            if attrs:
                _get_conv_manager().update_user_profile(user_id, attrs)
                logger.info(f"[{request_id}] Profile updated for {user_id}: {attrs}")
            response_text = (fast_response or "").strip()
            if not response_text:
                full_profile = _get_conv_manager().get_user_profile(user_id)
                response_text = await _generate_profile_ack(query_text, attrs or {}, full_profile)
            _get_conv_manager().add_message(user_id, "user", query_text, {"request_id": request_id})
            _get_conv_manager().add_message(user_id, "assistant", response_text, {
                "request_id": request_id, "query_type": "profile_update",
                "profile_attributes": attrs or {},
            })
            return QueryResponse(response=response_text, metadata={"request_id": request_id, "route": "PROFILE_UPDATE", "profile_attributes": attrs or {}, "elapsed_sec": round(time_module.time() - start_time, 3)})

        # FORMAT
        if route == "FORMAT" and previous_response:
            t0 = time_module.time()
            messages = [("system", "Reformat the previous response as requested. Keep all information."), ("user", f"Previous response:\n{previous_response}\n\nRequest: {query_text}")]
            text_response = await oci_chat_async(messages)
            answer_text = text_response
            timings["2_format"] = time_module.time() - t0
            _get_conv_manager().add_message(user_id, "user", query_text, {"request_id": request_id})
            _get_conv_manager().add_message(user_id, "assistant", answer_text, {"request_id": request_id, "query_type": "format"})
            return QueryResponse(response=format_gfm_to_html(answer_text), metadata={"request_id": request_id, "route": "FORMAT", "elapsed_sec": round(time_module.time() - start_time, 3)})

        # CLARIFICATION_ANSWER
        if route == "CLARIFICATION_ANSWER":
            if active_clarification:
                _get_clarification_tracker().add_answer(user_id, query_text)
            original_query = getattr(active_clarification, "original_query", "") if active_clarification else ""
            if not original_query and history:
                original_query = _extract_original_query_from_history(history)
            combined_query = f"{original_query} - {query_text}" if original_query else query_text
            search_result = await retrieve_fast(combined_query, user_id)
            context, sources = search_result["context"], search_result["sources"]
            _user_profile = _get_conv_manager().get_user_profile(user_id)
            _system_prompt = _SYSTEM_PROMPT_SIMPLE
            if _user_profile:
                _profile_lines = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in _user_profile.items())
                _system_prompt = f"{_system_prompt}\n\nUSER PROFILE (apply only when relevant):\n{_profile_lines}"
            messages = [("system", _system_prompt), ("user", f"Question: {combined_query}\n\nContext:\n{context}")]
            text_response = await oci_chat_async(messages)
            answer_text = text_response
            _get_clarification_tracker().complete_session(user_id)
            _get_conv_manager().add_message(user_id, "user", query_text, {"request_id": request_id})
            _get_conv_manager().add_message(user_id, "assistant", answer_text, {"request_id": request_id, "query_type": "clarification_answer"})
            return QueryResponse(response=format_gfm_to_html(answer_text), metadata={"request_id": request_id, "route": "CLARIFICATION_ANSWER", "sources": sources[:3], "elapsed_sec": round(time_module.time() - start_time, 3)})

        # RAG (SIMPLE)
        t0 = time_module.time()
        needs_rewrite = classification.needs_query_rewrite if classification else False
        if needs_rewrite and history:
            rewritten_query = await asyncio.to_thread(
                rewrite_query_with_history, history, query_text, user_id,
                clarification_tracker, oci_client, OCI_CHAT_MODEL,
            )
        else:
            rewritten_query = query_text
        timings["2_rewrite"] = time_module.time() - t0

        search_result = await retrieve_fast(rewritten_query, user_id)
        context, sources = search_result["context"], search_result["sources"]

        chain_depth = _get_clarification_tracker().get_chain_depth(user_id)
        if chain_depth >= 3:
            _user_profile = _get_conv_manager().get_user_profile(user_id)
            _system_prompt = _SYSTEM_PROMPT_SIMPLE
            if _user_profile:
                _profile_lines = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in _user_profile.items())
                _system_prompt = f"{_system_prompt}\n\nUSER PROFILE (apply only when relevant):\n{_profile_lines}"
            messages = [("system", _system_prompt), ("user", f"Question: {rewritten_query}\n\nContext:\n{context}")]
            text_response = await oci_chat_async(messages)
            answer_text = text_response
            _get_clarification_tracker().complete_session(user_id)
        else:
            answer_text = await _generate_answer_or_clarify(rewritten_query, context, _get_conv_manager().get_user_profile(user_id))
            if answer_text.startswith("CLARIFY:"):
                clarification_text = answer_text[len("CLARIFY:"):].strip()
                _get_clarification_tracker().create_session(user_id, rewritten_query, [clarification_text], context, sources)
                answer_text = clarification_text

        _get_conv_manager().add_message(user_id, "user", query_text, {"request_id": request_id})
        _get_conv_manager().add_message(user_id, "assistant", answer_text, {"request_id": request_id, "query_type": route.lower()})
        total_elapsed = time_module.time() - start_time
        return QueryResponse(
            response=format_gfm_to_html(answer_text),
            metadata={"request_id": request_id, "route": route, "sources": sources[:5], "elapsed_sec": round(total_elapsed, 3), "timings": {k: round(v, 3) for k, v in timings.items()}},
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}", exc_info=True)
        return QueryResponse(response="I apologize, but I encountered an error. Please try again.", metadata={"request_id": request_id, "error": str(e)})


# =====================================================================
# POST /query/stream (SSE streaming)
# =====================================================================
@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    async def generate() -> AsyncGenerator[str, None]:
        request_id = str(uuid.uuid4())[:8]
        start_time = time_module.time()
        timings = {}

        try:
            query_text = request.query.strip()
            user_id = request.user_id or "default_user"
            log_request(request_id, "OCI_STREAM_START", {"query": query_text})

            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...'})}\n\n"

            classifier = get_or_init_unified_classifier()
            history = get_user_history(_get_conv_manager(), user_id, use_summarization=False)
            active_clarification = _get_clarification_tracker().get_active_session(user_id)
            clarification_question = ""
            original_query = ""
            if active_clarification:
                clarification_question = getattr(active_clarification, 'questions', [''])[0] if active_clarification else ""
                original_query = getattr(active_clarification, 'original_query', "") if active_clarification else ""

            previous_response = ""
            if history:
                for msg in reversed(history):
                    if msg.get("role") == "assistant":
                        previous_response = msg.get("content", "")
                        break

            # ── Topic-change / ambiguity gate ─────────────────────────────
            if not active_clarification and _get_clarification_tracker().get_chain_depth(user_id) < 3:
                tc = await asyncio.to_thread(_topic_change_check, history, query_text)
                if tc is not None:
                    logger.info(f"[{request_id}] TOPIC_GATE_FIRED changed={tc.topic_changed} ambiguous={tc.is_ambiguous} conf={tc.confidence:.2f}")
                    _get_clarification_tracker().create_session(
                        user_id, query_text, [tc.suggested_question], "", []
                    )
                    _get_conv_manager().add_message(user_id, "user", query_text, {"request_id": request_id})
                    _get_conv_manager().add_message(user_id, "assistant", tc.suggested_question, {"request_id": request_id, "query_type": "intent_clarification"})
                    yield f"data: {json.dumps({'type': 'token', 'text': tc.suggested_question}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'INTENT_CLARIFY', 'topic_changed': tc.topic_changed, 'is_ambiguous': tc.is_ambiguous, 'confidence': tc.confidence}})}\n\n"
                    return

            # Classification — LLM-only via UnifiedClassifier
            # Profile is loaded BEFORE classify so PROFILE_UPDATE / clarification logic sees it.
            user_profile = _get_conv_manager().get_user_profile(user_id)

            t0 = time_module.time()
            classification = await asyncio.to_thread(
                classifier.classify, query=query_text, conversation_history=history,
                previous_response=previous_response, active_clarification=bool(active_clarification),
                clarification_question=clarification_question, original_query=original_query,
                user_profile=user_profile or None,
            )
            route = "CONVERSATIONAL" if classification.is_conversational else (classification.rag_route or "SIMPLE")
            fast_resp = classification.conversational_response if classification.is_conversational else None
            timings["1_classify"] = time_module.time() - t0
            if classification.is_clarification_answer and route != "CLARIFICATION_ANSWER":
                route = "CLARIFICATION_ANSWER"

            # CONVERSATIONAL
            if route == "CONVERSATIONAL":
                response_text = fast_resp or "Hi! I'm Dea. How can I help you with Azadea policies, operations, or procedures?"
                for word in response_text.split():
                    yield f"data: {json.dumps({'type': 'token', 'text': word + ' '}, ensure_ascii=False)}\n\n"
                _get_conv_manager().add_message(user_id, "user", query_text, {"request_id": request_id})
                _get_conv_manager().add_message(user_id, "assistant", response_text, {"request_id": request_id, "query_type": "conversational"})
                yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'CONVERSATIONAL'}})}\n\n"
                return

            # PROFILE_UPDATE - save attributes, stream the LLM acknowledgement, no retrieval
            if route == "PROFILE_UPDATE":
                attrs = getattr(classification, "profile_attributes", None) if classification else None
                if attrs:
                    _get_conv_manager().update_user_profile(user_id, attrs)
                    logger.info(f"[{request_id}] Profile updated for {user_id}: {attrs}")
                response_text = (fast_resp or "").strip()
                if not response_text:
                    full_profile = _get_conv_manager().get_user_profile(user_id)
                    response_text = await _generate_profile_ack(query_text, attrs or {}, full_profile)
                for word in response_text.split():
                    yield f"data: {json.dumps({'type': 'token', 'text': word + ' '}, ensure_ascii=False)}\n\n"
                _get_conv_manager().add_message(user_id, "user", query_text, {"request_id": request_id})
                _get_conv_manager().add_message(user_id, "assistant", response_text, {
                    "request_id": request_id, "query_type": "profile_update",
                    "profile_attributes": attrs or {},
                })
                yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'PROFILE_UPDATE', 'profile_attributes': attrs or {}}})}\n\n"
                return

            # FORMAT
            if route == "FORMAT" and previous_response:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Reformatting...'})}\n\n"
                messages = [("system", "Reformat the previous response as requested. Keep all information."), ("user", f"Previous response:\n{previous_response}\n\nUser request: {query_text}")]
                async for chunk in oci_stream_async(messages):
                    content = chunk
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'text': content}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'FORMAT'}})}\n\n"
                return

            # CLARIFICATION_ANSWER
            if route == "CLARIFICATION_ANSWER":
                if active_clarification:
                    _get_clarification_tracker().add_answer(user_id, query_text)
                if not original_query and history:
                    original_query = _extract_original_query_from_history(history)
                combined_query = f"{original_query} - {query_text}" if original_query else query_text

                yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"
                search_result = await retrieve_fast(combined_query, user_id)
                context, sources = search_result["context"], search_result["sources"]

                for src in sources[:3]:
                    yield f"data: {json.dumps({'type': 'source_found', 'source': src.get('source', ''), 'index': sources.index(src) + 1, 'score': src.get('score', 0)})}\n\n"

                _user_profile = _get_conv_manager().get_user_profile(user_id)
                _system_prompt = _SYSTEM_PROMPT_SIMPLE
                if _user_profile:
                    _profile_lines = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in _user_profile.items())
                    _system_prompt = f"{_system_prompt}\n\nUSER PROFILE (apply only when relevant):\n{_profile_lines}"
                messages = [("system", _system_prompt), ("user", f"Question: {combined_query}\n\nContext:\n{context}")]
                answer_text = ""
                async for chunk in oci_stream_async(messages):
                    content = chunk
                    if content:
                        answer_text += content
                        yield f"data: {json.dumps({'type': 'token', 'text': content}, ensure_ascii=False)}\n\n"

                _get_clarification_tracker().complete_session(user_id)
                _get_conv_manager().add_message(user_id, "user", query_text, {"request_id": request_id})
                _get_conv_manager().add_message(user_id, "assistant", answer_text, {"request_id": request_id, "query_type": "clarification_answer"})
                yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'CLARIFICATION_ANSWER', 'sources': sources[:5]}})}\n\n"
                return

            # RAG PROCESSING
            t0 = time_module.time()
            needs_rewrite = classification.needs_query_rewrite if classification else False
            if needs_rewrite and history:
                rewritten_query = await asyncio.to_thread(
                    rewrite_query_with_history, history, query_text, user_id,
                    clarification_tracker, oci_client, OCI_CHAT_MODEL,
                )
            else:
                rewritten_query = query_text
            timings["2_rewrite"] = (time_module.time() - t0)

            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"
            search_result = await retrieve_fast(rewritten_query, user_id)
            context, sources = search_result["context"], search_result["sources"]

            for idx, src in enumerate(sources[:5]):
                yield f"data: {json.dumps({'type': 'source_found', 'source': src.get('source', ''), 'index': idx + 1, 'score': src.get('score', 0)})}\n\n"

            yield f"data: {json.dumps({'type': 'progress', 'percentage': 60, 'message': 'Generating answer...'})}\n\n"

            chain_depth = _get_clarification_tracker().get_chain_depth(user_id)
            force_direct = chain_depth >= 3
            if force_direct:
                _get_clarification_tracker().complete_session(user_id)

            system_prompt = _SYSTEM_PROMPT_SIMPLE if force_direct else _SYSTEM_PROMPT_CLARIFY

            # Inject user profile (role/country/brand/department) into the system prompt — generation only
            _user_profile = _get_conv_manager().get_user_profile(user_id)
            if _user_profile:
                _profile_lines = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in _user_profile.items())
                system_prompt = f"{system_prompt}\n\nUSER PROFILE (apply only when relevant):\n{_profile_lines}"

            messages = [("system", system_prompt), ("user", f"Question: {rewritten_query}\n\nContext:\n{context}")]

            first_token_time = None
            answer_text = ""
            prefix_buffer = ""
            prefix_resolved = False

            async for chunk in oci_stream_async(messages):
                content = chunk
                if content:
                    answer_text += content
                    if not prefix_resolved:
                        prefix_buffer += content
                        if prefix_buffer.startswith("ANSWER:"):
                            prefix_resolved = True
                            remainder = prefix_buffer[len("ANSWER:"):].lstrip()
                            if remainder:
                                if first_token_time is None:
                                    first_token_time = time_module.time() - start_time
                                yield f"data: {json.dumps({'type': 'token', 'text': remainder}, ensure_ascii=False)}\n\n"
                        elif prefix_buffer.startswith("CLARIFY:"):
                            prefix_resolved = True
                            remainder = prefix_buffer[len("CLARIFY:"):].lstrip()
                            if remainder:
                                if first_token_time is None:
                                    first_token_time = time_module.time() - start_time
                                yield f"data: {json.dumps({'type': 'token', 'text': remainder}, ensure_ascii=False)}\n\n"
                        elif len(prefix_buffer) >= 10:
                            prefix_resolved = True
                            if first_token_time is None:
                                first_token_time = time_module.time() - start_time
                            yield f"data: {json.dumps({'type': 'token', 'text': prefix_buffer}, ensure_ascii=False)}\n\n"
                    else:
                        if first_token_time is None:
                            first_token_time = time_module.time() - start_time
                        yield f"data: {json.dumps({'type': 'token', 'text': content}, ensure_ascii=False)}\n\n"

            if not prefix_resolved and prefix_buffer:
                yield f"data: {json.dumps({'type': 'token', 'text': prefix_buffer}, ensure_ascii=False)}\n\n"

            # Handle CLARIFY session
            if answer_text.startswith("ANSWER:"):
                answer_text = answer_text[len("ANSWER:"):].strip()
            elif answer_text.startswith("CLARIFY:"):
                clarification_text = answer_text[len("CLARIFY:"):].strip()
                _get_clarification_tracker().create_session(user_id, rewritten_query, [clarification_text], context, sources)
                answer_text = clarification_text

            _get_conv_manager().add_message(user_id, "user", query_text, {"request_id": request_id})
            _get_conv_manager().add_message(user_id, "assistant", answer_text, {"request_id": request_id, "query_type": route.lower()})

            total_elapsed = time_module.time() - start_time
            yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': route, 'sources': sources[:5], 'elapsed_sec': round(total_elapsed, 3), 'ttft': round(first_token_time or 0, 3)}})}\n\n"

        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


# =====================================================================
# Health + Root
# =====================================================================
@app.get("/health")
async def health():
    try:
        db_health = oracle_vectordb.get_health()
        return {
            "status": db_health.get("status", "ok"),
            "tier": "oci",
            "backend": "oracle_26ai",
            "total_chunks": db_health.get("total_chunks", 0),
            "total_docs": db_health.get("total_docs", 0),
            "pipeline": "Gemini Flash DU + Cohere Embed v4.0 + Oracle AI Vector Search",
            "models": {"chat": OCI_CHAT_MODEL, "embedding": "cohere.embed-v4.0"},
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/")
async def root():
    return {
        "service": "OCI RAG API Server",
        "endpoints": {
            "/query": "POST — JSON response",
            "/query/stream": "POST — SSE streaming",
            "/health": "GET — service status",
        },
    }


if __name__ == "__main__":
    import uvicorn
    workers = int(os.getenv("RAG_WORKERS", "4"))
    uvicorn.run(
        "rag_server_oci:app",     # String import for multi-worker fork
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        workers=workers,
        timeout_keep_alive=300,
    )
