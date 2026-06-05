#!/usr/bin/env python3
"""
OPTIMIZED RAG API Service - Sub-4-Second / Sub-Second TTFT Version

Key optimizations:
1. Enriched collection (docs_hybrid_azure_azadea_multimodal_updated) - self-sufficient typed chunks
2. Chunk-only context (no full doc retrieval) - context drops from 50K+ to 3-5K tokens
3. Gemini 3.0 Flash Preview model (google/gemini-3-flash-preview)
4. Rule-based fast classification for obvious queries (0ms vs 1-2s)
5. Embedding LRU cache for repeated/similar queries
6. True SSE streaming endpoint (/query/stream) with token-level delivery
7. Async wraps for all blocking calls
8. Parallel dense+sparse embedding
9. Table-aware retrieval (full_table for table_summary chunks, neighbor expansion)
10. Concise system prompt for shorter, faster responses

Target: sub-second TTFT with streaming, <4s total response

Usage:
    uvicorn rag_server_gemini_optimized:app --host 0.0.0.0 --port 8001
"""

import os
import json
import asyncio
import logging
import uuid
import hashlib
import time as time_module
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Any, List, Optional, AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Import from original server to reuse common code
from rag_server_gemini import (
    # Clients
    openrouter_client, azure_embedding_client,
    OPENROUTER_API_KEY,
    # Conversation management
    conv_manager, clarification_tracker, get_user_history,
    # Utilities
    format_gfm_to_html, log_request, logger,
    count_tokens,
    # RAG implementation
    rewrite_query_with_history,
)

# Import the unified classifier
from unified_classifier import UnifiedClassifier

# Import for retrieval
import azure_doc_intelligence_qdrant as rag_impl
from qdrant_client import QdrantClient, models as qm

load_dotenv()

# =====================================================================
# CONFIGURATION
# =====================================================================

# Local Qdrant client — explicitly use localhost to avoid env var pollution
# (graphiti_core's load_dotenv() overrides QDRANT_URL to a cloud instance)
QDRANT_LOCAL_URL = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
qdrant_client = QdrantClient(url=QDRANT_LOCAL_URL, check_compatibility=False)

# Enriched collection with self-sufficient typed chunks (36 metadata fields, 7 chunk types)
COLLECTION_NAME_V2 = os.getenv(
    "QDRANT_COLLECTION_V2",
    "docs_hybrid_azure_azadea_multimodal_updated"
)

# Flash-Lite model: 2x faster, thinking OFF by default, 6x cheaper
FLASH_LITE_MODEL = os.getenv(
    "OPENROUTER_MODEL_FAST",
    "google/gemini-3-flash-preview"
)

# Full Flash model for complex queries that need deeper reasoning
FLASH_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemini-3-flash-preview"
)

# =====================================================================
# LLM Clients — Flash-Lite for speed, Flash for complex
# =====================================================================
# Gemini 3.0 Flash with reasoning minimal — for SIMPLE, FORMAT, CONVERSATIONAL
agent_llm_fast = ChatOpenAI(
    model=FLASH_LITE_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=2000,
    extra_body={
        "reasoning": {"effort": "minimal"},
        "provider": {"sort": "latency"},
    },
)

# Full Flash for COMPLEX queries — reasoning minimal
agent_llm_complex = ChatOpenAI(
    model=FLASH_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=2000,
    extra_body={
        "reasoning": {"effort": "minimal"},
        "provider": {"sort": "latency"},
    },
)

# ---------------------------------------------------------------------
# Embedding LRU Cache
# ---------------------------------------------------------------------
_EMBED_CACHE_MAX = 500
_embed_cache: OrderedDict = OrderedDict()

def _cached_embed_dense(text: str) -> List[float]:
    """Cache Azure embedding results. Deterministic for same input."""
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in _embed_cache:
        _embed_cache.move_to_end(key)
        return _embed_cache[key]
    vec = rag_impl.embed_dense_azure([text])[0]
    _embed_cache[key] = vec
    if len(_embed_cache) > _EMBED_CACHE_MAX:
        _embed_cache.popitem(last=False)
    return vec


# =====================================================================
# Rule-Based Fast Classification (0ms — no LLM call)
# =====================================================================
_GREETINGS = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
_THANKS = {"thanks", "thank you", "thanks a lot", "appreciate it", "thank you so much"}
_CASUAL = {"ok", "okay", "sure", "got it", "understood", "bye", "goodbye", "great", "perfect", "awesome",
           "ok thanks", "okay thanks", "ok thank you", "alright", "no thanks", "nope", "yes", "no"}
_FORMAT_KW = ["as table", "as a table", "in table", "table format",
              "as points", "bullet points", "as list", "as a list",
              "in arabic", "in french", "translate", "summarize this",
              "make it shorter", "shorter", "more detail", "elaborate"]


def _fast_classify(query: str, previous_response: str = "", active_clarification: bool = False):
    """
    Rule-based instant classifier. Returns (route, response) or (None, None).
    Covers ~30% of queries with zero latency.
    """
    q = query.strip().lower()
    words = q.split()
    n = len(words)

    # Greetings ALWAYS win — even during active clarification
    if n <= 4 and any(q == g or q.startswith(g + " ") or q.startswith(g + ",") or q.startswith(g + "!") for g in _GREETINGS):
        return "CONVERSATIONAL", "Hello! How can I help you with HR questions today?"

    # Thanks ALWAYS win — even during active clarification
    if n <= 5 and any(t in q for t in _THANKS):
        return "CONVERSATIONAL", "You're welcome! Let me know if you have any other questions."

    # Casual — exact match or multi-word casual phrases
    if q in _CASUAL:
        if q in ("bye", "goodbye"):
            return "CONVERSATIONAL", "Goodbye! Feel free to come back anytime."
        if q in ("no thanks", "nope", "no"):
            return "CONVERSATIONAL", "Alright! Let me know if you need anything else."
        return "CONVERSATIONAL", "Is there anything else I can help you with?"

    # FORMAT request with previous response
    if previous_response and any(kw in q for kw in _FORMAT_KW):
        return "FORMAT", None

    # Clarification answer (short response during active session)
    # Only if NOT a greeting/thanks/casual (checked above)
    if active_clarification and n <= 5:
        return "CLARIFICATION_ANSWER", None

    return None, None

# ---------------------------------------------------------------------
# Optimized App Setup
# ---------------------------------------------------------------------
app = FastAPI(title="RAG API Service - OPTIMIZED")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Initialize Unified Classifier
# ---------------------------------------------------------------------
_unified_classifier: Optional[UnifiedClassifier] = None

def get_or_init_unified_classifier() -> UnifiedClassifier:
    global _unified_classifier
    if _unified_classifier is None:
        _unified_classifier = UnifiedClassifier(
            llm_client=openrouter_client,
            deployment_name=FLASH_LITE_MODEL,  # Use Flash-Lite for classification too
            cache_enabled=True,
            cache_ttl_seconds=300
        )
        logger.info("Unified classifier initialized (Flash-Lite)")
    return _unified_classifier


# ---------------------------------------------------------------------
# OPTIMIZED Retrieval v2 - Chunk-Only Context
# ---------------------------------------------------------------------
async def retrieve_fast(
    query: str,
    user_id: str,
    top_k: int = 7
) -> Dict[str, Any]:
    """
    Fast retrieval using enriched collection with self-sufficient typed chunks.

    Key changes from v1:
    1. Uses enriched collection (COLLECTION_NAME_V2) with typed chunks
    2. Context built from chunk text + full_table payload (no disk I/O)
    3. Neighboring chunk expansion via indexed Qdrant scroll (~6ms)
    4. Embedding cached via LRU cache
    5. Async wraps for all blocking calls
    6. Context size: ~2-5K tokens (vs 50K+ with full docs)
    """
    sources = []
    retrieval_start = datetime.now()

    try:
        # 1. Embed query — parallel dense + sparse, with cache
        t0 = time_module.time()
        loop = asyncio.get_event_loop()

        dense_future = loop.run_in_executor(None, _cached_embed_dense, query)
        sparse_future = loop.run_in_executor(None, rag_impl.build_sparse_query_vector, query)

        dense_q, sparse_q = await asyncio.gather(dense_future, sparse_future)
        embed_time = time_module.time() - t0

        # 2. Run Qdrant hybrid search (dense + sparse with RRF fusion)
        t0 = datetime.now()
        try:
            content_search = await loop.run_in_executor(
                None,
                lambda: qdrant_client.query_points(
                    collection_name=COLLECTION_NAME_V2,
                    prefetch=[
                        qm.Prefetch(query=dense_q, using="dense", limit=20),
                        qm.Prefetch(
                            query=sparse_q,
                            using="sparse", limit=20
                        ),
                    ],
                    query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                    limit=top_k + 3,
                )
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return {"context": f"**Error**: Search failed: {type(e).__name__}", "sources": [], "images": []}

        search_time = (datetime.now() - t0).total_seconds()

        # Handle no results
        if not content_search or not content_search.points:
            return {
                "context": f"No documents found for: {query}",
                "sources": [],
                "images": [],
            }

        # 3. Build context from chunk payloads directly (no disk I/O)
        t0 = datetime.now()
        context_parts = []
        doc_ids_seen = set()
        retrieved_images = []

        for i, p in enumerate(content_search.points[:top_k]):
            pl = p.payload or {}
            src_file = pl.get("source_file", "unknown")
            chunk_type = pl.get("chunk_type", "text")
            doc_id = pl.get("doc_id", "unknown")
            text = pl.get("text", "")
            full_table = pl.get("full_table", "")

            doc_ids_seen.add(doc_id)

            sources.append({
                "id": p.id,
                "score": round(p.score or 0.5, 4),
                "source": src_file,
                "text_snippet": text[:200],
                "chunk_type": chunk_type,
            })

            # For table_summary chunks, use full_table for complete data
            if chunk_type == "table_summary" and full_table:
                context_parts.append(f"[{src_file}] Table:\n{full_table}")
            elif text:
                context_parts.append(f"[{src_file}] {text}")

        # 4. Fetch neighboring table chunks for docs that had page_context hits
        #    This ensures we get table data even when page_context was the top hit
        neighbor_doc_ids = list(doc_ids_seen)[:2]
        for doc_id in neighbor_doc_ids:
            try:
                table_chunks = await loop.run_in_executor(
                    None,
                    lambda did=doc_id: qdrant_client.scroll(
                        collection_name=COLLECTION_NAME_V2,
                        scroll_filter=qm.Filter(must=[
                            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=did)),
                            qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value="table_summary")),
                        ]),
                        limit=10,
                        with_payload=qm.PayloadSelectorInclude(
                            include=["text", "full_table", "source_file", "chunk_type", "page"]
                        ),
                        with_vectors=False,
                    )[0]
                )
                for tp in table_chunks:
                    tpl = tp.payload or {}
                    ft = tpl.get("full_table", "")
                    if ft and ft not in "\n".join(context_parts):
                        context_parts.append(
                            f"[{tpl.get('source_file', '')}] Table:\n{ft}"
                        )
            except Exception as e:
                logger.warning(f"Neighbor table fetch failed for {doc_id}: {e}")

        # Cap context at ~20K chars (~5K tokens) to prevent slow LLM responses on broad queries
        context = "\n\n".join(context_parts)
        if len(context) > 20000:
            context = context[:20000] + "\n\n[Context truncated for performance]"
        ctx_time = (datetime.now() - t0).total_seconds()

        total_time = (datetime.now() - retrieval_start).total_seconds()
        logger.info(
            f"FAST_RETRIEVAL_V2: embed={embed_time:.3f}s, search={search_time:.3f}s, "
            f"ctx={ctx_time:.3f}s, total={total_time:.3f}s, "
            f"ctx_chars={len(context)}, chunks={len(context_parts)}"
        )

        return {"context": context, "sources": sources, "images": retrieved_images}

    except Exception as e:
        logger.error(f"Fast retrieval error: {e}")
        return {"context": f"Error searching: {str(e)}", "sources": [], "images": []}


# ---------------------------------------------------------------------
# OPTIMIZED Query Endpoint
# ---------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"

class QueryResponse(BaseModel):
    response: str
    metadata: Dict[str, Any] = {}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint_optimized(request: QueryRequest):
    """
    Optimized query endpoint. Target: <4s total.

    Pipeline:
    1. Rule-based fast classification (0ms) → LLM classifier fallback
    2. Chunk-only retrieval from enriched collection (~0.5-1s)
    3. Flash-Lite generation with reasoning off (~1-2s)
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time_module.time()
    timings = {}

    try:
        query_text = request.query.strip()
        user_id = request.user_id or "default_user"

        log_request(request_id, "OPTIMIZED_V2_START", {"query": query_text})

        # Get conversation state (skip summarization — adds 1-2s LLM overhead)
        t_state = time_module.time()
        history = get_user_history(user_id, use_summarization=False)
        active_clarification = clarification_tracker.get_active_session(user_id)
        previous_response = ""
        if history:
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    previous_response = msg.get("content", "")
                    break
        state_time = time_module.time() - t_state
        if state_time > 0.1:
            logger.info(f"[{request_id}] SLOW state fetch: {state_time:.3f}s")

        # =============================================================
        # PHASE 1: Classification — rule-based first, LLM fallback
        # =============================================================
        t0 = time_module.time()

        fast_route, fast_response = _fast_classify(
            query_text, previous_response, bool(active_clarification)
        )

        if fast_route:
            route = fast_route
            classification = None
            timings["1_classify"] = 0.0
            logger.info(f"Fast classification: route={route}")
        else:
            classifier = get_or_init_unified_classifier()
            clarification_question = ""
            original_query = ""
            if active_clarification:
                clarification_question = getattr(active_clarification, 'questions', [''])[0] if active_clarification else ""
                original_query = getattr(active_clarification, 'original_query', "") if active_clarification else ""

            classification = await asyncio.to_thread(
                classifier.classify,
                query=query_text,
                conversation_history=history,
                previous_response=previous_response,
                active_clarification=bool(active_clarification),
                clarification_question=clarification_question,
                original_query=original_query,
            )
            route = "CONVERSATIONAL" if classification.is_conversational else (classification.rag_route or "SIMPLE")
            fast_response = classification.conversational_response if classification.is_conversational else None
            timings["1_classify"] = time_module.time() - t0
            logger.info(f"LLM classification: route={route}, confidence={classification.confidence:.2f}")

        # -------------------------------------------------------------
        # PHASE 2: Handle based on classification
        # -------------------------------------------------------------

        # --- CONVERSATIONAL (greeting, thanks, casual) ---
        if route == "CONVERSATIONAL":
            response_text = fast_response or "Hello! How can I help you with HR questions today?"
            total_elapsed = time_module.time() - start_time
            conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
            conv_manager.add_message(user_id, "assistant", response_text, {
                "request_id": request_id, "query_type": "conversational",
                "elapsed_sec": round(total_elapsed, 3),
            })
            log_request(request_id, "CONVERSATIONAL_COMPLETE", {"elapsed_sec": round(total_elapsed, 3)})
            return QueryResponse(
                response=response_text,
                metadata={"request_id": request_id, "route": "CONVERSATIONAL",
                          "elapsed_sec": round(total_elapsed, 3), "timings": timings},
            )

        # --- FORMAT (reformat previous response) ---
        if route == "FORMAT" and previous_response:
            t0 = time_module.time()
            messages = [
                ("system", "Reformat the previous response as requested. Keep all information."),
                ("user", f"Previous response:\n{previous_response}\n\nRequest: {query_text}"),
            ]
            response = await agent_llm_fast.ainvoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)
            timings["2_format"] = time_module.time() - t0

            total_elapsed = time_module.time() - start_time
            conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
            conv_manager.add_message(user_id, "assistant", answer_text, {
                "request_id": request_id, "query_type": "format",
                "elapsed_sec": round(total_elapsed, 3),
            })
            log_request(request_id, "FORMAT_COMPLETE", {"elapsed_sec": round(total_elapsed, 3)})
            return QueryResponse(
                response=format_gfm_to_html(answer_text),
                metadata={"request_id": request_id, "route": "FORMAT",
                          "elapsed_sec": round(total_elapsed, 3), "timings": timings},
            )

        # --- CLARIFICATION_ANSWER ---
        if route == "CLARIFICATION_ANSWER":
            t0 = time_module.time()
            if active_clarification:
                clarification_tracker.add_answer(user_id, query_text)

            original_query = getattr(active_clarification, "original_query", "") if active_clarification else ""
            combined_query = f"{original_query} - {query_text}" if original_query else query_text

            search_result = await retrieve_fast(combined_query, user_id)
            context = search_result["context"]
            sources = search_result["sources"]

            messages = [
                ("system", _SYSTEM_PROMPT_SIMPLE),
                ("user", f"Question: {combined_query}\n\nContext:\n{context}"),
            ]
            response = await agent_llm_fast.ainvoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)

            clarification_tracker.complete_session(user_id)
            timings["2_clarification"] = time_module.time() - t0

            total_elapsed = time_module.time() - start_time
            conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
            conv_manager.add_message(user_id, "assistant", answer_text, {
                "request_id": request_id, "query_type": "clarification_answer",
                "elapsed_sec": round(total_elapsed, 3),
            })
            log_request(request_id, "CLARIFICATION_COMPLETE", {"elapsed_sec": round(total_elapsed, 3)})
            return QueryResponse(
                response=format_gfm_to_html(answer_text),
                metadata={"request_id": request_id, "route": "CLARIFICATION_ANSWER",
                          "sources": sources[:3], "elapsed_sec": round(total_elapsed, 3), "timings": timings},
            )

        # =============================================================
        # PHASE 3: RAG Processing (SIMPLE, COMPLEX, GENERIC)
        # =============================================================

        # Query rewrite if conversation has history
        t0 = time_module.time()
        needs_rewrite = classification.needs_query_rewrite if classification else False
        if needs_rewrite and history:
            rewritten_query = await asyncio.to_thread(rewrite_query_with_history, history, query_text, user_id)
        else:
            rewritten_query = query_text
        timings["2_rewrite"] = time_module.time() - t0

        if route == "GENERIC":
            # Possibly ambiguous — retrieve and check if clarification needed
            t0 = time_module.time()
            search_result = await retrieve_fast(rewritten_query, user_id)
            context = search_result["context"]
            sources = search_result["sources"]

            messages = [
                ("system", _SYSTEM_PROMPT_GENERIC),
                ("user", f"Question: {rewritten_query}\n\nContext:\n{context}"),
            ]
            response = await agent_llm_fast.ainvoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)

            # Check if LLM is asking for clarification
            is_clarifying = any(p in answer_text.lower() for p in [
                "could you please tell me", "could you specify", "which country",
                "what is your", "can you provide", "please specify",
            ])
            if is_clarifying:
                clarification_tracker.create_session(user_id, rewritten_query, [answer_text], context, sources)

            timings["3_generic"] = time_module.time() - t0

        elif route == "COMPLEX":
            # Complex query — decompose and synthesize
            t0 = time_module.time()
            word_count = len(rewritten_query.split())

            # Short queries (<12 words): skip decomposition, single retrieval + flash
            # Long/multi-part queries: decompose into 2 sub-queries, parallel retrieve
            if word_count < 12:
                # Fast path: single retrieval with flash model (no decomposition overhead)
                search_result = await retrieve_fast(rewritten_query, user_id, top_k=7)
                combined_context = search_result["context"]
                sources = search_result.get("sources", [])
            else:
                # Decompose using Flash-Lite (fast, no reasoning)
                decompose_response = await asyncio.to_thread(
                    lambda: openrouter_client.chat.completions.create(
                        model=FLASH_LITE_MODEL,
                        messages=[{"role": "user", "content": f'Break this into exactly 2 focused sub-questions. Return JSON: {{"questions":["q1","q2"]}}.\n\nQuestion: {rewritten_query}'}],
                        temperature=0,
                        response_format={"type": "json_object"},
                        extra_body={"reasoning": {"effort": "minimal"}},
                    )
                )

                try:
                    sub_json = json.loads(decompose_response.choices[0].message.content)
                    sub_queries = sub_json.get("questions", sub_json.get("sub_questions", [rewritten_query]))
                except Exception:
                    sub_queries = [rewritten_query]

                # Execute sub-queries in parallel (max 2, reduced top_k=5 per sub-query)
                sub_results = await asyncio.gather(*[
                    retrieve_fast(sq, user_id, top_k=5) for sq in sub_queries[:2]
                ])

                combined_context = "\n\n".join([
                    f"### {sq}\n{result['context']}"
                    for sq, result in zip(sub_queries[:2], sub_results)
                ])
                sources = []
                for result in sub_results:
                    sources.extend(result.get("sources", []))

            # Synthesize with Gemini 3.0 Flash (reasoning=minimal)
            messages = [
                ("system", _SYSTEM_PROMPT_COMPLEX),
                ("user", f"Question: {rewritten_query}\n\nInformation:\n{combined_context}"),
            ]
            response = await agent_llm_fast.ainvoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)

            timings["3_complex"] = time_module.time() - t0

        else:
            # SIMPLE — direct RAG (most common path)
            t0 = time_module.time()
            search_result = await retrieve_fast(rewritten_query, user_id)
            context = search_result["context"]
            sources = search_result["sources"]

            answer_text = await _generate_answer(rewritten_query, context, agent_llm_fast)
            timings["3_simple"] = time_module.time() - t0

        # =============================================================
        # PHASE 4: Finalize
        # =============================================================
        t_pre_final = time_module.time()
        pre_final_elapsed = t_pre_final - start_time
        conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
        conv_manager.add_message(user_id, "assistant", answer_text, {
            "request_id": request_id, "query_type": route.lower(),
            "elapsed_sec": round(pre_final_elapsed, 3),
        })
        t_format = time_module.time()
        formatted = format_gfm_to_html(answer_text)
        format_time = time_module.time() - t_format
        total_elapsed = time_module.time() - start_time
        timings_sum = sum(timings.values())
        overhead = total_elapsed - timings_sum
        log_request(request_id, "OPTIMIZED_V2_COMPLETE", {
            "elapsed_sec": round(total_elapsed, 3), "route": route,
            "timings": {k: round(v, 3) for k, v in timings.items()},
            "overhead": round(overhead, 3), "format_time": round(format_time, 3),
        })
        return QueryResponse(
            response=formatted,
            metadata={
                "request_id": request_id, "route": route,
                "sources": sources[:5] if sources else [],
                "elapsed_sec": round(total_elapsed, 3),
                "timings": {k: round(v, 3) for k, v in timings.items()},
            },
        )

    except Exception as e:
        logger.error(f"[{request_id}] Optimized v2 error: {e}", exc_info=True)
        elapsed = time_module.time() - start_time
        return QueryResponse(
            response="I apologize, but I encountered an error. Please try again.",
            metadata={"request_id": request_id, "error": str(e), "elapsed_sec": round(elapsed, 3)},
        )


# ---------------------------------------------------------------------
# SSE Streaming Endpoint
# ---------------------------------------------------------------------
@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Streaming version of /query endpoint using Server-Sent Events.
    Delivers tokens as they are generated for sub-second TTFT.

    SSE protocol types (compatible with gradio_streaming_app.py):
    - status: intermediate status messages
    - progress: percentage progress updates
    - source_found: retrieved source info
    - token: text token for accumulation
    - done: final metadata
    - error: error message
    """
    async def generate() -> AsyncGenerator[str, None]:
        request_id = str(uuid.uuid4())[:8]
        start_time = time_module.time()
        timings = {}

        try:
            query_text = request.query.strip()
            user_id = request.user_id or "default_user"

            log_request(request_id, "STREAM_QUERY_START", {"query": query_text})

            # Progress: starting
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...'})}\n\n"

            # Get classifier + history
            classifier = get_or_init_unified_classifier()
            history = get_user_history(user_id, use_summarization=False)

            active_clarification = clarification_tracker.get_active_session(user_id)
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

            # Rule-based fast classification (0ms)
            t0 = time_module.time()
            fast_route, fast_resp = _fast_classify(
                query_text, previous_response, bool(active_clarification)
            )

            if fast_route:
                route = fast_route
                classification = None
                timings["1_classify"] = 0.0
            else:
                classification = await asyncio.to_thread(
                    classifier.classify,
                    query=query_text,
                    conversation_history=history,
                    previous_response=previous_response,
                    active_clarification=bool(active_clarification),
                    clarification_question=clarification_question,
                    original_query=original_query,
                )
                route = "CONVERSATIONAL" if classification.is_conversational else (classification.rag_route or "SIMPLE")
                fast_resp = classification.conversational_response if classification.is_conversational else None
                timings["1_classify"] = time_module.time() - t0

            # CONVERSATIONAL - stream the response directly
            if route == "CONVERSATIONAL":
                response_text = fast_resp or "Hello! How can I help you with HR questions today?"

                for word in response_text.split():
                    yield f"data: {json.dumps({'type': 'token', 'text': word + ' '}, ensure_ascii=False)}\n\n"

                conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
                conv_manager.add_message(user_id, "assistant", response_text, {
                    "request_id": request_id,
                    "query_type": "conversational",
                })

                total_elapsed = time_module.time() - start_time
                yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'CONVERSATIONAL', 'elapsed_sec': round(total_elapsed, 3), 'timings': {k: round(v, 3) if isinstance(v, float) else v for k, v in timings.items()}}})}\n\n"
                return

            # FORMAT - stream reformatted response
            if route == "FORMAT" and previous_response:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Reformatting response...'})}\n\n"
                t0 = datetime.now()
                messages = [
                    ("system", "Reformat the previous response as requested. Keep all information intact."),
                    ("user", f"Previous response:\n{previous_response}\n\nUser request: {query_text}"),
                ]
                async for chunk in agent_llm_fast.astream(messages):
                    content = chunk.content if hasattr(chunk, "content") else ""
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'text': content}, ensure_ascii=False)}\n\n"

                timings["2_format"] = (datetime.now() - t0).total_seconds()
                total_elapsed = time_module.time() - start_time
                yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'FORMAT', 'elapsed_sec': round(total_elapsed, 3), 'timings': {k: round(v, 3) if isinstance(v, float) else v for k, v in timings.items()}}})}\n\n"
                return

            # CLARIFICATION_ANSWER
            if route == "CLARIFICATION_ANSWER":
                if active_clarification:
                    clarification_tracker.add_answer(user_id, query_text)
                combined_query = f"{original_query} - {query_text}" if original_query else query_text

                yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"
                t0 = datetime.now()
                search_result = await retrieve_fast(combined_query, user_id)
                context = search_result["context"]
                sources = search_result["sources"]
                timings["2_retrieve"] = (datetime.now() - t0).total_seconds()

                for src in sources[:3]:
                    yield f"data: {json.dumps({'type': 'source_found', 'source': src.get('source', ''), 'index': sources.index(src) + 1, 'score': src.get('score', 0)})}\n\n"

                messages = [
                    ("system", _SYSTEM_PROMPT_SIMPLE),
                    ("user", f"Question: {combined_query}\n\nContext:\n{context}"),
                ]
                t0 = datetime.now()
                answer_text = ""
                async for chunk in agent_llm_fast.astream(messages):
                    content = chunk.content if hasattr(chunk, "content") else ""
                    if content:
                        answer_text += content
                        yield f"data: {json.dumps({'type': 'token', 'text': content}, ensure_ascii=False)}\n\n"
                timings["3_generate"] = (datetime.now() - t0).total_seconds()

                clarification_tracker.complete_session(user_id)
                conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
                conv_manager.add_message(user_id, "assistant", answer_text, {"request_id": request_id, "query_type": "clarification_answer"})

                total_elapsed = time_module.time() - start_time
                yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'CLARIFICATION_ANSWER', 'sources': sources[:5], 'elapsed_sec': round(total_elapsed, 3), 'timings': {k: round(v, 3) if isinstance(v, float) else v for k, v in timings.items()}}})}\n\n"
                return

            # RAG PROCESSING (SIMPLE, COMPLEX, GENERIC)
            t0 = datetime.now()
            needs_rewrite = classification.needs_query_rewrite if classification else False
            if needs_rewrite and history:
                rewritten_query = await asyncio.to_thread(
                    rewrite_query_with_history, history, query_text, user_id
                )
            else:
                rewritten_query = query_text
            timings["2_rewrite"] = (datetime.now() - t0).total_seconds()

            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"

            # Retrieve
            t0 = datetime.now()
            search_result = await retrieve_fast(rewritten_query, user_id)
            context = search_result["context"]
            sources = search_result["sources"]
            timings["3_retrieve"] = (datetime.now() - t0).total_seconds()

            # Send source_found events
            for idx, src in enumerate(sources[:5]):
                yield f"data: {json.dumps({'type': 'source_found', 'source': src.get('source', ''), 'index': idx + 1, 'score': src.get('score', 0)})}\n\n"

            yield f"data: {json.dumps({'type': 'progress', 'percentage': 60, 'message': 'Generating answer...'})}\n\n"

            # Determine LLM and prompt based on route
            if route == "COMPLEX":
                llm = agent_llm_complex
                system_prompt = _SYSTEM_PROMPT_COMPLEX
            elif route == "GENERIC":
                llm = agent_llm_fast
                system_prompt = _SYSTEM_PROMPT_GENERIC
            else:
                llm = agent_llm_fast
                system_prompt = _SYSTEM_PROMPT_SIMPLE

            messages = [
                ("system", system_prompt),
                ("user", f"Question: {rewritten_query}\n\nContext:\n{context}"),
            ]

            t0 = time_module.time()
            first_token_time = None
            answer_text = ""
            async for chunk in llm.astream(messages):
                content = chunk.content if hasattr(chunk, "content") else ""
                if content:
                    if first_token_time is None:
                        first_token_time = time_module.time() - start_time
                    answer_text += content
                    yield f"data: {json.dumps({'type': 'token', 'text': content}, ensure_ascii=False)}\n\n"
            timings["4_generate"] = time_module.time() - t0

            # Handle GENERIC clarification detection
            if route == "GENERIC":
                is_asking_clarification = any(phrase in answer_text.lower() for phrase in [
                    "could you please tell me", "could you specify", "which country",
                    "what is your", "can you provide", "please specify",
                ])
                if is_asking_clarification:
                    clarification_tracker.create_session(
                        user_id, rewritten_query, [answer_text], context, sources
                    )

            # Save to history
            conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
            conv_manager.add_message(user_id, "assistant", answer_text, {
                "request_id": request_id,
                "query_type": route.lower(),
            })

            total_elapsed = time_module.time() - start_time
            log_request(request_id, "STREAM_V2_COMPLETE", {
                "elapsed_sec": round(total_elapsed, 3),
                "ttft": round(first_token_time or 0, 3),
                "route": route,
                "timings": {k: round(v, 3) if isinstance(v, float) else v for k, v in timings.items()},
            })

            yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': route, 'sources': sources[:5], 'elapsed_sec': round(total_elapsed, 3), 'ttft': round(first_token_time or 0, 3), 'timings': {k: round(v, 3) if isinstance(v, float) else v for k, v in timings.items()}}})}\n\n"

        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =====================================================================
# System Prompts — Concise for faster generation
# =====================================================================
_SYSTEM_PROMPT_SIMPLE = """You are an HR assistant for Azadea Group. Answer using ONLY the provided context.
Rules:
- Include EXACT numbers, dates, percentages from the context
- Use bullet points or short numbered lists when listing multiple items, steps, or conditions
- The user's question may contain typos or synonyms. Match their intent to the context terminology (e.g. "dress code" = "uniform", "leave" = "vacation/annual leave", "fire" = "termination", "pay" = "salary/compensation")
- If the context does NOT contain information to answer the question, say "This information is not available in the HR documents I have access to." and STOP. Do NOT volunteer numbers, dates, or facts from unrelated parts of the context
- Respond in the SAME LANGUAGE as the user's question. If the question is in Arabic, respond in Arabic. If in English, respond in English
- Cite source documents naturally"""

_SYSTEM_PROMPT_COMPLEX = """You are an HR expert for Azadea Group. Synthesize the gathered information into a comprehensive answer.
Rules:
- Include EXACT numbers, dates, percentages from the context
- Cite source policy documents (e.g. "per HRD-GEN-001")
- Structure with bullet points or numbered steps when appropriate
- The user's question may contain typos or synonyms. Match their intent to the context terminology (e.g. "dress code" = "uniform", "leave" = "vacation/annual leave", "probation" = "trial period")
- If the context does NOT contain information to answer the question, say "This information is not available in the HR documents I have access to." and STOP. Do NOT volunteer numbers, dates, or facts from unrelated parts of the context
- Respond in the SAME LANGUAGE as the user's question. If the question is in Arabic, respond in Arabic. If in English, respond in English
- Be thorough but concise"""

_SYSTEM_PROMPT_GENERIC = """You are an HR assistant for Azadea Group. Analyze the context to answer the question.
- The user's question may contain typos or synonyms. Match their intent to the context terminology (e.g. "dress code" = "uniform", "leave" = "vacation/annual leave")
- Respond in the SAME LANGUAGE as the user's question

If the answer VARIES by country, position, or department AND the user hasn't specified:
- Ask ONE clarifying question: "To give you accurate information, could you please tell me [what's missing]?"

If you CAN answer with the context, provide the answer directly with specific details.

If the context does NOT contain information to answer the question, say "This information is not available in the HR documents I have access to." and STOP. Do NOT provide tangentially related information."""


# =====================================================================
# Helper Functions
# =====================================================================
async def _generate_answer(query: str, context: str, llm) -> str:
    """Generate answer using the specified LLM."""
    messages = [
        ("system", _SYSTEM_PROMPT_SIMPLE),
        ("user", f"Question: {query}\n\nContext:\n{context}"),
    ]
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)


def _finalize_response(
    request_id: str,
    start_time: float,
    timings: dict,
    user_id: str,
    query_text: str,
    response_text: str,
    route: str,
    sources: list,
) -> QueryResponse:
    """Save to history and return response."""
    elapsed = time_module.time() - start_time
    conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
    conv_manager.add_message(user_id, "assistant", response_text, {
        "request_id": request_id, "query_type": route.lower(),
        "elapsed_sec": round(elapsed, 3),
    })
    log_request(request_id, f"{route}_COMPLETE", {"elapsed_sec": round(elapsed, 3)})
    return QueryResponse(
        response=response_text,
        metadata={
            "request_id": request_id, "route": route,
            "sources": sources, "elapsed_sec": round(elapsed, 3),
            "timings": {k: round(v, 3) for k, v in timings.items()},
        },
    )


# =====================================================================
# Health Check & Info Endpoints
# =====================================================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "optimized-v2",
        "collection": COLLECTION_NAME_V2,
        "model_fast": FLASH_LITE_MODEL,
        "model_complex": FLASH_MODEL,
        "optimizations": [
            "enriched_collection",
            "chunk_only_retrieval",
            "flash_lite_model",
            "reasoning_effort_none",
            "rule_based_fast_classify",
            "embedding_cache",
            "parallel_embedding",
            "table_aware_retrieval",
            "sse_streaming",
        ],
    }


@app.get("/")
async def root():
    """Root endpoint with info."""
    return {
        "service": "RAG API Service - OPTIMIZED v2",
        "version": "2.0",
        "endpoints": {
            "/query": "POST - Optimized query endpoint (JSON response)",
            "/query/stream": "POST - SSE streaming endpoint (token-level)",
            "/health": "GET - Health check",
        },
        "target_latency": "<4 seconds total, sub-second TTFT",
    }


# ---------------------------------------------------------------------
# Run Server
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("Starting OPTIMIZED RAG Server v2...")
    print(f"  Collection: {COLLECTION_NAME_V2}")
    print(f"  Fast model: {FLASH_LITE_MODEL}")
    print(f"  Complex model: {FLASH_MODEL}")
    print(f"  Endpoints: /query (POST), /query/stream (POST SSE)")
    print(f"  Target: sub-second TTFT, <4s total")

    uvicorn.run(app, host="0.0.0.0", port=7867)
