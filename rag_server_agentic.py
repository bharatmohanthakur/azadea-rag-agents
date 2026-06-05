#!/usr/bin/env python3
"""
AGENTIC RAG API Service — Dual-Agent Architecture

Two-agent system where the LLM decides when full documents are needed:

  Agent 1 (Knowledge Analyst): Receives chunks from get_knowledge, decides:
    → Answer directly if chunks are sufficient
    → Request full document loading if chunks are incomplete

  Agent 2 (Document Analyst): Only activated when Agent 1 requests it.
    Receives query + chunks + full document, produces final answer.

No native function calling / tool_choice needed — avoids Gemini 3's
thought_signature issue entirely. Each agent is a single LLM call.

Port: 7868
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

from rag_server_gemini import (
    openrouter_client, azure_embedding_client,
    OPENROUTER_API_KEY,
    conv_manager, clarification_tracker, get_user_history,
    format_gfm_to_html, log_request, logger,
    count_tokens,
    rewrite_query_with_history,
)

from unified_classifier import UnifiedClassifier
import azure_doc_intelligence_qdrant as rag_impl
from qdrant_client import QdrantClient, models as qm

load_dotenv()

# =====================================================================
# CONFIGURATION
# =====================================================================
QDRANT_LOCAL_URL = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
qdrant_client = QdrantClient(url=QDRANT_LOCAL_URL, check_compatibility=False)

COLLECTION_NAME_V2 = os.getenv(
    "QDRANT_COLLECTION_V2",
    "docs_hybrid_azure_azadea_multimodal_updated"
)

FLASH_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemini-3-flash-preview"
)

# =====================================================================
# LLM Client — single model, reasoning minimal
# =====================================================================
agent_llm = ChatOpenAI(
    model=FLASH_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=4000,
    extra_body={
        "reasoning": {"effort": "minimal"},
        "provider": {"sort": "latency"},
    },
)

# =====================================================================
# Embedding LRU Cache
# =====================================================================
_EMBED_CACHE_MAX = 500
_embed_cache: OrderedDict = OrderedDict()


def _cached_embed_dense(text: str) -> List[float]:
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
# Retrieval — Chunks from Qdrant
# =====================================================================
async def retrieve_chunks(
    query: str,
    user_id: str,
    top_k: int = 7
) -> Dict[str, Any]:
    sources = []
    retrieval_start = datetime.now()

    try:
        t0 = time_module.time()
        loop = asyncio.get_event_loop()

        dense_future = loop.run_in_executor(None, _cached_embed_dense, query)
        sparse_future = loop.run_in_executor(None, rag_impl.build_sparse_query_vector, query)
        dense_q, sparse_q = await asyncio.gather(dense_future, sparse_future)
        embed_time = time_module.time() - t0

        t0 = datetime.now()
        try:
            content_search = await loop.run_in_executor(
                None,
                lambda: qdrant_client.query_points(
                    collection_name=COLLECTION_NAME_V2,
                    prefetch=[
                        qm.Prefetch(query=dense_q, using="dense", limit=20),
                        qm.Prefetch(query=sparse_q, using="sparse", limit=20),
                    ],
                    query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                    limit=top_k + 3,
                )
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return {"context": f"**Error**: Search failed: {type(e).__name__}", "sources": [], "source_files": []}

        search_time = (datetime.now() - t0).total_seconds()

        if not content_search or not content_search.points:
            return {"context": f"No documents found for: {query}", "sources": [], "source_files": []}

        t0 = datetime.now()
        context_parts = []
        doc_ids_seen = set()
        source_files_seen = set()

        for i, p in enumerate(content_search.points[:top_k]):
            pl = p.payload or {}
            src_file = pl.get("source_file", "unknown")
            chunk_type = pl.get("chunk_type", "text")
            doc_id = pl.get("doc_id", "unknown")
            text = pl.get("text", "")
            full_table = pl.get("full_table", "")

            doc_ids_seen.add(doc_id)
            source_files_seen.add(src_file)

            sources.append({
                "id": p.id,
                "score": round(p.score or 0.5, 4),
                "source": src_file,
                "text_snippet": text[:200],
                "chunk_type": chunk_type,
            })

            if chunk_type == "table_summary" and full_table:
                context_parts.append(f"[{src_file}] Table:\n{full_table}")
            elif text:
                context_parts.append(f"[{src_file}] {text}")

        # Fetch neighboring table chunks
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
                        context_parts.append(f"[{tpl.get('source_file', '')}] Table:\n{ft}")
            except Exception as e:
                logger.warning(f"Neighbor table fetch failed for {doc_id}: {e}")

        context = "\n\n".join(context_parts)
        if len(context) > 20000:
            context = context[:20000] + "\n\n[Context truncated]"
        ctx_time = (datetime.now() - t0).total_seconds()

        total_time = (datetime.now() - retrieval_start).total_seconds()
        logger.info(
            f"RETRIEVAL: embed={embed_time:.3f}s, search={search_time:.3f}s, "
            f"ctx={ctx_time:.3f}s, total={total_time:.3f}s, "
            f"ctx_chars={len(context)}, chunks={len(context_parts)}"
        )

        return {
            "context": context,
            "sources": sources,
            "source_files": list(source_files_seen),
        }

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {"context": f"Error searching: {str(e)}", "sources": [], "source_files": []}


# =====================================================================
# Full Document Retrieval
# =====================================================================
async def load_full_document(source_file: str) -> str:
    try:
        loop = asyncio.get_event_loop()
        possible_paths = [
            "/home/admincsp/multimodal-rag/azadea/md_out_data_multimodal",
            "/home/admincsp/multimodal-rag/azadea/md_out_data",
        ]
        # Try .md extension if original has .pdf
        filenames_to_try = [source_file]
        if source_file.endswith(".pdf"):
            filenames_to_try.append(source_file.replace(".pdf", ".md"))

        for base_path in possible_paths:
            for fname in filenames_to_try:
                full_path = os.path.join(base_path, fname)
                if os.path.exists(full_path):
                    content = await loop.run_in_executor(
                        None, lambda p=full_path: open(p, 'r', encoding='utf-8', errors='ignore').read()
                    )
                    logger.info(f"FULL_DOC loaded '{fname}': {len(content)} chars")
                    return content

        logger.warning(f"FULL_DOC not found: {source_file}")
        return ""
    except Exception as e:
        logger.warning(f"FULL_DOC error '{source_file}': {e}")
        return ""


# =====================================================================
# Rule-Based Fast Classification
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
    q = query.strip().lower()
    words = q.split()
    n = len(words)

    if n <= 4 and any(q == g or q.startswith(g + " ") or q.startswith(g + ",") or q.startswith(g + "!") for g in _GREETINGS):
        return "CONVERSATIONAL", "Hello! How can I help you with HR questions today?"
    if n <= 5 and any(t in q for t in _THANKS):
        return "CONVERSATIONAL", "You're welcome! Let me know if you have any other questions."
    if q in _CASUAL:
        if q in ("bye", "goodbye"):
            return "CONVERSATIONAL", "Goodbye! Feel free to come back anytime."
        if q in ("no thanks", "nope", "no"):
            return "CONVERSATIONAL", "Alright! Let me know if you need anything else."
        return "CONVERSATIONAL", "Is there anything else I can help you with?"
    if previous_response and any(kw in q for kw in _FORMAT_KW):
        return "FORMAT", None
    if active_clarification and n <= 5:
        return "CLARIFICATION_ANSWER", None

    return None, None


# =====================================================================
# AGENT 1 — Knowledge Analyst
# =====================================================================
_AGENT1_PROMPT = """You are Agent 1 (Knowledge Analyst) for Azadea Group HR.

You will receive a user's HR question along with CHUNKS retrieved from the knowledge base.

Your job: Analyze the chunks and decide ONE of two outcomes:

OPTION A — ANSWER DIRECTLY:
If the chunks contain enough information to fully answer the question, provide your answer.
Return JSON: {"action": "answer", "response": "your detailed answer here in markdown"}

OPTION B — REQUEST FULL DOCUMENT:
If the chunks are partial, missing key details (tables, numbers, conditions), or don't cover the topic well enough, request the full document.
Return JSON: {"action": "need_full_doc", "source_file": "exact_filename.md", "reason": "brief reason"}
Pick the SINGLE most relevant source file from the chunks.

Rules:
- Include EXACT numbers, dates, percentages from the chunks
- Use bullet points and markdown formatting in your answer
- Fix typos/synonyms: dress code=uniform, leave=vacation, fire=termination, pay=salary
- Respond in the SAME LANGUAGE as the user's question
- Cite source documents (e.g. "per HRD-GEN-001")
- If answer varies by country/position, ask ONE clarifying question in your response
- NEVER invent facts — only use information from the chunks
- If chunks contain NO relevant information at all, return: {"action": "answer", "response": "This information is not available in the HR documents I have access to."}

You MUST return valid JSON only. No text outside the JSON."""


async def run_agent1(query: str, chunks_context: str, source_files: List[str]) -> dict:
    """Agent 1: Analyze chunks, decide whether to answer or request full doc."""
    source_list = ", ".join(source_files[:5])
    messages = [
        ("system", _AGENT1_PROMPT),
        ("user", f"Question: {query}\n\nAvailable source files: {source_list}\n\nChunks:\n{chunks_context}"),
    ]

    response = await agent_llm.ainvoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    # Parse JSON from response — robust extraction
    try:
        cleaned = raw.strip()
        # Strip markdown code fences
        if "```" in cleaned:
            # Find content between first ``` and last ```
            parts = cleaned.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    cleaned = part
                    break

        # Find the JSON object in the text (skip any preamble)
        start_idx = cleaned.find("{")
        if start_idx >= 0:
            # Find matching closing brace
            depth = 0
            for i in range(start_idx, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = cleaned[start_idx:i + 1]
                        break
            else:
                json_str = cleaned[start_idx:]

            # Try strict parse first, then lenient (allow control chars)
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                # LLM often puts literal newlines inside JSON strings — fix them
                result = json.loads(json_str, strict=False)

            logger.info(f"AGENT1 decision: action={result.get('action')}, "
                        f"source_file={result.get('source_file', 'N/A')}")
            return result

        # No JSON object found
        logger.warning(f"AGENT1 no JSON found, using raw response")
        return {"action": "answer", "response": raw}
    except json.JSONDecodeError as e:
        logger.warning(f"AGENT1 JSON parse failed: {e}")
        # Last resort: extract action and response manually
        if '"action"' in raw and '"need_full_doc"' in raw:
            # Try to extract source_file
            import re
            sf_match = re.search(r'"source_file"\s*:\s*"([^"]+)"', raw)
            return {
                "action": "need_full_doc",
                "source_file": sf_match.group(1) if sf_match else "",
                "reason": "parsed from malformed JSON",
            }
        # Treat as direct answer — strip any JSON wrapper if visible
        answer = raw
        if '"response"' in answer:
            import re
            resp_match = re.search(r'"response"\s*:\s*"(.*)', answer, re.DOTALL)
            if resp_match:
                answer = resp_match.group(1).rstrip('"}').replace('\\"', '"').replace('\\n', '\n')
        return {"action": "answer", "response": answer}


# =====================================================================
# AGENT 2 — Document Analyst
# =====================================================================
_AGENT2_PROMPT = """You are Agent 2 (Document Analyst) for Azadea Group HR.

You receive a user's question, initial chunks from the knowledge base, AND the full source document.
Your job is to provide a comprehensive, detailed answer using ALL available information.

Rules:
- Include EXACT numbers, dates, percentages, conditions from the document
- Use bullet points, numbered lists, and tables for clarity
- Fix typos/synonyms: dress code=uniform, leave=vacation, fire=termination, pay=salary
- Respond in the SAME LANGUAGE as the user's question
- Cite the source document (e.g. "per HRD-GEN-001")
- If answer varies by country/position, ask ONE clarifying question
- NEVER invent facts — only use information from the provided document and chunks
- Be thorough — include all relevant details from the full document"""


async def run_agent2(query: str, chunks_context: str, full_doc: str) -> str:
    """Agent 2: Generate answer using chunks + full document."""
    # Cap full doc to prevent context overflow
    if len(full_doc) > 30000:
        full_doc = full_doc[:30000] + "\n\n[Document truncated]"

    messages = [
        ("system", _AGENT2_PROMPT),
        ("user", f"Question: {query}\n\nInitial chunks:\n{chunks_context}\n\n"
                 f"Full document:\n{full_doc}"),
    ]

    response = await agent_llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)


async def stream_agent2(query: str, chunks_context: str, full_doc: str):
    """Agent 2 streaming version — yields tokens as they arrive."""
    if len(full_doc) > 30000:
        full_doc = full_doc[:30000] + "\n\n[Document truncated]"

    messages = [
        ("system", _AGENT2_PROMPT),
        ("user", f"Question: {query}\n\nInitial chunks:\n{chunks_context}\n\n"
                 f"Full document:\n{full_doc}"),
    ]

    async for chunk in agent_llm.astream(messages):
        content = chunk.content if hasattr(chunk, "content") else ""
        if content:
            yield content


# =====================================================================
# App Setup
# =====================================================================
app = FastAPI(title="RAG API Service - AGENTIC DUAL-AGENT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Unified Classifier
_unified_classifier: Optional[UnifiedClassifier] = None


def get_or_init_unified_classifier() -> UnifiedClassifier:
    global _unified_classifier
    if _unified_classifier is None:
        _unified_classifier = UnifiedClassifier(
            llm_client=openrouter_client,
            deployment_name=FLASH_MODEL,
            cache_enabled=True,
            cache_ttl_seconds=300
        )
        logger.info("Unified classifier initialized")
    return _unified_classifier


# =====================================================================
# Query Endpoint — Dual-Agent
# =====================================================================
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"


class QueryResponse(BaseModel):
    response: str
    metadata: Dict[str, Any] = {}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    request_id = str(uuid.uuid4())[:8]
    start_time = time_module.time()
    timings = {}

    try:
        query_text = request.query.strip()
        user_id = request.user_id or "default_user"

        log_request(request_id, "AGENTIC_START", {"query": query_text})

        # State
        history = get_user_history(user_id, use_summarization=False)
        active_clarification = clarification_tracker.get_active_session(user_id)
        previous_response = ""
        if history:
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    previous_response = msg.get("content", "")
                    break

        # Phase 1: Classification
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
            logger.info(f"LLM classification: route={route}")

        # CONVERSATIONAL
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

        # FORMAT
        if route == "FORMAT" and previous_response:
            t0 = time_module.time()
            messages = [
                ("system", "Reformat the previous response as requested. Keep all information."),
                ("user", f"Previous response:\n{previous_response}\n\nRequest: {query_text}"),
            ]
            response = await agent_llm.ainvoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)
            timings["2_format"] = time_module.time() - t0

            total_elapsed = time_module.time() - start_time
            conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
            conv_manager.add_message(user_id, "assistant", answer_text, {
                "request_id": request_id, "query_type": "format",
            })
            return QueryResponse(
                response=format_gfm_to_html(answer_text),
                metadata={"request_id": request_id, "route": "FORMAT",
                          "elapsed_sec": round(total_elapsed, 3), "timings": timings},
            )

        # CLARIFICATION_ANSWER
        if route == "CLARIFICATION_ANSWER":
            t0 = time_module.time()
            if active_clarification:
                clarification_tracker.add_answer(user_id, query_text)
            original_query = getattr(active_clarification, "original_query", "") if active_clarification else ""
            combined_query = f"{original_query} - {query_text}" if original_query else query_text
            query_text_for_rag = combined_query
            route = "SIMPLE"  # Fall through to agentic RAG
            timings["2_clarification_prep"] = time_module.time() - t0
        else:
            query_text_for_rag = query_text

        # Query rewrite for follow-ups
        t0 = time_module.time()
        needs_rewrite = classification.needs_query_rewrite if classification else False
        if needs_rewrite and history:
            rewritten_query = await asyncio.to_thread(rewrite_query_with_history, history, query_text_for_rag, user_id)
        else:
            rewritten_query = query_text_for_rag
        timings["2_rewrite"] = time_module.time() - t0

        # =============================================================
        # AGENTIC RAG — Dual-Agent Flow
        # =============================================================

        # Step 1: Retrieve chunks
        t0 = time_module.time()
        search_result = await retrieve_chunks(rewritten_query, user_id)
        chunks_context = search_result["context"]
        sources = search_result["sources"]
        source_files = search_result.get("source_files", [])
        timings["3_retrieve"] = time_module.time() - t0

        # Step 2: Agent 1 — analyze chunks, decide action
        t0 = time_module.time()
        agent1_result = await run_agent1(rewritten_query, chunks_context, source_files)
        timings["4_agent1"] = time_module.time() - t0

        action = agent1_result.get("action", "answer")
        used_full_doc = False

        if action == "need_full_doc":
            # Step 3: Load full document
            requested_file = agent1_result.get("source_file", "")
            reason = agent1_result.get("reason", "")
            logger.info(f"AGENT1 → AGENT2: need_full_doc='{requested_file}', reason='{reason}'")

            t0 = time_module.time()
            full_doc = await load_full_document(requested_file)
            timings["5_load_doc"] = time_module.time() - t0

            if full_doc:
                # Step 4: Agent 2 — answer with full doc
                t0 = time_module.time()
                answer_text = await run_agent2(rewritten_query, chunks_context, full_doc)
                timings["6_agent2"] = time_module.time() - t0
                used_full_doc = True
            else:
                # Full doc not found, use Agent 1's chunk-based knowledge
                logger.warning(f"Full doc not found: '{requested_file}', falling back to Agent 1 re-answer")
                t0 = time_module.time()
                # Re-ask Agent 1 to answer with what it has
                fallback_messages = [
                    ("system", _AGENT2_PROMPT),
                    ("user", f"Question: {rewritten_query}\n\nContext:\n{chunks_context}\n\n"
                             f"Note: The full document '{requested_file}' was not found. "
                             f"Answer as best you can with the available chunks."),
                ]
                response = await agent_llm.ainvoke(fallback_messages)
                answer_text = response.content if hasattr(response, "content") else str(response)
                timings["6_fallback"] = time_module.time() - t0
        else:
            # Agent 1 answered directly
            answer_text = agent1_result.get("response", "")

        # Handle GENERIC clarification detection
        is_clarifying = any(p in answer_text.lower() for p in [
            "could you please tell me", "could you specify", "which country",
            "what is your", "can you provide", "please specify",
        ])
        if is_clarifying:
            clarification_tracker.create_session(user_id, rewritten_query, [answer_text], chunks_context, sources)

        # Finalize
        total_elapsed = time_module.time() - start_time
        conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
        conv_manager.add_message(user_id, "assistant", answer_text, {
            "request_id": request_id, "query_type": "agentic",
            "elapsed_sec": round(total_elapsed, 3),
        })

        if active_clarification:
            clarification_tracker.complete_session(user_id)

        log_request(request_id, "AGENTIC_COMPLETE", {
            "elapsed_sec": round(total_elapsed, 3),
            "agent1_action": action,
            "used_full_doc": used_full_doc,
            "timings": {k: round(v, 3) for k, v in timings.items()},
        })

        return QueryResponse(
            response=format_gfm_to_html(answer_text),
            metadata={
                "request_id": request_id,
                "route": "AGENTIC",
                "agent1_action": action,
                "used_full_doc": used_full_doc,
                "sources": sources[:5],
                "elapsed_sec": round(total_elapsed, 3),
                "timings": {k: round(v, 3) for k, v in timings.items()},
            },
        )

    except Exception as e:
        logger.error(f"[{request_id}] Agentic error: {e}", exc_info=True)
        elapsed = time_module.time() - start_time
        return QueryResponse(
            response="I apologize, but I encountered an error. Please try again.",
            metadata={"request_id": request_id, "error": str(e), "elapsed_sec": round(elapsed, 3)},
        )


# =====================================================================
# SSE Streaming Endpoint
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

            log_request(request_id, "STREAM_AGENTIC_START", {"query": query_text})
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...'})}\n\n"

            # Classification
            history = get_user_history(user_id, use_summarization=False)
            active_clarification = clarification_tracker.get_active_session(user_id)
            previous_response = ""
            if history:
                for msg in reversed(history):
                    if msg.get("role") == "assistant":
                        previous_response = msg.get("content", "")
                        break

            t0 = time_module.time()
            fast_route, fast_resp = _fast_classify(
                query_text, previous_response, bool(active_clarification)
            )

            if fast_route:
                route = fast_route
                classification = None
                timings["1_classify"] = 0.0
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
                fast_resp = classification.conversational_response if classification.is_conversational else None
                timings["1_classify"] = time_module.time() - t0

            # CONVERSATIONAL
            if route == "CONVERSATIONAL":
                response_text = fast_resp or "Hello! How can I help you with HR questions today?"
                for word in response_text.split():
                    yield f"data: {json.dumps({'type': 'token', 'text': word + ' '}, ensure_ascii=False)}\n\n"
                conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
                conv_manager.add_message(user_id, "assistant", response_text, {"request_id": request_id, "query_type": "conversational"})
                total_elapsed = time_module.time() - start_time
                yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'CONVERSATIONAL', 'elapsed_sec': round(total_elapsed, 3)}})}\n\n"
                return

            # FORMAT
            if route == "FORMAT" and previous_response:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Reformatting...'})}\n\n"
                messages = [
                    ("system", "Reformat the previous response as requested. Keep all information."),
                    ("user", f"Previous response:\n{previous_response}\n\nRequest: {query_text}"),
                ]
                answer_text = ""
                async for chunk in agent_llm.astream(messages):
                    content = chunk.content if hasattr(chunk, "content") else ""
                    if content:
                        answer_text += content
                        yield f"data: {json.dumps({'type': 'token', 'text': content}, ensure_ascii=False)}\n\n"
                conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
                conv_manager.add_message(user_id, "assistant", answer_text, {"request_id": request_id, "query_type": "format"})
                total_elapsed = time_module.time() - start_time
                yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'FORMAT', 'elapsed_sec': round(total_elapsed, 3)}})}\n\n"
                return

            # CLARIFICATION_ANSWER
            if route == "CLARIFICATION_ANSWER":
                if active_clarification:
                    clarification_tracker.add_answer(user_id, query_text)
                original_query_text = getattr(active_clarification, "original_query", "") if active_clarification else ""
                query_text_for_rag = f"{original_query_text} - {query_text}" if original_query_text else query_text
            else:
                query_text_for_rag = query_text

            # Query rewrite
            needs_rewrite = classification.needs_query_rewrite if classification else False
            if needs_rewrite and history:
                rewritten_query = await asyncio.to_thread(rewrite_query_with_history, history, query_text_for_rag, user_id)
            else:
                rewritten_query = query_text_for_rag

            # =============================================================
            # AGENTIC RAG — Streaming
            # =============================================================

            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"

            # Retrieve chunks
            t0 = time_module.time()
            search_result = await retrieve_chunks(rewritten_query, user_id)
            chunks_context = search_result["context"]
            sources = search_result["sources"]
            source_files = search_result.get("source_files", [])
            timings["3_retrieve"] = time_module.time() - t0

            for idx, src in enumerate(sources[:5]):
                yield f"data: {json.dumps({'type': 'source_found', 'source': src.get('source', ''), 'index': idx + 1, 'score': src.get('score', 0)})}\n\n"

            # Agent 1
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing knowledge base results...'})}\n\n"
            t0 = time_module.time()
            agent1_result = await run_agent1(rewritten_query, chunks_context, source_files)
            timings["4_agent1"] = time_module.time() - t0

            action = agent1_result.get("action", "answer")
            used_full_doc = False
            first_token_time = None

            if action == "need_full_doc":
                requested_file = agent1_result.get("source_file", "")
                logger.info(f"STREAM AGENT1 → AGENT2: need_full_doc='{requested_file}'")

                yield f"data: {json.dumps({'type': 'status', 'message': f'Loading full document...'})}\n\n"

                t0 = time_module.time()
                full_doc = await load_full_document(requested_file)
                timings["5_load_doc"] = time_module.time() - t0

                if full_doc:
                    yield f"data: {json.dumps({'type': 'progress', 'percentage': 60, 'message': 'Generating detailed answer...'})}\n\n"

                    t0 = time_module.time()
                    answer_text = ""
                    async for token in stream_agent2(rewritten_query, chunks_context, full_doc):
                        if first_token_time is None:
                            first_token_time = time_module.time() - start_time
                        answer_text += token
                        yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"
                    timings["6_agent2"] = time_module.time() - t0
                    used_full_doc = True
                else:
                    # Fallback — stream with chunks only
                    yield f"data: {json.dumps({'type': 'progress', 'percentage': 60, 'message': 'Generating answer...'})}\n\n"
                    fallback_messages = [
                        ("system", _AGENT2_PROMPT),
                        ("user", f"Question: {rewritten_query}\n\nContext:\n{chunks_context}"),
                    ]
                    answer_text = ""
                    async for chunk in agent_llm.astream(fallback_messages):
                        content = chunk.content if hasattr(chunk, "content") else ""
                        if content:
                            if first_token_time is None:
                                first_token_time = time_module.time() - start_time
                            answer_text += content
                            yield f"data: {json.dumps({'type': 'token', 'text': content}, ensure_ascii=False)}\n\n"
                    timings["6_fallback"] = time_module.time() - t0
            else:
                # Agent 1 answered directly — stream its response
                yield f"data: {json.dumps({'type': 'progress', 'percentage': 60, 'message': 'Generating answer...'})}\n\n"
                answer_text = agent1_result.get("response", "")
                first_token_time = time_module.time() - start_time
                # Stream the pre-generated answer in word chunks
                words = answer_text.split()
                for i in range(0, len(words), 3):
                    chunk_text = " ".join(words[i:i + 3])
                    if i + 3 < len(words):
                        chunk_text += " "
                    yield f"data: {json.dumps({'type': 'token', 'text': chunk_text}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)

            # Save to history
            conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
            conv_manager.add_message(user_id, "assistant", answer_text, {
                "request_id": request_id, "query_type": "agentic",
            })

            if active_clarification:
                clarification_tracker.complete_session(user_id)

            total_elapsed = time_module.time() - start_time
            log_request(request_id, "STREAM_AGENTIC_COMPLETE", {
                "elapsed_sec": round(total_elapsed, 3),
                "ttft": round(first_token_time or 0, 3),
                "agent1_action": action,
                "used_full_doc": used_full_doc,
                "timings": {k: round(v, 3) for k, v in timings.items()},
            })

            yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'AGENTIC', 'agent1_action': action, 'used_full_doc': used_full_doc, 'sources': sources[:5], 'elapsed_sec': round(total_elapsed, 3), 'ttft': round(first_token_time or 0, 3), 'timings': {k: round(v, 3) if isinstance(v, float) else v for k, v in timings.items()}}})}\n\n"

        except Exception as e:
            logger.error(f"[{request_id}] Stream agentic error: {e}", exc_info=True)
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
# Health Check & Info
# =====================================================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "agentic-dual-agent-v1",
        "collection": COLLECTION_NAME_V2,
        "model": FLASH_MODEL,
        "architecture": "dual-agent (knowledge-analyst + document-analyst)",
    }


@app.get("/")
async def root():
    return {
        "service": "RAG API Service - AGENTIC DUAL-AGENT",
        "version": "1.0",
        "architecture": "Agent 1 (chunks) → Agent 2 (full doc, if needed)",
        "endpoints": {
            "/query": "POST - Agentic query (JSON)",
            "/query/stream": "POST - SSE streaming",
            "/health": "GET - Health check",
        },
    }


# =====================================================================
# Run Server
# =====================================================================
if __name__ == "__main__":
    import uvicorn

    print("Starting AGENTIC DUAL-AGENT RAG Server...")
    print(f"  Architecture: Agent 1 (Knowledge Analyst) → Agent 2 (Document Analyst)")
    print(f"  Collection: {COLLECTION_NAME_V2}")
    print(f"  Model: {FLASH_MODEL}")
    print(f"  Port: 7868")

    uvicorn.run(app, host="0.0.0.0", port=7868)
