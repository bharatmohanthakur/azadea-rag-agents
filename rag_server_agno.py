#!/usr/bin/env python3
"""
RAG API Service - Agno Multi-Agent Architecture

Uses Agno Team with OpenRouter for autonomous multi-agent delegation.
Team leader delegates to knowledge_agent (chunks) or doc_agent (full docs).
The LLM decides when to delegate and to whom — no hardcoded flows.

Architecture: Agno Team with delegate_task_to_member
Port: 7869
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

from rag_server_gemini import (
    openrouter_client, azure_embedding_client,
    OPENROUTER_API_KEY,
    conv_manager, clarification_tracker, get_user_history,
    format_gfm_to_html, log_request, logger,
    count_tokens,
)

import azure_doc_intelligence_qdrant as rag_impl
from qdrant_client import QdrantClient, models as qm

from agno.agent import Agent
from agno.team import Team
from agno.tools import tool
from agno.models.openrouter import OpenRouter

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

FLASH_MODEL = "google/gemini-3-flash-preview"

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
# Retrieval — Chunk-Only Context
# =====================================================================
async def retrieve_fast(
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
            return {"context": f"**Error**: Search failed: {type(e).__name__}", "sources": [], "images": []}

        search_time = (datetime.now() - t0).total_seconds()

        if not content_search or not content_search.points:
            return {"context": f"No documents found for: {query}", "sources": [], "images": []}

        t0 = datetime.now()
        context_parts = []
        doc_ids_seen = set()

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

            if chunk_type == "table_summary" and full_table:
                context_parts.append(f"[{src_file}] Table:\n{full_table}")
            elif text:
                context_parts.append(f"[{src_file}] {text}")

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

        context = "\n\n".join(context_parts)
        ctx_time = (datetime.now() - t0).total_seconds()

        total_time = (datetime.now() - retrieval_start).total_seconds()
        logger.info(
            f"FAST_RETRIEVAL_V2: embed={embed_time:.3f}s, search={search_time:.3f}s, "
            f"ctx={ctx_time:.3f}s, total={total_time:.3f}s, "
            f"ctx_chars={len(context)}, chunks={len(context_parts)}"
        )

        return {"context": context, "sources": sources, "images": []}

    except Exception as e:
        logger.error(f"Fast retrieval error: {e}")
        return {"context": f"Error searching: {str(e)}", "sources": [], "images": []}


# =====================================================================
# Full Document Retrieval
# =====================================================================
async def _retrieve_complete_document(source_file: str) -> str:
    try:
        loop = asyncio.get_event_loop()
        possible_paths = [
            "/home/admincsp/multimodal-rag/azadea/md_out_data_multimodal",
            "/home/admincsp/multimodal-rag/azadea/md_out_data",
        ]

        for base_path in possible_paths:
            full_path = os.path.join(base_path, source_file)
            if os.path.exists(full_path):
                content = await loop.run_in_executor(
                    None, lambda p=full_path: open(p, 'r', encoding='utf-8', errors='ignore').read()
                )
                logger.info(f"FULL_DOC loaded '{source_file}': {len(content)} chars")
                return content

        base_name = os.path.splitext(source_file)[0]
        for ext in [".md", ".pdf"]:
            alt_file = base_name + ext
            for base_path in possible_paths:
                full_path = os.path.join(base_path, alt_file)
                if os.path.exists(full_path):
                    content = await loop.run_in_executor(
                        None, lambda p=full_path: open(p, 'r', encoding='utf-8', errors='ignore').read()
                    )
                    logger.info(f"FULL_DOC loaded '{alt_file}' (alt ext): {len(content)} chars")
                    return content

        logger.warning(f"FULL_DOC not found: {source_file}")
        return ""
    except Exception as e:
        logger.warning(f"FULL_DOC error '{source_file}': {e}")
        return ""


# =====================================================================
# Shared state
# =====================================================================
_current_user_id = "default_user"
_current_sources: List[dict] = []
_current_tools_called: List[str] = []
_loaded_docs: set = set()


# =====================================================================
# AGNO TOOL DEFINITIONS
# =====================================================================
@tool
def search_hr_knowledge_base(query: str) -> str:
    """Search the Azadea Group HR knowledge base for policy documents,
    procedures, benefits, leave policies, compensation, dress code, etc.
    Call this for any HR-related question. Fix spelling errors in the query
    before searching. Expand synonyms (dress code -> uniform, leave -> annual leave,
    fire -> termination, pay -> salary/compensation).

    Args:
        query: The search query about HR policies or procedures.

    Returns:
        Relevant HR knowledge base chunks with source file names.
    """
    global _current_sources, _current_tools_called
    _current_tools_called.append("get_knowledge")
    logger.info(f"TOOL get_knowledge: query='{query}'")

    result = asyncio.run(retrieve_fast(query, _current_user_id))

    _current_sources.extend(result.get("sources", []))
    content = result["context"]

    source_files = []
    seen = set()
    for s in result.get("sources", []):
        fname = s.get("source", "")
        if fname and fname not in seen:
            seen.add(fname)
            source_files.append(fname)

    if source_files:
        files_list = ", ".join(source_files[:3])
        content += f"\n\n[Source files: {files_list}]"

    return content


@tool
def get_conversation_history(last_n: int = 10) -> str:
    """Retrieve previous conversation messages with this user.
    Call this if the user references something said earlier,
    asks a follow-up, or says 'show as table', 'in arabic',
    'more detail', etc.

    Args:
        last_n: Number of recent messages to retrieve (default 10).

    Returns:
        Formatted conversation history.
    """
    global _current_tools_called
    _current_tools_called.append("get_conversation_history")
    logger.info(f"TOOL get_conversation_history: last_n={last_n}")

    history = get_user_history(_current_user_id, use_summarization=False)
    recent = history[-last_n:] if history else []
    formatted = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted) if formatted else "No conversation history found."


@tool
def load_full_document(source_file: str) -> str:
    """Load the COMPLETE source document from disk. Use this when the knowledge
    base chunks don't fully answer the question and you need the complete document.
    Pass the exact source file name from the knowledge base search results
    (e.g. 'HRD - GEN - 007 - Employee Attendance - P - 16.md').

    Args:
        source_file: Exact file name from knowledge base results.

    Returns:
        Full document content or error message.
    """
    global _current_tools_called
    _current_tools_called.append("get_full_document")
    logger.info(f"TOOL get_full_document: source_file='{source_file}'")

    if source_file in _loaded_docs:
        logger.info(f"TOOL get_full_document: DEDUP skip '{source_file}'")
        return f"Document '{source_file}' was already loaded. Use the content from before."

    content = asyncio.run(_retrieve_complete_document(source_file))
    if content:
        _loaded_docs.add(source_file)
        return content
    return f"Document not found: {source_file}"


# =====================================================================
# AGNO MODEL
# =====================================================================
agno_model = OpenRouter(
    id=FLASH_MODEL,
    api_key=OPENROUTER_API_KEY,
    max_tokens=4000,
)

# =====================================================================
# AGENTS
# =====================================================================
knowledge_agent = Agent(
    name="knowledge_researcher",
    model=agno_model,
    description="Searches the HR knowledge base chunks and answers questions using retrieved context. Has access to search_hr_knowledge_base and get_conversation_history tools.",
    instructions=[
        "You are a knowledge researcher for Azadea Group HR.",
        "Search the HR knowledge base and provide accurate, detailed answers.",
        "Include EXACT numbers, dates, percentages from the context.",
        "Use bullet points, numbered lists, and tables for clarity.",
        "Cite source documents (e.g. 'per HRD-GEN-001').",
        "Respond in the SAME LANGUAGE as the user's question.",
        "NEVER invent facts — only use information from tool results.",
    ],
    tools=[search_hr_knowledge_base, get_conversation_history],
    markdown=True,
)

doc_agent = Agent(
    name="document_specialist",
    model=agno_model,
    description="Loads and analyzes full HR policy documents when the knowledge base chunks are incomplete or missing key details like exact numbers, tables, conditions, or percentages.",
    instructions=[
        "You are a document specialist for Azadea Group HR.",
        "Load full documents using the exact source file name provided.",
        "Extract ALL relevant information for the question asked.",
        "Include exact numbers, dates, percentages, tables, conditions.",
        "Format with bullet points and markdown.",
        "Cite the source document.",
        "Respond in the SAME LANGUAGE as the original question.",
        "NEVER invent facts — only use information from the loaded document.",
    ],
    tools=[load_full_document],
    markdown=True,
)

# =====================================================================
# TEAM — Leader delegates to the right agent
# =====================================================================
hr_team = Team(
    name="hr_assistant_team",
    model=agno_model,
    members=[knowledge_agent, doc_agent],
    description="Azadea Group HR assistant team that answers employee questions about policies and procedures.",
    instructions=[
        "You are the Azadea Group HR Assistant team leader.",
        "For greetings (hi, hello, thanks) — respond directly without delegating.",
        "For HR questions — delegate to knowledge_researcher first.",
        "If knowledge_researcher's answer is incomplete or missing key details, delegate to document_specialist with the source file name.",
        "For follow-ups referencing earlier conversation — delegate to knowledge_researcher.",
        "Synthesize member responses into a clear, well-formatted answer.",
        "Use markdown: bullet points, numbered lists, tables.",
        "Respond in the SAME LANGUAGE as the user's question.",
        "NEVER invent facts.",
    ],
    respond_directly=False,
    delegate_to_all_members=False,
    show_members_responses=True,
    markdown=True,
)


# =====================================================================
# Run Team
# =====================================================================
async def run_agno_team(query: str, user_id: str) -> str:
    """Run the Agno multi-agent team for a single query."""
    global _current_user_id, _current_sources, _current_tools_called, _loaded_docs

    _current_user_id = user_id
    _current_sources = []
    _current_tools_called = []
    _loaded_docs = set()

    response = await hr_team.arun(query)
    return response.content if response and response.content else ""


# =====================================================================
# App Setup
# =====================================================================
app = FastAPI(title="RAG API Service - AGNO TEAM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# =====================================================================
# Query Endpoint
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

    try:
        query_text = request.query.strip()
        user_id = request.user_id or "default_user"

        log_request(request_id, "AGNO_START", {"query": query_text})

        t0 = time_module.time()
        answer_text = await run_agno_team(query_text, user_id)
        agent_time = time_module.time() - t0

        if not answer_text:
            answer_text = "I couldn't generate a response. Please try again."

        total_elapsed = time_module.time() - start_time
        conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
        conv_manager.add_message(user_id, "assistant", answer_text, {
            "request_id": request_id, "query_type": "agno_team",
            "tools_called": _current_tools_called,
            "elapsed_sec": round(total_elapsed, 3),
        })

        is_clarifying = any(p in answer_text.lower() for p in [
            "could you please tell me", "could you specify", "which country",
            "what is your", "can you provide", "please specify",
        ])
        if is_clarifying:
            clarification_tracker.create_session(user_id, query_text, [answer_text], "", _current_sources)

        log_request(request_id, "AGNO_COMPLETE", {
            "elapsed_sec": round(total_elapsed, 3),
            "agent_time": round(agent_time, 3),
            "tools_called": _current_tools_called,
        })

        return QueryResponse(
            response=format_gfm_to_html(answer_text),
            metadata={
                "request_id": request_id,
                "route": "AGNO_TEAM",
                "tools_called": _current_tools_called,
                "sources": _current_sources[:5],
                "elapsed_sec": round(total_elapsed, 3),
                "agent_time": round(agent_time, 3),
            },
        )

    except Exception as e:
        logger.error(f"[{request_id}] Agno error: {e}", exc_info=True)
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

        try:
            query_text = request.query.strip()
            user_id = request.user_id or "default_user"

            log_request(request_id, "STREAM_AGNO_START", {"query": query_text})
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...'})}\n\n"
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"

            answer_text = await run_agno_team(query_text, user_id)

            unique_sources = []
            seen_src = set()
            for s in _current_sources:
                key = s.get("source", "")
                if key not in seen_src:
                    seen_src.add(key)
                    unique_sources.append(s)
            for idx, src in enumerate(unique_sources[:5]):
                yield f"data: {json.dumps({'type': 'source_found', 'source': src.get('source', ''), 'index': idx + 1, 'score': src.get('score', 0)})}\n\n"

            yield f"data: {json.dumps({'type': 'progress', 'percentage': 60, 'message': 'Generating answer...'})}\n\n"

            if not answer_text:
                answer_text = "I couldn't generate a response."

            words = answer_text.split()
            for i in range(0, len(words), 3):
                chunk_text = " ".join(words[i:i + 3])
                if i + 3 < len(words):
                    chunk_text += " "
                yield f"data: {json.dumps({'type': 'token', 'text': chunk_text}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)

            conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
            conv_manager.add_message(user_id, "assistant", answer_text, {
                "request_id": request_id, "query_type": "agno_team",
            })

            total_elapsed = time_module.time() - start_time
            log_request(request_id, "STREAM_AGNO_COMPLETE", {
                "elapsed_sec": round(total_elapsed, 3),
                "tools_called": _current_tools_called,
            })

            yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'AGNO_TEAM', 'tools_called': _current_tools_called, 'sources': _current_sources[:5], 'elapsed_sec': round(total_elapsed, 3)}})}\n\n"

        except Exception as e:
            logger.error(f"[{request_id}] Stream Agno error: {e}", exc_info=True)
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
# Health Check & Info Endpoints
# =====================================================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "agno-team-v1",
        "collection": COLLECTION_NAME_V2,
        "model": FLASH_MODEL,
        "architecture": "agno-team-multi-agent",
        "agents": ["knowledge_researcher", "document_specialist"],
        "tools": ["search_hr_knowledge_base", "get_conversation_history", "load_full_document"],
    }


@app.get("/")
async def root():
    return {
        "service": "RAG API Service - AGNO TEAM",
        "version": "1.0",
        "architecture": "Agno Team — leader delegates to knowledge_researcher or document_specialist",
        "endpoints": {
            "/query": "POST - Agno team query (JSON response)",
            "/query/stream": "POST - SSE streaming",
            "/health": "GET - Health check",
        },
        "agents": {
            "knowledge_researcher": "Searches HR knowledge base chunks",
            "document_specialist": "Loads full HR policy documents",
        },
    }


# =====================================================================
# Run Server
# =====================================================================
if __name__ == "__main__":
    import uvicorn

    print("Starting AGNO TEAM RAG Server...")
    print(f"  Architecture: Agno Team (leader + 2 members)")
    print(f"  Collection: {COLLECTION_NAME_V2}")
    print(f"  Model: {FLASH_MODEL} (via OpenRouter)")
    print(f"  Agents: knowledge_researcher, document_specialist")
    print(f"  Tools: search_hr_knowledge_base, get_conversation_history, load_full_document")
    print(f"  Port: 7869")

    uvicorn.run(app, host="0.0.0.0", port=7869)
