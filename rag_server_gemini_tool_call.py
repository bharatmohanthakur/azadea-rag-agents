#!/usr/bin/env python3
"""
TOOL-CALL RAG API Service - LangGraph Agentic Architecture

Uses LangGraph ReAct agent with OpenRouter for automatic multi-round
tool calling. The LLM decides what tools to call and when to stop.

Tools:
  get_knowledge(query)              — search HR knowledge base chunks
  get_conversation_history(last_n)  — retrieve previous messages
  get_full_document(source_file)    — load complete source document

Architecture: LangGraph StateGraph with tools_condition routing
Port: 7867
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

# Import from original server to reuse common code
from rag_server_gemini import (
    openrouter_client, azure_embedding_client,
    OPENROUTER_API_KEY,
    conv_manager, clarification_tracker, get_user_history,
    format_gfm_to_html, log_request, logger,
    count_tokens,
)

# Import for retrieval
import azure_doc_intelligence_qdrant as rag_impl
from qdrant_client import QdrantClient, models as qm

# LangGraph + LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

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

FLASH_LITE_MODEL = os.getenv(
    "OPENROUTER_MODEL_FAST",
    "google/gemini-2.5-flash-lite"
)

FLASH_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemini-2.5-flash"
)

# =====================================================================
# Embedding LRU Cache
# =====================================================================
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
# OPTIMIZED Retrieval v2 — Chunk-Only Context
# =====================================================================
async def retrieve_fast(
    query: str,
    user_id: str,
    top_k: int = 7
) -> Dict[str, Any]:
    """
    Fast retrieval using enriched collection with self-sufficient typed chunks.
    Context built from chunk text + full_table payload (no disk I/O).
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

        # 2. Qdrant hybrid search (dense + sparse with RRF fusion)
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

        # 3. Build context from chunk payloads (no disk I/O)
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

            if chunk_type == "table_summary" and full_table:
                context_parts.append(f"[{src_file}] Table:\n{full_table}")
            elif text:
                context_parts.append(f"[{src_file}] {text}")

        # 4. Neighboring table chunks for docs that had page_context hits
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

        return {"context": context, "sources": sources, "images": retrieved_images}

    except Exception as e:
        logger.error(f"Fast retrieval error: {e}")
        return {"context": f"Error searching: {str(e)}", "sources": [], "images": []}


# =====================================================================
# Full Document Retrieval — fallback when chunks miss the answer
# =====================================================================
async def _retrieve_complete_document(source_file: str) -> str:
    """Load complete markdown document from disk."""
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
        logger.warning(f"FULL_DOC not found: {source_file}")
        return ""
    except Exception as e:
        logger.warning(f"FULL_DOC error '{source_file}': {e}")
        return ""


# =====================================================================
# Shared state for passing user_id and sources into tools
# =====================================================================
_current_user_id = "default_user"
_current_sources: List[dict] = []
_current_tools_called: List[str] = []
_loaded_docs: set = set()  # Dedup: already-loaded doc names per request
_context_quality_high: bool = False  # Set by get_knowledge based on retrieval scores


# =====================================================================
# LANGCHAIN TOOL DEFINITIONS — @tool decorated functions
# =====================================================================
def _sync_retrieve_fast(query: str, user_id: str) -> Dict[str, Any]:
    """Sync wrapper for retrieve_fast — runs async code in a new event loop."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, retrieve_fast(query, user_id))
        return future.result()


def _sync_retrieve_complete_document(source_file: str) -> str:
    """Sync wrapper for _retrieve_complete_document."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, _retrieve_complete_document(source_file))
        return future.result()


@tool
def get_knowledge(query: str) -> str:
    """Search the Azadea Group HR knowledge base for policy documents,
    procedures, benefits, leave policies, compensation, etc.
    MUST be called for any HR-related question. Fix spelling errors
    in the query before searching (e.g. 'dresscode polcy' -> 'dress code policy').
    The query should be a clear, corrected search query. Fix typos and expand
    synonyms (dress code -> uniform policy, leave -> vacation/annual leave,
    fire -> termination, pay -> salary/compensation)."""
    global _current_sources, _current_tools_called
    _current_tools_called.append("get_knowledge")
    logger.info(f"TOOL get_knowledge: query='{query}'")

    result = _sync_retrieve_fast(query, _current_user_id)

    _current_sources.extend(result.get("sources", []))
    content = result["context"]

    # Include source file names so LLM can pass them to get_full_document
    source_files = list({s.get("source", "") for s in result.get("sources", []) if s.get("source")})
    if source_files:
        files_list = ", ".join(source_files[:5])
        content += f"\n\n[Source files found: {files_list}]"

    return content


@tool
def get_conversation_history(last_n: int = 10) -> str:
    """Retrieve previous conversation messages with this user.
    ONLY call this if the user references something said earlier,
    asks a follow-up, or says 'show as table', 'in arabic',
    'more detail', etc. that requires knowing the previous response."""
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
def get_full_document(source_file: str) -> str:
    """Load the COMPLETE source document from disk. Use this when get_knowledge
    returned chunks that partially match but don't fully answer the question.
    Pass the exact source file name from get_knowledge results
    (e.g. 'HRD - GEN - 007 - Employee Attendance - P - 16.md').
    The full document contains all sections, tables, and details that chunks may have missed."""
    global _current_tools_called
    _current_tools_called.append("get_full_document")
    logger.info(f"TOOL get_full_document: source_file='{source_file}'")

    # Dedup: don't re-fetch the same document in the same request
    if source_file in _loaded_docs:
        logger.info(f"TOOL get_full_document: DEDUP skip '{source_file}'")
        return f"Document '{source_file}' was already loaded above. Use the content from the previous call."

    content = _sync_retrieve_complete_document(source_file)
    if content:
        _loaded_docs.add(source_file)
        return content
    return f"Document not found: {source_file}"


# Store for final_answer extraction
_final_answer_text = ""


@tool
def final_answer(answer: str) -> str:
    """Deliver your final answer to the user. You MUST call this tool when you are
    ready to respond. Pass the complete, formatted answer as the 'answer' parameter.
    Use markdown formatting (bullet points, tables, headers) in your answer."""
    global _final_answer_text, _current_tools_called
    _current_tools_called.append("final_answer")
    _final_answer_text = answer
    logger.info(f"TOOL final_answer: {len(answer)} chars")
    return "Answer delivered."


# =====================================================================
# SYSTEM PROMPT
# =====================================================================
TOOL_SYSTEM_PROMPT = """You are an HR assistant for Azadea Group. You MUST use tools for everything. You MUST always end by calling final_answer with your response.

## Your tools:
- **get_knowledge(query)** — Search HR knowledge base. Returns text chunks (snippets). Always call this first for any question.
- **get_conversation_history(last_n)** — Get previous conversation messages. Call this if the user's message seems like a follow-up or references something said earlier.
- **get_full_document(source_file)** — Load a COMPLETE document. ONLY use this when get_knowledge chunks are clearly incomplete or missing key details. Pass ONE source file name at a time.
- **final_answer(answer)** — Deliver your final response to the user. You MUST call this when done.

## Flow:
1. For greetings (hi, hello, thanks, bye) -> call final_answer directly
2. For follow-ups -> call get_conversation_history first, then get_knowledge, then final_answer
3. For HR questions -> call get_knowledge first. Then DECIDE:
   - **Chunks have the answer** (specific facts, numbers, details) -> call final_answer IMMEDIATELY. Do NOT call get_full_document.
   - **Chunks are incomplete** (mention a topic but lack specifics) -> call get_full_document for ONLY the single most relevant source file, then final_answer
   - **Chunks have no relevant info** -> call get_full_document for the top source file. If still no answer, say "not available"

## IMPORTANT — get_full_document rules:
- Call it for at most ONE file per round
- NEVER call it if get_knowledge already gave you enough details to answer
- NEVER call it for all source files — only the most relevant one
- After loading one full document, call final_answer with whatever you have

## Rules for final_answer:
- Give a DETAILED and COMPREHENSIVE answer covering ALL relevant facts, numbers, dates, percentages, conditions, exceptions, and steps
- Use bullet points, numbered lists, and tables for clarity
- Fix user typos/synonyms (dress code = uniform, leave = vacation, fire = termination, pay = salary)
- Respond in the SAME LANGUAGE as the user's question. Arabic question -> Arabic answer
- Cite source documents naturally (e.g. "per HRD-GEN-001")
- If answer varies by country/position, ask ONE clarifying question
- Do NOT invent facts. Only use information from tool results."""


# =====================================================================
# LangGraph Agent Setup — tool_choice="any" + final_answer pattern
# =====================================================================
AGENT_TOOLS = [get_knowledge, get_conversation_history, get_full_document, final_answer]

_base_llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model=FLASH_MODEL,
    temperature=0,
    max_tokens=2000,
)

# Normal mode: must call a tool (any tool), LLM decides which
llm_any = _base_llm.bind_tools(AGENT_TOOLS, tool_choice="any")

# Forced final: must call final_answer specifically
llm_force_final = _base_llm.bind_tools(
    [final_answer],
    tool_choice={"type": "function", "function": {"name": "final_answer"}},
)

MAX_TOOL_CALLS = 10   # Total tool calls before forcing final_answer
MAX_CALLS_PER_ROUND = 3  # Max parallel tool calls per LLM response


def call_model(state: MessagesState) -> dict:
    """LLM node — full freedom until tool call limit, then forced to deliver answer."""
    tool_call_count = len(_current_tools_called)
    if tool_call_count >= MAX_TOOL_CALLS:
        logger.info(f"AGENT: {tool_call_count} tool calls — forcing final_answer")
        response = llm_force_final.invoke(state["messages"])
    else:
        response = llm_any.invoke(state["messages"])

    # Cap parallel tool calls — Gemini sometimes emits 50+ in one message
    if hasattr(response, 'tool_calls') and len(response.tool_calls) > MAX_CALLS_PER_ROUND:
        logger.info(f"AGENT: Capping {len(response.tool_calls)} tool_calls → {MAX_CALLS_PER_ROUND}")
        response.tool_calls = response.tool_calls[:MAX_CALLS_PER_ROUND]

    return {"messages": [response]}


def route_after_model(state: MessagesState) -> str:
    """After model: always go to tools (tool_choice=any ensures tool calls)."""
    return "tools"


def route_after_tools(state: MessagesState) -> str:
    """After tools: if final_answer was called, stop. Otherwise back to model."""
    if _final_answer_text:
        return END
    return "model"


# Build the graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("model", call_model)
graph_builder.add_node("tools", ToolNode(AGENT_TOOLS))
graph_builder.set_entry_point("model")
graph_builder.add_edge("model", "tools")  # Model always produces tool calls (tool_choice=any)
graph_builder.add_conditional_edges("tools", route_after_tools, {"model": "model", END: END})
react_agent = graph_builder.compile()


# =====================================================================
# App Setup
# =====================================================================
app = FastAPI(title="RAG API Service - LANGGRAPH TOOL CALL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# =====================================================================
# Query Endpoint — LangGraph ReAct Agent
# =====================================================================
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"


class QueryResponse(BaseModel):
    response: str
    metadata: Dict[str, Any] = {}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Agentic query endpoint using LangGraph ReAct agent.
    The LLM decides what tools to call and chains them automatically.
    """
    global _current_user_id, _current_sources, _current_tools_called, _final_answer_text, _loaded_docs

    request_id = str(uuid.uuid4())[:8]
    start_time = time_module.time()

    try:
        query_text = request.query.strip()
        user_id = request.user_id or "default_user"

        # Reset shared state
        _current_user_id = user_id
        _current_sources = []
        _current_tools_called = []
        _final_answer_text = ""
        _loaded_docs = set()

        log_request(request_id, "LANGGRAPH_START", {"query": query_text})

        # No history injection — LLM must call get_conversation_history tool
        messages = [
            SystemMessage(content=TOOL_SYSTEM_PROMPT),
            HumanMessage(content=query_text),
        ]

        # Run the LangGraph ReAct agent (max 10 rounds via recursion_limit)
        t0 = time_module.time()
        result = await asyncio.to_thread(
            lambda: react_agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 25},  # ~10 tool rounds
            )
        )
        agent_time = time_module.time() - t0

        # Extract answer from final_answer tool
        answer_text = _final_answer_text
        if not answer_text:
            # Fallback: extract from last AI message
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, 'tool_calls', None):
                    answer_text = msg.content
                    break
        if not answer_text:
            answer_text = "I couldn't generate a response. Please try again."

        # Save conversation
        total_elapsed = time_module.time() - start_time
        conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
        conv_manager.add_message(user_id, "assistant", answer_text, {
            "request_id": request_id, "query_type": "langgraph_react",
            "tools_called": _current_tools_called,
            "elapsed_sec": round(total_elapsed, 3),
        })

        # Track clarification sessions
        is_clarifying = any(p in answer_text.lower() for p in [
            "could you please tell me", "could you specify", "which country",
            "what is your", "can you provide", "please specify",
        ])
        if is_clarifying:
            clarification_tracker.create_session(user_id, query_text, [answer_text], "", _current_sources)

        log_request(request_id, "LANGGRAPH_COMPLETE", {
            "elapsed_sec": round(total_elapsed, 3),
            "agent_time": round(agent_time, 3),
            "tools_called": _current_tools_called,
        })

        return QueryResponse(
            response=format_gfm_to_html(answer_text),
            metadata={
                "request_id": request_id,
                "route": "LANGGRAPH_REACT",
                "tools_called": _current_tools_called,
                "sources": _current_sources[:5],
                "elapsed_sec": round(total_elapsed, 3),
                "agent_time": round(agent_time, 3),
            },
        )

    except Exception as e:
        logger.error(f"[{request_id}] LangGraph error: {e}", exc_info=True)
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
    """
    Streaming version of /query using Server-Sent Events.
    Runs LangGraph agent, then fake-streams the final answer.
    """
    async def generate() -> AsyncGenerator[str, None]:
        global _current_user_id, _current_sources, _current_tools_called, _final_answer_text, _loaded_docs

        request_id = str(uuid.uuid4())[:8]
        start_time = time_module.time()

        try:
            query_text = request.query.strip()
            user_id = request.user_id or "default_user"

            _current_user_id = user_id
            _current_sources = []
            _current_tools_called = []
            _final_answer_text = ""
            _loaded_docs = set()

            log_request(request_id, "STREAM_LANGGRAPH_START", {"query": query_text})
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...'})}\n\n"

            messages = [
                SystemMessage(content=TOOL_SYSTEM_PROMPT),
                HumanMessage(content=query_text),
            ]

            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"

            # Run LangGraph agent
            result = await asyncio.to_thread(
                lambda: react_agent.invoke(
                    {"messages": messages},
                    config={"recursion_limit": 25},
                )
            )

            # Send source events
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

            # Extract answer from final_answer tool
            answer_text = _final_answer_text
            if not answer_text:
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage) and msg.content and not getattr(msg, 'tool_calls', None):
                        answer_text = msg.content
                        break
            if not answer_text:
                answer_text = "I couldn't generate a response."

            # Fake-stream the answer in 3-word chunks
            words = answer_text.split()
            for i in range(0, len(words), 3):
                chunk_text = " ".join(words[i:i + 3])
                if i + 3 < len(words):
                    chunk_text += " "
                yield f"data: {json.dumps({'type': 'token', 'text': chunk_text}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)

            # Save conversation
            conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
            conv_manager.add_message(user_id, "assistant", answer_text, {
                "request_id": request_id, "query_type": "langgraph_react",
            })

            total_elapsed = time_module.time() - start_time
            log_request(request_id, "STREAM_LANGGRAPH_COMPLETE", {
                "elapsed_sec": round(total_elapsed, 3),
                "tools_called": _current_tools_called,
            })

            yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'LANGGRAPH_REACT', 'tools_called': _current_tools_called, 'sources': _current_sources[:5], 'elapsed_sec': round(total_elapsed, 3)}})}\n\n"

        except Exception as e:
            logger.error(f"[{request_id}] Stream LangGraph error: {e}", exc_info=True)
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
        "version": "langgraph-react-v1",
        "collection": COLLECTION_NAME_V2,
        "model": FLASH_MODEL,
        "architecture": "langgraph-react-agent",
        "tools": ["get_knowledge", "get_conversation_history", "get_full_document"],
    }


@app.get("/")
async def root():
    return {
        "service": "RAG API Service - LANGGRAPH REACT",
        "version": "1.0",
        "architecture": "LangGraph ReAct agent — LLM decides what to call, auto-chains tools",
        "endpoints": {
            "/query": "POST - LangGraph ReAct query (JSON response)",
            "/query/stream": "POST - SSE streaming (token-level)",
            "/health": "GET - Health check",
        },
        "tools": ["get_knowledge", "get_conversation_history", "get_full_document"],
    }


# =====================================================================
# Run Server
# =====================================================================
if __name__ == "__main__":
    import uvicorn

    print("Starting LANGGRAPH REACT RAG Server...")
    print(f"  Architecture: LangGraph ReAct agent")
    print(f"  Collection: {COLLECTION_NAME_V2}")
    print(f"  Model: {FLASH_MODEL}")
    print(f"  Tools: get_knowledge, get_conversation_history, get_full_document")
    print(f"  Port: 7867")
    print(f"  Endpoints: /query (POST), /query/stream (POST SSE)")

    uvicorn.run(app, host="0.0.0.0", port=7867)
