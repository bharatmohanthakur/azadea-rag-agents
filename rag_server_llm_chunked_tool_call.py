#!/usr/bin/env python3
"""
TOOL-CALL RAG API Service — LLM-Chunked Collection + LangGraph Agentic Architecture

Combines:
- Retrieval from rag_server_llm_chunked.py (collection docs_llm_chunked_azadea,
  table-first context, dedup, ALL-doc neighbor tables)
- Tool-calling architecture from rag_server_gemini_tool_call.py (LangGraph ReAct
  agent, @tool decorators, tool_choice="any", final_answer pattern)

What's removed vs rag_server_llm_chunked.py:
- UnifiedClassifier — no classification step at all
- _fast_classify() — no rule-based classification
- clarification_tracker — no CLARIFY:/ANSWER: prefix system
- rewrite_query_with_history() — LLM handles follow-ups via tools

Architecture:
  User query → LangGraph ReAct Agent (tool_choice="any")
                 ├─ greetings        → final_answer() directly
                 ├─ history question  → get_conversation_history() → final_answer()
                 ├─ HR question       → get_knowledge() → final_answer()
                 └─ follow-up         → get_conversation_history() + get_knowledge() → final_answer()

Tools: get_knowledge, get_conversation_history, final_answer
Port: 7871
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
    conv_manager, get_user_history,
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
from langgraph.prebuilt import ToolNode

load_dotenv()

# =====================================================================
# CONFIGURATION
# =====================================================================
QDRANT_LOCAL_URL = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
qdrant_client = QdrantClient(url=QDRANT_LOCAL_URL, check_compatibility=False)

# LLM-guided semantic chunks collection
COLLECTION_NAME_V2 = "docs_llm_chunked_azadea"

FLASH_MODEL = "google/gemini-3-flash-preview"


# =====================================================================
# LLM Clients
# =====================================================================
# Agent LLM — for tool-calling decisions (full reasoning)
agent_llm = ChatOpenAI(
    model=FLASH_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=4096,
)

# Fast LLM — for streaming answer generation (reasoning=minimal)
agent_llm_fast = ChatOpenAI(
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
# OPTIMIZED Retrieval v2 — Chunk-Only Context (table-first, ALL-doc neighbors)
# =====================================================================
async def retrieve_fast(
    query: str,
    user_id: str,
    top_k: int = 7
) -> Dict[str, Any]:
    """
    Fast retrieval using enriched collection with self-sufficient typed chunks.

    Key features:
    1. Uses docs_llm_chunked_azadea collection with typed chunks
    2. Context built from chunk text + full_table payload (no disk I/O)
    3. Tables placed FIRST in context (highest priority — exact numbers)
    4. Neighboring table chunks fetched for ALL docs that had hits
    5. Dedup via table_texts_seen set
    6. Embedding cached via LRU cache
    7. Async wraps for all blocking calls
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

        # 3. Build context from chunk payloads — tables FIRST (highest priority)
        t0 = datetime.now()
        table_parts = []   # Tables go first — they have exact numbers
        text_parts = []    # Page context goes after
        doc_ids_seen = set()
        table_texts_seen = set()  # Dedup tables
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
                if full_table not in table_texts_seen:
                    table_parts.append(f"[{src_file}] Table:\n{full_table}")
                    table_texts_seen.add(full_table)
            elif text:
                text_parts.append(f"[{src_file}] {text}")

        # 4. Fetch neighboring table chunks for ALL docs that had hits
        #    This ensures we get table data even when page_context was the top hit
        for doc_id in doc_ids_seen:
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
                    if ft and ft not in table_texts_seen:
                        table_parts.append(
                            f"[{tpl.get('source_file', '')}] Table:\n{ft}"
                        )
                        table_texts_seen.add(ft)
            except Exception as e:
                logger.warning(f"Neighbor table fetch failed for {doc_id}: {e}")

        # Tables first, then page context — tables have exact data the LLM needs most
        context = "\n\n".join(table_parts + text_parts)
        ctx_time = (datetime.now() - t0).total_seconds()

        total_time = (datetime.now() - retrieval_start).total_seconds()
        logger.info(
            f"FAST_RETRIEVAL_V2: embed={embed_time:.3f}s, search={search_time:.3f}s, "
            f"ctx={ctx_time:.3f}s, total={total_time:.3f}s, "
            f"ctx_chars={len(context)}, chunks={len(table_parts) + len(text_parts)}"
        )

        return {"context": context, "sources": sources, "images": retrieved_images}

    except Exception as e:
        logger.error(f"Fast retrieval error: {e}")
        return {"context": f"Error searching: {str(e)}", "sources": [], "images": []}


# =====================================================================
# Shared state for passing user_id and sources into tools
# =====================================================================
_current_user_id = "default_user"
_current_sources: List[dict] = []
_current_tools_called: List[str] = []
_final_answer_text = ""
_current_context = ""  # Collected context from get_knowledge (used by streaming phase 2)


# =====================================================================
# Sync wrapper for calling async retrieve from sync tools
# =====================================================================
def _sync_retrieve_fast(query: str, user_id: str) -> Dict[str, Any]:
    """Sync wrapper for retrieve_fast — runs async code in a new event loop."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, retrieve_fast(query, user_id))
        return future.result()


# =====================================================================
# LANGCHAIN TOOL DEFINITIONS — @tool decorated functions
# =====================================================================
@tool
def get_knowledge(query: str) -> str:
    """Search the Azadea Group HR knowledge base for policy documents,
    procedures, benefits, leave policies, compensation, etc.
    MUST be called for any HR-related question. Fix spelling errors
    in the query before searching (e.g. 'dresscode polcy' -> 'dress code policy').
    The query should be a clear, corrected search query. Fix typos and expand
    synonyms (dress code -> uniform policy, leave -> vacation/annual leave,
    fire -> termination, pay -> salary/compensation)."""
    global _current_sources, _current_tools_called, _current_context
    _current_tools_called.append("get_knowledge")
    logger.info(f"TOOL get_knowledge: query='{query}'")

    result = _sync_retrieve_fast(query, _current_user_id)

    _current_sources.extend(result.get("sources", []))
    content = result["context"]

    # Store context for streaming phase 2
    _current_context = content

    # Include source file names in tool output
    source_files = list({s.get("source", "") for s in result.get("sources", []) if s.get("source")})
    if source_files:
        files_list = ", ".join(source_files[:5])
        content += f"\n\n[Source files found: {files_list}]"

    return content


@tool
def get_conversation_history(last_n: int = 10) -> str:
    """Retrieve previous conversation messages with this user.
    Call this if the user asks 'what did I ask', 'show my history',
    references something said earlier, asks a follow-up, or says
    'show as table', 'in arabic', 'more detail', etc."""
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
# SYSTEM PROMPT — merged tool-calling flow + HR-specific rules
# =====================================================================
TOOL_SYSTEM_PROMPT = """You are an HR assistant for Azadea Group. You MUST use tools for everything. You MUST always end by calling final_answer with your response.

## Your tools:
- **get_knowledge(query)** — Search HR knowledge base. Returns text chunks and tables from HR policy documents. Always call this for any HR-related question.
- **get_conversation_history(last_n)** — Get previous conversation messages. Call this if the user asks about what they asked before, references something said earlier, or wants to see their history.
- **final_answer(answer)** — Deliver your final response to the user. You MUST call this when done.

## Flow:
1. For greetings (hi, hello, thanks, bye) → call final_answer directly with a friendly response
2. For "what did I ask" / "show my history" → call get_conversation_history, then final_answer
3. For follow-ups ("what about UAE", "show as table", "in arabic") → call get_conversation_history first to understand context, then get_knowledge if needed, then final_answer
4. For HR questions → call get_knowledge first, then final_answer
5. For combined (history + knowledge needed) → call both tools, then final_answer

## CRITICAL — Clarification-first rule:
After calling get_knowledge, READ the context carefully. Check if the answer depends on a detail the user did NOT provide:
  - Country of employment (if policy varies by country)
  - Employee type (shop/back office/part-time)
  - Department or brand
  - Specific situation details

If the answer VARIES by country, employee type, or role → you MUST ask a clarifying follow-up instead of dumping all variations. Your follow-up MUST list 2-3 specific options found in the context (e.g. "I found leave policies for Lebanon, UAE, Jordan, KSA, and Egypt. Which country are you asking about?").
Only answer directly when the policy is universal/identical across all countries and roles, OR the user already specified their country/role.
PREFER asking clarification over listing every country's variation. Do NOT list all variations — narrow it down first.
ALWAYS mention the specific countries/options you found in the context in your follow-up question.

## Rules for final_answer:
- Include EXACT numbers, dates, percentages from the context
- Use bullet points, numbered lists, and tables for clarity
- Fix user typos/synonyms (dress code = uniform, leave = vacation, fire = termination, pay = salary)
- Respond in the SAME LANGUAGE as the user's question. Arabic question → Arabic answer
- Cite source documents naturally (e.g. "per HRD-GEN-001")
- Maximum 3 clarification options — pick the most relevant from the context
- If the context does NOT contain information to answer the question, say "This information is not available in the HR documents I have access to."
- Do NOT invent facts. Only use information from tool results."""


# =====================================================================
# LangGraph Agent Setup — tool_choice="any" + final_answer pattern
# =====================================================================
AGENT_TOOLS = [get_knowledge, get_conversation_history, final_answer]

# Normal mode: must call a tool (any tool), LLM decides which
llm_any = agent_llm.bind_tools(AGENT_TOOLS, tool_choice="any")

# Forced final: must call final_answer specifically
llm_force_final = agent_llm.bind_tools(
    [final_answer],
    tool_choice={"type": "function", "function": {"name": "final_answer"}},
)

MAX_TOOL_CALLS = 8   # Total tool calls before forcing final_answer
MAX_CALLS_PER_ROUND = 3  # Max parallel tool calls per LLM response


def call_model(state: MessagesState) -> dict:
    """LLM node — full freedom until tool call limit, then forced to deliver answer."""
    tool_call_count = len(_current_tools_called)
    msg_count = len(state["messages"])
    logger.info(f"AGENT call_model: tool_calls_so_far={tool_call_count}, messages={msg_count}")

    if tool_call_count >= MAX_TOOL_CALLS:
        logger.info(f"AGENT: {tool_call_count} tool calls — forcing final_answer")
        response = llm_force_final.invoke(state["messages"])
    else:
        response = llm_any.invoke(state["messages"])

    # Log what the model decided to call
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [tc.get("name", "?") for tc in response.tool_calls]
        logger.info(f"AGENT model returned tool_calls: {tool_names}")
    else:
        logger.info(f"AGENT model returned NO tool_calls, content length: {len(response.content) if response.content else 0}")

    # Cap parallel tool calls — Gemini sometimes emits many in one message
    if hasattr(response, 'tool_calls') and len(response.tool_calls) > MAX_CALLS_PER_ROUND:
        logger.info(f"AGENT: Capping {len(response.tool_calls)} tool_calls → {MAX_CALLS_PER_ROUND}")
        response.tool_calls = response.tool_calls[:MAX_CALLS_PER_ROUND]

    return {"messages": [response]}


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
app = FastAPI(title="RAG API Service - LLM CHUNKED TOOL CALL")

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
    global _current_user_id, _current_sources, _current_tools_called
    global _final_answer_text, _current_context

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
        _current_context = ""

        log_request(request_id, "TOOL_CALL_START", {"query": query_text})

        # No history injection — LLM must call get_conversation_history tool if needed
        messages = [
            SystemMessage(content=TOOL_SYSTEM_PROMPT),
            HumanMessage(content=query_text),
        ]

        # Run the LangGraph ReAct agent
        t0 = time_module.time()
        result = await asyncio.to_thread(
            lambda: react_agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 25},
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
            "request_id": request_id, "query_type": "langgraph_tool_call",
            "tools_called": _current_tools_called,
            "elapsed_sec": round(total_elapsed, 3),
        })

        log_request(request_id, "TOOL_CALL_COMPLETE", {
            "elapsed_sec": round(total_elapsed, 3),
            "agent_time": round(agent_time, 3),
            "tools_called": _current_tools_called,
        })

        return QueryResponse(
            response=format_gfm_to_html(answer_text),
            metadata={
                "request_id": request_id,
                "route": "LANGGRAPH_TOOL_CALL",
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
# SSE Streaming Endpoint — Two-Phase Real Streaming
# =====================================================================
@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Streaming version of /query using Server-Sent Events.

    Two-phase approach:
    - Phase 1: Run LangGraph agent non-streamed (tool calls, collect context)
    - Phase 2: Stream answer via agent_llm_fast.astream() with collected context
    - Short-circuit: If no context collected (greetings), fake-stream final_answer text
    """
    async def generate() -> AsyncGenerator[str, None]:
        global _current_user_id, _current_sources, _current_tools_called
        global _final_answer_text, _current_context

        request_id = str(uuid.uuid4())[:8]
        start_time = time_module.time()

        try:
            query_text = request.query.strip()
            user_id = request.user_id or "default_user"

            _current_user_id = user_id
            _current_sources = []
            _current_tools_called = []
            _final_answer_text = ""
            _current_context = ""

            log_request(request_id, "STREAM_TOOL_CALL_START", {"query": query_text})
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...'})}\n\n"

            messages = [
                SystemMessage(content=TOOL_SYSTEM_PROMPT),
                HumanMessage(content=query_text),
            ]

            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing query...'})}\n\n"

            # ── Phase 1: Run LangGraph agent (non-streamed) ──
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

            # ── Phase 2: Stream the answer ──
            context_collected = _current_context
            agent_answer = _final_answer_text

            if not agent_answer:
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage) and msg.content and not getattr(msg, 'tool_calls', None):
                        agent_answer = msg.content
                        break
            if not agent_answer:
                agent_answer = "I couldn't generate a response."

            if context_collected:
                # Real streaming: re-generate answer with context using fast LLM
                stream_messages = [
                    ("system", """You are an HR assistant for Azadea Group. Answer using ONLY the provided context.

CRITICAL — Clarification-first rule:
Read the context carefully. If the answer VARIES by country, employee type, or role and the user did NOT specify theirs → ask a clarifying follow-up listing 2-3 specific options from the context (e.g. "I found leave policies for Lebanon, UAE, and Jordan. Which country are you asking about?").
PREFER asking clarification over listing every country's variation. Do NOT list all variations — narrow it down first.
Only answer directly when the policy is universal/identical across all countries and roles, OR the user already specified their country/role.

Rules:
- Include EXACT numbers, dates, percentages from the context
- Use bullet points, numbered lists, and tables for clarity
- Fix user typos/synonyms (dress code = uniform, leave = vacation, fire = termination, pay = salary)
- Respond in the SAME LANGUAGE as the user's question
- Cite source documents naturally (e.g. "per HRD-GEN-001")
- Maximum 3 clarification options
- If the context does NOT contain information to answer, say "This information is not available in the HR documents I have access to."
- Do NOT invent facts. Only use information from the provided context."""),
                    ("user", f"Question: {query_text}\n\nContext:\n{context_collected}"),
                ]

                answer_text = ""
                first_token_time = None
                async for chunk in agent_llm_fast.astream(stream_messages):
                    content = chunk.content if hasattr(chunk, "content") else ""
                    if content:
                        answer_text += content
                        if first_token_time is None:
                            first_token_time = time_module.time() - start_time
                        yield f"data: {json.dumps({'type': 'token', 'text': content}, ensure_ascii=False)}\n\n"
            else:
                # Short-circuit: no context collected (greetings, history-only)
                # Fake-stream the agent's final_answer text in 3-word chunks
                answer_text = agent_answer
                first_token_time = None
                words = answer_text.split()
                for i in range(0, len(words), 3):
                    chunk_text = " ".join(words[i:i + 3])
                    if i + 3 < len(words):
                        chunk_text += " "
                    if first_token_time is None:
                        first_token_time = time_module.time() - start_time
                    yield f"data: {json.dumps({'type': 'token', 'text': chunk_text}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)

            # Save conversation
            conv_manager.add_message(user_id, "user", query_text, {"request_id": request_id})
            conv_manager.add_message(user_id, "assistant", answer_text, {
                "request_id": request_id, "query_type": "langgraph_tool_call",
                "tools_called": _current_tools_called,
            })

            total_elapsed = time_module.time() - start_time
            log_request(request_id, "STREAM_TOOL_CALL_COMPLETE", {
                "elapsed_sec": round(total_elapsed, 3),
                "ttft": round(first_token_time or 0, 3),
                "tools_called": _current_tools_called,
            })

            yield f"data: {json.dumps({'type': 'done', 'metadata': {'request_id': request_id, 'route': 'LANGGRAPH_TOOL_CALL', 'tools_called': _current_tools_called, 'sources': _current_sources[:5], 'elapsed_sec': round(total_elapsed, 3), 'ttft': round(first_token_time or 0, 3)}})}\n\n"

        except Exception as e:
            logger.error(f"[{request_id}] Stream tool-call error: {e}", exc_info=True)
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
        "version": "llm-chunked-tool-call-v1",
        "collection": COLLECTION_NAME_V2,
        "model": FLASH_MODEL,
        "architecture": "langgraph-react-agent",
        "tools": ["get_knowledge", "get_conversation_history", "final_answer"],
    }


@app.get("/")
async def root():
    return {
        "service": "RAG API Service - LLM CHUNKED TOOL CALL",
        "version": "1.0",
        "architecture": "LangGraph ReAct agent with LLM-chunked retrieval",
        "endpoints": {
            "/query": "POST - LangGraph ReAct query (JSON response)",
            "/query/stream": "POST - SSE streaming (two-phase: agent + real stream)",
            "/health": "GET - Health check",
        },
        "tools": ["get_knowledge", "get_conversation_history", "final_answer"],
        "collection": COLLECTION_NAME_V2,
    }


# =====================================================================
# Run Server
# =====================================================================
if __name__ == "__main__":
    import uvicorn

    print("Starting LLM-CHUNKED TOOL CALL RAG Server...")
    print(f"  Architecture: LangGraph ReAct agent (tool_choice=any)")
    print(f"  Collection: {COLLECTION_NAME_V2}")
    print(f"  Model: {FLASH_MODEL}")
    print(f"  Tools: get_knowledge, get_conversation_history, final_answer")
    print(f"  Port: 7871")
    print(f"  Endpoints: /query (POST), /query/stream (POST SSE)")

    uvicorn.run(app, host="0.0.0.0", port=7871)
