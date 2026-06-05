#!/usr/bin/env python3
"""
Unified FastAPI: Graphiti LangGraph Agent + RAG API (port 8095) + LLM synthesis

- RAG is called via HTTP POST to RAG_API_URL (default: http://localhost:8095/ask)
  Request:  { "query": "...", "top_k": <int>, "max_tokens_ctx": <int> }
  Response: { "answer": <str>, "sources": [ ... ] }

- KG answer comes from GraphitiLangGraphAgent.

- Final answer is produced by Azure OpenAI chat model that *synthesizes* both:
  * If both present: the LLM merges them.
  * If only one is present: the LLM reformats that one.
"""

import os
import json
import asyncio
from typing import Dict, Any, AsyncGenerator, Optional, Tuple, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------
# GFM Markdown formatting
# ---------------------------------------------------------------------
from markdown_it import MarkdownIt
from mdit_py_plugins.tasklists import tasklists_plugin

# Initialize markdown-it with GFM-like features
md = MarkdownIt("gfm-like").use(tasklists_plugin)

def format_gfm_to_html(text: str) -> str:
    """Convert markdown text to HTML using GFM-like formatting."""
    if not text or not text.strip():
        return text
    return md.render(text)

# ---------------------------------------------------------------------
# Graphiti KG agent (unchanged on your side)
# ---------------------------------------------------------------------
from langgraph_graphiti_agent import GraphitiLangGraphAgent

# HTTP client for RAG API call
try:
    import httpx
except ImportError as _e:
    raise RuntimeError("Install httpx: pip install httpx") from _e

# Azure OpenAI client (for synthesis)
try:
    from openai import AzureOpenAI
except ImportError as _e:
    raise RuntimeError("Install openai: pip install openai") from _e


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
REQUIRED_ENV = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "AZURE_OPENAI_API_KEY"]

def env_ok() -> Tuple[bool, List[str]]:
    missing = [v for v in REQUIRED_ENV if not os.getenv(v)]
    return (len(missing) == 0, missing)

# — RAG API (this replaces deepsearch) —
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8095/ask").rstrip("/")

# RAG defaults to forward to the RAG API
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MAX_TOKENS_CTX = int(os.getenv("RAG_MAX_TOKENS_CTX", "7000"))

RAG_API_TIMEOUT = float(os.getenv("RAG_API_TIMEOUT", "90"))

# Graphiti defaults
DEFAULT_USER = os.getenv("GRAPHITI_DEFAULT_USER", "APIUser")
DEFAULT_THREAD = os.getenv("GRAPHITI_DEFAULT_THREAD", "api_session")

# Azure OpenAI for synthesis
AZURE_OPENAI_ENDPOINT  = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
AZURE_OPENAI_API_KEY   = os.getenv("AZURE_OPENAI_API_KEY")  or os.getenv("OPENAI_API_KEY")
AZURE_CHAT_DEPLOYMENT  = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
if not AZURE_CHAT_DEPLOYMENT:
    raise RuntimeError("Set AZURE_OPENAI_CHAT_DEPLOYMENT.")

aoai = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01",
)


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
app = FastAPI(title="Unified Graphiti + RAG API (LLM-synth)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Singletons
_agent: Optional[GraphitiLangGraphAgent] = None
_agent_lock = asyncio.Lock()


async def _ensure_agent() -> GraphitiLangGraphAgent:
    global _agent
    async with _agent_lock:
        if _agent is not None:
            return _agent
        ok, missing = env_ok()
        if not ok:
            raise RuntimeError(f"Missing required env vars: {missing}.")
        agent = GraphitiLangGraphAgent()
        init_ok = await agent.initialize()
        if not init_ok:
            raise RuntimeError("Failed to initialize GraphitiLangGraphAgent.")
        await agent.setup_user(DEFAULT_USER)
        try:
            await agent.load_existing_data()
        except Exception:
            pass
        _agent = agent
        return _agent


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    metadata: Dict[str, Any] = {}


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def make_serializable(obj):
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {k: make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


async def _call_rag_api(query: str, top_k: int, max_tokens_ctx: int) -> Dict[str, Any]:
    """
    Calls the RAG API you provided (port 8095).
    Request:  {query, top_k, max_tokens_ctx}
    Response: {answer, sources}
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    payload = {
        "query": query,
        "top_k": top_k,
        "max_tokens_ctx": max_tokens_ctx,
    }

    print(f"🔗 RAG API → POST {RAG_API_URL} :: {payload}")
    async with httpx.AsyncClient(timeout=RAG_API_TIMEOUT) as client:
        try:
            r = await client.post(RAG_API_URL, json=payload)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            err = f"RAG API HTTP {e.response.status_code}: {e.response.text}"
            print(f"❌ {err}")
            return {"error": err}
        except Exception as e:
            err = f"RAG API call failed: {e}"
            print(f"❌ {err}")
            return {"error": err}

    # Normalize keys expected downstream
    data.setdefault("answer", "")
    data.setdefault("sources", [])
    return data


# ---------------- LLM synthesis of KG + RAG ----------------
SYNTH_SYSTEM = """You are a precise assistant. Merge two answers into one final response:
- Prefer grounded facts from the RAG answer (document-backed).
- Incorporate useful structure or relationships from the KG answer.
- If they conflict, clearly note the conflict and prefer RAG unless the KG has explicit corroboration.
- Keep it concise, cite sources from RAG inline as [1], [2], ... if present in the RAG answer.
- If user requested a specific document, only use that document's content from RAG (ignore others and KG).
"""

def synthesize_answers_llm(user_query: str, rag_answer: str, kg_answer: str) -> str:
    """
    Uses Azure OpenAI chat to synthesize a single, polished response from:
    - RAG answer (document-grounded)
    - KG answer (graph reasoning)
    """
    rag_answer = (rag_answer or "").strip()
    kg_answer = (kg_answer or "").strip()

    # Minimal guardrails
    if rag_answer and not kg_answer:
        # Reformat RAG for clarity, keep citations
        user_content = f"User question:\n{user_query}\n\nRAG answer:\n{rag_answer}\n\nPlease present a clear final answer. Keep existing citations."
    elif kg_answer and not rag_answer:
        user_content = f"User question:\n{user_query}\n\nKG answer:\n{kg_answer}\n\nReturn a concise final answer. If info seems ungrounded, say so."
    else:
        user_content = (
            f"User question:\n{user_query}\n\n"
            f"RAG answer (document-grounded):\n{rag_answer}\n\n"
            f"KG answer (graph-based):\n{kg_answer}\n\n"
            "Merge them per the system instructions."
        )

    msgs = [
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    resp = aoai.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=msgs,
        temperature=0.2,
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/health")
async def health_check():
    ok, missing = env_ok()
    return {
        "status": "healthy" if ok else "degraded",
        "service": "unified-graphiti-ragapi",
        "env_ok": ok,
        "missing_env": missing,
        "rag_api_url": RAG_API_URL,
        "rag_defaults": {
            "top_k": RAG_TOP_K,
            "max_tokens_ctx": RAG_MAX_TOKENS_CTX,
        },
    }


@app.get("/agent/info")
async def get_agent_info():
    return {
        "name": "Unified Graphiti + RAG API (LLM-synth)",
        "version": "2.0.0",
        "description": "KG-backed Graphiti + RAG API (port 8095) with LLM answer synthesis.",
        "capabilities": [
            "KG-backed QA (Graphiti)",
            "Hybrid RAG via external API /ask",
            "LLM-based synthesis of KG + RAG",
            "Streaming tokens",
        ],
        "env_required": REQUIRED_ENV,
    }


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    try:
        agent = await _ensure_agent()
        thread_id = DEFAULT_THREAD

        async def run_kg():
            try:
                ans = await agent.chat(request.query, thread_id)
                return ans if isinstance(ans, str) else json.dumps(make_serializable(ans), ensure_ascii=False)
            except Exception as e:
                return {"error": f"Graphiti error: {e}"}

        async def run_rag():
            return await _call_rag_api(
                query=request.query,
                top_k=RAG_TOP_K,
                max_tokens_ctx=RAG_MAX_TOKENS_CTX,
            )

        kg_task, rag_task = await asyncio.gather(run_kg(), run_rag())

        # Parse KG
        kg_answer, kg_meta = None, {}
        if isinstance(kg_task, dict) and "error" in kg_task:
            kg_meta["error"] = kg_task["error"]
        else:
            kg_answer = kg_task
            kg_meta["last_graph_query"] = getattr(agent, "_last_query", None)

        # Parse RAG
        rag_answer, rag_meta = None, {}
        if isinstance(rag_task, dict) and "error" in rag_task and "answer" not in rag_task:
            rag_meta["error"] = rag_task["error"]
        else:
            rag_answer = (rag_task or {}).get("answer", "")
            rag_sources = (rag_task or {}).get("sources", [])
            rag_meta = {"sources": rag_sources}

        # LLM synthesis
        final_text = synthesize_answers_llm(request.query, rag_answer or "", kg_answer or "")
        
        # Format response using GFM to HTML
        formatted_response = format_gfm_to_html(final_text)

        metadata = {
            "user_id": DEFAULT_USER,
            "thread_id": thread_id,
            "query": request.query,
            "rag_api": rag_meta,
            "graphiti": kg_meta,
        }
        return QueryResponse(response=formatted_response, metadata=metadata)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    async def generate() -> AsyncGenerator[bytes, None]:
        try:
            agent = await _ensure_agent()
            thread_id = DEFAULT_THREAD

            async def run_kg():
                try:
                    ans = await agent.chat(request.query, thread_id)
                    return ans if isinstance(ans, str) else json.dumps(make_serializable(ans), ensure_ascii=False)
                except Exception as e:
                    return {"error": f"Graphiti error: {e}"}

            async def run_rag():
                return await _call_rag_api(
                    query=request.query,
                    top_k=RAG_TOP_K,
                    max_tokens_ctx=RAG_MAX_TOKENS_CTX,
                )

            kg_task, rag_task = await asyncio.gather(run_kg(), run_rag())

            kg_answer, kg_meta = None, {}
            if isinstance(kg_task, dict) and "error" in kg_task:
                kg_meta["error"] = kg_task["error"]
            else:
                kg_answer = kg_task
                kg_meta["last_graph_query"] = getattr(agent, "_last_query", None)

            rag_answer, rag_meta = None, {}
            if isinstance(rag_task, dict) and "error" in rag_task and "answer" not in rag_task:
                rag_meta["error"] = rag_task["error"]
            else:
                rag_answer = (rag_task or {}).get("answer", "")
                rag_sources = (rag_task or {}).get("sources", [])
                rag_meta = {"sources": rag_sources}

            # LLM synthesis
            final_text = synthesize_answers_llm(request.query, rag_answer or "", kg_answer or "")
            
            # Format response using GFM to HTML
            formatted_response = format_gfm_to_html(final_text)

            # simple sentence-chunk streaming
            buf = ""
            for ch in formatted_response:
                buf += ch
                if ch in ".!?\n" and len(buf) >= 40:
                    yield f"data: {json.dumps({'type':'token','text':buf}, ensure_ascii=False)}\n\n".encode("utf-8")
                    buf = ""
            if buf:
                yield f"data: {json.dumps({'type':'token','text':buf}, ensure_ascii=False)}\n\n".encode("utf-8")

            done = {
                "type": "final",
                "done": True,
                "metadata": {
                    "user_id": DEFAULT_USER,
                    "thread_id": thread_id,
                    "query": request.query,
                    "rag_api": rag_meta,
                    "graphiti": kg_meta,
                },
            }
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n".encode("utf-8")

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n".encode("utf-8")

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Content-Type": "text/plain; charset=utf-8"},
    )


@app.on_event("shutdown")
async def shutdown_event():
    global _agent
    if _agent is not None:
        try:
            await _agent.close()
        except Exception:
            pass
        _agent = None


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8054"))
    uvicorn.run(app, host=host, port=port)
