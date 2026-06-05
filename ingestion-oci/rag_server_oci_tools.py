#!/usr/bin/env python3
"""
RAG Server — OCI tier (TOOL-CALLING AGENT)

Tool-calling agent for the OCI tier. Uses OCI Generative AI (Gemini 2.5 Flash)
for the LLM, OCI Cohere Embed v4.0 (1536-dim) for embeddings, and Qdrant
collection `docs_oci_ingested_azadea` for hybrid (dense+sparse RRF) retrieval.

Tools:
  - get_history(limit)             → recent conversation messages
  - get_document_knowledge(query)  → Qdrant dense+sparse hybrid (RRF fusion)
  - get_user_profile()             → stored user attributes
  - save_user_profile(attributes)  → persist user-disclosed attributes

Endpoints (parity with rag-azure-7867 production frontend contract):
  POST /query         — sync agent loop, JSON response
  POST /query/stream  — SSE streaming (status / source_found / progress / token / done)
  GET  /health        — service status
"""

import asyncio
import json
import logging
import os
import sys
import time as time_module
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

# Local modules
SERVICE_DIR = Path(__file__).parent.resolve()
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from oci_clients import OCI_CHAT_MODEL
from oci_chat import oci_chat_with_tools_async
from conversation_manager import get_conversation_manager
from rag_utils import format_gfm_to_html
from agent_tools import (
    TOOL_DEFINITIONS, execute_tool, get_tool_names,
    qdrant_client, COLLECTION_NAME,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("RAG-OCI-Tools")


# ─────────────────────────────────────────────────────────────────────────────
# Agent system prompt
# ─────────────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are Dea — the internal knowledge assistant for Azadea Group employees.
Your knowledge base covers Azadea policies, procedures, and SOPs across many domains
(HR, Operations, Finance, IT, Stock Management, F&B, Marketing, BCP, Real Estate, and more).

You have these tools:
- get_history(limit) — fetch recent messages in this conversation. Use this when the
  user's current message contains pronouns or ellipsis or otherwise depends on prior
  turns to be interpreted.
- get_document_knowledge(query) — search the corporate knowledge base. The query you
  pass MUST be a clear standalone phrase.
  Query construction rule: pass the user's question verbatim as the query whenever
  it is already a clear, standalone question. Do not rephrase, shorten, or summarize
  it — every word in the user's phrasing affects retrieval ranking.
  When the user's profile or the recent conversation (both visible in your system
  prompt) reveal context that scopes the answer — who the user is, where they work,
  what topic they have been discussing — fold that context into the retrieval query
  so the returned chunks match THIS user's situation. Then use the same profile and
  conversation context when interpreting retrieved content: if multiple sections of
  the policy apply differently to different categories of user, pick the section
  that fits the current user's profile, and tell them that section's answer — not
  a generic enumeration of all categories.
- get_user_profile() — retrieve the user's stored attributes (role, country, brand,
  department, etc.). Call this before answering any question whose answer might
  depend on the user's role/country/brand so you can personalise rather than ask.
- save_user_profile(attributes) — store profile attributes the user just disclosed
  about themselves. Call this whenever the user shares a personal fact (their role,
  country, brand, department, employment type, etc.) so it is remembered across
  turns. Save only what the user explicitly stated; do not infer.

After get_document_knowledge returns, you may see an additional
[CLARIFICATION VERDICT from specialist model] block appended to the tool
result. When present and it lists suggested_questions, ask those questions
(or a concise rewrite, picking 2-3 most important) to the user BEFORE giving
the policy answer. When the verdict is absent or says no clarification is
needed, proceed to answer directly.

Decide what to do based on the user's message (check rules in order — first match wins):
- The user discloses any personal fact about themselves, whether stated
  explicitly or implied by phrasing → call save_user_profile with the
  attributes you can infer. A disclosure still counts when the message also
  reads as a greeting or small talk.
  After saving, if the user's own message carries a real question, answer it
  (retrieve as needed). If the message is only a bare fact with no question of
  its own, call get_history next to check whether your previous turn asked a
  question that this fact answers — if so, resume that task and answer it. Do
  not stop at a thank-you when there is an open question.
- Retrieve first, ask second. Only ask clarifying questions after you have
  seen the retrieved documents — when those documents themselves reveal that
  the correct answer depends on missing user attributes, splits across
  multiple policies or variants, varies by scenario, time period, or other
  context. You may ask multiple clarifying questions in a single turn when
  several pieces of context would each change the answer. Do not ask
  clarifying questions preemptively before retrieval; do not ask for context
  the retrieved documents do not actually require.
- Short follow-up that depends on prior context → call get_history first.
- Any question that could be answered from the knowledge base → call
  get_document_knowledge first. The profile already in your system prompt
  is sufficient context; do not delay retrieval by asking the user for more
  attributes upfront. Only after seeing the retrieved documents should you
  decide whether the answer needs clarification.
- Pure greeting or casual chat with no disclosure and no question → reply warmly
  without calling any tools.
- Question you genuinely cannot interpret even with history → ask one short clarifying
  question as your reply (no tool call needed; just respond with the question).

Call ONE tool at a time — never emit more than a single tool call in the same
turn. If a message needs two actions (for example, saving a fact the user just
disclosed AND searching the knowledge base), do the FIRST tool call now, wait
for its result, then make the SECOND tool call on your next turn. One tool per
turn, always — never two tool calls together.

Tools are how you act in this conversation; words are how you report results.
If your reasoning concludes that a search, lookup, save, or any other tool action
is needed, emit the corresponding tool call in this same turn. Describing what
you are about to do is not a substitute for doing it — only the final reply,
once you already have what you need, should be plain text.

When you cannot answer a question from what you already know, retrieve before
declining or redirecting. Do not pre-judge a question as out of scope, personal,
or unanswerable — the retrieved documents are the only authority on whether the
answer exists. Sending the user elsewhere without first searching is wrong.

These rules apply to every turn independently. If your earlier responses in this
conversation conflict with the rules above, do not perpetuate them — your next
response must follow the rules, even when the user is repeating or rephrasing a
question you previously handled differently.

When you have what you need, write your final answer:
- Be warm, friendly, and concise. Avoid corporate-manual phrasing.
- Apply the user's profile (role/country/brand/department) to your answer when it
  is genuinely relevant; otherwise ignore it.
- When multiple retrieved chunks could answer the question, lead with the one
  whose content best matches the current user's profile and conversation
  context — not the one with the highest retrieval score. Retrieval scores
  measure surface keyword overlap, not which chunk actually applies to THIS
  user. A lower-scored or neighbor-pulled chunk that contains the specific
  information for the user's situation should take priority over a
  higher-scored chunk that describes a different category or audience.
- Cite source documents naturally when you used the knowledge base.
- If the documents don't actually answer the question, say so honestly and suggest
  1-2 closest topics that ARE in the retrieved chunks.
- Reply in the language the user is using, unless they explicitly ask for a
  different language — in that case, use the language they requested. Never
  claim you cannot respond in a particular language; you can.
- Do NOT make up answers. Use only retrieved content for factual claims.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────────────────

MAX_AGENT_ITERS = int(os.getenv("AGENT_MAX_ITERS", "5"))
HISTORY_TURNS = int(os.getenv("AGENT_HISTORY_TURNS", "10"))
# OCI thinking budget control. Accepted: NONE, MINIMAL, LOW, MEDIUM, HIGH.
# Empty/unset → OCI default (dynamic thinking, no cap). Pro cannot truly be
# disabled — NONE/MINIMAL both bottom out near the 128-token floor.
OCI_REASONING_EFFORT = os.getenv("OCI_REASONING_EFFORT", "").strip() or None

# Values the upstream UI sends as placeholders for "not provided" — never persist
# these into the user profile. Case-insensitive comparison.
_USER_CONTEXT_PLACEHOLDERS = {
    "anonymous user", "", "0", "false", "none", "null", "undefined",
}


def _parse_user_context(raw_query: str) -> Dict[str, str]:
    """Parse the upstream `User Context:` block preceding the `Request:` line.
    Returns a dict of non-placeholder attributes."""
    if "User Context:" not in raw_query or "Request:" not in raw_query:
        return {}
    header = raw_query.split("Request:", 1)[0]
    header = header.split("User Context:", 1)[-1]
    attrs: Dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if not key or not value:
            continue
        if value.lower() in _USER_CONTEXT_PLACEHOLDERS:
            continue
        attrs[key] = value
    return attrs


def _sync_user_context_to_profile(raw_query: str, user_id: str, request_id: str) -> None:
    """Merge non-placeholder upstream User Context attributes into the user_profile.
    Only fills keys that are not already present — explicit user disclosures
    saved via the agent's save_user_profile tool always win over UI-supplied
    placeholders."""
    attrs = _parse_user_context(raw_query)
    if not attrs:
        return
    try:
        cm = get_conversation_manager()
        existing = cm.get_user_profile(user_id) or {}
        new_attrs = {k: v for k, v in attrs.items() if k not in existing}
        if new_attrs:
            cm.update_user_profile(user_id, new_attrs)
            logger.info(f"[{request_id}] synced UI context → profile: {new_attrs}")
    except Exception as e:
        logger.warning(f"[{request_id}] failed to sync UI context: {e}")


def _system_prompt_with_profile(user_id: str, request_id: str) -> str:
    """Mandatory server-side profile + recent-history injection into the system
    prompt. The LLM sees both the user's stored attributes AND a digest of the
    recent conversation as part of its instructions every turn — never needs to
    call get_user_profile or get_history to know identity or context, and can
    never claim ignorance of an attribute or prior turn that is actually stored."""
    try:
        profile = get_conversation_manager().get_user_profile(user_id) or {}
    except Exception as e:
        logger.warning(f"[{request_id}] failed to load profile: {e}")
        profile = {}
    try:
        history = _load_recent_history(user_id, HISTORY_TURNS) or []
    except Exception as e:
        logger.warning(f"[{request_id}] failed to load history: {e}")
        history = []

    blocks = []
    _COMMON_IDENTITY_KEYS = ["name", "role", "country", "brand", "department",
                              "grade", "employment_type", "employee_id", "title"]
    have = profile or {}
    missing = [k for k in _COMMON_IDENTITY_KEYS if k not in have]

    have_lines = [f"- {k}: {v}" for k, v in have.items()] if have else ["- (nothing stored yet)"]
    miss_lines = [f"- {k}: NOT ON FILE" for k in missing]

    blocks.append(
        "\n\nUser profile (the ONLY authoritative source of who the user is):\n"
        "STORED — treat as authoritative, do not contradict:\n"
        + "\n".join(have_lines)
        + "\n\nNOT STORED — you do NOT know these. If the user asks about any "
        "attribute listed below, say honestly that it is not on file. Never "
        "invent a value for an attribute marked NOT ON FILE, never offer a "
        "placeholder, never preface with 'based on your profile' for these.\n"
        + "\n".join(miss_lines)
    )
    if history:
        turns = []
        for m in history:
            role = "User" if m["role"] == "user" else "You (assistant)"
            content = (m.get("content") or "").strip()
            if content:
                turns.append(f"- {role}: {content}")
        if turns:
            blocks.append(
                "\n\nRecent conversation with this user (treat everything here "
                "as established context the user has already provided — do not ask "
                "them to repeat any of it, and actively use these prior turns to "
                "interpret the current message, infer who they are, and shape "
                "your next response):\n"
                + "\n".join(turns)
            )
    return AGENT_SYSTEM_PROMPT + "".join(blocks)


def _load_recent_history(user_id: str, limit: int) -> List[Dict[str, Any]]:
    """Pull the last `limit` messages from Redis as plain user/assistant turns.
    Strips the upstream User Context wrapper.

    Rehydration policy: keep all user messages plus only the most recent
    assistant message. Older assistant turns are dropped to prevent the
    model from pattern-matching its own past responses — replaying a wall
    of prior refusals or wrong answers strongly biases the next turn to
    repeat them. The most recent assistant turn is kept so the agent can
    still resolve "you asked X, I'm answering Y" linkage."""
    try:
        prior = get_conversation_manager().get_history(user_id, limit=limit) or []
    except Exception as e:
        logger.warning(f"history load failed for {user_id}: {e}")
        return []
    cleaned: List[Dict[str, Any]] = []
    for m in prior:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content") or ""
        if "Request:" in content:
            content = content.split("Request:", 1)[-1].lstrip("\r\n").strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    last_assistant_idx = None
    for i in range(len(cleaned) - 1, -1, -1):
        if cleaned[i]["role"] == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None:
        return cleaned
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(cleaned):
        if m["role"] == "assistant" and i != last_assistant_idx:
            continue
        out.append(m)
    return out


async def agent_loop(user_query: str, user_id: str, request_id: str) -> Dict[str, Any]:
    """Run the OCI agent until it produces a final response or hits max iterations.
    Returns: {response, tools_used, iterations, sources, timings}"""
    # History is baked into the system prompt (via _system_prompt_with_profile),
    # so the message list contains only the current user turn — no separate
    # history spread. This avoids the conversational-consistency poisoning seen
    # when prior bad assistant turns are replayed in the message list.
    system_prompt = _system_prompt_with_profile(user_id, request_id)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    tools_used: List[Dict[str, Any]] = []
    sources_collected: List[Dict[str, Any]] = []
    seen_chunk_ids = set()
    timings: Dict[str, float] = {"3_retrieve": 0.0, "4_generate": 0.0}

    for iteration in range(MAX_AGENT_ITERS):
        t0 = time_module.time()
        try:
            result = await oci_chat_with_tools_async(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                model=OCI_CHAT_MODEL,
                temperature=0.4,
                # Gemini 2.5 Pro is a thinking model — reasoning tokens count
                # against this budget. 16000 (was 8000) leaves more room for
                # visible output after reasoning, mitigating the known empty-
                # completion failure mode on the synthesis hop.
                max_tokens=16000,
                reasoning_effort=OCI_REASONING_EFFORT,
            )
        except Exception as e:
            logger.exception(f"[{request_id}] LLM call failed at iter {iteration+1}: {e}")
            return {
                "response": "I hit an error working on that — please try again in a moment.",
                "tools_used": tools_used, "iterations": iteration + 1,
                "sources": sources_collected, "timings": timings, "error": str(e),
            }

        llm_secs = round(time_module.time() - t0, 2)
        timings["4_generate"] += llm_secs

        text = result.get("content", "") or ""
        calls = result.get("tool_calls", []) or []

        if not calls:
            final = text.strip()
            # Empty-completion guard — GENERATION ONLY. Gemini 2.5 Pro
            # intermittently returns empty text on the final synthesis hop
            # (thinking-budget exhaustion + a known backend bug). We re-issue
            # ONLY this generation call (messages already carry tool results —
            # we do NOT re-run tools or restart the iteration) up to 2x before
            # falling back to a graceful message, so the user never sees blank.
            gen_retry = 0
            while not final and gen_retry < 2:
                gen_retry += 1
                logger.warning(f"[{request_id}] empty completion at iter {iteration+1}; generation retry {gen_retry}/2")
                try:
                    retry_result = await oci_chat_with_tools_async(
                        messages=messages,
                        tools=TOOL_DEFINITIONS,
                        model=OCI_CHAT_MODEL,
                        temperature=0.4,
                        max_tokens=16000,
                        reasoning_effort=OCI_REASONING_EFFORT,
                    )
                    final = (retry_result.get("content", "") or "").strip()
                except Exception as e:
                    logger.warning(f"[{request_id}] generation retry {gen_retry} failed: {e}")
                    break
            if not final:
                logger.warning(f"[{request_id}] still empty after {gen_retry} generation retries; using fallback")
                final = ("I found relevant information but had trouble putting together a "
                         "complete answer just now. Could you rephrase or narrow your question "
                         "a little, and I'll try again?")
            logger.info(f"[{request_id}] agent done at iter {iteration+1} ({llm_secs}s, {len(tools_used)} tools used)")
            return {
                "response": final, "tools_used": tools_used,
                "iterations": iteration + 1,
                "sources": sources_collected, "timings": timings,
            }

        # Append the assistant turn (carrying tool_calls) and execute each call
        messages.append({"role": "assistant", "content": text or "", "tool_calls": calls})

        for tc in calls:
            name = tc.get("name") or ""
            args_json = tc.get("arguments") or "{}"
            tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            logger.info(f"[{request_id}] iter {iteration+1} → {name}({args_json[:120]})")

            t_tool = time_module.time()
            tool_result = execute_tool(name, args_json, user_id)
            tool_secs = time_module.time() - t_tool

            if name == "get_document_knowledge":
                timings["3_retrieve"] += tool_secs
                # Extract sources so streaming endpoint can emit source_found events
                try:
                    parsed = json.loads(tool_result)
                    for r in (parsed.get("results", []) or []):
                        src_file = r.get("source") or ""
                        if not src_file:
                            continue
                        key = f"{src_file}|{r.get('page')}|{r.get('chunk_type')}"
                        if key in seen_chunk_ids:
                            continue
                        seen_chunk_ids.add(key)
                        sources_collected.append({
                            "source": src_file,
                            "score": r.get("score"),
                            "chunk_type": r.get("chunk_type", ""),
                            "text_snippet": (r.get("text") or "")[:200],
                            "id": key,
                        })
                except Exception:
                    pass

                # Server-side automatic clarification consultation, gated by the
                # AUTO_CLARIFY env var (default OFF). When enabled, after a
                # successful retrieval we ask Pro whether the retrieved chunks
                # justify answering directly or whether clarifying questions
                # should be asked first; the verdict is appended to the tool
                # result. This is a SECOND Pro call (oci_chat_json) that adds
                # 20-80s per retrieval — with Pro as the main model its own
                # reasoning already covers this, so it stays off by default.
                if os.getenv("AUTO_CLARIFY", "0") == "1":
                    try:
                        t_clari = time_module.time()
                        verdict_json = execute_tool(
                            "get_clarification",
                            json.dumps({"query": user_query, "retrieved_chunks_json": tool_result}),
                            user_id,
                        )
                        verdict = json.loads(verdict_json)
                        timings["4_generate"] += (time_module.time() - t_clari)
                        if verdict.get("needs_clarification") and verdict.get("questions"):
                            clarif_note = (
                                "\n\n[CLARIFICATION VERDICT from specialist model]\n"
                                f"needs_clarification: true\n"
                                f"rationale: {verdict.get('rationale','')}\n"
                                f"suggested_questions: {json.dumps(verdict.get('questions',[]), ensure_ascii=False)}\n"
                                "ACTION: ask the user the suggested_questions (or a "
                                "concise rewrite of them) BEFORE giving the policy "
                                "answer. Pick at most 2-3 of the most important."
                            )
                            tool_result = tool_result + clarif_note
                            logger.info(f"[{request_id}] clarification verdict: {len(verdict.get('questions',[]))} questions to ask")
                        else:
                            logger.info(f"[{request_id}] clarification verdict: no clarification needed")
                    except Exception as e:
                        logger.warning(f"[{request_id}] clarification consultation failed: {e}")

            tools_used.append({"name": name, "args": args_json[:200], "result_chars": len(tool_result)})

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": tool_result,
            })

    logger.warning(f"[{request_id}] hit MAX_AGENT_ITERS={MAX_AGENT_ITERS}; returning best-effort")
    return {
        "response": "I'm having trouble finding the answer to that — could you rephrase or give me a bit more detail?",
        "tools_used": tools_used,
        "iterations": MAX_AGENT_ITERS,
        "sources": sources_collected, "timings": timings,
    }


def _route_for_tools(tools_used_names: List[str]) -> str:
    """Map tool usage to a route label compatible with the production frontend."""
    if not tools_used_names:
        return "CONVERSATIONAL"
    if "get_document_knowledge" in tools_used_names:
        return "SIMPLE"
    return "CONVERSATIONAL"


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="OCI RAG (Tool-Calling Agent)", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"


class QueryResponse(BaseModel):
    response: str
    metadata: Dict[str, Any] = {}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    request_id = str(uuid.uuid4())[:8]
    start = time_module.time()
    user_id = request.user_id or "default_user"
    raw_query = request.query.strip()

    # Strip the upstream User Context wrapper so the agent sees just the actual user request.
    query_text = raw_query
    if "Request:" in raw_query:
        query_text = raw_query.split("Request:", 1)[-1].lstrip("\r\n").strip()

    logger.info(f"[{request_id}] AGENT_QUERY_START | user={user_id} | q={query_text[:120]}")

    # Pre-fill profile from upstream UI's User Context (name / language / country / etc.)
    _sync_user_context_to_profile(raw_query, user_id, request_id)

    try:
        result = await agent_loop(query_text, user_id, request_id)
    except Exception as e:
        logger.exception(f"[{request_id}] agent_loop failed: {e}")
        return QueryResponse(
            response="I hit an error working on that — please try again in a moment.",
            metadata={"request_id": request_id, "error": str(e)},
        )

    final = result["response"]

    # Persist to Redis (keep User Context wrapper on the user message for parity)
    try:
        cm = get_conversation_manager()
        cm.add_message(user_id, "user", raw_query, {"request_id": request_id, "backend": "oci-agent"})
        cm.add_message(user_id, "assistant", final, {"request_id": request_id, "backend": "oci-agent"})
    except Exception as e:
        logger.warning(f"[{request_id}] persistence failed: {e}")

    elapsed = round(time_module.time() - start, 2)
    tools_used_names = [t["name"] for t in result.get("tools_used", [])]
    route = _route_for_tools(tools_used_names)
    timings = result.get("timings", {})
    timings_rounded = {k: round(v, 3) for k, v in timings.items()}

    logger.info(f"[{request_id}] AGENT_QUERY_DONE | iters={result['iterations']} tools={len(tools_used_names)} elapsed={elapsed}s")

    return QueryResponse(
        response=format_gfm_to_html(final),
        metadata={
            "request_id": request_id,
            "route": route,
            "sources": (result.get("sources") or [])[:5],
            "elapsed_sec": elapsed,
            "timings": timings_rounded,
            # extras (not in production schema, but useful for observability)
            "backend": "oci-agent",
            "iterations": result.get("iterations"),
            "tools_used": tools_used_names,
            "model": OCI_CHAT_MODEL,
        },
    )


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """SSE streaming endpoint — matches production rag-oci-7874's /query/stream protocol.
    Event types:
      - status        : intermediate progress messages
      - source_found  : retrieved sources during the agent's tool calls
      - progress      : milestone progress events (e.g. "Generating answer...")
      - token         : streamed answer text (word-at-a-time)
      - done          : final metadata (request_id, route, sources, elapsed_sec, ttft, timings)
      - error         : error message
    """
    async def generate():
        request_id = str(uuid.uuid4())[:8]
        start_time = time_module.time()
        user_id = request.user_id or "default_user"
        raw_query = request.query.strip()

        query_text = raw_query
        if "Request:" in raw_query:
            query_text = raw_query.split("Request:", 1)[-1].lstrip("\r\n").strip()

        logger.info(f"[{request_id}] STREAM_QUERY_START | user={user_id} | q={query_text[:120]}")

        # Pre-fill profile from upstream UI's User Context (name / language / country / etc.)
        _sync_user_context_to_profile(raw_query, user_id, request_id)

        # opening status
        yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...'})}\n\n"

        try:
            result = await agent_loop(query_text, user_id, request_id)
        except Exception as e:
            logger.exception(f"[{request_id}] agent_loop failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        if result.get("error"):
            yield f"data: {json.dumps({'type': 'error', 'message': result['error']})}\n\n"
            return

        final_text = result.get("response", "") or ""
        sources = result.get("sources", []) or []
        timings = result.get("timings", {}) or {}
        tools_used_names = [t["name"] for t in result.get("tools_used", [])]
        route = _route_for_tools(tools_used_names)

        # source_found events — match production frontend
        for idx, src in enumerate(sources[:5], 1):
            yield f"data: {json.dumps({'type': 'source_found', 'source': src.get('source', ''), 'index': idx, 'score': src.get('score', 0) or 0})}\n\n"

        # progress milestone before generation tokens
        if route == "SIMPLE":
            yield f"data: {json.dumps({'type': 'progress', 'percentage': 60, 'message': 'Generating answer...'})}\n\n"

        # Stream the final answer in markdown-preserving chunks. We hold whitespace
        # runs (\n\n, list indentation, code spans) intact by grouping `\S+\s*`
        # segments — splitting plainly on whitespace would erase formatting and
        # the frontend would render a wall of text.
        import re as _re
        segments = _re.findall(r'\S+\s*', final_text)
        CHUNK_WORDS = 10
        first_token_time = None
        for i in range(0, len(segments), CHUNK_WORDS):
            chunk_text = "".join(segments[i:i + CHUNK_WORDS])
            if first_token_time is None:
                first_token_time = round(time_module.time() - start_time, 3)
            yield f"data: {json.dumps({'type': 'token', 'text': chunk_text}, ensure_ascii=False)}\n\n"

        # Persist conversation
        try:
            cm = get_conversation_manager()
            cm.add_message(user_id, "user", raw_query, {"request_id": request_id, "backend": "oci-agent"})
            cm.add_message(user_id, "assistant", final_text, {"request_id": request_id, "backend": "oci-agent"})
        except Exception as e:
            logger.warning(f"[{request_id}] persistence failed: {e}")

        elapsed = round(time_module.time() - start_time, 3)
        timings_rounded = {k: round(v, 3) for k, v in timings.items()}

        # done event — match production metadata shape
        done_metadata = {
            "request_id": request_id,
            "route": route,
            "sources": sources[:5],
            "elapsed_sec": elapsed,
            "ttft": first_token_time,
            "timings": timings_rounded,
        }
        yield f"data: {json.dumps({'type': 'done', 'metadata': done_metadata})}\n\n"
        logger.info(f"[{request_id}] STREAM_V2_COMPLETE | iters={result.get('iterations')} tools={tools_used_names} elapsed={elapsed}s")

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
def health():
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "status": "ok",
            "service": "rag-oci-tools",
            "model": OCI_CHAT_MODEL,
            "tools": get_tool_names(),
            "max_iters": MAX_AGENT_ITERS,
            "qdrant_collection": COLLECTION_NAME,
            "qdrant_points": info.points_count,
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/")
def root():
    return {"service": "rag-oci-tools", "endpoints": ["/query", "/query/stream", "/health"]}


if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("SERVICE_PORT", "7884"))
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
