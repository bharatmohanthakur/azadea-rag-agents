#!/usr/bin/env python3
"""
RAG Server — AZURE tier (TOOL-CALLING AGENT, port 7887)

Parallel to rag_server_llm_chunked.py (port 7867). Uses the same Qdrant collection
(docs_llm_chunked_azadea), same Azure OpenAI embedder (text-embedding-3-large),
same Redis conversation store. Difference: instead of a hand-coded
classify→rewrite→retrieve→generate pipeline, this runs a tool-calling agent loop.

Tools:
  - get_history(limit)             → recent conversation messages
  - get_document_knowledge(query)  → Qdrant dense+sparse hybrid (RRF fusion)

Endpoints:
  POST /query   — sync agent loop, returns final answer
  GET  /health  — service status
"""

import asyncio
import json
import logging
import os
import time as time_module
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Reuse existing singletons — no new init for the running services
from rag_server_gemini import openrouter_client, conv_manager, format_gfm_to_html
from agent_tools_oci_claude import TOOLS_OPENAI, execute_tool, get_tool_names, qdrant_client, COLLECTION_NAME_V2

# Pluggable chat backend: set LLM_BASE_URL (+ LLM_API_KEY) to point the agent's
# chat completions at a different OpenAI-compatible provider (e.g. Fireworks AI:
# https://api.fireworks.ai/inference/v1). Only the CHAT client is swapped — the
# reranker in agent_tools_azure_rerank stays on OpenRouter (Fireworks has no
# /rerank endpoint), and embeddings stay on Azure OpenAI.
if os.getenv("LLM_BASE_URL"):
    from openai import OpenAI as _OpenAI
    openrouter_client = _OpenAI(api_key=os.getenv("LLM_API_KEY", ""),
                                base_url=os.getenv("LLM_BASE_URL"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("RAG-Azure-Rerank")


# Model for the main agent loop. Default is Gemini 2.5 Pro for stronger
# tool-call decisions; reasoning.effort=low (set on each call below) keeps
# thinking tokens bounded so latency stays workable.
MODEL = os.getenv("OPENROUTER_MODEL_FAST", "google/gemini-2.5-pro")
MAX_AGENT_ITERS = int(os.getenv("AGENT_MAX_ITERS", "5"))
# Real token-by-token streaming on /query/stream (default on). Claude streams
# incrementally; Gemini buffers to a late burst (so it falls back to looking
# the same). Set REAL_STREAM=0 to force the legacy chunk-after-complete path.
REAL_STREAM = os.getenv("REAL_STREAM", "1") == "1"


# Stream the model's reasoning (Claude exposes it via delta.reasoning; Gemini
# does not). When on, we request low-effort thinking and surface it to the user
# as `status` events during processing — no new event type, no frontend change.
STREAM_THINKING = os.getenv("STREAM_THINKING", "1") == "1"


def _reasoning_extra_body() -> Dict[str, Any]:
    """reasoning.effort=low for Gemini (bounds its thinking budget). For Claude,
    enable low-effort thinking only when STREAM_THINKING is on (so we can stream
    delta.reasoning); otherwise omit it to keep first-token latency minimal."""
    if MODEL.startswith("google/"):
        return {"reasoning": {"effort": "low"}}
    if STREAM_THINKING:
        return {"reasoning": {"effort": "low"}}
    return {}


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
  When the user's profile (visible in your system prompt) — or the recent
  conversation, which you can fetch with get_history — reveals context that scopes
  the answer — who the user is, where they work, what topic they have been
  discussing — fold that context into the retrieval query so the returned chunks
  match THIS user's situation. Then use the same profile and
  conversation context when interpreting retrieved content: if multiple sections of
  the policy apply differently to different categories of user, pick the section
  that fits the current user's profile, and tell them that section's answer — not
  a generic enumeration of all categories.
- list_documents(category) — list ALL documents in a policy area (a complete
  catalog, not a search). Use this when the user asks to "list / share / give me
  all the X policies" or "all the documents/links related to X" (e.g. all IT
  policies, all HR policies). For a question about what a specific policy SAYS,
  use get_document_knowledge instead.
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

HISTORY_TURNS = int(os.getenv("AGENT_HISTORY_TURNS", "10"))

# In this variant conversation history is NOT pre-injected into the system prompt
# every turn. It's fetched on demand via the get_history tool only — so the model
# pulls prior turns when a message actually depends on them, instead of paying to
# re-send the whole digest on every query. The user PROFILE (identity: name, role,
# country, …) is still injected, since identity scopes almost every answer. Set
# INJECT_HISTORY=1 to restore the always-injected behaviour of the base agent.
INJECT_HISTORY = os.getenv("INJECT_HISTORY", "0") == "1"

# Values the upstream UI sends as placeholders for "not provided" — never persist
# these into the user profile. Case-insensitive comparison.
_USER_CONTEXT_PLACEHOLDERS = {
    "anonymous user", "", "0", "false", "none", "null", "undefined",
}


def _parse_user_context(raw_query: str) -> Dict[str, str]:
    """Parse the upstream `User Context:` block (Name / Username / Language /
    Country / IsCustomer / ... : value pairs) preceding the `Request:` line.
    Returns a dict of non-placeholder attributes ready to merge into the profile."""
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
        existing = conv_manager.get_user_profile(user_id) or {}
        new_attrs = {k: v for k, v in attrs.items() if k not in existing}
        if new_attrs:
            conv_manager.update_user_profile(user_id, new_attrs)
            logger.info(f"[{request_id}] synced UI context → profile: {new_attrs}")
    except Exception as e:
        logger.warning(f"[{request_id}] failed to sync UI context: {e}")


def _system_prompt_with_profile(user_id: str, request_id: str) -> str:
    """Server-side profile injection into the system prompt. The LLM always sees
    the user's stored attributes as part of its instructions — so it never has to
    call get_user_profile to know identity and can't claim ignorance of a stored
    attribute. Conversation history is NOT injected here unless INJECT_HISTORY=1;
    by default the model fetches it on demand via the get_history tool."""
    try:
        profile = conv_manager.get_user_profile(user_id) or {}
    except Exception as e:
        logger.warning(f"[{request_id}] failed to load profile: {e}")
        profile = {}
    if INJECT_HISTORY:
        try:
            history = _load_recent_history(user_id, HISTORY_TURNS) or []
        except Exception as e:
            logger.warning(f"[{request_id}] failed to load history: {e}")
            history = []
    else:
        history = []   # on-demand via get_history tool, not pre-injected

    blocks = []
    # The stored profile is the ONLY authoritative source of user attributes.
    # Two blocks are built: (1) what we have, (2) explicit "NOT ON FILE" rows
    # for every common identity attribute that's missing. The explicit-absence
    # list is essential — without it Pro has a strong training prior to
    # invent plausible identities when asked. Listing each missing attribute
    # gives Pro a structured signal to refuse.
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


# Prompt caching (Claude via OpenRouter). The STATIC instructions are sent as a
# cacheable content block (cache_control) so they bill at ~10% on reads; the
# per-user profile/history go in a separate UNcached block (they change per user,
# so caching them would break the cache key). Gemini doesn't support this — skip.
PROMPT_CACHE = os.getenv("PROMPT_CACHE", "1") == "1"


def _build_system_message(user_id: str, request_id: str) -> Dict[str, Any]:
    """Build the system message. With caching on (and a Claude model), the static
    base prompt is a cacheable block and the dynamic profile/history is a separate
    uncached block. Otherwise a plain string (current behaviour)."""
    full = _system_prompt_with_profile(user_id, request_id)
    dynamic = full[len(AGENT_SYSTEM_PROMPT):]   # full == AGENT_SYSTEM_PROMPT + dynamic
    if PROMPT_CACHE and not MODEL.startswith("google/"):
        return {
            "role": "system",
            "content": [
                {"type": "text", "text": AGENT_SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": dynamic},
            ],
        }
    return {"role": "system", "content": full}


def _load_recent_history(user_id: str, limit: int) -> List[Dict[str, Any]]:
    """Pull the last `limit` messages from Redis and shape them as plain
    chat messages. Strips the upstream User Context wrapper.

    Rehydration policy: keep all user messages plus only the most recent
    assistant message. Older assistant turns are dropped to prevent the
    model from pattern-matching its own past responses — replaying a wall
    of prior refusals or wrong answers strongly biases the next turn to
    repeat them. The most recent assistant turn is kept so the agent can
    still resolve "you asked X, I'm answering Y" linkage."""
    try:
        prior = conv_manager.get_history(user_id, limit=limit) or []
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
    # Find the index of the most recent assistant message (if any) and drop
    # all earlier assistant turns — keep all user turns intact.
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


def _agent_loop_sync(user_query: str, user_id: str, request_id: str) -> Dict[str, Any]:
    """Run the agent until it produces a final response or hits MAX_AGENT_ITERS.
    Returns: {response, tools_used, iterations, sources, timings}

    History is baked into the system prompt (via _system_prompt_with_profile),
    so the message list contains only the current user turn — no separate
    history spread. This avoids the conversational-consistency poisoning seen
    when prior bad assistant turns are replayed in the message list."""
    messages: List[Dict[str, Any]] = [
        _build_system_message(user_id, request_id),
        {"role": "user", "content": user_query},
    ]
    tools_used: List[Dict[str, Any]] = []
    sources_collected: List[Dict[str, Any]] = []   # surfaced from get_document_knowledge tool results
    seen_chunk_ids = set()
    timings: Dict[str, float] = {"3_retrieve": 0.0, "4_generate": 0.0}

    for iteration in range(MAX_AGENT_ITERS):
        t0 = time_module.time()
        try:
            response = openrouter_client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS_OPENAI,
                temperature=0.4,
                # Cap thinking at ~20% of max_tokens — Pro and Flash are both
                # thinking models. Without this cap Pro can spend almost the
                # entire budget on internal reasoning and emit empty text.
                extra_body=_reasoning_extra_body(),
                # 16000 (was 8000): with effort=low, thinking ≈ 20% of this
                # budget, so a larger ceiling leaves more room for visible
                # output after reasoning — mitigates the known Gemini 2.5 Pro
                # empty-completion (finish_reason=MAX_TOKENS) failure mode.
                max_tokens=16000,
                timeout=120.0,
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
        msg = response.choices[0].message
        content = msg.content or ""
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            final = content.strip()
            # Empty-completion guard — GENERATION ONLY. Gemini 2.5 Pro
            # intermittently returns empty text on the final synthesis hop
            # (thinking-budget exhaustion and a known backend bug; see
            # github.com/googleapis/python-genai#811). We re-issue ONLY this
            # generation call (the messages already carry any tool results —
            # we do NOT re-run tools or restart the iteration) up to 2x before
            # falling back to a graceful message, so the user never sees blank.
            gen_retry = 0
            while not final and gen_retry < 2:
                gen_retry += 1
                logger.warning(f"[{request_id}] empty completion at iter {iteration+1}; generation retry {gen_retry}/2")
                try:
                    retry_resp = openrouter_client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=TOOLS_OPENAI,
                        temperature=0.4,
                        extra_body=_reasoning_extra_body(),
                        max_tokens=16000,
                        timeout=120.0,
                    )
                    final = (retry_resp.choices[0].message.content or "").strip()
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
                "iterations": iteration + 1, "sources": sources_collected, "timings": timings,
            }

        # Append the assistant message (with tool_calls)
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })

        # Execute each tool call and append the tool message
        for tc in tool_calls:
            name = tc.function.name
            args_json = tc.function.arguments or "{}"
            logger.info(f"[{request_id}] iter {iteration+1} → {name}({args_json[:120]})")
            t_tool = time_module.time()
            # Inject the user's ORIGINAL question for the OCI dual-pass retrieval
            # (rewrite-robust recall widening). Not part of the model-facing schema.
            if name == "get_document_knowledge":
                try:
                    _a = json.loads(args_json); _a["_user_query"] = user_query
                    args_json = json.dumps(_a)
                except Exception:
                    pass
            tool_result = execute_tool(name, args_json, user_id)
            tool_secs = time_module.time() - t_tool
            if name == "get_document_knowledge":
                timings["3_retrieve"] += tool_secs
                # Extract sources from tool result so streaming endpoint can emit source_found events
                try:
                    parsed = json.loads(tool_result)
                    for r in (parsed.get("results", []) or []):
                        src_file = r.get("source") or ""
                        if not src_file:
                            continue
                        # dedupe by (source_file, page) so we don't re-emit identical chunks
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

                # Auto-clarification consult is gated by AUTO_CLARIFY env var
                # (default off when MODEL is Pro — Pro's own reasoning already
                # covers what the secondary consult would add, calling Pro
                # twice is redundant). Set AUTO_CLARIFY=1 to force-enable.
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
                "role": "tool", "tool_call_id": tc.id, "content": tool_result,
            })

    logger.warning(f"[{request_id}] hit MAX_AGENT_ITERS={MAX_AGENT_ITERS}; returning best-effort")
    return {
        "response": "I'm having trouble finding the answer to that — could you rephrase or give me a bit more detail?",
        "tools_used": tools_used, "iterations": MAX_AGENT_ITERS,
        "sources": sources_collected, "timings": timings,
    }


async def agent_loop(user_query: str, user_id: str, request_id: str) -> Dict[str, Any]:
    """Async wrapper — runs the sync agent loop in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(_agent_loop_sync, user_query, user_id, request_id)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Azure RAG (Tool-Calling Agent + Reranker)", version="0.1.0")
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

    # Strip upstream User Context wrapper so the agent sees just the actual user request
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
        conv_manager.add_message(user_id, "user", raw_query, {"request_id": request_id, "backend": "azure-agent"})
        conv_manager.add_message(user_id, "assistant", final, {"request_id": request_id, "backend": "azure-agent"})
    except Exception as e:
        logger.warning(f"[{request_id}] persistence failed: {e}")

    elapsed = round(time_module.time() - start, 2)
    logger.info(f"[{request_id}] AGENT_QUERY_DONE | iters={result['iterations']} tools={len(result['tools_used'])} elapsed={elapsed}s")

    tools_used_names = [t["name"] for t in result.get("tools_used", [])]
    if not tools_used_names:
        route = "CONVERSATIONAL"
    elif "get_document_knowledge" in tools_used_names:
        route = "SIMPLE"
    else:
        route = "CONVERSATIONAL"

    timings = result.get("timings", {})
    timings_rounded = {k: round(v, 3) for k, v in timings.items()}

    return QueryResponse(
        response=format_gfm_to_html(final),
        metadata={
            "request_id": request_id,
            "route": route,
            "sources": (result.get("sources") or [])[:5],
            "elapsed_sec": elapsed,
            "timings": timings_rounded,
            # extras (not in production schema, but useful for observability)
            "backend": "azure-agent",
            "iterations": result.get("iterations"),
            "tools_used": tools_used_names,
            "model": MODEL,
        },
    )


def _stream_agent_worker(query_text, user_id, request_id, loop, queue, start_time):
    """Thread worker for REAL token streaming. Runs the tool-calling agent with
    stream=True on every turn: tool-call turns are consumed internally (emitting
    source_found events), and the final answer turn streams `token` events as the
    model emits them. Pushes SSE event dicts onto `queue`; ends with a {'__final__'}
    summary then None. Token-by-token only materialises for models that stream
    incrementally (Claude); Gemini still bursts at the end but remains correct."""
    def put(ev):
        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)
    try:
        messages = [_build_system_message(user_id, request_id),
                    {"role": "user", "content": query_text}]
        sources, seen, tools_used = [], set(), []
        final_text = ""
        first_token_time = None
        tok_in = tok_out = tok_cached = 0   # summed across every LLM call
        for iteration in range(MAX_AGENT_ITERS):
            stream = openrouter_client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS_OPENAI, temperature=0.4,
                max_tokens=16000, stream=True, extra_body=_reasoning_extra_body(),
                stream_options={"include_usage": True}, timeout=120.0)
            content_parts, tcs, tool_seen = [], {}, False
            reason_buf = ""        # accumulates reasoning until a sentence/clause boundary
            for chunk in stream:
                # usage arrives on the final chunk (include_usage) — sum it
                u = getattr(chunk, "usage", None)
                if u:
                    tok_in += getattr(u, "prompt_tokens", 0) or 0
                    tok_out += getattr(u, "completion_tokens", 0) or 0
                    ptd = getattr(u, "prompt_tokens_details", None)
                    if ptd is not None:
                        tok_cached += (getattr(ptd, "cached_tokens", 0)
                                       if not isinstance(ptd, dict)
                                       else ptd.get("cached_tokens", 0)) or 0
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # Stream the model's THINKING via `status` events (no new event
                # type → no frontend change). Flush on sentence/clause boundaries
                # so the status line updates readably instead of per-token flicker.
                rd = getattr(delta, "reasoning", None)
                if rd:
                    reason_buf += rd
                    if len(reason_buf) >= 60 and reason_buf[-1] in ".!?;\n":
                        put({"type": "status", "message": "💭 " + reason_buf.strip()[:220]})
                        reason_buf = ""
                dtc = getattr(delta, "tool_calls", None)
                if dtc:
                    tool_seen = True
                    for tc in dtc:
                        a = tcs.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                        if tc.id:
                            a["id"] = tc.id
                        if getattr(tc, "function", None):
                            if tc.function.name:
                                a["name"] = tc.function.name
                            if tc.function.arguments:
                                a["args"] += tc.function.arguments
                elif getattr(delta, "content", None) and not tool_seen:
                    if first_token_time is None:
                        first_token_time = round(time_module.time() - start_time, 3)
                    content_parts.append(delta.content)
                    put({"type": "token", "text": delta.content})
            if tcs:
                messages.append({
                    "role": "assistant", "content": "".join(content_parts) or None,
                    "tool_calls": [{"id": a["id"], "type": "function",
                                    "function": {"name": a["name"], "arguments": a["args"]}}
                                   for a in tcs.values()],
                })
                for a in tcs.values():
                    tools_used.append(a["name"])
                    logger.info(f"[{request_id}] iter {iteration+1} → {a['name']}({(a['args'] or '')[:120]})")
                    # Stream a live status for this processing step so the user
                    # sees activity during the otherwise-silent tool/retrieval
                    # phase (instead of a static spinner).
                    _status = {
                        "get_document_knowledge": "Searching the knowledge base…",
                        "get_user_profile": "Checking your profile…",
                        "save_user_profile": "Saving your details…",
                        "get_history": "Reviewing the conversation…",
                        "get_clarification": "Analysing the documents…",
                    }.get(a["name"], "Working on it…")
                    put({"type": "status", "message": _status})
                    _args = a["args"] or "{}"
                    if a["name"] == "get_document_knowledge":
                        try:
                            _p = json.loads(_args); _p["_user_query"] = query_text
                            _args = json.dumps(_p)
                        except Exception:
                            pass
                    res = execute_tool(a["name"], _args, user_id)
                    if a["name"] == "get_document_knowledge":
                        try:
                            parsed = json.loads(res)
                            for r in (parsed.get("results") or []):
                                sf = r.get("source") or ""
                                if not sf:
                                    continue
                                k = f"{sf}|{r.get('page')}|{r.get('chunk_type')}"
                                if k in seen:
                                    continue
                                seen.add(k)
                                sources.append({"source": sf, "score": r.get("score")})
                                put({"type": "source_found", "source": sf,
                                     "index": len(sources), "score": r.get("score") or 0})
                        except Exception:
                            pass
                    messages.append({"role": "tool", "tool_call_id": a["id"], "content": res})
                continue
            final_text = "".join(content_parts).strip()
            break
        if not final_text:
            final_text = ("I found relevant information but had trouble putting together a "
                          "complete answer just now. Could you rephrase or narrow your question?")
            put({"type": "token", "text": final_text})
        put({"__final__": True, "text": final_text, "sources": sources,
             "tools": tools_used, "ttft": first_token_time,
             "tok_in": tok_in, "tok_out": tok_out, "tok_cached": tok_cached})
    except Exception as e:
        logger.exception(f"[{request_id}] stream agent failed: {e}")
        put({"type": "error", "message": str(e)})
    finally:
        put(None)


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """SSE streaming endpoint — matches production rag-azure-7867's /query/stream protocol.
    Event types (for compatibility with existing frontend):
      - status        : intermediate progress messages
      - source_found  : retrieved sources during the agent's tool calls
      - token         : streamed answer text (word-at-a-time)
      - done          : final metadata (request_id, route, sources, elapsed_sec, timings, ttft)
      - error         : error message
    """
    from fastapi.responses import StreamingResponse

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

        # opening status — matches production
        yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...'})}\n\n"

        # ── REAL streaming path ──────────────────────────────────────────────
        # Stream every agent turn; consume tool turns internally and stream the
        # final answer token-by-token as the model emits it. Falls back to the
        # legacy chunk-after-complete path when REAL_STREAM=0.
        if REAL_STREAM:
            import threading
            queue: asyncio.Queue = asyncio.Queue()
            run_loop = asyncio.get_running_loop()
            threading.Thread(
                target=_stream_agent_worker,
                args=(query_text, user_id, request_id, run_loop, queue, start_time),
                daemon=True,
            ).start()
            final_text, sources, tools_used_names, ttft = "", [], [], None
            tok_in = tok_out = tok_cached = 0
            progress_sent = False
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                if ev.get("__final__"):
                    final_text = ev.get("text", "")
                    sources = ev.get("sources", [])
                    tools_used_names = ev.get("tools", [])
                    ttft = ev.get("ttft")
                    tok_in = ev.get("tok_in", 0)
                    tok_out = ev.get("tok_out", 0)
                    tok_cached = ev.get("tok_cached", 0)
                    continue
                if ev.get("type") == "error":
                    yield f"data: {json.dumps(ev)}\n\n"
                    return
                # emit the 'progress' milestone once, right before the first token
                # (preserves the legacy event sequence the frontend expects)
                if ev.get("type") == "token" and not progress_sent:
                    progress_sent = True
                    yield f"data: {json.dumps({'type': 'progress', 'percentage': 60, 'message': 'Generating answer...'})}\n\n"
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            try:
                conv_manager.add_message(user_id, "user", raw_query, {"request_id": request_id, "backend": "azure-agent"})
                conv_manager.add_message(user_id, "assistant", final_text, {"request_id": request_id, "backend": "azure-agent"})
            except Exception as e:
                logger.warning(f"[{request_id}] persistence failed: {e}")
            elapsed = round(time_module.time() - start_time, 3)
            route = "SIMPLE" if "get_document_knowledge" in tools_used_names else "CONVERSATIONAL"
            done_metadata = {
                "request_id": request_id, "route": route, "sources": sources[:5],
                "elapsed_sec": elapsed, "ttft": ttft, "timings": {},
                "tokens_in": tok_in, "tokens_out": tok_out, "tokens_cached": tok_cached,
            }
            yield f"data: {json.dumps({'type': 'done', 'metadata': done_metadata})}\n\n"
            logger.info(f"[{request_id}] STREAM_V2_COMPLETE | tools={tools_used_names} "
                        f"elapsed={elapsed}s ttft={ttft} tokens_in={tok_in} tokens_out={tok_out} tokens_cached={tok_cached}")
            return
        # ── legacy chunk-after-complete path (REAL_STREAM=0) ─────────────────

        try:
            result = await asyncio.to_thread(_agent_loop_sync, query_text, user_id, request_id)
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

        # Map tool usage to a route label compatible with production frontend
        if not tools_used_names:
            route = "CONVERSATIONAL"
        elif "get_document_knowledge" in tools_used_names:
            route = "SIMPLE"
        else:
            route = "CONVERSATIONAL"

        # source_found events for sources used during retrieval — matches production
        for idx, src in enumerate(sources[:5], 1):
            yield f"data: {json.dumps({'type': 'source_found', 'source': src.get('source', ''), 'index': idx, 'score': src.get('score', 0) or 0})}\n\n"

        # progress milestone before generation tokens — matches production sequence
        if route == "SIMPLE":
            yield f"data: {json.dumps({'type': 'progress', 'percentage': 60, 'message': 'Generating answer...'})}\n\n"

        # Stream the answer in markdown-preserving chunks. We hold whitespace runs
        # (including \n\n, list indentation, code spans) intact by grouping
        # `\S+\s*` segments — splitting plainly on whitespace would erase all
        # formatting and the frontend would render a wall of text.
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
            conv_manager.add_message(user_id, "user", raw_query, {"request_id": request_id, "backend": "azure-agent"})
            conv_manager.add_message(user_id, "assistant", final_text, {"request_id": request_id, "backend": "azure-agent"})
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
        info = qdrant_client.get_collection(COLLECTION_NAME_V2)
        return {
            "status": "ok",
            "service": "rag-azure-tools",
            "model": MODEL,
            "tools": get_tool_names(),
            "max_iters": MAX_AGENT_ITERS,
            "qdrant_collection": COLLECTION_NAME_V2,
            "qdrant_points": info.points_count,
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/")
def root():
    return {"service": "rag-azure-tools", "endpoints": ["/query", "/health"]}


if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("SERVICE_PORT", "7868"))   # rerank variant — distinct from base 7867/7887
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
