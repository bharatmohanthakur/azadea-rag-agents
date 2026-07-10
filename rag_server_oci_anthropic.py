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

# Reuse existing singletons — conv_manager (Redis) + HTML formatter.
from rag_server_gemini import conv_manager, format_gfm_to_html
# Retrieval tools + dispatch are provider-agnostic (execute_tool takes a tool
# name + JSON args). The reranker inside agent_tools_azure_rerank keeps using
# OpenRouter's /rerank endpoint; only the CHAT model moves to native Anthropic.
from agent_tools_oci_claude import TOOLS_OPENAI, execute_tool, get_tool_names, qdrant_client, COLLECTION_NAME_V2

import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("RAG-OCI-Anthropic")

# ── Native Anthropic chat backend ────────────────────────────────────────────
# Direct first-party Claude API (NOT OpenRouter, NOT the OpenAI-compat layer).
# Going native is what unlocks real prompt caching — Anthropic's OpenAI-compat
# endpoint explicitly does NOT support caching. Key comes from ANTHROPIC_API_KEY.
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Native model id: dash form, no "anthropic/" prefix (that's OpenRouter's scheme).
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_AGENT_ITERS = int(os.getenv("AGENT_MAX_ITERS", "5"))

# Per-1M-token USD prices for the cost estimate returned in query metadata.
# Keyed by model-id prefix; falls back to Sonnet 4.6 rates. Override input/output
# via env if the model changes. tok_in as logged already folds in cache reads +
# writes, so we price the cached portion at the read rate and the rest at input.
_PRICING = {
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00, "cache_read": 0.30},
    "claude-opus-4-8":   {"in": 5.00, "out": 25.00, "cache_read": 0.50},
    "claude-haiku-4-5":  {"in": 1.00, "out": 5.00,  "cache_read": 0.10},
}


def _estimate_cost_usd(tok_in: int, tok_out: int, tok_cached: int) -> float:
    """Approximate the API cost of one query from its token counts.
    tok_in includes cache reads; tok_cached is the cache-read portion (billed at
    0.1x). The remainder is priced at the input rate (cache writes, a small
    fraction, are approximated at the input rate)."""
    p = _PRICING.get(MODEL, _PRICING["claude-sonnet-4-6"])
    p_in = float(os.getenv("PRICE_IN_PER_M", p["in"]))
    p_out = float(os.getenv("PRICE_OUT_PER_M", p["out"]))
    p_cache = float(os.getenv("PRICE_CACHE_READ_PER_M", p["cache_read"]))
    uncached = max(0, tok_in - tok_cached)
    cost = (tok_cached * p_cache + uncached * p_in + tok_out * p_out) / 1_000_000
    return round(cost, 6)
REAL_STREAM = os.getenv("REAL_STREAM", "1") == "1"
# Native prompt caching via cache_control on the system+tools prefix (bills at
# 0.1x on hits). Sonnet 4.6's minimum cacheable prefix is 2048 tokens; the
# static system prompt + tool schemas (~2.2K) clears it.
PROMPT_CACHE = os.getenv("PROMPT_CACHE", "1") == "1"
# Stream Claude's thinking as 💭 status events. Off by default for lowest TTFT;
# enabling it requires an SDK/model that accepts the adaptive thinking param.
STREAM_THINKING = os.getenv("STREAM_THINKING", "0") == "1"


def _tools_anthropic():
    """Convert the OpenAI-format tool list to native Anthropic tool schema:
    {type:function, function:{name,description,parameters}}  →
    {name, description, input_schema}.  execute_tool() is unchanged — it dispatches
    on the tool name regardless of provider."""
    out = []
    for t in TOOLS_OPENAI:
        fn = t.get("function", t)
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


TOOLS_ANTHROPIC = _tools_anthropic()


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


def _build_system_blocks(user_id: str, request_id: str) -> List[Dict[str, Any]]:
    """Build the native-Anthropic top-level `system` parameter as content blocks.
    The STATIC instructions are one cacheable block (cache_control: ephemeral) —
    in render order tools → system → messages, a breakpoint on this block caches
    the tool schemas + static prompt together (bills ~0.1x on reads). The dynamic
    profile/history goes in a separate UNcached block (changes per user, so
    caching it would never hit). With caching off, both ship uncached."""
    full = _system_prompt_with_profile(user_id, request_id)
    dynamic = full[len(AGENT_SYSTEM_PROMPT):]   # full == AGENT_SYSTEM_PROMPT + dynamic
    static_block = {"type": "text", "text": AGENT_SYSTEM_PROMPT}
    if PROMPT_CACHE:
        static_block["cache_control"] = {"type": "ephemeral"}
    blocks = [static_block]
    if dynamic.strip():
        blocks.append({"type": "text", "text": dynamic})
    return blocks


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
    system_blocks = _build_system_blocks(user_id, request_id)
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": user_query},
    ]
    tools_used: List[Dict[str, Any]] = []
    sources_collected: List[Dict[str, Any]] = []   # surfaced from get_document_knowledge tool results
    seen_chunk_ids = set()
    timings: Dict[str, float] = {"3_retrieve": 0.0, "4_generate": 0.0}

    def _call():
        return anthropic_client.messages.create(
            model=MODEL, system=system_blocks, messages=messages,
            tools=TOOLS_ANTHROPIC, temperature=0.4, max_tokens=16000, timeout=120.0)

    for iteration in range(MAX_AGENT_ITERS):
        t0 = time_module.time()
        try:
            response = _call()
        except Exception as e:
            logger.exception(f"[{request_id}] LLM call failed at iter {iteration+1}: {e}")
            return {
                "response": "I hit an error working on that — please try again in a moment.",
                "tools_used": tools_used, "iterations": iteration + 1,
                "sources": sources_collected, "timings": timings, "error": str(e),
            }

        llm_secs = round(time_module.time() - t0, 2)
        timings["4_generate"] += llm_secs
        content = "".join(b.text for b in response.content if b.type == "text")
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            final = content.strip()
            # Empty-completion guard — re-issue ONLY the generation call (messages
            # already carry any tool results; we do NOT re-run tools) up to 2x.
            gen_retry = 0
            while not final and gen_retry < 2:
                gen_retry += 1
                logger.warning(f"[{request_id}] empty completion at iter {iteration+1}; generation retry {gen_retry}/2")
                try:
                    retry_resp = _call()
                    final = "".join(b.text for b in retry_resp.content if b.type == "text").strip()
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

        # Append the assistant turn (raw content blocks — includes the tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool call and collect tool_result blocks for one user turn
        tool_results: List[Dict[str, Any]] = []
        for tu in tool_uses:
            name = tu.name
            tool_args = dict(tu.input or {})   # Anthropic gives a parsed dict
            if name == "get_document_knowledge":
                # Server-side injection: the user's ORIGINAL question, so the tool
                # can run a rewrite-robust second retrieval pass + rerank against
                # it. Never part of the model-facing schema.
                tool_args["_user_query"] = user_query
            args_json = json.dumps(tool_args)
            logger.info(f"[{request_id}] iter {iteration+1} → {name}({args_json[:120]})")
            t_tool = time_module.time()
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
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": tool_result})

        # All tool_result blocks go back in a SINGLE user message (native format)
        messages.append({"role": "user", "content": tool_results})

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

app = FastAPI(title="OCI RAG (Anthropic native + Reranker)", version="0.1.0")
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
        system_blocks = _build_system_blocks(user_id, request_id)
        messages = [{"role": "user", "content": query_text}]
        sources, seen, tools_used = [], set(), []
        final_text = ""
        first_token_time = None
        tok_in = tok_out = tok_cached = 0   # summed across every LLM call
        _STATUS = {
            "get_document_knowledge": "Searching the knowledge base…",
            "get_user_profile": "Checking your profile…",
            "save_user_profile": "Saving your details…",
            "get_history": "Reviewing the conversation…",
            "get_clarification": "Analysing the documents…",
            "list_documents": "Gathering the document list…",
        }
        for iteration in range(MAX_AGENT_ITERS):
            content_parts, tool_seen = [], False
            reason_buf = ""        # accumulates thinking until a sentence/clause boundary
            with anthropic_client.messages.stream(
                model=MODEL, system=system_blocks, messages=messages,
                tools=TOOLS_ANTHROPIC, temperature=0.4, max_tokens=16000) as stream:
                for event in stream:
                    et = event.type
                    if et == "content_block_start":
                        cb = event.content_block
                        if getattr(cb, "type", None) == "tool_use":
                            # A tool block opened → emit its status as soon as we
                            # know the name (args still streaming). Stops further
                            # text from this turn being treated as the answer.
                            tool_seen = True
                            put({"type": "status",
                                 "message": _STATUS.get(cb.name, "Working on it…")})
                    elif et == "content_block_delta":
                        d = event.delta
                        dt = getattr(d, "type", None)
                        if dt == "text_delta" and not tool_seen:
                            if first_token_time is None:
                                first_token_time = round(time_module.time() - start_time, 3)
                            content_parts.append(d.text)
                            put({"type": "token", "text": d.text})
                        elif dt == "thinking_delta" and STREAM_THINKING:
                            reason_buf += getattr(d, "thinking", "") or ""
                            if len(reason_buf) >= 60 and reason_buf[-1] in ".!?;\n":
                                put({"type": "status", "message": "💭 " + reason_buf.strip()[:220]})
                                reason_buf = ""
                final_msg = stream.get_final_message()
            # Native usage: input_tokens is the UNcached prompt; cache_read /
            # cache_creation are separate. Report tok_in as the full prompt
            # (comparable to OpenRouter's prompt_tokens) and tok_cached as reads.
            u = final_msg.usage
            cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
            tok_in += (getattr(u, "input_tokens", 0) or 0) + cache_read + cache_write
            tok_out += getattr(u, "output_tokens", 0) or 0
            tok_cached += cache_read

            tool_uses = [b for b in final_msg.content if b.type == "tool_use"]
            if tool_uses:
                messages.append({"role": "assistant", "content": final_msg.content})
                tool_results = []
                for tu in tool_uses:
                    tools_used.append(tu.name)
                    tool_args = dict(tu.input or {})
                    if tu.name == "get_document_knowledge":
                        # Server-side injection of the user's ORIGINAL question —
                        # enables the rewrite-robust second retrieval pass in the
                        # tool. Not part of the model-facing schema.
                        tool_args["_user_query"] = query_text
                    args_json = json.dumps(tool_args)
                    logger.info(f"[{request_id}] iter {iteration+1} → {tu.name}({args_json[:120]})")
                    res = execute_tool(tu.name, args_json, user_id)
                    if tu.name == "get_document_knowledge":
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
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})
                messages.append({"role": "user", "content": tool_results})
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
                "cost_usd": _estimate_cost_usd(tok_in, tok_out, tok_cached),
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
    PORT = int(os.getenv("SERVICE_PORT", "7875"))   # OCI-retrieval native-Anthropic variant
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
