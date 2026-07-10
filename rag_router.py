"""
Model router — port 7960.

Routes each /query/stream request to the right backend over the SAME v2
retrieval stack (both backends share collection docs_oci_claude_v2, reranker,
dual-pass, Redis conversation history — the user sees one continuous thread):

  EASY        → Haiku 4.5  (:7996)  — 2x cheaper, ~1s TTFT
  HARD        → Sonnet 4.6 (:7969)  — synthesis, follow-ups, reasoning
  FRUSTRATED  → Sonnet 4.6 + sticky escalation (whole session stays on Sonnet)

Decision order per turn:
  0. sticky flag set for this user            → Sonnet
  1. trivial fast-path (greeting/thanks)      → Haiku   (no classifier call)
  2. loud frustration markers (rules)         → Sonnet + sticky
  3. Haiku classifier (~400ms, ~$0.0002):
       query + this user's previous turn context → EASY|HARD|FRUSTRATED
  4. classifier error / backend down          → Sonnet  (fail-safe)

Escalation is also PREVENTIVE: if an EASY answer streams back containing a
failure marker ("couldn't find", "please clarify"...), the sticky flag is set
so the user's next turn lands on Sonnet before irritation shows.

Every decision is appended to logs/router_decisions.jsonl for audit/tuning.
SHADOW=1 sends everything to Sonnet while still logging what WOULD have routed.
"""
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
import redis
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env")
import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | RAG-Router | %(message)s")
logger = logging.getLogger("rag_router")

PORT = int(os.getenv("SERVICE_PORT", "7960"))
EASY_BACKEND = os.getenv("EASY_BACKEND", "http://localhost:7996")   # Haiku 4.5
HARD_BACKEND = os.getenv("HARD_BACKEND", "http://localhost:7969")   # Sonnet 4.6 (prod)
SHADOW = os.getenv("SHADOW", "0") == "1"                            # log-only mode
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "claude-haiku-4-5")
STICKY_TTL = int(os.getenv("STICKY_TTL_SECONDS", "1800"))           # 30 min
DECISION_LOG = Path(__file__).parent / "logs" / "router_decisions.jsonl"
DECISION_LOG.parent.mkdir(exist_ok=True)

rds = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)
anthropic_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

app = FastAPI(title="RAG Model Router (Haiku/Sonnet)")


class QueryRequest(BaseModel):
    query: str
    user_id: str | None = None


# ── rules ────────────────────────────────────────────────────────────────────
GREETING_RE = re.compile(
    r"^\s*(hi|hii+|hello|hey|good (morning|afternoon|evening)|salam|marhaba|"
    r"thank(s| you).{0,20}|ok(ay)?|great|bye|شكرا|مرحبا)\s*[!. ]*$", re.I)
LOUD_FRUSTRATION_RE = re.compile(
    r"(!{2,}|\?{3,}|\b(wrong|useless|not what i asked|i already (said|asked)|"
    r"again\?|why can'?t you|this is not)\b)", re.I)
FAILURE_MARKERS = ("couldn't find", "could not find", "wasn't able to find",
                   "could you clarify", "rephrase", "not found in the knowledge base",
                   "i hit an error")

CLASSIFIER_SYSTEM = """You route employee questions for Azadea Group's internal policy chatbot to one of two models. Output EXACTLY one word: EASY, HARD, or FRUSTRATED.

EASY — a single factual lookup a small model handles well: one policy/brand/leave type + one attribute; short how-to; greeting/smalltalk.
Examples: "Zara refund policy" | "annual leave days entitlement" | "how to apply for medical leave" | "maternity leave Jordan" | "employee discount limit grade 8 Egypt"

HARD — needs reasoning, synthesis across documents, or context resolution: comparisons ("vs", "difference", "all brands"), why/explain questions, multi-part questions, follow-ups whose meaning depends on earlier turns ("how do I apply for it?"), Arabic or mixed-language, vague/underspecified asks.
Examples: "attendance policy fashion shop vs F&B employees" | "why does the organization reserve the right not to issue recommendation letters" | "can I apply for both a certification and a master's" | "العمل من المنزل policy"

FRUSTRATED — the user shows irritation or is stuck: repeating/rephrasing the same question, correcting the bot ("no, I meant…", "that's wrong"), emphasis (caps, !!), or their PREVIOUS ANSWER (provided as context) failed to help.

If unsure between EASY and HARD, answer HARD."""


def _sticky_key(uid: str) -> str: return f"router:esc:{uid}"
def _last_key(uid: str) -> str: return f"router:last:{uid}"


def _log_decision(rec: dict):
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        with open(DECISION_LOG, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


async def classify(query: str, uid: str) -> tuple[str, str]:
    """Returns (label, reason). Rules first, then the Haiku classifier."""
    try:
        if rds.exists(_sticky_key(uid)):
            return "HARD", "sticky-escalated"
    except Exception:
        pass
    if GREETING_RE.match(query):
        return "EASY", "rule-greeting"
    if LOUD_FRUSTRATION_RE.search(query):
        return "FRUSTRATED", "rule-loud-frustration"

    # context for the classifier: this user's previous turn as the router saw it
    ctx = ""
    try:
        last = rds.get(_last_key(uid))
        if last:
            l = json.loads(last)
            ctx = (f"\n\nContext — user's previous question: {l.get('q','')[:200]}"
                   f"\nPrevious answer ending: …{l.get('ans_tail','')[:200]}"
                   f"\nPrevious answer failed to help: {l.get('failed', False)}")
            if l.get("q", "").strip().casefold() == query.strip().casefold():
                return "FRUSTRATED", "rule-exact-repeat"
    except Exception:
        pass
    try:
        resp = await anthropic_client.messages.create(
            model=CLASSIFIER_MODEL, max_tokens=5,
            system=[{"type": "text", "text": CLASSIFIER_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"Question: {query}{ctx}"}])
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        label = (raw.split() or [""])[0].strip(".:,").upper()
        if label not in ("EASY", "HARD", "FRUSTRATED"):
            return "HARD", f"classifier-unparseable:{raw[:20]}"
        return label, "classifier"
    except Exception as e:
        return "HARD", f"classifier-error:{type(e).__name__}"


def _set_sticky(uid: str, why: str):
    try:
        rds.setex(_sticky_key(uid), STICKY_TTL, why)
    except Exception:
        pass


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    uid = request.user_id or "default_user"
    query = request.query.strip()
    t0 = time.time()

    label, reason = await classify(query, uid)
    if label == "FRUSTRATED":
        _set_sticky(uid, reason)
    backend = EASY_BACKEND if (label == "EASY" and not SHADOW) else HARD_BACKEND
    routed_model = "haiku-4.5" if backend == EASY_BACKEND else "sonnet-4.6"
    classify_ms = round((time.time() - t0) * 1000)
    logger.info(f"[{uid[:12]}] {label} ({reason}, {classify_ms}ms) → {routed_model} | q={query[:80]}")

    async def generate():
        answer_parts = []
        sent_done = False
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5)) as client:
                async with client.stream("POST", f"{backend}/query/stream",
                                         json={"query": query, "user_id": uid}) as resp:
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        try:
                            ev = json.loads(line[6:])
                        except Exception:
                            yield line + "\n\n"
                            continue
                        if ev.get("type") == "token":
                            answer_parts.append(ev.get("text") or "")
                        if ev.get("type") == "done":
                            ev.setdefault("metadata", {}).update({
                                "routed_to": routed_model, "route_label": label,
                                "route_reason": reason, "classify_ms": classify_ms,
                                "shadow": SHADOW,
                            })
                            sent_done = True
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception(f"[{uid[:12]}] backend {routed_model} failed: {e}")
            if backend == EASY_BACKEND:
                # failover: replay the whole request on Sonnet
                _set_sticky(uid, "easy-backend-failure")
                async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5)) as client:
                    async with client.stream("POST", f"{HARD_BACKEND}/query/stream",
                                             json={"query": query, "user_id": uid}) as resp:
                        async for line in resp.aiter_lines():
                            if line and line.startswith("data: "):
                                yield line + "\n\n"
                sent_done = True
            elif not sent_done:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        # post-answer bookkeeping: remember this turn; preventive escalation
        answer = "".join(answer_parts)
        failed = any(m in answer.lower() for m in FAILURE_MARKERS)
        if failed and routed_model.startswith("haiku"):
            _set_sticky(uid, "preventive-failure-marker")
        try:
            rds.setex(_last_key(uid), STICKY_TTL,
                      json.dumps({"q": query, "ans_tail": answer[-300:],
                                  "failed": failed, "model": routed_model}))
        except Exception:
            pass
        _log_decision({"user": uid, "q": query, "label": label, "reason": reason,
                       "routed_to": routed_model, "classify_ms": classify_ms,
                       "answer_failed": failed, "shadow": SHADOW})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    out = {"status": "ok", "service": "rag-router", "shadow": SHADOW,
           "easy_backend": EASY_BACKEND, "hard_backend": HARD_BACKEND}
    async with httpx.AsyncClient(timeout=4) as client:
        for name, url in (("easy", EASY_BACKEND), ("hard", HARD_BACKEND)):
            try:
                r = await client.get(f"{url}/health")
                out[f"{name}_model"] = r.json().get("model")
                out[f"{name}_ok"] = r.status_code == 200
            except Exception as e:
                out[f"{name}_ok"] = False
                out[f"{name}_error"] = str(e)[:60]
    return JSONResponse(out)


@app.get("/router/stats")
def stats():
    counts, esc = {}, 0
    try:
        with open(DECISION_LOG) as f:
            for line in f:
                d = json.loads(line)
                counts[d.get("label")] = counts.get(d.get("label"), 0) + 1
    except FileNotFoundError:
        pass
    try:
        esc = len(list(rds.scan_iter("router:esc:*", count=500)))
    except Exception:
        pass
    return {"decisions": counts, "currently_escalated_users": esc, "shadow": SHADOW}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
