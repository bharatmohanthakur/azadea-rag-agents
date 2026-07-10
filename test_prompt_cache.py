#!/usr/bin/env python3
"""
Standalone test: does Anthropic prompt caching actually engage via OpenRouter?

Sends the SAME large static system block (content-blocks format with
cache_control) on two back-to-back calls. Expectation:
  call 1 → cache WRITE  (cache_creation tokens > 0)
  call 2 → cache READ   (cached tokens > 0, billed at ~10%)
"""
import sys, time, json
sys.path.insert(0, ".")
from rag_server_gemini import openrouter_client

MODEL = "anthropic/claude-sonnet-4.6"

# A static block that must exceed Sonnet's 1024-token minimum to be cacheable.
STATIC = ("You are Dea, Azadea Group's internal knowledge assistant. "
          "Follow these rules carefully. " + ("Policy guidance and procedure detail. " * 200))

def make_system(cached: bool):
    """System message as content-blocks; mark the static block cacheable."""
    block = {"type": "text", "text": STATIC}
    if cached:
        block["cache_control"] = {"type": "ephemeral"}
    return {"role": "system", "content": [block]}

def call(cached: bool, user_msg: str):
    t0 = time.time()
    r = openrouter_client.chat.completions.create(
        model=MODEL,
        messages=[make_system(cached), {"role": "user", "content": user_msg}],
        max_tokens=60,
    )
    dt = round(time.time() - t0, 2)
    u = r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage)
    details = u.get("prompt_tokens_details") or {}
    cached_tok = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    return {
        "secs": dt,
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "cached_tokens": cached_tok,
        "cache_write": u.get("cache_creation_input_tokens") or u.get("cache_write_tokens"),
        "raw_usage": u,
    }

print(f"static block ~chars={len(STATIC)} (~{len(STATIC)//4} tokens)\n")

print("=== call 1 (cache WRITE expected) ===")
r1 = call(True, "Say 'one'.")
print(f"  prompt={r1['prompt_tokens']} cached={r1['cached_tokens']} write={r1['cache_write']} ({r1['secs']}s)")

time.sleep(2)

print("\n=== call 2 (cache READ / hit expected) ===")
r2 = call(True, "Say 'two'.")
print(f"  prompt={r2['prompt_tokens']} cached={r2['cached_tokens']} write={r2['cache_write']} ({r2['secs']}s)")

print("\n=== control: same call WITHOUT cache_control ===")
r3 = call(False, "Say 'three'.")
print(f"  prompt={r3['prompt_tokens']} cached={r3['cached_tokens']} write={r3['cache_write']} ({r3['secs']}s)")

print("\n--- raw usage of call 2 (to see exact field names) ---")
print(json.dumps(r2["raw_usage"], indent=2))

hit = (r2["cached_tokens"] or 0) > 0
print(f"\n>>> PROMPT CACHING WORKS: {hit}  (call 2 cached_tokens={r2['cached_tokens']})")
