"""
Utility functions for OCI RAG server.
Copied from rag_server_gemini.py to avoid importing it (which triggers Azure + Graphiti init).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from markdown_it import MarkdownIt
from mdit_py_plugins.tasklists import tasklists_plugin

logger = logging.getLogger("RAG-Server-OCI")

# Markdown renderer
_md = MarkdownIt("gfm-like").use(tasklists_plugin)


def format_gfm_to_html(text: str) -> str:
    if not text or not text.strip():
        return text
    return _md.render(text)


def log_request(request_id: str, step: str, data: Any, level: str = "info") -> None:
    try:
        data_str = json.dumps(data, default=str, ensure_ascii=False)[:500] if not isinstance(data, str) else data[:500]
    except Exception:
        data_str = str(data)[:500]
    msg = f"[{request_id}] {step} | {data_str}"
    if level == "error":
        logger.error(msg)
    elif level == "warning":
        logger.warning(msg)
    else:
        logger.info(msg)


def count_tokens(text: str, method: str = "approximate") -> int:
    if not text:
        return 0
    if method == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass
    return max(1, len(text) // 4)


def get_user_history(conv_manager, user_id: str, use_summarization: bool = False) -> List[Dict[str, str]]:
    """Get conversation history. Always use_summarization=False for OCI (no summarizer)."""
    history = conv_manager.get_history(user_id, limit=20)
    return [{"role": msg.get("role"), "content": msg.get("content")} for msg in history]


def rewrite_query_with_history(
    history: List[Dict[str, str]],
    latest_query: str,
    user_id: str = None,
    clarification_tracker=None,
    llm_client=None,
    model_name: str = "google.gemini-2.5-flash",
) -> str:
    """
    Rewrite contextual query into standalone question using OCI LLM.
    Extracted from rag_server_gemini.py with injectable dependencies.
    """
    # Check for active clarification
    if user_id and clarification_tracker:
        active_session = clarification_tracker.get_active_session(user_id)
        if active_session:
            if clarification_tracker.is_clarification_response(user_id, latest_query):
                return latest_query

    if not history:
        return latest_query

    # Get original question if available
    original_question = None
    try:
        from conversation_manager import get_conversation_manager
        conv_mgr = get_conversation_manager()
        original_question = conv_mgr.get_original_question(user_id, within_last_n=15)
    except Exception:
        pass

    # Filter greetings from history
    greeting_patterns = ["hi", "hello", "hey", "thanks", "thank you", "okay", "ok",
                         "sure", "great", "awesome", "perfect"]
    filtered_history = []
    for msg in history[-10:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "").strip().lower()
        if role == "user":
            is_greeting = any(p in content for p in greeting_patterns) and len(content.split()) <= 5
            if is_greeting:
                continue
        filtered_history.append(msg)

    if not filtered_history:
        return latest_query

    history_str = ""
    for msg in filtered_history:
        history_str += f"{msg.get('role', 'unknown')}: {msg.get('content', '')}\n"

    original_context = f"\n**IMPORTANT - Original Question**: {original_question}\n" if original_question else ""

    prompt = f"""You are an AI assistant. Your task is to rewrite the latest user question into a standalone question.
{original_context}
**RULES**:
1. **Preserve User's Question**: If the latest input is a complete question, keep its core topic and intent intact.
2. **Add Context Only When Needed**: Only add context from history to resolve pronouns (it, they, that) or ambiguous references.
3. **Format Requests**: If latest query is "give me as table" / "provide as points", keep it as-is - it's a format request.
4. **Clarification Answers**: If user is answering a clarification question, combine their answer with the ORIGINAL QUESTION.
5. **Ignore Greetings**: Do NOT include greetings (hi, hello, thanks) in the rewritten query.
6. **Do Not Force-Merge Topics**: If the user switches to a NEW topic, respect that - don't force-merge with previous topics.
7. **Do Not Hallucinate**: Only use info from the provided history.

Conversation History (greetings filtered out):
{history_str}

Latest User Input: {latest_query}

Standalone Question:"""

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        if original_question and len(rewritten.split()) < 5:
            rewritten = f"{original_question} - {latest_query}"
        return rewritten
    except Exception as e:
        logger.error(f"Error rewriting query: {e}")
        return latest_query
