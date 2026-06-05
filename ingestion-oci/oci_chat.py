"""
Native OCI GenAI chat helpers — no LangChain, no OpenAI SDK.
Direct OCI SDK calls for reliable, low-overhead LLM access.

Provides:
  - oci_chat(messages, **kwargs)              → full response text
  - oci_chat_json(messages, schema, **kwargs) → parsed JSON dict (JSON_SCHEMA mode)
  - oci_chat_stream(messages, **kwargs)       → generator yielding text chunks
  - oci_chat_async(messages, **kwargs)        → async wrapper for non-streaming
  - oci_stream_async(messages, **kwargs)      → async generator for streaming
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

import oci
from oci.generative_ai_inference.models import (
    AssistantMessage,
    ChatDetails,
    FunctionCall,
    FunctionDefinition,
    GenericChatRequest,
    JsonSchemaResponseFormat,
    OnDemandServingMode,
    ResponseJsonSchema,
    SystemMessage,
    TextContent,
    ToolMessage,
    UserMessage,
)

from oci_clients import (
    OCI_CHAT_MODEL,
    OCI_COMPARTMENT_ID,
    get_vision_client,
)

logger = logging.getLogger("oci_chat")


def _build_messages(messages: List[tuple]) -> list:
    """Convert tuple-style messages to OCI message objects.
    Input: [("system", "..."), ("user", "...")]
    """
    oci_messages = []
    for role, content in messages:
        if role == "system":
            oci_messages.append(SystemMessage(content=[TextContent(text=content)]))
        elif role == "user":
            oci_messages.append(UserMessage(content=[TextContent(text=content)]))
    return oci_messages


def _make_request(
    messages: List[tuple],
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    is_stream: bool = False,
    json_schema: Optional[Dict] = None,
) -> ChatDetails:
    model = model or OCI_CHAT_MODEL

    response_format = None
    if json_schema:
        response_format = JsonSchemaResponseFormat(
            type="JSON_SCHEMA",
            json_schema=ResponseJsonSchema(
                name="structured_output",
                schema=json_schema,
            ),
        )

    chat_request = GenericChatRequest(
        messages=_build_messages(messages),
        temperature=temperature,
        max_tokens=max_tokens,
        is_stream=is_stream,
        response_format=response_format,
    )
    return ChatDetails(
        compartment_id=OCI_COMPARTMENT_ID,
        serving_mode=OnDemandServingMode(model_id=model),
        chat_request=chat_request,
    )


def _extract_text(response) -> str:
    """Extract text from OCI chat response."""
    cr = response.data.chat_response
    if cr.choices and cr.choices[0].message and cr.choices[0].message.content:
        return cr.choices[0].message.content[0].text.strip()
    return ""


# ---------------------------------------------------------------------------
# Synchronous
# ---------------------------------------------------------------------------

def oci_chat(
    messages: List[tuple],
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 8000,
    retries: int = 3,
) -> str:
    """Non-streaming chat → returns full response text."""
    client = get_vision_client()
    details = _make_request(messages, model, temperature, max_tokens)

    for attempt in range(retries):
        try:
            response = client.chat(details)
            return _extract_text(response)
        except Exception as e:
            if attempt < retries - 1:
                sleep_s = 2 ** attempt
                logger.warning(f"oci_chat retry {attempt+1}/{retries}: {e} (sleep {sleep_s}s)")
                time.sleep(sleep_s)
            else:
                raise


def oci_chat_json(
    messages: List[tuple],
    schema: Dict[str, Any],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    retries: int = 3,
) -> Dict[str, Any]:
    """Chat with JSON_SCHEMA enforcement → returns parsed dict."""
    client = get_vision_client()
    details = _make_request(messages, model, temperature, max_tokens, json_schema=schema)

    for attempt in range(retries):
        try:
            response = client.chat(details)
            text = _extract_text(response)
            if not text:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {}
            # Strip markdown code blocks if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3].strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            if attempt < retries - 1:
                logger.warning(f"oci_chat_json parse retry {attempt+1}: {e}")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"oci_chat_json failed to parse: {text[:200]}")
                raise
        except Exception as e:
            if attempt < retries - 1:
                sleep_s = 2 ** attempt
                logger.warning(f"oci_chat_json retry {attempt+1}/{retries}: {e} (sleep {sleep_s}s)")
                time.sleep(sleep_s)
            else:
                raise


def oci_chat_stream(
    messages: List[tuple],
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Generator[str, None, None]:
    """Streaming chat → yields text chunks."""
    client = get_vision_client()
    details = _make_request(messages, model, temperature, max_tokens, is_stream=True)
    response = client.chat(details)

    for event in response.data.events():
        try:
            parsed = json.loads(event.data)
            msg = parsed.get("message", {})
            content = msg.get("content", [])
            if content and isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("text"):
                        yield c["text"]
        except json.JSONDecodeError:
            pass


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------

async def oci_chat_async(
    messages: List[tuple],
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 8000,
) -> str:
    """Async non-streaming chat."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: oci_chat(messages, model, temperature, max_tokens)
    )


async def oci_stream_async(
    messages: List[tuple],
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 8000,
) -> AsyncGenerator[str, None]:
    """Async streaming chat — yields text chunks."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _stream_worker():
        try:
            for chunk in oci_chat_stream(messages, model, temperature, max_tokens):
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    loop.run_in_executor(None, _stream_worker)

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk


# ---------------------------------------------------------------------------
# Tool-calling (NEW — used by the agent service; existing helpers untouched)
# ---------------------------------------------------------------------------

def _build_messages_for_tools(messages: List[Dict[str, Any]]) -> list:
    """Convert OpenAI-style dict messages to OCI message objects, with tool-call support.

    Accepts message dicts with these roles:
      {"role": "system",    "content": str}
      {"role": "user",      "content": str}
      {"role": "assistant", "content": str|None, "tool_calls": [{"id","name","arguments"}]?}
      {"role": "tool",      "tool_call_id": str, "content": str}
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=[TextContent(text=content)]))
        elif role == "user":
            out.append(UserMessage(content=[TextContent(text=content)]))
        elif role == "assistant":
            tcs_in = m.get("tool_calls") or []
            tcs_out = []
            for tc in tcs_in:
                tcs_out.append(FunctionCall(
                    type="FUNCTION",
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"]),
                ))
            am = AssistantMessage(
                content=[TextContent(text=content)] if content else None,
                tool_calls=tcs_out or None,
            )
            out.append(am)
        elif role == "tool":
            out.append(ToolMessage(
                role="TOOL",
                tool_call_id=m["tool_call_id"],
                content=[TextContent(text=str(content))],
            ))
    return out


def oci_chat_with_tools(
    messages: List[Dict[str, Any]],
    tools: List[FunctionDefinition],
    model: str = None,
    temperature: float = 0.4,
    max_tokens: int = 4000,
    retries: int = 3,
    reasoning_effort: str = None,
) -> Dict[str, Any]:
    """Call OCI Gemini with tool definitions. Returns a dict:
        {
          "content": str | None,        # final text if any
          "tool_calls": [               # populated when the model wants to call tools
              {"id": str, "name": str, "arguments": str (JSON)},
              ...
          ],
        }
    Designed for an agent loop: caller checks tool_calls, executes them,
    appends assistant + tool messages, calls again until tool_calls is empty.
    """
    client = get_vision_client()
    model = model or OCI_CHAT_MODEL

    chat_kwargs = dict(
        messages=_build_messages_for_tools(messages),
        temperature=temperature,
        max_tokens=max_tokens,
        is_stream=False,
        tools=tools,
        # Force one tool call per turn. OCI/Gemini rejects multi-call turns
        # with 400 "number of function response parts must equal the number
        # of function call parts" because our message builder emits one
        # ToolMessage per response (OCI reads them as separate turns) while
        # Gemini expects the responses grouped to match the parallel calls.
        # Disabling parallel calls sidesteps this entirely — the model just
        # takes an extra iteration to chain tools sequentially.
        is_parallel_tool_calls=False,
    )
    if reasoning_effort:
        chat_kwargs["reasoning_effort"] = reasoning_effort
    chat_request = GenericChatRequest(**chat_kwargs)
    details = ChatDetails(
        compartment_id=OCI_COMPARTMENT_ID,
        serving_mode=OnDemandServingMode(model_id=model),
        chat_request=chat_request,
    )

    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat(details)
            cr = response.data.chat_response
            if not (cr.choices and cr.choices[0].message):
                return {"content": "", "tool_calls": []}
            msg = cr.choices[0].message

            # Extract text content (may be None when only tool_calls)
            text = ""
            if msg.content:
                for c in msg.content:
                    t = getattr(c, "text", None)
                    if t:
                        text += t

            # Extract tool_calls — synthesize an id if the model didn't supply one
            # (OCI Gemini sometimes omits id, but OCI's API requires it on the
            # subsequent assistant message round-trip).
            import uuid as _uuid
            tool_calls = []
            for tc in (getattr(msg, "tool_calls", None) or []):
                tc_id = getattr(tc, "id", None) or f"call_{_uuid.uuid4().hex[:12]}"
                tool_calls.append({
                    "id": tc_id,
                    "name": getattr(tc, "name", None),
                    "arguments": getattr(tc, "arguments", "{}"),
                })

            return {"content": text.strip(), "tool_calls": tool_calls}
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"oci_chat_with_tools failed after {retries} attempts: {e}")
                raise


async def oci_chat_with_tools_async(
    messages: List[Dict[str, Any]],
    tools: List[FunctionDefinition],
    model: str = None,
    temperature: float = 0.4,
    max_tokens: int = 4000,
    reasoning_effort: str = None,
) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: oci_chat_with_tools(
            messages, tools, model, temperature, max_tokens,
            reasoning_effort=reasoning_effort,
        ),
    )
