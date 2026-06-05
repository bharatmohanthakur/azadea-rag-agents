"""
Thin adapter: makes native OCI GenAI look like openai.OpenAI for UnifiedClassifier/LLMClassifier.

These classifiers call: client.chat.completions.create(model=..., messages=[...], response_format=..., ...)
This adapter translates that to native OCI SDK calls.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from oci_chat import oci_chat, oci_chat_json
from oci_clients import OCI_CHAT_MODEL

logger = logging.getLogger("oci_adapter")


@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _ChatCompletion:
    choices: List[_Choice] = field(default_factory=list)
    model: str = ""


class _Completions:
    """Mimics openai.resources.chat.Completions interface."""

    def create(
        self,
        model: str = None,
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        response_format: Optional[Dict] = None,
        timeout: float = None,
        tools: Optional[List] = None,
        tool_choice: Optional[Any] = None,
        **kwargs,
    ) -> _ChatCompletion:
        # Convert OpenAI-style messages to tuple format
        msg_tuples = []
        for msg in (messages or []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("system", "user"):
                msg_tuples.append((role, content))
            # Skip assistant messages (not supported in our simple adapter)

        model = model or OCI_CHAT_MODEL

        # If response_format requests JSON — always use native JSON_SCHEMA
        if response_format and response_format.get("type") in ("json_object", "json_schema"):
            # Build a permissive schema for json_object mode
            schema = {"type": "object", "additionalProperties": True}

            # Use oci_chat_json which sends JsonSchemaResponseFormat natively
            try:
                result = oci_chat_json(
                    msg_tuples, schema=schema,
                    model=model, temperature=temperature,
                    max_tokens=max(max_tokens, 2048),  # Gemini needs room for thinking
                )
                text = json.dumps(result)
            except Exception as e:
                logger.error(f"oci_chat_json failed: {e}")
                raise
        else:
            text = oci_chat(
                msg_tuples, model=model,
                temperature=temperature, max_tokens=max_tokens,
            )

        return _ChatCompletion(
            choices=[_Choice(message=_Message(content=text))],
            model=model,
        )


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class OciAsOpenAI:
    """
    Drop-in replacement for openai.OpenAI that routes to native OCI SDK.

    Usage:
        client = OciAsOpenAI()
        # Works like: openai.OpenAI().chat.completions.create(...)
        response = client.chat.completions.create(
            model="google.gemini-2.5-flash",
            messages=[{"role": "user", "content": "hello"}],
            response_format={"type": "json_object"},
        )
        print(response.choices[0].message.content)
    """

    def __init__(self):
        self.chat = _Chat()
