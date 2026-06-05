"""
Topic-change & ambiguity classifier — single LLM call gate (Azure tier).

Sits BEFORE UnifiedClassifier. Given the last 7 user queries plus the current
user query, decides:
  - has the topic changed?
  - is the current query too short/vague to be confident?
If either is true AND a clarifying question is producible, fires INTENT_CLARIFY.

Outcome-oriented prompt: no phrase lists, no taxonomies, no thresholds.
The same model (google/gemini-2.5-flash via OpenRouter) decides everything.

Two production-hardenings vs OCI v1:
  1. Empty-question guard — caller checks suggested_question.strip() before firing
  2. Greeting-not-topic clause — greetings in history are NOT a topic, so the
     first real question after a greeting is NOT a topic change
"""

import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("TopicChangeClassifier")


class TopicChangeResult(BaseModel):
    topic_changed: bool = Field(
        description="True if the current query is about a different topic than the recent ones."
    )
    is_ambiguous: bool = Field(
        description="True if the current query is too short or vague to be sure what the user means."
    )
    confidence: float = Field(
        default=0.0,
        description="0.0 to 1.0 — how confident you are in the topic_changed and is_ambiguous decisions.",
    )
    suggested_question: str = Field(
        default="",
        description="If topic_changed or is_ambiguous, write ONE short friendly clarifying question. Otherwise empty.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of your decision.",
    )


class TopicChangeClassifier:
    def __init__(self, llm_client, deployment_name: str = "google/gemini-2.5-flash"):
        self.client = llm_client
        self.model = deployment_name
        logger.info(f"TopicChangeClassifier initialized with model: {deployment_name}")

    def _build_prompt(self, recent_user_queries: List[str], current_user_query: str) -> str:
        recent_block = "\n".join(
            f"  {i+1}. {q}" for i, q in enumerate(recent_user_queries)
        ) or "  (none yet)"

        return f"""You are analyzing a conversation to detect topic changes and ambiguity.

RECENT USER QUERIES (oldest to most recent):
{recent_block}

CURRENT USER QUERY:
{current_user_query}

Looking at the trajectory of recent queries and the current one, decide:

1. topic_changed — Has the user moved to a different topic than what they were
   asking about? Consider the dominant topic in recent queries; a brief detour
   doesn't count if they're now back on the main thread.

   IMPORTANT: A greeting in the recent queries (hi, hello, hey, hola, مرحبا,
   كيف حالك, good morning, etc.) is NOT a "topic" — it's just an opener. If
   the user's only prior queries were greetings or pleasantries, the current
   query is the START of the conversation, not a topic change. Set
   topic_changed=false in that case.

2. is_ambiguous — Is the current query too short, vague, or context-dependent
   to be confident what the user means without asking? Short pronoun or
   fragment messages ("first action?", "the formula", "and another thing")
   are usually ambiguous when prior context could be interpreted multiple ways.

If topic_changed OR is_ambiguous is true, write ONE short, friendly clarifying
question that would resolve the specific uncertainty. Reference the actual
content of the conversation in the question — not generic placeholders. Frame
it as a check, not an interrogation. Keep it under 30 words.

If neither flag is true, set suggested_question to empty string.

Respond with valid JSON matching this shape:
{{
  "topic_changed": <bool>,
  "is_ambiguous": <bool>,
  "confidence": <0.0-1.0>,
  "suggested_question": "<string>",
  "reasoning": "<short explanation>"
}}
"""

    def classify(
        self,
        recent_user_queries: List[str],
        current_user_query: str,
    ) -> TopicChangeResult:
        """Synchronous call. Returns TopicChangeResult.
        On any error, returns a no-op result (no clarification, no flags)."""
        if not current_user_query.strip():
            return TopicChangeResult(
                topic_changed=False, is_ambiguous=False, confidence=1.0,
                suggested_question="", reasoning="empty current query",
            )

        prompt = self._build_prompt(recent_user_queries, current_user_query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You analyze conversation for topic changes and ambiguity. Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=400,
                response_format={"type": "json_object"},
                timeout=8.0,
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            result = TopicChangeResult(
                topic_changed=bool(data.get("topic_changed", False)),
                is_ambiguous=bool(data.get("is_ambiguous", False)),
                confidence=float(data.get("confidence", 0.0)),
                suggested_question=str(data.get("suggested_question", "") or "").strip(),
                reasoning=str(data.get("reasoning", "") or "").strip(),
            )
            logger.info(
                f"Topic-change: changed={result.topic_changed}, "
                f"ambiguous={result.is_ambiguous}, conf={result.confidence:.2f}, "
                f"reason='{result.reasoning[:80]}'"
            )
            return result
        except Exception as e:
            logger.warning(f"TopicChangeClassifier failed: {e} — defaulting to no-op")
            return TopicChangeResult(
                topic_changed=False, is_ambiguous=False, confidence=0.0,
                suggested_question="", reasoning=f"classifier error: {e}",
            )


_topic_change_classifier: Optional[TopicChangeClassifier] = None


def get_or_init_topic_change_classifier(llm_client, deployment_name: str) -> TopicChangeClassifier:
    global _topic_change_classifier
    if _topic_change_classifier is None:
        _topic_change_classifier = TopicChangeClassifier(llm_client, deployment_name)
    return _topic_change_classifier
