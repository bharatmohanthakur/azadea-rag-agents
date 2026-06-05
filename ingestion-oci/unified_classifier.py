"""
Unified Classifier - Single LLM call for all query classification.

Replaces 3 separate LLM calls:
1. general_query_handler.classify_query() - greeting/casual detection
2. greeting_detection_node (llm_classifier) - greeting detection
3. router_node - SIMPLE/COMPLEX/FORMAT/GENERIC routing

Saves 4-8 seconds per query by consolidating into ONE LLM call.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
# No OpenAI SDK needed — using OciAsOpenAI adapter
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger("UnifiedClassifier")


class UnifiedClassificationResult(BaseModel):
    """
    Single classification result that replaces 3 separate classifiers.
    """
    # Layer 1: Conversational detection (replaces general_handler + greeting_detection)
    is_conversational: bool = Field(
        default=False,
        description="True if this is a greeting, casual message, thanks, or expression - should respond directly without RAG"
    )
    conversational_type: Optional[str] = Field(
        default=None,
        description="Type of conversational message: greeting, casual, thanks, expression, farewell, about_assistant"
    )
    conversational_response: Optional[str] = Field(
        default=None,
        description="Direct response for conversational queries (short, friendly)"
    )

    # Layer 2: RAG routing (replaces router_node) - only relevant if NOT conversational
    rag_route: Optional[Literal["SIMPLE", "FORMAT", "CLARIFICATION_ANSWER", "PROFILE_UPDATE"]] = Field(
        default=None,
        description="RAG strategy - only set if is_conversational=False. SIMPLE covers all knowledge queries; PROFILE_UPDATE saves user attributes without retrieval."
    )

    # Layer 2b: Profile attributes — set when rag_route="PROFILE_UPDATE"
    profile_attributes: Optional[Dict[str, str]] = Field(
        default=None,
        description="User self-disclosure attributes (e.g., {'role': 'shop_manager', 'country': 'Lebanon'}). Only set when rag_route='PROFILE_UPDATE'."
    )

    # Layer 3: Special context
    is_clarification_answer: bool = Field(
        default=False,
        description="True if user is answering a previous clarification question"
    )
    is_follow_up: bool = Field(
        default=False,
        description="True if this is a follow-up to a previous knowledge query"
    )
    needs_query_rewrite: bool = Field(
        default=False,
        description="True if query references previous context and needs rewriting"
    )

    # Metadata
    confidence: float = Field(
        default=0.8,
        description="Classification confidence 0.0-1.0"
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for the classification"
    )


class UnifiedClassifier:
    """
    Single LLM classifier that handles all query classification in one call.

    Replaces:
    - GeneralQueryHandler.classify_query() + generate_response()
    - greeting_detection_node with llm_classifier
    - router_node

    Benefits:
    - 4-8 seconds faster (1 LLM call instead of 3)
    - Consistent classification logic
    - Better context awareness across all dimensions
    """

    def __init__(
        self,
        llm_client,
        deployment_name: str,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 300  # 5 min cache
    ):
        """
        Initialize unified classifier.

        Args:
            llm_client: OpenAI-compatible client
            deployment_name: Model name/deployment
            cache_enabled: Enable caching for repeated queries
            cache_ttl_seconds: Cache TTL
        """
        self.client = llm_client
        self.deployment = deployment_name
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple] = {}

        logger.info(f"UnifiedClassifier initialized with model: {deployment_name}")

    def _get_cache_key(self, query: str, has_history: bool, has_clarification: bool, history_hash: str = "") -> str:
        """Generate cache key including conversation history hash for context-aware caching."""
        key_str = f"{query}|{has_history}|{has_clarification}|{history_hash}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[UnifiedClassificationResult]:
        """Get cached result if valid."""
        if not self.cache_enabled or key not in self._cache:
            return None

        result, timestamp = self._cache[key]
        if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
            logger.info(f"Cache hit for classification")
            return result

        del self._cache[key]
        return None

    def _set_cached(self, key: str, value: UnifiedClassificationResult):
        """Cache a result."""
        if self.cache_enabled:
            self._cache[key] = (value, datetime.now())

    def classify(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        previous_response: Optional[str] = None,
        active_clarification: bool = False,
        clarification_question: Optional[str] = None,
        original_query: Optional[str] = None,
        user_profile: Optional[Dict[str, str]] = None,
    ) -> UnifiedClassificationResult:
        """
        Classify query in a SINGLE LLM call.

        Args:
            query: User's current query
            conversation_history: Recent conversation history
            previous_response: Last assistant response (for FORMAT detection)
            active_clarification: Whether there's an active clarification session
            clarification_question: The clarification question asked
            original_query: Original query before clarification
            user_profile: Stored user attributes (role, country, brand, etc.) — informs
                          PROFILE_UPDATE detection (don't re-store known facts) and
                          clarification logic (don't ask for already-known info).

        Returns:
            UnifiedClassificationResult with all classification info
        """
        # Check cache (include history hash + profile hash for context-aware caching)
        history_hash = ""
        if conversation_history:
            last_msgs = conversation_history[-4:]  # Last 2 turns for cache key
            history_hash = hashlib.md5(
                json.dumps(last_msgs, default=str).encode()
            ).hexdigest()[:8]
        if user_profile:
            history_hash += "_" + hashlib.md5(
                json.dumps(user_profile, sort_keys=True).encode()
            ).hexdigest()[:6]
        cache_key = self._get_cache_key(query, bool(conversation_history), active_clarification, history_hash)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Build context strings
        history_str = ""
        if conversation_history:
            history_str = "\n".join([
                f"[{m.get('role', 'unknown').upper()}]: {m.get('content', '')}"
                for m in conversation_history
            ])

        clarification_context = ""
        if active_clarification:
            clarification_context = f"""
ACTIVE CLARIFICATION SESSION:
- Original Question: {original_query or 'Unknown'}
- Clarification Asked: {clarification_question or 'Unknown'}
- User's Current Response: {query}
NOTE: There is an active clarification session. If the user's response answers the clarification question, set is_clarification_answer=true.
"""
        else:
            # Even without an active session, the LLM should check conversation history
            clarification_context = """
NO ACTIVE CLARIFICATION SESSION — but check the conversation history above.
If the assistant's last message asked a question and the user's current query is answering it,
set is_clarification_answer=true and rag_route="CLARIFICATION_ANSWER".
"""

        previous_context = ""
        if previous_response:
            previous_context = f"\nPREVIOUS ASSISTANT RESPONSE:\n{previous_response}"

        # Profile context — informs PROFILE_UPDATE (avoid re-saving known attrs) and
        # clarification logic (don't ask for facts the user already disclosed).
        profile_context = ""
        if user_profile:
            attrs_str = "\n".join(f"  - {k}: {v}" for k, v in user_profile.items())
            profile_context = f"""
KNOWN USER PROFILE (already stored from previous turns):
{attrs_str}
NOTE:
- Do NOT issue PROFILE_UPDATE for attributes already in this list — that wastes a turn.
  Only issue PROFILE_UPDATE if the user discloses a NEW attribute or CHANGES an existing one.
- Do NOT ask clarifying questions for attributes already known above.
  Example: if country is already "Lebanon", don't ask "which country?" — just use Lebanon.
"""

        # Build the unified prompt
        prompt = f"""Analyze this user query and classify it in ONE pass.

<query>{query}</query>

<context>
CONVERSATION HISTORY:
{history_str or "(No prior conversation)"}
{profile_context}
{clarification_context}
{previous_context}
</context>

<task>
Determine ALL of the following in one analysis:

1. **IS_CONVERSATIONAL**: Is this a greeting, casual message, thanks, or expression?
   - Greetings: "hi", "hello", "hey", "good morning/afternoon/evening"
   - Casual: "ok", "sure", "got it", "bye", "see you"
   - Thanks: "thanks", "thank you", "appreciate it"
   - Expressions: "awesome", "great", "perfect", "I love you"
   - About assistant: "who are you", "what can you do"

   If YES: Set is_conversational=true and provide a short, friendly conversational_response.

   PERSONA — when crafting conversational_response (greetings, "who are you", etc.):
     The bot's name is "Dea". It is the internal knowledge assistant for
     Azadea Group employees, covering HR, Operations, Finance & Accounting, IT, Stock
     Management, F&B, Marketing, BCP, and compliance. NEVER frame it as an "HR assistant"
     or say "HR questions". Frame it as the multi-domain Azadea knowledge assistant.
     Greetings should invite a question across any of these domains, vary wording each turn,
     and stay under 30 words.
     Good: "Hi! I'm Dea. Ask me anything about Azadea policies or procedures."
     Good: "Welcome back. What can I look up for you?"
     Bad:  "Hello! How can I help you with HR questions today?" (too narrow)

2. **RAG_ROUTE** (only if NOT conversational):
   - "SIMPLE": ANY knowledge question about Azadea Group — factual, complex, ambiguous, all go here.
     The knowledge base spans HR, Operations, Finance & Accounting, IT, Stock Management, F&B,
     Marketing, BCP, and compliance. This is NOT an HR-only bot.
     Examples: "What is the dress code?", "How many maternity leave days?",
               "tell me about the DCR process", "what is the policy for shipment receiving?",
               "how do I get JDE access?", "uniform allowance for shop manager"
   - "FORMAT": Asking to reformat/restructure PREVIOUS response
     Examples: "Put that in a table", "Make it bullet points", "as list"
     CRITICAL: Must have previous_response AND format keywords
   - "PROFILE_UPDATE": User STATES a fact about themselves (role, country, brand, employment type, location)
     WITHOUT asking a question. The user is providing context for future questions.
     Examples: "I'm a shop manager", "I work in Lebanon", "Im a full time employee in Jordan",
               "I work in eataly as f&b manager", "Im based in UAE", "back office in jordan"
     CRITICAL: Set profile_attributes as a JSON object with keys like "role", "country", "brand",
               "employment_type", "department". Examples:
               "I'm a shop manager"            -> {{"role": "shop_manager"}}
               "I work in Lebanon"             -> {{"country": "Lebanon"}}
               "f&b manager at eataly"         -> {{"role": "f&b_manager", "brand": "eataly"}}
               "im a full time employee in Jordan" -> {{"employment_type": "full_time", "country": "Jordan"}}
     ALSO write a friendly conversational_response acknowledging the role and inviting a question.
     Examples:
       "Got it — you're a shop manager. What can I help you with regarding store operations, payroll, or shifts?"
       "Noted, you work in Lebanon. What would you like to know?"
     DO NOT route as PROFILE_UPDATE if the user is asking a question that mentions their role
     (e.g. "as a shop manager what's my uniform allowance?" -> SIMPLE).

3. **SPECIAL CASES**:
   - is_clarification_answer: True if the user is answering a clarification question.
     IMPORTANT: Check the conversation history — if the assistant's last message asked a question
     (e.g. "Which country are you in?", "Are you asking about...?") and the current query
     looks like a direct answer to that question (e.g. "Lebanon", "back office", "annual leave"),
     set is_clarification_answer=true and rag_route="CLARIFICATION_ANSWER".
     This applies whether or not there is an active clarification session.
   - is_follow_up: True if references previous topic
   - needs_query_rewrite: True if uses pronouns/references needing context

Respond in JSON format:
{{
    "is_conversational": boolean,
    "conversational_type": "greeting" | "casual" | "thanks" | "expression" | "farewell" | "about_assistant" | null,
    "conversational_response": "short friendly response" | null,
    "rag_route": "SIMPLE" | "FORMAT" | "CLARIFICATION_ANSWER" | "PROFILE_UPDATE" | null,
    "profile_attributes": {{"role": "...", "country": "...", "brand": "...", "employment_type": "...", "department": "..."}} | null,
    "is_clarification_answer": boolean,
    "is_follow_up": boolean,
    "needs_query_rewrite": boolean,
    "confidence": float (0.0-1.0),
    "reasoning": "brief explanation"
}}
</task>"""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise query classifier. Respond with valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=500,
                response_format={"type": "json_object"},
                timeout=8.0
            )

            result_text = response.choices[0].message.content.strip()
            result_json = json.loads(result_text)

            # Normalize legacy routes to SIMPLE
            raw_route = result_json.get("rag_route")
            if raw_route in ("COMPLEX", "GENERIC"):
                raw_route = "SIMPLE"

            # Build result
            profile_attrs = result_json.get("profile_attributes")
            if profile_attrs and not isinstance(profile_attrs, dict):
                profile_attrs = None

            result = UnifiedClassificationResult(
                is_conversational=result_json.get("is_conversational", False),
                conversational_type=result_json.get("conversational_type"),
                conversational_response=result_json.get("conversational_response"),
                rag_route=raw_route,
                profile_attributes=profile_attrs if raw_route == "PROFILE_UPDATE" else None,
                is_clarification_answer=result_json.get("is_clarification_answer", False),
                is_follow_up=result_json.get("is_follow_up", False),
                needs_query_rewrite=result_json.get("needs_query_rewrite", False),
                confidence=result_json.get("confidence", 0.8),
                reasoning=result_json.get("reasoning", "")
            )

            # Handle FORMAT detection - must have previous response
            if result.rag_route == "FORMAT" and not previous_response:
                result.rag_route = "SIMPLE"  # No previous response to format

            logger.info(f"Unified classification: conversational={result.is_conversational}, "
                       f"route={result.rag_route}, confidence={result.confidence:.2f}")

            # Cache result
            self._set_cached(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Unified classification failed: {e}")
            # Return safe default
            return self._fallback_classify(query, previous_response, active_clarification)

    def _fallback_classify(
        self,
        query: str,
        previous_response: Optional[str],
        active_clarification: bool
    ) -> UnifiedClassificationResult:
        """
        Fallback classification using pattern matching.
        Used when LLM call fails.
        """
        query_lower = query.lower().strip()

        # Check greetings
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if any(query_lower == g or query_lower.startswith(g + " ") for g in greetings) and len(query.split()) <= 3:
            return UnifiedClassificationResult(
                is_conversational=True,
                conversational_type="greeting",
                conversational_response="Hi! I'm Dea — I can help with policies and procedures across HR, Operations, IT, Finance, and more. What do you need?",
                confidence=0.9,
                reasoning="Pattern match: greeting detected"
            )

        # Check thanks
        if any(t in query_lower for t in ["thanks", "thank you", "appreciate"]) and len(query.split()) <= 5:
            return UnifiedClassificationResult(
                is_conversational=True,
                conversational_type="thanks",
                conversational_response="You're welcome! Let me know if you need anything else.",
                confidence=0.9,
                reasoning="Pattern match: thanks detected"
            )

        # Check casual
        casual = ["ok", "okay", "sure", "got it", "understood", "bye", "goodbye"]
        if query_lower in casual:
            return UnifiedClassificationResult(
                is_conversational=True,
                conversational_type="casual",
                conversational_response="Is there anything else I can help you with?",
                confidence=0.9,
                reasoning="Pattern match: casual detected"
            )

        # Check FORMAT
        format_keywords = ["as table", "as a table", "in table", "table format",
                         "as points", "bullet points", "as list", "as a list"]
        if previous_response and any(kw in query_lower for kw in format_keywords):
            return UnifiedClassificationResult(
                is_conversational=False,
                rag_route="FORMAT",
                confidence=0.85,
                reasoning="Pattern match: format request with previous response"
            )

        # Default to SIMPLE (covers all knowledge queries — HR, Ops, Finance, IT, etc.)
        return UnifiedClassificationResult(
            is_conversational=False,
            rag_route="SIMPLE",
            confidence=0.5,
            reasoning="Fallback: defaulting to SIMPLE"
        )


# Global instance
_unified_classifier: Optional[UnifiedClassifier] = None


def init_unified_classifier(llm_client, deployment_name: str, **kwargs):
    """Initialize the global unified classifier."""
    global _unified_classifier
    _unified_classifier = UnifiedClassifier(llm_client, deployment_name, **kwargs)
    return _unified_classifier


def get_unified_classifier() -> Optional[UnifiedClassifier]:
    """Get the global unified classifier instance."""
    return _unified_classifier
