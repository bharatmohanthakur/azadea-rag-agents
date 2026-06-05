"""
Advanced Conversation Features - Critical Missing Pieces
Implements 7 critical features for complete best-in-class status:
1. Explicit Correction Handling
2. Conversational Repair
3. Reasoning Explanation
4. Why/How Question Handling
5. Comparison Intelligence
6. Enhanced Uncertainty Expression
7. Session Continuity
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger("AdvancedConversation")


class CorrectionType(Enum):
    """Types of user corrections."""
    ENTITY_CORRECTION = "entity_correction"  # "I meant Lebanon, not UAE"
    INTENT_CORRECTION = "intent_correction"  # "No, I'm asking about bonus not leave"
    CLARIFICATION = "clarification"  # "Actually, I'm a manager"
    NEGATION = "negation"  # "Not that, something else"


class RepairStrategy(Enum):
    """Conversational repair strategies."""
    ACKNOWLEDGE_AND_RESTART = "acknowledge_restart"
    OFFER_ALTERNATIVES = "offer_alternatives"
    SIMPLIFY_QUESTION = "simplify_question"
    PROVIDE_EXAMPLES = "provide_examples"
    ESCALATE_TO_HUMAN = "escalate"


@dataclass
class CorrectionDetection:
    """Detected user correction."""
    is_correction: bool
    correction_type: Optional[CorrectionType] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    entity_type: Optional[str] = None
    confidence: float = 0.0
    repair_instruction: Optional[str] = None


@dataclass
class ReasoningPath:
    """System's reasoning path for transparency."""
    steps: List[str] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    assumptions_made: List[str] = field(default_factory=list)
    alternatives_considered: List[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Structured comparison result."""
    items_compared: List[str]
    dimensions: List[str]
    comparison_table: Dict[str, Dict[str, Any]]
    winner: Optional[str] = None
    summary: str = ""
    recommendation: Optional[str] = None


@dataclass
class UncertaintyInfo:
    """Enhanced uncertainty information."""
    confidence_score: float
    confidence_label: str  # "Very High", "High", "Medium", "Low"
    reasons_for_uncertainty: List[str] = field(default_factory=list)
    verification_needed: bool = False
    alternative_interpretations: List[str] = field(default_factory=list)
    scope_limitations: List[str] = field(default_factory=list)
    recommendation_text: str = ""


class ExplicitCorrectionHandler:
    """
    Handles explicit user corrections gracefully.
    Detects when user corrects previous input and updates context.
    """

    def __init__(self, llm_client=None, deployment_name: str = None):
        """
        Initialize correction handler.

        Args:
            llm_client: LLM client for advanced detection
            deployment_name: Azure deployment name
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name

        # Correction patterns
        self.correction_patterns = [
            # Entity corrections
            (r"(?:no|not|actually),?\s*(?:i meant|i said|i mean)\s+(\w+)(?:\s+not\s+(\w+))?", CorrectionType.ENTITY_CORRECTION),
            (r"(?:i meant|i said)\s+(\w+)(?:,?\s*not\s+(\w+))?", CorrectionType.ENTITY_CORRECTION),
            (r"not\s+(\w+)(?:,?\s*but\s+(\w+))", CorrectionType.ENTITY_CORRECTION),
            (r"(\w+),?\s*not\s+(\w+)", CorrectionType.ENTITY_CORRECTION),

            # Intent corrections
            (r"(?:no|not)\s*(?:that|this)(?:\.|,)?\s*(?:i(?:'m| am) asking about|i meant|i want)\s+(.+)", CorrectionType.INTENT_CORRECTION),
            (r"(?:actually|forget that)(?:\.|,)?\s*(?:i want to know about|tell me about)\s+(.+)", CorrectionType.INTENT_CORRECTION),

            # Negations
            (r"(?:no|nope|not that|none of these)", CorrectionType.NEGATION),
            (r"(?:forget|ignore|skip)\s+(?:that|this|it)", CorrectionType.NEGATION),
        ]

    def detect_correction(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        context: Any  # ConversationContext
    ) -> CorrectionDetection:
        """
        Detect if user is making a correction.

        Args:
            query: Current user query
            conversation_history: Recent conversation
            context: Conversation context

        Returns:
            CorrectionDetection with details
        """
        query_lower = query.lower().strip()

        # Quick pattern matching
        for pattern, correction_type in self.correction_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                return self._handle_pattern_match(match, correction_type, query, context)

        # LLM-based detection for complex cases
        if self.llm_client and len(conversation_history) > 0:
            return self._detect_with_llm(query, conversation_history, context)

        return CorrectionDetection(is_correction=False)

    def _handle_pattern_match(
        self,
        match: re.Match,
        correction_type: CorrectionType,
        query: str,
        context: Any
    ) -> CorrectionDetection:
        """Handle pattern match for correction."""
        groups = match.groups()

        if correction_type == CorrectionType.ENTITY_CORRECTION:
            # Extract old and new values
            new_value = groups[0] if groups else None
            old_value = groups[1] if len(groups) > 1 else None

            # Determine entity type from context
            entity_type = self._infer_entity_type(new_value, old_value, context)

            return CorrectionDetection(
                is_correction=True,
                correction_type=correction_type,
                old_value=old_value,
                new_value=new_value,
                entity_type=entity_type,
                confidence=0.9,
                repair_instruction=f"Update {entity_type or 'entity'} to {new_value}"
            )

        elif correction_type == CorrectionType.NEGATION:
            return CorrectionDetection(
                is_correction=True,
                correction_type=correction_type,
                confidence=0.95,
                repair_instruction="User rejects current path, offer alternatives"
            )

        elif correction_type == CorrectionType.INTENT_CORRECTION:
            new_intent = groups[0] if groups else None
            return CorrectionDetection(
                is_correction=True,
                correction_type=correction_type,
                new_value=new_intent,
                confidence=0.85,
                repair_instruction=f"Switch to new topic: {new_intent}"
            )

        return CorrectionDetection(is_correction=False)

    def _infer_entity_type(
        self,
        new_value: Optional[str],
        old_value: Optional[str],
        context: Any
    ) -> Optional[str]:
        """Infer entity type from values and context."""
        if not new_value:
            return None

        new_lower = new_value.lower()

        # Country patterns
        countries = ["lebanon", "uae", "egypt", "saudi", "kuwait", "qatar", "jordan"]
        if new_lower in countries or (old_value and old_value.lower() in countries):
            return "country"

        # Position patterns
        positions = ["manager", "director", "employee", "supervisor", "executive"]
        if new_lower in positions or (old_value and old_value.lower() in positions):
            return "position"

        # Check context for recent entity types
        if hasattr(context, 'entities'):
            if context.entities:
                # Use most recently asked entity type
                entity_types = list(context.entities.keys())
                if entity_types:
                    return entity_types[-1]

        return None

    def _detect_with_llm(
        self,
        query: str,
        history: List[Dict[str, Any]],
        context: Any
    ) -> CorrectionDetection:
        """Use LLM to detect corrections."""
        # Get last exchange
        last_bot_msg = ""
        if history:
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    last_bot_msg = msg.get("content", "")
                    break

        prompt = f"""Analyze if the user is making a correction or negation.

Last Bot Message: "{last_bot_msg}"
User Response: "{query}"

Is the user:
1. Correcting a previous answer (e.g., "I meant Lebanon not UAE")?
2. Rejecting and wanting something else (e.g., "No, not that")?
3. Clarifying their previous input?
4. Just answering normally?

Respond in JSON:
{{
    "is_correction": true/false,
    "correction_type": "entity_correction|intent_correction|negation|none",
    "old_value": "what they said before or null",
    "new_value": "what they want now or null",
    "entity_type": "country|position|policy_type|null",
    "confidence": 0.0-1.0
}}"""

        try:
            model_param = self.deployment_name if self.deployment_name else "gpt-4o"
            response = self.llm_client.chat.completions.create(
                model=model_param,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            if result.get("is_correction"):
                correction_type_str = result.get("correction_type")
                correction_type_map = {
                    "entity_correction": CorrectionType.ENTITY_CORRECTION,
                    "intent_correction": CorrectionType.INTENT_CORRECTION,
                    "negation": CorrectionType.NEGATION
                }

                return CorrectionDetection(
                    is_correction=True,
                    correction_type=correction_type_map.get(correction_type_str),
                    old_value=result.get("old_value"),
                    new_value=result.get("new_value"),
                    entity_type=result.get("entity_type"),
                    confidence=result.get("confidence", 0.8)
                )

        except Exception as e:
            logger.error(f"LLM correction detection failed: {e}")

        return CorrectionDetection(is_correction=False)

    def apply_correction(
        self,
        correction: CorrectionDetection,
        context: Any,
        conversation_manager: Any
    ) -> str:
        """
        Apply correction to context and generate response.

        Args:
            correction: Detected correction
            context: Conversation context
            conversation_manager: ConversationManager instance

        Returns:
            Acknowledgment message
        """
        if correction.correction_type == CorrectionType.ENTITY_CORRECTION:
            # Update entity in context
            if correction.entity_type and correction.new_value and hasattr(context, 'entities'):
                # Remove old entity
                if correction.old_value and correction.entity_type in context.entities:
                    context.entities[correction.entity_type] = [
                        e for e in context.entities[correction.entity_type]
                        if e.value.lower() != correction.old_value.lower()
                    ]

                # Add new entity
                from conversation_context import Entity
                new_entity = Entity(
                    type=correction.entity_type,
                    value=correction.new_value.title(),
                    confidence=0.95,
                    turn=context.turn_count if hasattr(context, 'turn_count') else 0,
                    source="user_correction"
                )
                context.add_entity(new_entity)

            response = f"✅ Got it! I've updated to **{correction.new_value.title()}**. Let me find the right information for you."

        elif correction.correction_type == CorrectionType.INTENT_CORRECTION:
            response = f"✅ Understood! Switching focus to **{correction.new_value}**. Let me help you with that."

        elif correction.correction_type == CorrectionType.NEGATION:
            response = "I understand you're looking for something different. Let me offer some alternatives:\n\nWhat would you like to know about?"

        else:
            response = "✅ Thanks for clarifying! Let me adjust my understanding."

        return response


class ConversationalRepair:
    """
    Handles conversational breakdowns and repairs gracefully.
    Detects when user is confused, stuck, or wants to abandon.
    """

    def __init__(self, llm_client=None, deployment_name: str = None):
        """Initialize conversational repair."""
        self.llm_client = llm_client
        self.deployment_name = deployment_name

        # Confusion/abandonment signals
        self.confusion_signals = [
            "i don't understand", "what do you mean", "confused", "what?",
            "huh?", "what are you asking", "i'm lost", "not sure what you mean"
        ]

        self.abandonment_signals = [
            "forget it", "never mind", "i don't know", "skip this",
            "i give up", "this is too complicated", "not working",
            "start over", "forget that question"
        ]

        self.uncertainty_signals = [
            "i don't know", "not sure", "maybe", "i think", "probably",
            "i guess", "could be", "not certain"
        ]

    def detect_need_for_repair(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        context: Any
    ) -> Tuple[bool, Optional[RepairStrategy], str]:
        """
        Detect if conversation needs repair.

        Args:
            query: Current user query
            conversation_history: Conversation history
            context: Conversation context

        Returns:
            Tuple of (needs_repair, strategy, reason)
        """
        query_lower = query.lower().strip()

        # Check for confusion
        if any(signal in query_lower for signal in self.confusion_signals):
            return True, RepairStrategy.SIMPLIFY_QUESTION, "User is confused"

        # Check for abandonment
        if any(signal in query_lower for signal in self.abandonment_signals):
            return True, RepairStrategy.ACKNOWLEDGE_AND_RESTART, "User wants to abandon"

        # Check for uncertainty
        if any(signal in query_lower for signal in self.uncertainty_signals):
            return True, RepairStrategy.PROVIDE_EXAMPLES, "User is uncertain"

        # Check for repeated questions (user not getting answers)
        if self._is_repeated_question(query, conversation_history):
            return True, RepairStrategy.OFFER_ALTERNATIVES, "User repeating question"

        # Check for very short answers after multiple turns (frustration)
        if len(query.split()) <= 2 and hasattr(context, 'turn_count') and context.turn_count >= 4:
            return True, RepairStrategy.SIMPLIFY_QUESTION, "User showing frustration"

        return False, None, ""

    def _is_repeated_question(
        self,
        query: str,
        history: List[Dict[str, Any]]
    ) -> bool:
        """Check if user is repeating a question."""
        if not history:
            return False

        query_words = set(query.lower().split())

        # Check last 3 user messages
        user_messages = [m for m in history if m.get("role") == "user"][-3:]

        for msg in user_messages:
            content = msg.get("content", "").lower()
            content_words = set(content.split())

            # If >70% word overlap, consider it repeated
            overlap = len(query_words & content_words) / max(len(query_words), 1)
            if overlap > 0.7:
                return True

        return False

    def generate_repair_response(
        self,
        strategy: RepairStrategy,
        query: str,
        context: Any,
        reason: str
    ) -> str:
        """
        Generate appropriate repair response.

        Args:
            strategy: Repair strategy to use
            query: User's query
            context: Conversation context
            reason: Reason for repair

        Returns:
            Repair response message
        """
        if strategy == RepairStrategy.ACKNOWLEDGE_AND_RESTART:
            return self._acknowledge_and_restart(context)

        elif strategy == RepairStrategy.SIMPLIFY_QUESTION:
            return self._simplify_question(context)

        elif strategy == RepairStrategy.PROVIDE_EXAMPLES:
            return self._provide_examples(context)

        elif strategy == RepairStrategy.OFFER_ALTERNATIVES:
            return self._offer_alternatives(context)

        elif strategy == RepairStrategy.ESCALATE_TO_HUMAN:
            return self._escalate_message()

        return "I understand you're having difficulty. How can I help you better?"

    def _acknowledge_and_restart(self, context: Any) -> str:
        """Acknowledge and offer fresh start."""
        topic = context.primary_topic if hasattr(context, 'primary_topic') else "this"

        return f"""No problem! Let's start fresh.

🔄 Would you like to:
• Ask a completely different question?
• Rephrase your question about {topic}?
• Talk to a human for help?

What would help you most?"""

    def _simplify_question(self, context: Any) -> str:
        """Simplify the question."""
        return """Let me make this simpler.

Instead of asking multiple questions, let's go one step at a time.

**First, what's the main thing you want to know?**
(For example: "Maternity leave information" or "Bonus structure")

Just give me the topic, and I'll handle the rest."""

    def _provide_examples(self, context: Any) -> str:
        """Provide examples to help user."""
        return """I understand you're not sure. Let me give you some examples:

**If you're asking about leave policies:**
• "Maternity leave for employees in Lebanon"
• "Vacation days for managers"

**If you're asking about compensation:**
• "Bonus structure for sales"
• "Commission rates"

**If you're asking about benefits:**
• "Health insurance coverage"
• "Relocation allowance"

Which category is closest to what you need?"""

    def _offer_alternatives(self, context: Any) -> str:
        """Offer alternative paths."""
        return """I'm not finding what you need. Let's try a different approach:

📍 **Option 1:** Describe your situation
   (e.g., "I'm a manager in Lebanon expecting a baby")

📍 **Option 2:** Choose a category from:
   • Leave & Time Off
   • Compensation & Bonuses
   • Benefits & Insurance
   • Other HR Policies

📍 **Option 3:** Talk to a human HR representative

What works best for you?"""

    def _escalate_message(self) -> str:
        """Message for escalation."""
        return """I want to make sure you get the right information.

🤝 **Let me connect you with a human HR representative** who can help you directly.

In the meantime, is there anything else I can help clarify?"""


class ReasoningExplainer:
    """
    Explains system's reasoning and decision-making process.
    Provides transparency into how answers are generated.
    """

    def __init__(self, llm_client=None, deployment_name: str = None):
        """Initialize reasoning explainer."""
        self.llm_client = llm_client
        self.deployment_name = deployment_name

    def create_reasoning_path(
        self,
        query: str,
        context: Any,
        confidence_score: Any,
        sources: List[Dict[str, Any]],
        entities_extracted: Dict[str, str]
    ) -> ReasoningPath:
        """
        Create reasoning path for transparency.

        Args:
            query: User query
            context: Conversation context
            confidence_score: Confidence score object
            sources: Source documents used
            entities_extracted: Entities extracted

        Returns:
            ReasoningPath object
        """
        steps = []
        sources_used = []
        confidence_factors = {}
        assumptions = []

        # Step 1: Query understanding
        steps.append(f"1. **Understood your question**: {query}")

        # Step 2: Entity extraction
        if entities_extracted:
            entity_str = ", ".join([f"{k}={v}" for k, v in entities_extracted.items()])
            steps.append(f"2. **Identified key details**: {entity_str}")
        else:
            steps.append("2. **No specific details provided yet**")
            assumptions.append("Using general information (not country/position specific)")

        # Step 3: Topic detection
        if hasattr(context, 'primary_topic'):
            steps.append(f"3. **Determined topic**: {context.primary_topic}")

        # Step 4: Source retrieval
        if sources:
            source_names = [s.get('title', 'Document') for s in sources[:3]]
            steps.append(f"4. **Retrieved relevant documents**: {', '.join(source_names)}")
            sources_used = source_names
        else:
            steps.append("4. **No specific sources found**")
            assumptions.append("Answering based on general knowledge")

        # Step 5: Confidence assessment
        if confidence_score and hasattr(confidence_score, 'factors'):
            conf_str = f"{confidence_score.overall:.0%}"
            steps.append(f"5. **Assessed confidence**: {conf_str} confident")

            # Add confidence factors
            for factor, score in confidence_score.factors.items():
                confidence_factors[factor] = score

        # Step 6: Answer generation
        steps.append("6. **Generated answer** based on above information")

        return ReasoningPath(
            steps=steps,
            sources_used=sources_used,
            confidence_factors=confidence_factors,
            assumptions_made=assumptions
        )

    def explain_why_question_asked(
        self,
        question: str,
        context: Any
    ) -> str:
        """
        Explain why a clarifying question was asked.

        Args:
            question: The clarifying question
            context: Conversation context

        Returns:
            Explanation text
        """
        # Identify what was asked
        question_lower = question.lower()

        if "country" in question_lower or "location" in question_lower:
            return """**Why I asked about country:**
HR policies vary significantly by country due to local labor laws. Maternity leave, vacation days, and other benefits are different in Lebanon, UAE, Egypt, etc.

To give you accurate information specific to your situation, I need to know which country's policies apply to you."""

        elif "position" in question_lower or "role" in question_lower:
            return """**Why I asked about your position:**
Some policies have different terms based on job level. For example:
• Managers might have different leave entitlements
• Executive positions may have additional benefits
• Entry-level vs senior positions may have different bonus structures

Knowing your position helps me provide the most relevant information."""

        elif "policy" in question_lower or "type" in question_lower:
            return """**Why I asked about policy type:**
Your question could relate to several different policies:
• Leave policies (maternity, paternity, vacation, sick leave)
• Compensation policies (bonus, commission, raises)
• Insurance policies (health, dental, life)

Specifying which policy helps me give you the exact information you need rather than overwhelming you with everything."""

        else:
            return f"""**Why I asked this question:**
To provide accurate, relevant information, I need to understand your specific situation. This detail helps me:
• Find the right policy documents
• Give you applicable information (not general)
• Save your time by focusing on what matters to you"""

    def explain_source_selection(
        self,
        sources: List[Dict[str, Any]],
        query: str
    ) -> str:
        """Explain why specific sources were selected."""
        if not sources:
            return "No specific sources were found matching your query."

        explanations = ["**How I selected these sources:**\n"]

        for i, source in enumerate(sources[:3], 1):
            title = source.get('title', 'Document')
            score = source.get('score', 0)

            explanations.append(f"{i}. **{title}** (Relevance: {score:.0%})")

            # Explain relevance
            if score > 0.85:
                explanations.append("   - Very high match to your question")
            elif score > 0.7:
                explanations.append("   - Good match with relevant information")
            else:
                explanations.append("   - Moderate match, may have partial information")

        return "\n".join(explanations)


class WhyHowQuestionHandler:
    """
    Handles meta-questions about the system itself.
    Users asking "why", "how", "what source", etc.
    """

    def __init__(self, reasoning_explainer: ReasoningExplainer):
        """Initialize with reasoning explainer."""
        self.reasoning_explainer = reasoning_explainer

        # Meta-question patterns
        self.meta_patterns = {
            'why_asked': [
                r"why did you ask",
                r"why do you need",
                r"why are you asking",
                r"what's the reason"
            ],
            'how_calculated': [
                r"how (?:did you|do you) (?:calculate|compute|determine|arrive at)",
                r"how is (?:this|that) calculated",
                r"where did you get"
            ],
            'what_source': [
                r"what(?:'s| is) (?:the|your) source",
                r"where did (?:this|that|you) come from",
                r"which document",
                r"what page"
            ],
            'how_confident': [
                r"how (?:sure|confident|certain) are you",
                r"are you sure",
                r"is (?:this|that) correct"
            ]
        }

    def is_meta_question(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if query is a meta-question.

        Args:
            query: User query

        Returns:
            Tuple of (is_meta, meta_type)
        """
        query_lower = query.lower()

        for meta_type, patterns in self.meta_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return True, meta_type

        return False, None

    def handle_meta_question(
        self,
        meta_type: str,
        query: str,
        context: Any,
        last_question: Optional[str],
        last_answer: Optional[str],
        reasoning_path: Optional[ReasoningPath],
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Handle meta-question appropriately.

        Args:
            meta_type: Type of meta-question
            query: User's query
            context: Conversation context
            last_question: Last clarifying question asked
            last_answer: Last answer provided
            reasoning_path: Reasoning path object
            sources: Sources used

        Returns:
            Explanation text
        """
        if meta_type == 'why_asked':
            if last_question:
                return self.reasoning_explainer.explain_why_question_asked(last_question, context)
            else:
                return "I asked that question to better understand your specific situation and provide accurate information."

        elif meta_type == 'how_calculated':
            if reasoning_path:
                steps_text = "\n".join(reasoning_path.steps)
                return f"""**Here's how I arrived at this answer:**

{steps_text}

{self._format_confidence_factors(reasoning_path.confidence_factors)}"""
            else:
                return "I determined this answer by searching our HR policy documents and matching them to your specific situation."

        elif meta_type == 'what_source':
            return self.reasoning_explainer.explain_source_selection(sources, query)

        elif meta_type == 'how_confident':
            if reasoning_path and reasoning_path.confidence_factors:
                conf_text = self._format_confidence_factors(reasoning_path.confidence_factors)
                return f"""**About my confidence in this answer:**

{conf_text}

{self._get_verification_recommendation(reasoning_path.confidence_factors)}"""
            else:
                return "I'm reasonably confident based on the sources I found, but I recommend verifying with your HR representative for your specific situation."

        return "I'm not sure what you're asking about. Could you rephrase?"

    def _format_confidence_factors(self, factors: Dict[str, float]) -> str:
        """Format confidence factors for display."""
        if not factors:
            return ""

        lines = ["**Confidence Breakdown:**"]
        for factor, score in factors.items():
            emoji = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🟠"
            factor_name = factor.replace('_', ' ').title()
            lines.append(f"{emoji} {factor_name}: {score:.0%}")

        return "\n".join(lines)

    def _get_verification_recommendation(self, factors: Dict[str, float]) -> str:
        """Get verification recommendation based on confidence."""
        if not factors:
            return ""

        overall = sum(factors.values()) / len(factors)

        if overall > 0.85:
            return "✅ **High confidence** - This information should be accurate."
        elif overall > 0.7:
            return "⚠️ **Good confidence** - I recommend confirming details for your specific case."
        else:
            return "⚠️ **Moderate confidence** - Please verify with your HR representative to be certain."


class ComparisonIntelligence:
    """
    Specialized handling for comparison queries.
    Formats comparisons side-by-side with structured output.
    """

    def __init__(self, llm_client, deployment_name: str = None):
        """Initialize comparison intelligence."""
        self.llm_client = llm_client
        self.deployment_name = deployment_name

    def is_comparison_query(self, query: str) -> bool:
        """Detect if query is asking for comparison."""
        query_lower = query.lower()

        comparison_indicators = [
            "compare", "versus", "vs", "difference between",
            "better", "which is", "what's the difference",
            "or", "rather than", "instead of"
        ]

        return any(indicator in query_lower for indicator in comparison_indicators)

    def extract_comparison_items(self, query: str, context: Any) -> List[str]:
        """Extract what's being compared."""
        # Simple extraction - can be enhanced with LLM
        query_clean = query.lower()

        # Remove comparison words
        for word in ["compare", "versus", "vs", "difference between", "or", "and"]:
            query_clean = query_clean.replace(word, "|")

        # Split and clean
        items = [item.strip() for item in query_clean.split("|") if item.strip()]

        return items[:5]  # Limit to 5 items

    def format_comparison(
        self,
        items: List[str],
        answers: Dict[str, str],
        sources: Dict[str, List[Dict]]
    ) -> ComparisonResult:
        """
        Format comparison result with structure.

        Args:
            items: Items being compared
            answers: Dictionary of item -> answer text
            sources: Dictionary of item -> sources

        Returns:
            ComparisonResult object
        """
        # Build comparison table
        comparison_table = {}

        for item in items:
            comparison_table[item] = {
                "answer": answers.get(item, "Information not found"),
                "sources": sources.get(item, [])
            }

        # Generate summary (can be enhanced with LLM)
        summary = self._generate_comparison_summary(comparison_table)

        return ComparisonResult(
            items_compared=items,
            dimensions=list(comparison_table.keys()),
            comparison_table=comparison_table,
            summary=summary
        )

    def _generate_comparison_summary(self, table: Dict[str, Dict]) -> str:
        """Generate comparison summary."""
        items = list(table.keys())

        if len(items) == 2:
            return f"Here's a comparison of {items[0]} vs {items[1]}:"
        else:
            return f"Here's a comparison of {', '.join(items)}:"


class EnhancedUncertaintyExpression:
    """
    Enhanced uncertainty communication beyond simple confidence scores.
    Expresses uncertainty naturally with recommendations.
    """

    @staticmethod
    def express_uncertainty(
        confidence_score: float,
        context: Any,
        sources: List[Dict[str, Any]],
        answer: str
    ) -> UncertaintyInfo:
        """
        Create enhanced uncertainty information.

        Args:
            confidence_score: Confidence score (0-1)
            context: Conversation context
            sources: Sources used
            answer: Generated answer

        Returns:
            UncertaintyInfo object
        """
        # Determine confidence label
        if confidence_score > 0.9:
            label = "Very High"
        elif confidence_score > 0.8:
            label = "High"
        elif confidence_score > 0.7:
            label = "Good"
        elif confidence_score > 0.6:
            label = "Moderate"
        elif confidence_score > 0.5:
            label = "Low"
        else:
            label = "Very Low"

        # Identify reasons for uncertainty
        reasons = []
        verification_needed = False
        limitations = []

        if confidence_score < 0.7:
            if not sources:
                reasons.append("No specific source documents found")
            elif len(sources) < 2:
                reasons.append("Only one source available for verification")

            if hasattr(context, 'entities'):
                entities = context.get_all_entities() if hasattr(context, 'get_all_entities') else {}
                if not entities.get('country'):
                    reasons.append("Country not specified - using general information")
                    limitations.append("Specific country policies may differ")

                if not entities.get('position'):
                    reasons.append("Position not specified")
                    limitations.append("Benefits may vary by position level")

            verification_needed = True

        # Generate recommendation
        recommendation = EnhancedUncertaintyExpression._generate_recommendation(
            confidence_score,
            reasons,
            verification_needed
        )

        return UncertaintyInfo(
            confidence_score=confidence_score,
            confidence_label=label,
            reasons_for_uncertainty=reasons,
            verification_needed=verification_needed,
            scope_limitations=limitations,
            recommendation_text=recommendation
        )

    @staticmethod
    def _generate_recommendation(
        confidence_score: float,
        reasons: List[str],
        verification_needed: bool
    ) -> str:
        """Generate recommendation text."""
        if confidence_score > 0.85:
            return "✅ This information should be accurate based on our policy documents."

        elif confidence_score > 0.7:
            return "⚠️ This information is generally accurate, but I recommend confirming specific details for your situation."

        elif confidence_score > 0.5:
            if verification_needed:
                return "⚠️ **Please verify this information** with your HR representative. I have moderate confidence in this answer."
            else:
                return "⚠️ I have moderate confidence in this answer. Consider confirming with HR."

        else:
            return "⚠️ **Important**: I'm not very confident in this answer. Please consult your HR representative directly for accurate information."

    @staticmethod
    def format_uncertainty_in_answer(
        answer: str,
        uncertainty_info: UncertaintyInfo
    ) -> str:
        """
        Add uncertainty information to answer naturally.

        Args:
            answer: Original answer
            uncertainty_info: Uncertainty information

        Returns:
            Answer with uncertainty expressed
        """
        if uncertainty_info.confidence_score > 0.85:
            # High confidence - no special formatting needed
            return answer

        # Add uncertainty prefix
        prefix = ""
        if uncertainty_info.confidence_score < 0.7:
            prefix = "Based on available information, "
        elif uncertainty_info.confidence_score < 0.8:
            prefix = "According to our policy documents, "

        # Add limitations
        limitations_text = ""
        if uncertainty_info.scope_limitations:
            limitations_text = "\n\n**Note:** " + " ".join(uncertainty_info.scope_limitations)

        # Add recommendation
        recommendation_text = f"\n\n{uncertainty_info.recommendation_text}"

        return f"{prefix}{answer}{limitations_text}{recommendation_text}"


class SessionContinuity:
    """
    Maintains continuity across user sessions.
    Remembers what was discussed and offers to continue.
    """

    def __init__(self, conversation_manager):
        """Initialize session continuity."""
        self.conv_manager = conversation_manager

    def get_last_session_summary(
        self,
        user_id: str,
        time_threshold_hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Get summary of last session if within time threshold.

        Args:
            user_id: User identifier
            time_threshold_hours: Hours to consider "recent"

        Returns:
            Session summary or None
        """
        history = self.conv_manager.get_history(user_id, limit=20)

        if not history:
            return None

        # Check if last interaction was recent
        last_msg = history[-1]
        timestamp_str = last_msg.get('timestamp')

        if timestamp_str:
            try:
                last_time = datetime.fromisoformat(timestamp_str)
                time_diff = datetime.now() - last_time

                if time_diff > timedelta(hours=time_threshold_hours):
                    return None  # Too old
            except:
                pass

        # Extract last topics/entities
        topics = set()
        entities = {}

        for msg in history[-10:]:
            metadata = msg.get('metadata', {})

            # Extract topic if available
            if 'topic' in metadata:
                topics.add(metadata['topic'])

            # Extract entities if available
            if 'entities' in metadata:
                entities.update(metadata['entities'])

        if not topics and not entities:
            return None

        return {
            'topics': list(topics),
            'entities': entities,
            'message_count': len(history),
            'last_interaction': timestamp_str
        }

    def generate_welcome_back_message(
        self,
        user_id: str,
        session_summary: Dict[str, Any]
    ) -> str:
        """
        Generate welcome back message with context.

        Args:
            user_id: User identifier
            session_summary: Summary of last session

        Returns:
            Welcome message
        """
        topics = session_summary.get('topics', [])
        entities = session_summary.get('entities', {})

        topic_str = topics[0] if topics else "HR policies"

        entity_parts = []
        if entities:
            for k, v in list(entities.items())[:2]:
                entity_parts.append(f"{k}: {v}")

        entity_str = " (" + ", ".join(entity_parts) + ")" if entity_parts else ""

        message = f"""👋 **Welcome back!**

Last time we discussed **{topic_str}**{entity_str}.

Would you like to:
• Continue where we left off?
• Ask a related question?
• Start something completely new?

What can I help you with today?"""

        return message
