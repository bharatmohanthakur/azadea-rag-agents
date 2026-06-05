"""
Confidence-Based Confirmation and Proactive Context Recovery
Prevents bad answers by confirming understanding and recovering from context loss.
Best-in-class implementation with multi-factor confidence scoring.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from conversation_context import ConversationContext
from intent_analyzer import IntentAnalysis

logger = logging.getLogger("ConfidenceAndRecovery")


class ConfidenceLevel(Enum):
    """Confidence levels for understanding."""
    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"  # 0.8-0.9
    MEDIUM = "medium"  # 0.6-0.8
    LOW = "low"  # 0.4-0.6
    VERY_LOW = "very_low"  # < 0.4


@dataclass
class ConfidenceScore:
    """Multi-factor confidence scoring."""
    overall: float  # 0-1 overall confidence
    level: ConfidenceLevel
    factors: Dict[str, float]  # Individual confidence factors
    should_confirm: bool  # Whether to ask for confirmation
    should_recover: bool  # Whether context recovery is needed
    reasoning: str


@dataclass
class ConfirmationRequest:
    """Request for user confirmation."""
    message: str  # Confirmation message to show user
    entities_summary: Dict[str, str]  # Entities extracted
    query_interpretation: str  # How we understood the query
    confidence_score: float


@dataclass
class RecoveryAction:
    """Context recovery action."""
    action_type: str  # "confirm", "ask_clarification", "restart"
    message: str
    suggested_query: Optional[str] = None
    missing_entities: List[str] = None


class ConfidenceScorer:
    """
    Multi-factor confidence scoring system.
    Evaluates confidence based on multiple signals.
    """

    def __init__(self, llm_client=None, deployment_name: str = None):
        """
        Initialize confidence scorer.

        Args:
            llm_client: Optional LLM client for advanced scoring
            deployment_name: Azure deployment name
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name

        # Confidence thresholds
        self.CONFIRM_THRESHOLD = 0.7  # Below this, ask for confirmation
        self.RECOVERY_THRESHOLD = 0.5  # Below this, proactive recovery

    def score_confidence(
        self,
        context: ConversationContext,
        intent_analysis: IntentAnalysis,
        rewritten_query: str,
        original_query: str,
        retrieval_results: Optional[List[Dict]] = None
    ) -> ConfidenceScore:
        """
        Calculate multi-factor confidence score.

        Args:
            context: Conversation context
            intent_analysis: Intent analysis result
            rewritten_query: Rewritten standalone query
            original_query: Original user query
            retrieval_results: Optional retrieval results

        Returns:
            ConfidenceScore with multi-factor analysis
        """
        factors = {}

        # Factor 1: Intent confidence
        factors['intent_confidence'] = intent_analysis.confidence

        # Factor 2: Context completeness (do we have necessary entities?)
        factors['context_completeness'] = self._score_context_completeness(context)

        # Factor 3: Query transformation quality
        factors['transformation_quality'] = self._score_transformation(
            original_query,
            rewritten_query,
            context
        )

        # Factor 4: Semantic drift (how far have we drifted from original?)
        factors['semantic_alignment'] = intent_analysis.similarity_score

        # Factor 5: Turn count penalty (more turns = higher risk of drift)
        factors['turn_freshness'] = self._score_turn_freshness(context.turn_count)

        # Factor 6: Retrieval quality (if available)
        if retrieval_results:
            factors['retrieval_quality'] = self._score_retrieval_quality(retrieval_results)
        else:
            factors['retrieval_quality'] = 0.8  # Neutral if not available

        # Calculate weighted overall confidence
        weights = {
            'intent_confidence': 0.25,
            'context_completeness': 0.20,
            'transformation_quality': 0.15,
            'semantic_alignment': 0.20,
            'turn_freshness': 0.10,
            'retrieval_quality': 0.10
        }

        overall = sum(factors[k] * weights[k] for k in factors.keys())

        # Determine level
        level = self._determine_confidence_level(overall)

        # Determine if we should confirm or recover
        should_confirm = overall < self.CONFIRM_THRESHOLD
        should_recover = overall < self.RECOVERY_THRESHOLD

        # Generate reasoning
        reasoning = self._generate_confidence_reasoning(factors, overall)

        return ConfidenceScore(
            overall=overall,
            level=level,
            factors=factors,
            should_confirm=should_confirm,
            should_recover=should_recover,
            reasoning=reasoning
        )

    def _score_context_completeness(self, context: ConversationContext) -> float:
        """
        Score how complete the context is.
        Checks if we have the typical entities needed for HR queries.
        """
        # Define typical entities for HR queries
        typical_entities = ['country', 'position', 'policy_type']

        # Check how many we have
        present = sum(1 for e in typical_entities if context.has_entity(e))

        # If we have topic but no entities yet, it's still okay (early in conversation)
        if context.turn_count <= 1:
            return 0.9  # High confidence for first turn

        # Otherwise score based on presence
        if present == 0:
            return 0.4  # Low - we're missing key context
        elif present == 1:
            return 0.6  # Medium - partial context
        elif present == 2:
            return 0.8  # High - good context
        else:
            return 1.0  # Very high - complete context

    def _score_transformation(
        self,
        original: str,
        rewritten: str,
        context: ConversationContext
    ) -> float:
        """
        Score the quality of query transformation.
        Good transformation should preserve intent and add context.
        """
        # If queries are identical, transformation didn't add value
        if original.lower() == rewritten.lower():
            return 0.7  # Neutral

        # Check if rewritten contains original keywords
        original_words = set(original.lower().split())
        rewritten_words = set(rewritten.lower().split())

        # Calculate keyword preservation
        preserved = len(original_words & rewritten_words) / max(len(original_words), 1)

        # Check if entities were added
        entities = context.get_all_entities()
        entities_added = sum(
            1 for entity_val in entities.values()
            if entity_val.lower() in rewritten.lower() and entity_val.lower() not in original.lower()
        )

        # Good transformation preserves original + adds entities
        if preserved > 0.7 and entities_added > 0:
            return 0.95  # Excellent transformation
        elif preserved > 0.5:
            return 0.8  # Good preservation
        elif preserved > 0.3:
            return 0.6  # Moderate preservation
        else:
            return 0.4  # Poor preservation - might have lost intent

    def _score_turn_freshness(self, turn_count: int) -> float:
        """
        Score based on turn count.
        More turns = higher risk of context drift.
        """
        if turn_count <= 1:
            return 1.0
        elif turn_count <= 3:
            return 0.9
        elif turn_count <= 5:
            return 0.7
        elif turn_count <= 8:
            return 0.5
        else:
            return 0.3  # Very long conversation, high drift risk

    def _score_retrieval_quality(self, results: List[Dict]) -> float:
        """Score retrieval quality based on results."""
        if not results:
            return 0.3  # No results = low confidence

        # Check if we have good scores
        if hasattr(results[0], 'score'):
            top_score = results[0].score
            if top_score > 0.8:
                return 0.95
            elif top_score > 0.6:
                return 0.8
            else:
                return 0.6

        # If no scores, check result count
        if len(results) >= 3:
            return 0.8  # Good number of results
        elif len(results) >= 1:
            return 0.6  # Some results
        else:
            return 0.4  # Few results

    def _determine_confidence_level(self, score: float) -> ConfidenceLevel:
        """Map score to confidence level."""
        if score > 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score > 0.8:
            return ConfidenceLevel.HIGH
        elif score > 0.6:
            return ConfidenceLevel.MEDIUM
        elif score > 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _generate_confidence_reasoning(self, factors: Dict[str, float], overall: float) -> str:
        """Generate human-readable reasoning."""
        # Find the weakest factor
        weakest = min(factors.items(), key=lambda x: x[1])
        strongest = max(factors.items(), key=lambda x: x[1])

        parts = [f"Overall: {overall:.2f}"]

        if overall > 0.8:
            parts.append(f"High confidence due to {strongest[0]}")
        elif overall < 0.6:
            parts.append(f"Low confidence due to {weakest[0]} ({weakest[1]:.2f})")
        else:
            parts.append("Moderate confidence")

        return "; ".join(parts)


class ConfirmationGenerator:
    """
    Generates natural confirmation requests for users.
    Asks users to confirm understanding before providing potentially wrong answers.
    """

    def __init__(self, llm_client=None, deployment_name: str = None):
        """
        Initialize confirmation generator.

        Args:
            llm_client: Optional LLM client for natural language generation
            deployment_name: Azure deployment name
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name

    def generate_confirmation(
        self,
        context: ConversationContext,
        rewritten_query: str,
        confidence_score: ConfidenceScore
    ) -> ConfirmationRequest:
        """
        Generate a confirmation request for the user.

        Args:
            context: Conversation context
            rewritten_query: Rewritten query
            confidence_score: Confidence score

        Returns:
            ConfirmationRequest with message
        """
        entities = context.get_all_entities()

        # Generate natural confirmation message
        if self.llm_client:
            message = self._generate_with_llm(context, rewritten_query, entities)
        else:
            message = self._generate_template(context, rewritten_query, entities)

        return ConfirmationRequest(
            message=message,
            entities_summary=entities,
            query_interpretation=rewritten_query,
            confidence_score=confidence_score.overall
        )

    def _generate_template(
        self,
        context: ConversationContext,
        rewritten_query: str,
        entities: Dict[str, str]
    ) -> str:
        """Generate confirmation using templates."""
        # Build entity string
        if entities:
            entity_parts = [f"{k.replace('_', ' ')}: {v}" for k, v in entities.items()]
            entity_str = ", ".join(entity_parts)
            return f"Just to confirm, you're asking about **{context.primary_topic}** with these specifics: {entity_str}. Is that correct?"
        else:
            return f"Just to confirm, you're asking about **{context.primary_topic}**. Is that correct?"

    def _generate_with_llm(
        self,
        context: ConversationContext,
        rewritten_query: str,
        entities: Dict[str, str]
    ) -> str:
        """Generate natural confirmation using LLM."""
        try:
            entities_str = json.dumps(entities)

            prompt = f"""Generate a natural, friendly confirmation message to verify understanding.

Original Question: {context.original_question}
Current Understanding: {rewritten_query}
Extracted Details: {entities_str}
Topic: {context.primary_topic}

Generate a brief, natural confirmation like:
- "Just to confirm, you're asking about [topic] for [entities]?"
- "Let me make sure I understand - you want to know about [topic] specifically for [entities]?"

Keep it conversational and under 30 words. Use bold (**text**) for key terms.

Confirmation:"""

            model_param = self.deployment_name if self.deployment_name else "gpt-4o"
            response = self.llm_client.chat.completions.create(
                model=model_param,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )

            message = response.choices[0].message.content.strip()
            return message

        except Exception as e:
            logger.error(f"LLM confirmation generation failed: {e}")
            return self._generate_template(context, rewritten_query, entities)


class ContextRecovery:
    """
    Proactive context recovery when confidence is very low.
    Detects and recovers from context loss before providing bad answers.
    """

    def __init__(self, llm_client=None, deployment_name: str = None):
        """
        Initialize context recovery.

        Args:
            llm_client: LLM client
            deployment_name: Azure deployment name
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name

    def should_trigger_recovery(
        self,
        confidence_score: ConfidenceScore,
        context: ConversationContext
    ) -> bool:
        """Determine if context recovery should be triggered."""
        # Trigger if:
        # 1. Confidence is very low
        if confidence_score.should_recover:
            return True

        # 2. Missing critical entities after multiple turns
        if context.turn_count >= 3 and not context.has_entity('country') and not context.has_entity('position'):
            return True

        # 3. Semantic drift is high
        if confidence_score.factors.get('semantic_alignment', 1.0) < 0.3:
            return True

        return False

    def generate_recovery_action(
        self,
        context: ConversationContext,
        confidence_score: ConfidenceScore,
        rewritten_query: str
    ) -> RecoveryAction:
        """
        Generate a recovery action to get conversation back on track.

        Args:
            context: Conversation context
            confidence_score: Confidence score
            rewritten_query: Rewritten query

        Returns:
            RecoveryAction with recovery strategy
        """
        # Analyze what went wrong
        weakest_factor = min(confidence_score.factors.items(), key=lambda x: x[1])

        # Identify missing entities
        typical_entities = ['country', 'position', 'policy_type']
        missing = [e for e in typical_entities if not context.has_entity(e)]

        # Generate recovery based on issue
        if weakest_factor[0] == 'context_completeness' and missing:
            # Missing entities - ask for them
            return self._generate_missing_entity_recovery(context, missing)

        elif weakest_factor[0] == 'semantic_alignment':
            # Semantic drift - reconfirm original question
            return self._generate_drift_recovery(context, rewritten_query)

        elif weakest_factor[0] == 'turn_freshness':
            # Too many turns - offer restart
            return self._generate_restart_recovery(context)

        else:
            # Generic recovery - confirm understanding
            return self._generate_generic_recovery(context, rewritten_query)

    def _generate_missing_entity_recovery(
        self,
        context: ConversationContext,
        missing: List[str]
    ) -> RecoveryAction:
        """Generate recovery for missing entities."""
        entity_names = {
            'country': 'which country',
            'position': 'what position/role',
            'policy_type': 'which policy or benefit'
        }

        # Ask for first missing entity
        first_missing = missing[0]
        entity_question = entity_names.get(first_missing, first_missing.replace('_', ' '))

        message = f"To give you the most accurate answer about **{context.primary_topic}**, could you specify {entity_question}?"

        return RecoveryAction(
            action_type="ask_clarification",
            message=message,
            missing_entities=missing
        )

    def _generate_drift_recovery(
        self,
        context: ConversationContext,
        rewritten_query: str
    ) -> RecoveryAction:
        """Generate recovery for semantic drift."""
        message = f"""I want to make sure I'm on the right track. Your original question was about **{context.primary_topic}**.

Are you still asking about that, or have we moved to a different topic?"""

        return RecoveryAction(
            action_type="confirm",
            message=message,
            suggested_query=context.original_question
        )

    def _generate_restart_recovery(self, context: ConversationContext) -> RecoveryAction:
        """Generate recovery for long conversations."""
        entities = context.get_all_entities()
        entities_str = ", ".join(f"{k}: {v}" for k, v in entities.items()) if entities else "no specific details yet"

        message = f"""We've been chatting for a while. Let me summarize what I understand:

**Original Question**: {context.original_question}
**Topic**: {context.primary_topic}
**Details**: {entities_str}

Is this still what you're looking for, or would you like to start fresh with a new question?"""

        return RecoveryAction(
            action_type="restart",
            message=message,
            suggested_query=context.original_question
        )

    def _generate_generic_recovery(
        self,
        context: ConversationContext,
        rewritten_query: str
    ) -> RecoveryAction:
        """Generic recovery confirmation."""
        entities = context.get_all_entities()

        if entities:
            entity_parts = [f"**{k}**: {v}" for k, v in entities.items()]
            entity_str = ", ".join(entity_parts)
            message = f"Let me confirm I understand correctly:\n\n**Question**: {context.primary_topic}\n**Details**: {entity_str}\n\nIs this accurate?"
        else:
            message = f"Let me make sure I understand - you're asking about **{context.primary_topic}**. Is that right?"

        return RecoveryAction(
            action_type="confirm",
            message=message,
            suggested_query=rewritten_query
        )
