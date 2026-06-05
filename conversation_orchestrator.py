"""
Conversation Orchestrator - Best-in-Class Integration
Integrates all conversation enhancement modules into a cohesive system.
Provides a single interface for advanced conversation management.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import all our best-in-class modules
from conversation_context import (
    ContextManager,
    ConversationContext,
    ConversationState
)
from intent_analyzer import (
    SemanticIntentAnalyzer,
    MultiIntentDisambiguator,
    IntentAnalysis,
    IntentRelationship
)
from confidence_and_recovery import (
    ConfidenceScorer,
    ConfirmationGenerator,
    ContextRecovery,
    ConfidenceScore,
    ConfirmationRequest,
    RecoveryAction
)
from conversation_compression import (
    ConversationCompressor,
    ConversationMemoryRAG,
    CompressedConversation
)
from user_preferences import (
    UserPreferenceLearner,
    UserProfile
)
from conversation_analytics import (
    ConversationAnalytics,
    ConversationMetrics
)

logger = logging.getLogger("ConversationOrchestrator")


@dataclass
class OrchestrationResult:
    """Result from conversation orchestration."""
    # Query processing
    rewritten_query: str
    should_use_query: str  # The final query to use for retrieval

    # Context
    conversation_context: ConversationContext
    compressed_history: Optional[CompressedConversation] = None

    # Intent
    intent_analysis: Optional[IntentAnalysis] = None

    # Confidence
    confidence_score: Optional[ConfidenceScore] = None
    confirmation_request: Optional[ConfirmationRequest] = None
    recovery_action: Optional[RecoveryAction] = None

    # Recommendations
    user_recommendations: Dict[str, Any] = None

    # Flags
    should_confirm: bool = False
    should_recover: bool = False
    topic_switched: bool = False

    # Performance
    processing_time_ms: float = 0.0


class ConversationOrchestrator:
    """
    Master orchestrator for all conversation enhancements.
    Integrates context tracking, intent analysis, confidence scoring,
    compression, user preferences, and analytics.
    """

    def __init__(
        self,
        conversation_manager,
        llm_client,
        embedder_client,
        qdrant_client,
        deployment_name: str = None,
        enable_analytics: bool = True,
        enable_compression: bool = True,
        enable_memory_rag: bool = False  # Disabled by default (requires setup)
    ):
        """
        Initialize conversation orchestrator.

        Args:
            conversation_manager: ConversationManager instance
            llm_client: LLM client for AI operations
            embedder_client: OpenAI client for embeddings
            qdrant_client: Qdrant client for vector storage
            deployment_name: Azure deployment name
            enable_analytics: Enable conversation analytics
            enable_compression: Enable conversation compression
            enable_memory_rag: Enable conversation memory with RAG
        """
        self.conv_manager = conversation_manager
        self.llm_client = llm_client
        self.deployment_name = deployment_name

        # Initialize all components
        logger.info("Initializing Conversation Orchestrator...")

        # Core components (always enabled)
        self.context_manager = ContextManager(
            conversation_manager,
            llm_client,
            deployment_name
        )

        self.intent_analyzer = SemanticIntentAnalyzer(
            embedder_client,
            similarity_threshold=0.7
        )

        self.intent_disambiguator = MultiIntentDisambiguator(
            llm_client,
            deployment_name
        )

        self.confidence_scorer = ConfidenceScorer(
            llm_client,
            deployment_name
        )

        self.confirmation_generator = ConfirmationGenerator(
            llm_client,
            deployment_name
        )

        self.context_recovery = ContextRecovery(
            llm_client,
            deployment_name
        )

        self.user_preference_learner = UserPreferenceLearner(
            conversation_manager
        )

        # Optional components
        self.compressor = None
        if enable_compression:
            self.compressor = ConversationCompressor(
                llm_client,
                deployment_name
            )

        self.memory_rag = None
        if enable_memory_rag:
            try:
                self.memory_rag = ConversationMemoryRAG(
                    qdrant_client,
                    embedder_client
                )
                logger.info("✅ Conversation memory RAG enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize memory RAG: {e}")

        self.analytics = None
        if enable_analytics:
            self.analytics = ConversationAnalytics(conversation_manager)

        # Tracking
        self._conversation_start_times: Dict[str, float] = {}
        self._intent_history: Dict[str, List[IntentAnalysis]] = {}
        self._confidence_history: Dict[str, List[ConfidenceScore]] = {}

        logger.info("✅ Conversation Orchestrator initialized successfully")

    def process_query(
        self,
        user_id: str,
        query: str,
        conversation_history: List[Dict[str, Any]],
        retrieval_results: Optional[List[Dict]] = None
    ) -> OrchestrationResult:
        """
        Process a user query through all enhancement layers.

        Args:
            user_id: User identifier
            query: User's query
            conversation_history: Full conversation history
            retrieval_results: Optional retrieval results for confidence scoring

        Returns:
            OrchestrationResult with all analyses and recommendations
        """
        start_time = time.time()

        # Track conversation start
        if user_id not in self._conversation_start_times:
            self._conversation_start_times[user_id] = start_time

        # Step 1: Get or create conversation context
        context = self.context_manager.get_or_create_context(user_id, query)

        # Step 2: Analyze intent
        intent_analysis = self.intent_analyzer.analyze_intent(
            query,
            context,
            conversation_history
        )

        # Disambiguate if needed
        if intent_analysis.confidence < 0.85:
            intent_analysis = self.intent_disambiguator.disambiguate(
                query,
                context,
                intent_analysis
            )

        # Track intent history
        if user_id not in self._intent_history:
            self._intent_history[user_id] = []
        self._intent_history[user_id].append(intent_analysis)

        # Step 3: Handle topic switches
        if intent_analysis.topic_shift:
            logger.info(f"Topic switch detected for {user_id}: {context.primary_topic}")
            # Reset context for new topic
            context = self.context_manager._create_new_context(user_id, query)
            self.context_manager._update_context_from_query(context, query)

        # Step 4: Rewrite query with full context
        rewritten_query = self._rewrite_query_with_context(
            query,
            context,
            intent_analysis,
            conversation_history
        )

        # Step 5: Compress conversation if needed
        compressed = None
        if self.compressor and self.compressor.should_compress(conversation_history, context):
            compressed = self.compressor.compress_conversation(conversation_history, context)
            logger.info(f"Compressed conversation: {compressed.original_turn_count} → {compressed.compressed_turn_count} turns")

        # Step 6: Score confidence
        confidence_score = self.confidence_scorer.score_confidence(
            context,
            intent_analysis,
            rewritten_query,
            query,
            retrieval_results
        )

        # Track confidence history
        if user_id not in self._confidence_history:
            self._confidence_history[user_id] = []
        self._confidence_history[user_id].append(confidence_score)

        # Step 7: Check if confirmation needed
        confirmation = None
        if confidence_score.should_confirm:
            confirmation = self.confirmation_generator.generate_confirmation(
                context,
                rewritten_query,
                confidence_score
            )
            logger.info(f"Confirmation requested for {user_id}: confidence={confidence_score.overall:.2f}")

        # Step 8: Check if recovery needed
        recovery = None
        if self.context_recovery.should_trigger_recovery(confidence_score, context):
            recovery = self.context_recovery.generate_recovery_action(
                context,
                confidence_score,
                rewritten_query
            )
            logger.warning(f"Context recovery triggered for {user_id}: {recovery.action_type}")

        # Step 9: Get user recommendations
        user_recommendations = self.user_preference_learner.get_recommendations(user_id)

        # Step 10: Determine final query to use
        should_use_query = self._determine_final_query(
            query,
            rewritten_query,
            confidence_score,
            recovery
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms

        return OrchestrationResult(
            rewritten_query=rewritten_query,
            should_use_query=should_use_query,
            conversation_context=context,
            compressed_history=compressed,
            intent_analysis=intent_analysis,
            confidence_score=confidence_score,
            confirmation_request=confirmation,
            recovery_action=recovery,
            user_recommendations=user_recommendations,
            should_confirm=confirmation is not None,
            should_recover=recovery is not None,
            topic_switched=intent_analysis.topic_shift,
            processing_time_ms=processing_time
        )

    def _rewrite_query_with_context(
        self,
        query: str,
        context: ConversationContext,
        intent_analysis: IntentAnalysis,
        history: List[Dict[str, Any]]
    ) -> str:
        """
        Rewrite query using full context and intent understanding.

        Args:
            query: Original query
            context: Conversation context
            intent_analysis: Intent analysis
            history: Conversation history

        Returns:
            Rewritten query
        """
        # Build context string
        original_question = context.original_question
        entities = context.get_all_entities()
        topic = context.primary_topic

        # Build entity string
        entity_parts = [f"{k}: {v}" for k, v in entities.items()] if entities else []
        entity_str = ", ".join(entity_parts)

        # Build history string (recent turns only)
        recent_history = history[-6:] if len(history) > 6 else history
        history_str = ""
        for msg in recent_history:
            role = msg.get("role")
            content = msg.get("content", "")[:100]  # Truncate
            history_str += f"{role}: {content}\n"

        # Build prompt based on intent relationship
        if intent_analysis.relationship == IntentRelationship.CONTINUATION:
            prompt_instruction = "This is a continuation. Combine the current query with the original question and known details."
        elif intent_analysis.relationship == IntentRelationship.COMPARISON:
            prompt_instruction = "This is a comparison. Maintain reference to original while adding the comparison."
        elif intent_analysis.relationship == IntentRelationship.RELATED:
            prompt_instruction = "This is related to the original. Connect both contexts."
        else:
            # New topic
            return query  # Use as-is

        prompt = f"""Rewrite the current query into a standalone question that preserves all context.

**Original Question**: {original_question}
**Topic**: {topic}
**Known Details**: {entity_str if entity_str else "none yet"}

**Intent**: {prompt_instruction}

Recent Conversation:
{history_str}

**Current Query**: {query}

Rewrite into a complete, standalone question that:
1. Preserves the original question's intent
2. Incorporates all known details ({entity_str})
3. Handles the current query appropriately based on intent

Standalone Question:"""

        try:
            model_param = self.deployment_name if self.deployment_name else "gpt-4o"
            response = self.llm_client.chat.completions.create(
                model=model_param,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )

            rewritten = response.choices[0].message.content.strip()

            # Remove quotes if present
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]

            logger.info(f"Query rewrite: '{query}' → '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            # Fallback: combine manually
            if entities:
                return f"{original_question} ({entity_str}) - {query}"
            return query

    def _determine_final_query(
        self,
        original_query: str,
        rewritten_query: str,
        confidence: ConfidenceScore,
        recovery: Optional[RecoveryAction]
    ) -> str:
        """Determine which query to actually use for retrieval."""
        # If recovery was triggered, might use original question
        if recovery and recovery.suggested_query:
            return recovery.suggested_query

        # If confidence is very low, use rewritten (it has more context)
        if confidence.overall < 0.5:
            return rewritten_query

        # Otherwise use rewritten
        return rewritten_query

    def finalize_conversation(
        self,
        user_id: str,
        conversation_id: str,
        conversation_history: List[Dict[str, Any]],
        context: ConversationContext,
        final_feedback: Optional[str] = None
    ):
        """
        Finalize conversation - update preferences and record analytics.

        Args:
            user_id: User identifier
            conversation_id: Conversation ID
            conversation_history: Full conversation history
            context: Conversation context
            final_feedback: Optional user feedback
        """
        # Calculate duration
        start_time = self._conversation_start_times.get(user_id, time.time())
        duration = time.time() - start_time

        # Update user preferences
        self.user_preference_learner.update_from_conversation(
            user_id,
            conversation_history,
            context,
            final_feedback
        )

        # Record analytics
        if self.analytics:
            intent_history = self._intent_history.get(user_id, [])
            confidence_history = self._confidence_history.get(user_id, [])

            # Check if recovery was triggered
            recovery_triggered = any(
                score.should_recover
                for score in confidence_history
            )

            # Check if confirmation was requested
            confirmation_requested = any(
                score.should_confirm
                for score in confidence_history
            )

            # Get confidence scores
            confidence_scores = [score.overall for score in confidence_history]

            self.analytics.record_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                conversation_history=conversation_history,
                context=context,
                confidence_scores=confidence_scores,
                intent_analysis_history=intent_history,
                recovery_triggered=recovery_triggered,
                confirmation_requested=confirmation_requested,
                duration_seconds=duration,
                final_feedback=final_feedback
            )

        # Store in conversation memory RAG if enabled
        if self.memory_rag:
            for i, msg in enumerate(conversation_history):
                if i % 2 == 0 and i + 1 < len(conversation_history):  # User + Assistant pairs
                    user_msg = conversation_history[i]
                    assistant_msg = conversation_history[i + 1]

                    if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                        self.memory_rag.store_conversation_turn(
                            user_id=user_id,
                            turn_number=i // 2 + 1,
                            user_query=user_msg.get("content", ""),
                            assistant_response=assistant_msg.get("content", ""),
                            context=context,
                            session_id=context.session_id
                        )

        # Cleanup tracking
        if user_id in self._conversation_start_times:
            del self._conversation_start_times[user_id]
        if user_id in self._intent_history:
            del self._intent_history[user_id]
        if user_id in self._confidence_history:
            del self._confidence_history[user_id]

        logger.info(f"Finalized conversation {conversation_id} for {user_id}")

    def get_analytics_insights(self, time_period: str = "last_7d") -> Dict[str, Any]:
        """Get analytics insights."""
        if not self.analytics:
            return {"error": "Analytics not enabled"}

        return self.analytics.generate_insights(time_period)

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile and preferences."""
        profile = self.user_preference_learner.get_or_create_profile(user_id)
        return profile.to_dict()
