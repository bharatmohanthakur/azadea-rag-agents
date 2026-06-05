"""
Conversation Analytics and Metrics
Tracks conversation quality, context loss rate, and system performance.
Provides insights for continuous improvement.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger("ConversationAnalytics")


@dataclass
class ConversationMetrics:
    """Metrics for a single conversation."""
    conversation_id: str
    user_id: str
    timestamp: str

    # Turn metrics
    total_turns: int = 0
    clarification_turns: int = 0
    avg_query_length: float = 0.0
    avg_response_length: float = 0.0

    # Context metrics
    original_question: str = ""
    topic: str = ""
    entities_extracted: int = 0
    context_loss_detected: bool = False

    # Quality metrics
    confidence_scores: List[float] = field(default_factory=list)
    avg_confidence: float = 0.0
    recovery_triggered: bool = False
    confirmation_requested: bool = False

    # Intent metrics
    intent_switches: int = 0  # How many times intent changed
    semantic_drift: float = 0.0  # Final similarity to original

    # Outcome
    completed_successfully: bool = False
    user_satisfaction_signal: Optional[str] = None  # "positive", "negative", "neutral"

    # Performance
    total_duration_seconds: float = 0.0
    avg_response_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AggregateMetrics:
    """Aggregate metrics across conversations."""
    time_period: str  # e.g., "last_24h", "last_7d"
    total_conversations: int = 0
    total_turns: int = 0

    # Turn metrics
    avg_turns_per_conversation: float = 0.0
    max_turns: int = 0
    conversations_over_5_turns: int = 0

    # Context metrics
    context_loss_rate: float = 0.0  # % of conversations with context loss
    avg_entities_extracted: float = 0.0
    recovery_trigger_rate: float = 0.0

    # Quality metrics
    avg_confidence_score: float = 0.0
    low_confidence_rate: float = 0.0  # % with confidence < 0.6

    # Topic distribution
    top_topics: List[Dict[str, Any]] = field(default_factory=list)  # [{topic: count}]

    # User satisfaction
    satisfaction_rate: float = 0.0  # % positive signals
    frustration_rate: float = 0.0  # % negative signals

    # Performance
    avg_duration: float = 0.0
    avg_response_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ConversationAnalytics:
    """
    Tracks and analyzes conversation metrics.
    Provides insights for system improvement.
    """

    def __init__(self, conversation_manager):
        """
        Initialize conversation analytics.

        Args:
            conversation_manager: ConversationManager for storage
        """
        self.conv_manager = conversation_manager

        # In-memory metrics storage (can be persisted to DB later)
        self._metrics: List[ConversationMetrics] = []

    def record_conversation(
        self,
        conversation_id: str,
        user_id: str,
        conversation_history: List[Dict[str, Any]],
        context: Any,  # ConversationContext
        confidence_scores: List[float],
        intent_analysis_history: List[Any],
        recovery_triggered: bool,
        confirmation_requested: bool,
        duration_seconds: float,
        final_feedback: Optional[str] = None
    ) -> ConversationMetrics:
        """
        Record metrics for a completed conversation.

        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
            conversation_history: Full conversation history
            context: ConversationContext
            confidence_scores: List of confidence scores across turns
            intent_analysis_history: List of intent analyses
            recovery_triggered: Was context recovery triggered
            confirmation_requested: Was confirmation requested
            duration_seconds: Total conversation duration
            final_feedback: Optional user feedback

        Returns:
            ConversationMetrics
        """
        # Calculate turn metrics
        user_messages = [m for m in conversation_history if m.get("role") == "user"]
        assistant_messages = [m for m in conversation_history if m.get("role") == "assistant"]

        total_turns = len(user_messages)

        # Count clarification turns (assistant messages with "?")
        clarification_turns = sum(1 for m in assistant_messages if "?" in m.get("content", ""))

        # Calculate average lengths
        avg_query_len = statistics.mean([len(m.get("content", "").split()) for m in user_messages]) if user_messages else 0
        avg_response_len = statistics.mean([len(m.get("content", "").split()) for m in assistant_messages]) if assistant_messages else 0

        # Get context info
        original_question = context.original_question if hasattr(context, 'original_question') else ""
        topic = context.primary_topic if hasattr(context, 'primary_topic') else ""
        entities_count = len(context.get_all_entities()) if hasattr(context, 'get_all_entities') else 0

        # Calculate average confidence
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0

        # Detect context loss (if confidence dropped significantly)
        context_loss = self._detect_context_loss(confidence_scores)

        # Count intent switches
        intent_switches = self._count_intent_switches(intent_analysis_history)

        # Get final semantic drift
        semantic_drift = 0.0
        if intent_analysis_history and hasattr(intent_analysis_history[-1], 'similarity_score'):
            semantic_drift = intent_analysis_history[-1].similarity_score

        # Detect user satisfaction
        satisfaction = self._detect_satisfaction(final_feedback, assistant_messages)

        # Calculate response time
        avg_response_time = duration_seconds / max(total_turns, 1)

        # Create metrics
        metrics = ConversationMetrics(
            conversation_id=conversation_id,
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            total_turns=total_turns,
            clarification_turns=clarification_turns,
            avg_query_length=avg_query_len,
            avg_response_length=avg_response_len,
            original_question=original_question,
            topic=topic,
            entities_extracted=entities_count,
            context_loss_detected=context_loss,
            confidence_scores=confidence_scores,
            avg_confidence=avg_confidence,
            recovery_triggered=recovery_triggered,
            confirmation_requested=confirmation_requested,
            intent_switches=intent_switches,
            semantic_drift=semantic_drift,
            completed_successfully=not context_loss,
            user_satisfaction_signal=satisfaction,
            total_duration_seconds=duration_seconds,
            avg_response_time=avg_response_time
        )

        # Store metrics
        self._metrics.append(metrics)
        self._save_metrics(metrics)

        logger.info(f"Recorded metrics for conversation {conversation_id}: {total_turns} turns, confidence={avg_confidence:.2f}")

        return metrics

    def get_aggregate_metrics(self, time_period: str = "last_24h") -> AggregateMetrics:
        """
        Get aggregate metrics for a time period.

        Args:
            time_period: Time period ("last_24h", "last_7d", "last_30d")

        Returns:
            AggregateMetrics
        """
        # Load recent metrics
        recent_metrics = self._load_recent_metrics(time_period)

        if not recent_metrics:
            return AggregateMetrics(time_period=time_period)

        # Calculate aggregates
        total_conversations = len(recent_metrics)
        total_turns = sum(m.total_turns for m in recent_metrics)

        # Turn metrics
        avg_turns = total_turns / max(total_conversations, 1)
        max_turns_val = max(m.total_turns for m in recent_metrics) if recent_metrics else 0
        conversations_over_5 = sum(1 for m in recent_metrics if m.total_turns > 5)

        # Context metrics
        context_loss_count = sum(1 for m in recent_metrics if m.context_loss_detected)
        context_loss_rate = context_loss_count / max(total_conversations, 1)

        avg_entities = statistics.mean([m.entities_extracted for m in recent_metrics]) if recent_metrics else 0

        recovery_count = sum(1 for m in recent_metrics if m.recovery_triggered)
        recovery_rate = recovery_count / max(total_conversations, 1)

        # Quality metrics
        all_confidences = [score for m in recent_metrics for score in m.confidence_scores]
        avg_conf = statistics.mean(all_confidences) if all_confidences else 0

        low_conf_count = sum(1 for conf in all_confidences if conf < 0.6)
        low_conf_rate = low_conf_count / max(len(all_confidences), 1)

        # Topic distribution
        topic_counts = defaultdict(int)
        for m in recent_metrics:
            if m.topic:
                topic_counts[m.topic] += 1

        top_topics = [
            {"topic": topic, "count": count}
            for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        # Satisfaction
        satisfaction_count = sum(1 for m in recent_metrics if m.user_satisfaction_signal == "positive")
        satisfaction_rate = satisfaction_count / max(total_conversations, 1)

        frustration_count = sum(1 for m in recent_metrics if m.user_satisfaction_signal == "negative")
        frustration_rate = frustration_count / max(total_conversations, 1)

        # Performance
        avg_duration = statistics.mean([m.total_duration_seconds for m in recent_metrics]) if recent_metrics else 0
        avg_resp_time = statistics.mean([m.avg_response_time for m in recent_metrics]) if recent_metrics else 0

        return AggregateMetrics(
            time_period=time_period,
            total_conversations=total_conversations,
            total_turns=total_turns,
            avg_turns_per_conversation=avg_turns,
            max_turns=max_turns_val,
            conversations_over_5_turns=conversations_over_5,
            context_loss_rate=context_loss_rate,
            avg_entities_extracted=avg_entities,
            recovery_trigger_rate=recovery_rate,
            avg_confidence_score=avg_conf,
            low_confidence_rate=low_conf_rate,
            top_topics=top_topics,
            satisfaction_rate=satisfaction_rate,
            frustration_rate=frustration_rate,
            avg_duration=avg_duration,
            avg_response_time=avg_resp_time
        )

    def _detect_context_loss(self, confidence_scores: List[float]) -> bool:
        """Detect if context loss occurred based on confidence scores."""
        if len(confidence_scores) < 2:
            return False

        # Check if confidence dropped significantly
        for i in range(1, len(confidence_scores)):
            drop = confidence_scores[i-1] - confidence_scores[i]
            if drop > 0.3:  # 30% drop
                return True

        # Check if confidence stayed low
        if len(confidence_scores) >= 3:
            recent_avg = statistics.mean(confidence_scores[-3:])
            if recent_avg < 0.5:
                return True

        return False

    def _count_intent_switches(self, intent_history: List[Any]) -> int:
        """Count how many times intent relationship changed."""
        if not intent_history:
            return 0

        switches = 0
        for i in range(1, len(intent_history)):
            if hasattr(intent_history[i], 'topic_shift') and intent_history[i].topic_shift:
                switches += 1

        return switches

    def _detect_satisfaction(
        self,
        final_feedback: Optional[str],
        assistant_messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Detect user satisfaction from feedback or conversation patterns."""
        if final_feedback:
            feedback_lower = final_feedback.lower()

            positive = ["thanks", "thank you", "great", "perfect", "awesome", "excellent", "helpful", "good"]
            negative = ["wrong", "not what", "incorrect", "bad", "confused", "frustrated", "useless"]

            if any(word in feedback_lower for word in positive):
                return "positive"
            elif any(word in feedback_lower for word in negative):
                return "negative"

        # Check if conversation ended naturally (last message didn't have clarification)
        if assistant_messages:
            last_msg = assistant_messages[-1].get("content", "")
            if "?" not in last_msg:
                return "neutral"  # Likely got answer

        return None

    def _save_metrics(self, metrics: ConversationMetrics):
        """Save metrics to storage."""
        try:
            key = f"analytics:{metrics.conversation_id}"
            data = json.dumps(metrics.to_dict(), ensure_ascii=False)

            if self.conv_manager.redis_client:
                # Save to Redis with 90-day TTL
                self.conv_manager.redis_client.setex(
                    key,
                    timedelta(days=90),
                    data
                )
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def _load_recent_metrics(self, time_period: str) -> List[ConversationMetrics]:
        """Load recent metrics from storage."""
        # For simplicity, return in-memory metrics
        # In production, this would query Redis/DB with time filter

        now = datetime.now()

        if time_period == "last_24h":
            cutoff = now - timedelta(hours=24)
        elif time_period == "last_7d":
            cutoff = now - timedelta(days=7)
        elif time_period == "last_30d":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = now - timedelta(hours=24)

        # Filter metrics by timestamp
        recent = []
        for m in self._metrics:
            try:
                metric_time = datetime.fromisoformat(m.timestamp)
                if metric_time >= cutoff:
                    recent.append(m)
            except:
                continue

        return recent

    def generate_insights(self, time_period: str = "last_7d") -> Dict[str, Any]:
        """
        Generate actionable insights from metrics.

        Args:
            time_period: Time period to analyze

        Returns:
            Dictionary of insights and recommendations
        """
        metrics = self.get_aggregate_metrics(time_period)

        insights = {
            "summary": {
                "total_conversations": metrics.total_conversations,
                "avg_confidence": f"{metrics.avg_confidence_score:.2f}",
                "satisfaction_rate": f"{metrics.satisfaction_rate*100:.1f}%"
            },
            "concerns": [],
            "recommendations": [],
            "highlights": []
        }

        # Identify concerns
        if metrics.context_loss_rate > 0.15:  # >15%
            insights["concerns"].append(f"High context loss rate: {metrics.context_loss_rate*100:.1f}%")
            insights["recommendations"].append("Review query rewriting logic and entity extraction")

        if metrics.low_confidence_rate > 0.30:  # >30%
            insights["concerns"].append(f"Many low confidence answers: {metrics.low_confidence_rate*100:.1f}%")
            insights["recommendations"].append("Improve clarification triggering threshold")

        if metrics.frustration_rate > 0.10:  # >10%
            insights["concerns"].append(f"User frustration detected: {metrics.frustration_rate*100:.1f}%")
            insights["recommendations"].append("Reduce clarification turns, improve answer quality")

        # Identify highlights
        if metrics.satisfaction_rate > 0.70:  # >70%
            insights["highlights"].append(f"High user satisfaction: {metrics.satisfaction_rate*100:.1f}%")

        if metrics.avg_confidence_score > 0.80:  # >80%
            insights["highlights"].append(f"Strong confidence in answers: {metrics.avg_confidence_score:.2f}")

        if metrics.recovery_trigger_rate < 0.05:  # <5%
            insights["highlights"].append("Low need for context recovery - good context preservation")

        return insights
