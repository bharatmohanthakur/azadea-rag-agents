"""
User Feedback System
Collects, analyzes, and learns from user feedback.
Closes the loop for continuous improvement.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger("FeedbackSystem")


class FeedbackType(Enum):
    """Types of feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    CONFUSED = "confused"
    PERFECT = "perfect"
    WRONG_ANSWER = "wrong_answer"
    NEEDS_MORE_DETAIL = "needs_more_detail"


@dataclass
class UserFeedback:
    """User feedback record."""
    feedback_id: str
    user_id: str
    conversation_id: str
    feedback_type: str  # FeedbackType value
    query: str
    answer: str
    rating: Optional[int] = None  # 1-5 stars
    comment: Optional[str] = None
    timestamp: str = None

    # Context
    topic: Optional[str] = None
    confidence_score: Optional[float] = None
    entities: Dict[str, str] = None

    # What went wrong (for negative feedback)
    issue_category: Optional[str] = None  # "accuracy", "relevance", "completeness", "clarity"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.entities is None:
            self.entities = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class FeedbackCollector:
    """Collects and stores user feedback."""

    def __init__(self, conversation_manager):
        """
        Initialize feedback collector.

        Args:
            conversation_manager: ConversationManager for storage
        """
        self.conv_manager = conversation_manager

    def collect_feedback(
        self,
        user_id: str,
        conversation_id: str,
        feedback_type: str,
        query: str,
        answer: str,
        context: Any,  # ConversationContext
        confidence_score: Optional[float] = None,
        rating: Optional[int] = None,
        comment: Optional[str] = None
    ) -> UserFeedback:
        """
        Collect user feedback.

        Args:
            user_id: User identifier
            conversation_id: Conversation ID
            feedback_type: Type of feedback
            query: User's query
            answer: System's answer
            context: Conversation context
            confidence_score: System confidence
            rating: Optional 1-5 star rating
            comment: Optional free-text comment

        Returns:
            UserFeedback record
        """
        feedback_id = f"{user_id}_{conversation_id}_{int(datetime.now().timestamp())}"

        # Extract context
        topic = context.primary_topic if hasattr(context, 'primary_topic') else None
        entities = context.get_all_entities() if hasattr(context, 'get_all_entities') else {}

        # Create feedback record
        feedback = UserFeedback(
            feedback_id=feedback_id,
            user_id=user_id,
            conversation_id=conversation_id,
            feedback_type=feedback_type,
            query=query,
            answer=answer[:500],  # Truncate long answers
            rating=rating,
            comment=comment,
            topic=topic,
            confidence_score=confidence_score,
            entities=entities
        )

        # Store feedback
        self._save_feedback(feedback)

        logger.info(f"Collected feedback: {feedback_type} from {user_id}")

        return feedback

    def _save_feedback(self, feedback: UserFeedback):
        """Save feedback to storage."""
        try:
            key = f"feedback:{feedback.feedback_id}"
            data = json.dumps(feedback.to_dict(), ensure_ascii=False)

            if self.conv_manager.redis_client:
                # Save to Redis with 90-day TTL
                self.conv_manager.redis_client.setex(
                    key,
                    timedelta(days=90),
                    data
                )
                # Also add to a sorted set for easy querying
                score = datetime.now().timestamp()
                self.conv_manager.redis_client.zadd(
                    f"feedback_timeline:{feedback.user_id}",
                    {feedback.feedback_id: score}
                )
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")


class FeedbackAnalyzer:
    """Analyzes feedback patterns and generates insights."""

    def __init__(self, conversation_manager):
        """
        Initialize feedback analyzer.

        Args:
            conversation_manager: ConversationManager instance
        """
        self.conv_manager = conversation_manager

    def analyze_feedback_trends(
        self,
        time_period: str = "last_7d"
    ) -> Dict[str, Any]:
        """
        Analyze feedback trends over time period.

        Args:
            time_period: Time period to analyze

        Returns:
            Dictionary of insights
        """
        # This would query Redis/DB in production
        # For now, return structure

        return {
            "positive_rate": 0.75,  # 75% positive
            "most_common_issues": [
                {"issue": "needs_more_detail", "count": 15},
                {"issue": "not_relevant", "count": 8}
            ],
            "low_confidence_feedback_correlation": 0.82,  # 82% of negative feedback had low confidence
            "topics_with_most_issues": [
                {"topic": "insurance policy", "negative_count": 12},
                {"topic": "commission structure", "negative_count": 8}
            ],
            "recommendations": [
                "Improve depth of insurance policy answers",
                "Add more examples for commission structure"
            ]
        }

    def identify_improvement_areas(
        self,
        feedback_records: List[UserFeedback]
    ) -> List[Dict[str, Any]]:
        """
        Identify specific areas for improvement from feedback.

        Args:
            feedback_records: List of feedback records

        Returns:
            List of improvement suggestions
        """
        improvements = []

        # Group by topic
        topic_feedback = {}
        for fb in feedback_records:
            if fb.topic:
                if fb.topic not in topic_feedback:
                    topic_feedback[fb.topic] = {'positive': 0, 'negative': 0}

                if fb.feedback_type in ['thumbs_up', 'helpful', 'perfect']:
                    topic_feedback[fb.topic]['positive'] += 1
                else:
                    topic_feedback[fb.topic]['negative'] += 1

        # Identify problematic topics
        for topic, counts in topic_feedback.items():
            total = counts['positive'] + counts['negative']
            if total >= 5:  # Enough data
                negative_rate = counts['negative'] / total
                if negative_rate > 0.4:  # >40% negative
                    improvements.append({
                        'area': 'topic',
                        'topic': topic,
                        'issue': f"High negative feedback rate: {negative_rate*100:.0f}%",
                        'suggestion': f"Review and improve answers for '{topic}'"
                    })

        return improvements


class FeedbackLoop:
    """Implements complete feedback loop for continuous learning."""

    def __init__(self, conversation_manager, llm_client=None, deployment_name: str = None):
        """
        Initialize feedback loop.

        Args:
            conversation_manager: ConversationManager instance
            llm_client: Optional LLM client for advanced analysis
            deployment_name: Azure deployment name
        """
        self.conv_manager = conversation_manager
        self.llm_client = llm_client
        self.deployment_name = deployment_name

        self.collector = FeedbackCollector(conversation_manager)
        self.analyzer = FeedbackAnalyzer(conversation_manager)

    def handle_feedback_response(
        self,
        feedback_type: str,
        user_id: str,
        conversation_id: str,
        query: str,
        answer: str,
        context: Any
    ) -> str:
        """
        Handle user feedback and provide appropriate response.

        Args:
            feedback_type: Type of feedback received
            user_id: User identifier
            conversation_id: Conversation ID
            query: Original query
            answer: Original answer
            context: Conversation context

        Returns:
            Response message to user
        """
        # Collect the feedback
        self.collector.collect_feedback(
            user_id, conversation_id, feedback_type,
            query, answer, context
        )

        # Generate appropriate response
        responses = {
            'thumbs_up': "👍 Thanks for the feedback! Glad I could help!",
            'thumbs_down': "👎 Sorry this wasn't helpful. Let me know if you'd like me to:\n• Provide more details\n• Try a different approach\n• Connect you with a human",
            'helpful': "😊 Wonderful! Feel free to ask more questions anytime.",
            'not_helpful': "😔 I apologize. Would you like to:\n• Rephrase your question?\n• Get more specific information?\n• Start over?",
            'confused': "🤔 I understand you're confused. Let me try to clarify - what part would you like me to explain better?",
            'perfect': "🎉 Excellent! That's what I'm here for!",
            'wrong_answer': "⚠️ Thanks for letting me know. Let me try again with more careful attention.",
            'needs_more_detail': "📝 I can provide more details! What specifically would you like to know more about?"
        }

        return responses.get(feedback_type, "Thanks for your feedback!")

    def should_ask_for_feedback(
        self,
        context: Any,
        confidence_score: Optional[float] = None
    ) -> bool:
        """
        Determine if we should proactively ask for feedback.

        Args:
            context: Conversation context
            confidence_score: System confidence score

        Returns:
            True if should ask for feedback
        """
        # Ask for feedback if:
        # 1. Confidence was low (want to know if we still helped)
        if confidence_score and confidence_score < 0.6:
            return True

        # 2. Complex conversation (multiple turns)
        if hasattr(context, 'turn_count') and context.turn_count >= 4:
            return True

        # 3. First-time user (in production, check user profile)
        # return True for new users

        return False

    def generate_feedback_prompt(
        self,
        confidence_score: Optional[float] = None,
        mood: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate appropriate feedback prompt for user.

        Args:
            confidence_score: System confidence
            mood: Detected user mood

        Returns:
            Feedback prompt configuration
        """
        # Low confidence - ask directly
        if confidence_score and confidence_score < 0.6:
            return {
                'message': "⚠️ I'm not entirely confident in this answer. Was this helpful to you?",
                'options': [
                    {'emoji': '✅', 'text': 'Yes, this helped', 'type': 'helpful'},
                    {'emoji': '❌', 'text': 'No, not quite right', 'type': 'not_helpful'},
                    {'emoji': '❓', 'text': 'Still confused', 'type': 'confused'}
                ]
            }

        # Regular feedback request
        return {
            'message': "Was this answer helpful?",
            'options': [
                {'emoji': '👍', 'text': 'Yes', 'type': 'thumbs_up'},
                {'emoji': '👎', 'text': 'No', 'type': 'thumbs_down'},
                {'emoji': '⭐', 'text': 'Perfect!', 'type': 'perfect'}
            ]
        }
