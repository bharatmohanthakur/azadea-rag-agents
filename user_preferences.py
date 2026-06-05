"""
User Preference Learning System
Learns user communication patterns and preferences over time.
Provides personalized conversation experience.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger("UserPreferences")


class ClarificationStyle(Enum):
    """User's preferred clarification style."""
    STEP_BY_STEP = "step_by_step"  # Prefers one question at a time
    ALL_AT_ONCE = "all_at_once"  # Prefers all questions together
    MINIMAL = "minimal"  # Prefers minimal clarification
    COMPREHENSIVE = "comprehensive"  # Wants detailed clarifications


class DetailLevel(Enum):
    """User's preferred level of detail in answers."""
    BRIEF = "brief"  # Short, concise answers
    STANDARD = "standard"  # Normal detail level
    DETAILED = "detailed"  # Comprehensive, detailed answers
    TECHNICAL = "technical"  # Technical depth with specifics


class CommunicationStyle(Enum):
    """User's communication style."""
    DIRECT = "direct"  # Gets straight to the point
    CONVERSATIONAL = "conversational"  # Likes friendly chat
    FORMAL = "formal"  # Prefers formal communication
    CASUAL = "casual"  # Very casual, uses pronouns frequently


@dataclass
class UserProfile:
    """User's learned preferences and patterns."""
    user_id: str

    # Communication preferences
    clarification_style: str = ClarificationStyle.STEP_BY_STEP.value
    detail_level: str = DetailLevel.STANDARD.value
    communication_style: str = CommunicationStyle.CONVERSATIONAL.value

    # Behavioral patterns
    avg_query_length: float = 0.0  # Average words per query
    uses_pronouns_frequently: bool = False  # Uses "it", "they", etc.
    asks_follow_ups: bool = False  # Tends to ask follow-up questions
    provides_context_upfront: bool = False  # Gives all details in first query

    # Interaction history
    total_conversations: int = 0
    total_turns: int = 0
    total_clarifications_needed: int = 0
    avg_turns_per_conversation: float = 0.0

    # Topic preferences
    common_topics: List[str] = field(default_factory=list)
    common_entities: Dict[str, List[str]] = field(default_factory=dict)  # Frequently asked about

    # Timing patterns
    typical_response_time: float = 0.0  # Avg seconds between messages
    session_count: int = 0
    last_interaction: Optional[str] = None

    # Quality metrics
    satisfaction_signals: int = 0  # Positive feedback ("thanks", "great", etc.)
    frustration_signals: int = 0  # Negative signals

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create from dictionary."""
        return cls(**data)


class UserPreferenceLearner:
    """
    Learns user preferences from interaction patterns.
    Adapts conversation flow based on learned preferences.
    """

    def __init__(self, conversation_manager):
        """
        Initialize user preference learner.

        Args:
            conversation_manager: ConversationManager for storage
        """
        self.conv_manager = conversation_manager

        # Cache of user profiles
        self._profiles: Dict[str, UserProfile] = {}

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """
        Get existing profile or create new one.

        Args:
            user_id: User identifier

        Returns:
            UserProfile
        """
        # Check cache
        if user_id in self._profiles:
            return self._profiles[user_id]

        # Load from storage
        profile = self._load_profile(user_id)

        if not profile:
            profile = UserProfile(user_id=user_id)
            logger.info(f"Created new profile for {user_id}")

        # Cache it
        self._profiles[user_id] = profile
        return profile

    def update_from_conversation(
        self,
        user_id: str,
        conversation_history: List[Dict[str, Any]],
        context: Any,  # ConversationContext
        final_feedback: Optional[str] = None
    ):
        """
        Update user profile based on conversation.

        Args:
            user_id: User identifier
            conversation_history: Full conversation history
            context: ConversationContext
            final_feedback: Optional final user feedback
        """
        profile = self.get_or_create_profile(user_id)

        # Update basic stats
        profile.total_conversations += 1
        profile.session_count += 1
        profile.last_interaction = datetime.now().isoformat()

        # Count turns (user messages)
        user_messages = [m for m in conversation_history if m.get("role") == "user"]
        turn_count = len(user_messages)
        profile.total_turns += turn_count

        # Update average turns per conversation
        profile.avg_turns_per_conversation = profile.total_turns / max(profile.total_conversations, 1)

        # Learn communication style
        self._learn_communication_style(profile, user_messages)

        # Learn clarification preferences
        self._learn_clarification_style(profile, conversation_history)

        # Learn topic preferences
        if hasattr(context, 'primary_topic') and context.primary_topic:
            if context.primary_topic not in profile.common_topics:
                profile.common_topics.append(context.primary_topic)
                # Keep only top 10
                profile.common_topics = profile.common_topics[-10:]

        # Learn entity preferences
        if hasattr(context, 'get_all_entities'):
            entities = context.get_all_entities()
            for entity_type, entity_value in entities.items():
                if entity_type not in profile.common_entities:
                    profile.common_entities[entity_type] = []
                if entity_value not in profile.common_entities[entity_type]:
                    profile.common_entities[entity_type].append(entity_value)
                    # Keep only top 5 per type
                    profile.common_entities[entity_type] = profile.common_entities[entity_type][-5:]

        # Detect satisfaction/frustration
        if final_feedback:
            self._detect_satisfaction(profile, final_feedback)

        profile.updated_at = datetime.now().isoformat()

        # Save profile
        self._save_profile(profile)

        logger.info(f"Updated profile for {user_id}: {turn_count} turns, style={profile.communication_style}")

    def _learn_communication_style(
        self,
        profile: UserProfile,
        user_messages: List[Dict[str, Any]]
    ):
        """Learn user's communication style from messages."""
        if not user_messages:
            return

        # Calculate average query length
        total_words = sum(len(m.get("content", "").split()) for m in user_messages)
        profile.avg_query_length = total_words / len(user_messages)

        # Detect pronoun usage
        all_text = " ".join(m.get("content", "").lower() for m in user_messages)
        pronouns = ["it", "they", "them", "that", "this", "those", "these"]
        pronoun_count = sum(all_text.count(f" {p} ") for p in pronouns)

        # If pronouns appear frequently relative to total words
        profile.uses_pronouns_frequently = pronoun_count / max(total_words, 1) > 0.05

        # Detect if user provides context upfront (first message is long and detailed)
        if user_messages:
            first_msg_length = len(user_messages[0].get("content", "").split())
            profile.provides_context_upfront = first_msg_length > 15

        # Detect if user asks follow-ups
        profile.asks_follow_ups = len(user_messages) > 2

        # Determine communication style
        if profile.avg_query_length < 5:
            profile.communication_style = CommunicationStyle.DIRECT.value
        elif profile.avg_query_length > 20:
            profile.communication_style = CommunicationStyle.DETAILED.value
        elif profile.uses_pronouns_frequently:
            profile.communication_style = CommunicationStyle.CASUAL.value
        else:
            profile.communication_style = CommunicationStyle.CONVERSATIONAL.value

    def _learn_clarification_style(
        self,
        profile: UserProfile,
        conversation_history: List[Dict[str, Any]]
    ):
        """Learn user's preferred clarification style."""
        # Count clarifications (assistant messages with questions)
        clarifications = [
            m for m in conversation_history
            if m.get("role") == "assistant" and "?" in m.get("content", "")
        ]

        profile.total_clarifications_needed += len(clarifications)

        # If user needs many clarifications, prefer step-by-step
        if len(clarifications) > 2:
            profile.clarification_style = ClarificationStyle.STEP_BY_STEP.value
        elif len(clarifications) == 0 and profile.provides_context_upfront:
            # User provides everything upfront, prefer minimal
            profile.clarification_style = ClarificationStyle.MINIMAL.value
        else:
            # Default to standard
            profile.clarification_style = ClarificationStyle.STEP_BY_STEP.value

    def _detect_satisfaction(self, profile: UserProfile, feedback: str):
        """Detect satisfaction or frustration from feedback."""
        feedback_lower = feedback.lower()

        positive_words = ["thanks", "thank you", "great", "perfect", "awesome", "excellent", "helpful"]
        negative_words = ["wrong", "not what", "incorrect", "bad", "confused", "frustrated"]

        if any(word in feedback_lower for word in positive_words):
            profile.satisfaction_signals += 1

        if any(word in feedback_lower for word in negative_words):
            profile.frustration_signals += 1

    def get_recommendations(self, user_id: str) -> Dict[str, Any]:
        """
        Get recommendations for handling this user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary of recommendations
        """
        profile = self.get_or_create_profile(user_id)

        recommendations = {
            "clarification_approach": self._recommend_clarification_approach(profile),
            "detail_level": self._recommend_detail_level(profile),
            "communication_tone": self._recommend_tone(profile),
            "should_use_entities": len(profile.common_entities) > 0,
            "common_entities": profile.common_entities,
            "anticipate_follow_ups": profile.asks_follow_ups
        }

        return recommendations

    def _recommend_clarification_approach(self, profile: UserProfile) -> str:
        """Recommend how to approach clarifications."""
        if profile.clarification_style == ClarificationStyle.MINIMAL.value:
            return "Minimize clarifications - user provides context upfront"
        elif profile.clarification_style == ClarificationStyle.STEP_BY_STEP.value:
            return "Ask one clarifying question at a time"
        elif profile.clarification_style == ClarificationStyle.ALL_AT_ONCE.value:
            return "Ask all clarifying questions together"
        else:
            return "Standard clarification approach"

    def _recommend_detail_level(self, profile: UserProfile) -> str:
        """Recommend answer detail level."""
        if profile.avg_query_length < 5:
            return "brief"  # Short questions = want brief answers
        elif profile.avg_query_length > 20:
            return "detailed"  # Long questions = want detailed answers
        else:
            return "standard"

    def _recommend_tone(self, profile: UserProfile) -> str:
        """Recommend communication tone."""
        style = profile.communication_style

        if style == CommunicationStyle.DIRECT.value:
            return "direct_and_concise"
        elif style == CommunicationStyle.FORMAL.value:
            return "formal_and_professional"
        elif style == CommunicationStyle.CASUAL.value:
            return "casual_and_friendly"
        else:
            return "conversational_and_helpful"

    def _save_profile(self, profile: UserProfile):
        """Save profile to storage."""
        try:
            key = f"user_profile:{profile.user_id}"
            data = json.dumps(profile.to_dict(), ensure_ascii=False)

            if self.conv_manager.redis_client:
                # Save to Redis with 90-day TTL
                self.conv_manager.redis_client.setex(
                    key,
                    timedelta(days=90),
                    data
                )
            else:
                # Fallback to memory
                if not hasattr(self.conv_manager, '_user_profiles'):
                    self.conv_manager._user_profiles = {}
                self.conv_manager._user_profiles[profile.user_id] = data
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")

    def _load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load profile from storage."""
        try:
            key = f"user_profile:{user_id}"

            if self.conv_manager.redis_client:
                data = self.conv_manager.redis_client.get(key)
                if data:
                    return UserProfile.from_dict(json.loads(data))
            else:
                # Fallback to memory
                if hasattr(self.conv_manager, '_user_profiles'):
                    data = self.conv_manager._user_profiles.get(user_id)
                    if data:
                        return UserProfile.from_dict(json.loads(data))

            return None
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
            return None
