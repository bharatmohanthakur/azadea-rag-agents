"""
Conversation State Machine - Natural conversation flow management.
Implements smooth transitions like ChatGPT/Claude/Gemini.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("ConversationStateMachine")


class ConversationState(Enum):
    """Conversation states."""
    GREETING = "greeting"  # Initial greeting exchange
    CONTEXT_GATHERING = "context_gathering"  # Building user profile
    ANSWERING = "answering"  # Providing answers (main state)
    CLARIFYING = "clarifying"  # ONE clarification question active
    TOPIC_CHANGING = "topic_changing"  # User changed topic
    WRAPPING_UP = "wrapping_up"  # Concluding conversation
    FRUSTRATED = "frustrated"  # User showing frustration


@dataclass
class ConversationSession:
    """Conversation session state."""
    user_id: str
    current_state: ConversationState = ConversationState.GREETING
    previous_state: Optional[ConversationState] = None
    turn_count: int = 0
    successful_answers: int = 0
    clarification_count: int = 0
    topic_changes: int = 0
    frustration_signals: List[str] = field(default_factory=list)
    current_topic: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def transition_to(self, new_state: ConversationState, reason: Optional[str] = None):
        """Transition to a new state."""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.last_updated = datetime.now().isoformat()

        logger.info(f"State transition: {self.previous_state.value} -> {new_state.value}" +
                   (f" (reason: {reason})" if reason else ""))


class ConversationStateMachine:
    """
    Manages conversation flow with state transitions.
    Ensures natural, user-friendly interactions.
    """

    def __init__(self):
        """Initialize conversation state machine."""
        self.sessions: Dict[str, ConversationSession] = {}

        # State transition rules
        self.transition_rules = {
            ConversationState.GREETING: [
                ConversationState.CONTEXT_GATHERING,
                ConversationState.ANSWERING
            ],
            ConversationState.CONTEXT_GATHERING: [
                ConversationState.ANSWERING,
                ConversationState.CLARIFYING
            ],
            ConversationState.ANSWERING: [
                ConversationState.ANSWERING,  # Can stay in answering
                ConversationState.CLARIFYING,
                ConversationState.TOPIC_CHANGING,
                ConversationState.WRAPPING_UP,
                ConversationState.FRUSTRATED
            ],
            ConversationState.CLARIFYING: [
                ConversationState.ANSWERING,
                ConversationState.TOPIC_CHANGING,  # Can change topic during clarification
                ConversationState.FRUSTRATED
            ],
            ConversationState.TOPIC_CHANGING: [
                ConversationState.ANSWERING,
                ConversationState.CLARIFYING
            ],
            ConversationState.FRUSTRATED: [
                ConversationState.ANSWERING,
                ConversationState.WRAPPING_UP
            ],
            ConversationState.WRAPPING_UP: [
                ConversationState.GREETING,  # Can restart
                ConversationState.ANSWERING
            ]
        }

    def get_session(self, user_id: str) -> ConversationSession:
        """Get or create conversation session."""
        if user_id not in self.sessions:
            self.sessions[user_id] = ConversationSession(user_id=user_id)
        return self.sessions[user_id]

    def can_transition(
        self,
        user_id: str,
        target_state: ConversationState
    ) -> bool:
        """Check if transition to target state is allowed."""
        session = self.get_session(user_id)
        allowed_states = self.transition_rules.get(session.current_state, [])
        return target_state in allowed_states

    def handle_greeting(self, user_id: str) -> Dict[str, Any]:
        """Handle greeting state."""
        session = self.get_session(user_id)

        # Transition to answering (ready to help)
        session.transition_to(ConversationState.ANSWERING, "greeting acknowledged")

        return {
            "state": session.current_state.value,
            "should_greet": True,
            "message": "How can I help you today?"
        }

    def handle_query(
        self,
        user_id: str,
        query: str,
        is_greeting: bool = False,
        confidence: Optional[str] = None,
        topic_change_detected: bool = False,
        frustration_detected: bool = False
    ) -> Dict[str, Any]:
        """
        Handle user query and determine appropriate state.

        Args:
            user_id: User ID
            query: User query
            is_greeting: Is this a greeting?
            confidence: Answer confidence (if answering)
            topic_change_detected: Was topic change detected?
            frustration_detected: Was frustration detected?

        Returns:
            Dictionary with state and action recommendations
        """
        session = self.get_session(user_id)
        session.turn_count += 1

        # Handle greeting
        if is_greeting:
            return self.handle_greeting(user_id)

        # Handle frustration
        if frustration_detected:
            session.frustration_signals.append(query[:50])
            if session.current_state == ConversationState.CLARIFYING:
                # Exit clarification immediately
                session.transition_to(ConversationState.ANSWERING, "frustration detected")
                session.clarification_count += 1
                return {
                    "state": session.current_state.value,
                    "action": "answer_immediately",
                    "abandon_clarification": True,
                    "message": "Understood, let me give you the information I have."
                }
            else:
                session.transition_to(ConversationState.FRUSTRATED, "frustration detected")
                return {
                    "state": session.current_state.value,
                    "action": "apologize_and_simplify",
                    "message": "I apologize for any confusion. Let me help you more directly."
                }

        # Handle topic change
        if topic_change_detected:
            session.topic_changes += 1
            if session.current_state == ConversationState.CLARIFYING:
                # Abandon clarification
                session.transition_to(ConversationState.TOPIC_CHANGING, "topic changed")
                return {
                    "state": session.current_state.value,
                    "action": "switch_topic",
                    "abandon_clarification": True,
                    "message": "Sure, let's talk about that instead."
                }
            else:
                session.transition_to(ConversationState.TOPIC_CHANGING, "topic changed")
                return {
                    "state": session.current_state.value,
                    "action": "acknowledge_and_answer",
                    "message": None  # Seamless transition
                }

        # Handle based on current state
        if session.current_state == ConversationState.CLARIFYING:
            # User is answering clarification
            session.transition_to(ConversationState.ANSWERING, "clarification answered")
            session.clarification_count += 1
            return {
                "state": session.current_state.value,
                "action": "answer_with_clarification",
                "message": None
            }

        elif session.current_state in [ConversationState.ANSWERING, ConversationState.TOPIC_CHANGING]:
            # Decide whether to answer directly or clarify
            # RULE: Maximum 1 clarification per conversation flow

            if confidence == "very_low" and session.clarification_count == 0:
                # Only clarify if we haven't clarified yet
                session.transition_to(ConversationState.CLARIFYING, "need critical info")
                return {
                    "state": session.current_state.value,
                    "action": "ask_one_question",
                    "max_questions": 1,  # ONLY ONE QUESTION!
                    "message": None
                }
            else:
                # Answer with best guess
                session.transition_to(ConversationState.ANSWERING, "providing answer")
                session.successful_answers += 1
                return {
                    "state": session.current_state.value,
                    "action": "answer_with_best_guess",
                    "message": None
                }

        # Default: answer
        session.transition_to(ConversationState.ANSWERING, "default")
        return {
            "state": session.current_state.value,
            "action": "answer",
            "message": None
        }

    def should_ask_clarification(self, user_id: str) -> bool:
        """
        Determine if clarification is allowed.
        RULE: Maximum 1 clarification per session.

        Args:
            user_id: User ID

        Returns:
            True if clarification is allowed
        """
        session = self.get_session(user_id)
        return session.clarification_count == 0

    def get_conversation_quality(self, user_id: str) -> Dict[str, Any]:
        """
        Get conversation quality metrics.

        Args:
            user_id: User ID

        Returns:
            Quality metrics
        """
        session = self.get_session(user_id)

        # Calculate metrics
        avg_turns_per_answer = (session.turn_count / session.successful_answers
                               if session.successful_answers > 0 else 0)

        quality_score = 1.0
        # Penalize for high turns per answer
        if avg_turns_per_answer > 2:
            quality_score -= 0.2
        # Penalize for multiple clarifications
        if session.clarification_count > 1:
            quality_score -= 0.3
        # Penalize for frustration
        if len(session.frustration_signals) > 0:
            quality_score -= 0.2 * len(session.frustration_signals)
        # Penalize for excessive topic changes
        if session.topic_changes > 3:
            quality_score -= 0.1

        quality_score = max(0.0, min(1.0, quality_score))

        return {
            "quality_score": quality_score,
            "turn_count": session.turn_count,
            "successful_answers": session.successful_answers,
            "clarification_count": session.clarification_count,
            "topic_changes": session.topic_changes,
            "frustration_count": len(session.frustration_signals),
            "avg_turns_per_answer": round(avg_turns_per_answer, 2),
            "current_state": session.current_state.value
        }

    def transition_to_answering(self, user_id: str):
        """
        Convenience method to transition to ANSWERING state.
        Called when ready to provide an answer.
        """
        session = self.get_session(user_id)
        session.transition_to(ConversationState.ANSWERING, "ready to answer")

    def get_state(self, user_id: str) -> str:
        """
        Get current conversation state for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Current state as string
        """
        session = self.get_session(user_id)
        return session.current_state.value

    def has_clarified(self, user_id: str) -> bool:
        """
        Check if user has already been asked a clarification question.
        RULE: Only one clarification per conversation.
        
        Args:
            user_id: User ID
            
        Returns:
            True if clarification has already been done
        """
        session = self.get_session(user_id)
        return session.clarification_count > 0

    def transition_to_clarifying(self, user_id: str):
        """
        Transition to CLARIFYING state.
        Called when asking a clarification question.
        """
        session = self.get_session(user_id)
        session.transition_to(ConversationState.CLARIFYING, "asking clarification")

    def mark_clarification_done(self, user_id: str):
        """
        Mark that clarification has been completed.
        Increments the clarification counter to prevent future clarifications.
        """
        session = self.get_session(user_id)
        session.clarification_count += 1
        logger.info(f"Clarification done for {user_id}, count: {session.clarification_count}")

    def reset_session(self, user_id: str):
        """Reset conversation session."""
        if user_id in self.sessions:
            del self.sessions[user_id]
        logger.info(f"Reset conversation session for {user_id}")


# Global instance
_conversation_state_machine: Optional[ConversationStateMachine] = None


def get_conversation_state_machine() -> ConversationStateMachine:
    """Get or create global conversation state machine."""
    global _conversation_state_machine
    if _conversation_state_machine is None:
        _conversation_state_machine = ConversationStateMachine()
        logger.info("Initialized conversation state machine")
    return _conversation_state_machine
