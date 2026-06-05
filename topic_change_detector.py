"""
Topic Change Detector - Detects when user changes conversation topic.
Enables smooth transitions like ChatGPT/Claude/Gemini.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger("TopicChangeDetector")


class TopicChangeType(Enum):
    """Types of topic changes."""
    NO_CHANGE = "no_change"  # Continuing same topic
    SLIGHT_SHIFT = "slight_shift"  # Related topic shift
    MAJOR_CHANGE = "major_change"  # Completely different topic
    RETURN_TO_PREVIOUS = "return_to_previous"  # Returning to earlier topic


class TopicChangeDetector:
    """
    Detects topic changes in conversation using semantic similarity.
    Helps system smoothly transition between topics.
    """

    def __init__(self, embedding_function):
        """
        Initialize topic change detector.

        Args:
            embedding_function: Function to embed text (returns List[float])
        """
        self.embedding_function = embedding_function

        # Thresholds for topic change detection
        self.major_change_threshold = 0.4  # Similarity < 0.4 = major change
        self.slight_shift_threshold = 0.7  # Similarity < 0.7 = slight shift

        # Topic keywords for quick detection
        self.hr_topics = {
            "leave": ["leave", "vacation", "time off", "annual leave", "sick leave", "maternity"],
            "insurance": ["insurance", "medical", "health", "coverage", "benefits"],
            "salary": ["salary", "pay", "compensation", "bonus", "allowance"],
            "policy": ["policy", "procedure", "rule", "guideline", "regulation"],
            "onboarding": ["onboarding", "joining", "new hire", "orientation", "induction"],
            "performance": ["performance", "appraisal", "review", "evaluation", "rating"],
            "training": ["training", "development", "learning", "course", "workshop"],
            "benefits": ["benefits", "perks", "entitlements", "privileges"],
            "termination": ["termination", "resignation", "notice period", "exit", "offboarding"],
            "attendance": ["attendance", "working hours", "shift", "schedule", "timesheet"],
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _detect_topic_keywords(self, text: str) -> List[str]:
        """Detect topics using keyword matching (fast path)."""
        text_lower = text.lower()
        detected = []

        for topic, keywords in self.hr_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                detected.append(topic)

        return detected

    def detect_topic_change(
        self,
        current_query: str,
        recent_context: List[str],
        topic_history: Optional[List[str]] = None
    ) -> Tuple[TopicChangeType, float, Optional[str]]:
        """
        Detect if current query represents a topic change.

        Args:
            current_query: Current user query
            recent_context: Recent conversation messages (last 3-5)
            topic_history: Optional list of topics discussed

        Returns:
            Tuple of (change_type, similarity_score, detected_topic)
        """
        if not recent_context or len(recent_context) == 0:
            return TopicChangeType.NO_CHANGE, 1.0, None

        # Fast path: Detect topics using keywords
        current_topics = self._detect_topic_keywords(current_query)

        if topic_history and current_topics:
            # Check if returning to a previous topic
            for topic in current_topics:
                if topic in topic_history[:-1]:  # Exclude most recent
                    logger.info(f"Detected return to previous topic: {topic}")
                    return TopicChangeType.RETURN_TO_PREVIOUS, 0.8, topic

        # Semantic similarity check using embeddings
        try:
            # Embed current query
            current_embedding = self.embedding_function(current_query)
            current_vec = np.array(current_embedding)

            # Compute similarity with recent context
            similarities = []
            for context_msg in recent_context[-3:]:  # Last 3 messages
                if not context_msg.strip():
                    continue

                context_embedding = self.embedding_function(context_msg)
                context_vec = np.array(context_embedding)

                similarity = self._cosine_similarity(current_vec, context_vec)
                similarities.append(similarity)

            if not similarities:
                return TopicChangeType.NO_CHANGE, 1.0, None

            # Use average similarity
            avg_similarity = np.mean(similarities)

            # Determine change type
            detected_topic = current_topics[0] if current_topics else None

            if avg_similarity < self.major_change_threshold:
                logger.info(f"Major topic change detected (similarity: {avg_similarity:.3f})")
                return TopicChangeType.MAJOR_CHANGE, avg_similarity, detected_topic
            elif avg_similarity < self.slight_shift_threshold:
                logger.info(f"Slight topic shift detected (similarity: {avg_similarity:.3f})")
                return TopicChangeType.SLIGHT_SHIFT, avg_similarity, detected_topic
            else:
                return TopicChangeType.NO_CHANGE, avg_similarity, detected_topic

        except Exception as e:
            logger.error(f"Error in topic change detection: {e}")
            # Fallback to keyword-based detection
            if len(current_topics) > 0:
                recent_topics = []
                for msg in recent_context[-2:]:
                    recent_topics.extend(self._detect_topic_keywords(msg))

                if not any(topic in recent_topics for topic in current_topics):
                    return TopicChangeType.MAJOR_CHANGE, 0.3, current_topics[0]

            return TopicChangeType.NO_CHANGE, 1.0, None

    def should_abandon_clarification(
        self,
        change_type: TopicChangeType,
        similarity: float
    ) -> bool:
        """
        Determine if clarification session should be abandoned due to topic change.

        Args:
            change_type: Type of topic change
            similarity: Similarity score

        Returns:
            True if clarification should be abandoned
        """
        # Abandon on major change
        if change_type == TopicChangeType.MAJOR_CHANGE:
            logger.info("Abandoning clarification due to major topic change")
            return True

        # Abandon on significant shift
        if change_type == TopicChangeType.SLIGHT_SHIFT and similarity < 0.5:
            logger.info("Abandoning clarification due to significant topic shift")
            return True

        return False

    def generate_transition_message(
        self,
        change_type: TopicChangeType,
        new_topic: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate smooth transition message for topic changes.

        Args:
            change_type: Type of topic change
            new_topic: Optional detected new topic

        Returns:
            Transition message or None
        """
        if change_type == TopicChangeType.MAJOR_CHANGE:
            if new_topic:
                return f"Sure, let's talk about {new_topic} instead."
            else:
                return "Sure, let me help you with that."

        elif change_type == TopicChangeType.RETURN_TO_PREVIOUS:
            if new_topic:
                return f"Going back to {new_topic}."
            else:
                return "Returning to your earlier question."

        elif change_type == TopicChangeType.SLIGHT_SHIFT:
            return None  # No explicit transition needed

        return None

    def detect_transition(
        self,
        previous_query: str = "",
        current_query: str = "",
        conversation_history: Optional[List[Dict]] = None
    ) -> 'TopicTransitionResult':
        """
        Detect topic transition between queries.
        Uses LLM classifier with conversation history for natural, context-aware detection with CoT reasoning.
        
        Args:
            previous_query: Previous user query
            current_query: Current user query
            conversation_history: Conversation history (optional)
            
        Returns:
            TopicTransitionResult with changed and acknowledgment fields
        """
        # Try to use LLM classifier for intelligent topic change detection
        from llm_classifier import get_llm_classifier
        llm_classifier = get_llm_classifier()
        
        if llm_classifier:
            try:
                # Build recent queries list for LLM
                recent_queries = []
                if previous_query:
                    recent_queries.append(previous_query)
                if conversation_history:
                    for msg in conversation_history[-5:]:
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            if content and content not in recent_queries:
                                recent_queries.append(content)
                
                # Detect current topic from previous query
                current_topic = None
                if previous_query:
                    # Extract topic from previous query (simple keyword-based for now)
                    prev_topics = self._detect_topic_keywords(previous_query)
                    current_topic = prev_topics[0] if prev_topics else None
                
                # Use LLM classifier with conversation history
                result = llm_classifier.detect_topic_change(
                    current_query=current_query,
                    recent_queries=recent_queries,
                    current_topic=current_topic
                )
                
                # Map LLM result to TopicChangeType
                if result.is_major_change:
                    change_type = TopicChangeType.MAJOR_CHANGE
                elif result.is_minor_shift:
                    change_type = TopicChangeType.SLIGHT_SHIFT
                else:
                    change_type = TopicChangeType.NO_CHANGE
                
                changed = result.is_major_change or result.is_minor_shift
                detected_topic = result.new_topic
                acknowledgment = result.acknowledgment if result.should_acknowledge else None
                
                logger.info(f"🔄 LLM Topic Change: {change_type.value} "
                           f"(similarity: {result.similarity:.2f}, reasoning: {result.reasoning[:100]})")
                
                return TopicTransitionResult(
                    changed=changed,
                    change_type=change_type.value,
                    similarity=result.similarity,
                    new_topic=detected_topic,
                    acknowledgment=acknowledgment
                )
                
            except Exception as e:
                logger.warning(f"LLM classifier failed for topic change detection, using fallback: {e}")
        
        # Fallback to original method if LLM classifier not available or fails
        # Build recent context from previous query
        recent_context = [previous_query] if previous_query else []
        if conversation_history:
            for msg in conversation_history[-5:]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if content and content not in recent_context:
                        recent_context.append(content)
        
        # Detect topic change using original method
        change_type, similarity, detected_topic = self.detect_topic_change(
            current_query, recent_context, None
        )
        
        # Build result
        changed = change_type in [TopicChangeType.MAJOR_CHANGE, TopicChangeType.SLIGHT_SHIFT]
        acknowledgment = self.generate_transition_message(change_type, detected_topic)
        
        return TopicTransitionResult(
            changed=changed,
            change_type=change_type.value,
            similarity=similarity,
            new_topic=detected_topic,
            acknowledgment=acknowledgment
        )


class TopicTransitionResult:
    """Result of topic transition detection."""
    def __init__(
        self,
        changed: bool = False,
        change_type: str = "no_change",
        similarity: float = 1.0,
        new_topic: Optional[str] = None,
        acknowledgment: Optional[str] = None
    ):
        self.changed = changed
        self.change_type = change_type
        self.similarity = similarity
        self.new_topic = new_topic
        self.acknowledgment = acknowledgment


# Global instance
_topic_change_detector: Optional[TopicChangeDetector] = None


def get_topic_change_detector() -> Optional[TopicChangeDetector]:
    """Get global topic change detector instance."""
    return _topic_change_detector


def init_topic_change_detector(embedding_function):
    """
    Initialize global topic change detector.

    Args:
        embedding_function: Function to embed text
    """
    global _topic_change_detector
    _topic_change_detector = TopicChangeDetector(embedding_function)
    logger.info("Initialized topic change detector")
