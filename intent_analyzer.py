"""
Semantic Intent Analyzer
Uses embeddings to detect intent shifts, topic changes, and intent relationships.
Best-in-class implementation with semantic similarity and multi-intent disambiguation.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from conversation_context import IntentRelationship, ConversationContext

logger = logging.getLogger("IntentAnalyzer")


class IntentType(Enum):
    """Types of user intents."""
    QUESTION = "question"  # Initial question
    CLARIFICATION_ANSWER = "clarification_answer"  # Answering a clarification
    FOLLOW_UP = "follow_up"  # Follow-up question
    REFINEMENT = "refinement"  # Refining/narrowing previous question
    COMPARISON = "comparison"  # Comparing options
    NEW_TOPIC = "new_topic"  # Completely new topic
    GREETING = "greeting"
    FEEDBACK = "feedback"


@dataclass
class IntentAnalysis:
    """Result of intent analysis."""
    intent_type: IntentType
    relationship: IntentRelationship
    confidence: float
    similarity_score: float  # Semantic similarity to original
    reasoning: str
    topic_shift: bool = False


class SemanticIntentAnalyzer:
    """
    Analyzes user intent using embeddings and semantic similarity.
    Detects topic shifts, intent relationships, and conversation flow.
    """

    def __init__(
        self,
        embedder_client,
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.7
    ):
        """
        Initialize semantic intent analyzer.

        Args:
            embedder_client: OpenAI client for embeddings
            embedding_model: Embedding model name
            similarity_threshold: Threshold for considering queries similar
        """
        self.embedder_client = embedder_client
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

        # Cache embeddings to avoid recomputation
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def analyze_intent(
        self,
        current_query: str,
        context: Optional[ConversationContext] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> IntentAnalysis:
        """
        Analyze the intent of current query relative to context.

        Args:
            current_query: Current user query
            context: Conversation context
            conversation_history: Recent conversation history

        Returns:
            IntentAnalysis with detected intent and relationship
        """
        # Quick heuristics for obvious cases
        quick_intent = self._quick_intent_detection(current_query, conversation_history)
        if quick_intent:
            return quick_intent

        # If no context, this is initial question
        if not context or not context.original_question:
            return IntentAnalysis(
                intent_type=IntentType.QUESTION,
                relationship=IntentRelationship.CONTINUATION,
                confidence=1.0,
                similarity_score=1.0,
                reasoning="Initial question"
            )

        # Compute semantic similarity
        similarity = self._compute_similarity(current_query, context.original_question)

        # Analyze relationship based on similarity and context
        intent_type, relationship = self._classify_intent_and_relationship(
            current_query,
            context,
            similarity,
            conversation_history
        )

        # Determine if this is a topic shift
        topic_shift = self._is_topic_shift(similarity, relationship)

        return IntentAnalysis(
            intent_type=intent_type,
            relationship=relationship,
            confidence=self._calculate_confidence(similarity, intent_type),
            similarity_score=similarity,
            reasoning=self._generate_reasoning(intent_type, relationship, similarity),
            topic_shift=topic_shift
        )

    def _quick_intent_detection(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]]
    ) -> Optional[IntentAnalysis]:
        """Quick heuristic-based intent detection."""
        query_lower = query.lower().strip()

        # Greetings
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if any(query_lower.startswith(g) for g in greetings) and len(query.split()) <= 3:
            return IntentAnalysis(
                intent_type=IntentType.GREETING,
                relationship=IntentRelationship.UNRELATED,
                confidence=1.0,
                similarity_score=0.0,
                reasoning="Greeting detected"
            )

        # Feedback
        feedback_words = ["thanks", "thank you", "great", "perfect", "awesome"]
        if query_lower in feedback_words or (len(query.split()) <= 3 and any(w in query_lower for w in feedback_words)):
            return IntentAnalysis(
                intent_type=IntentType.FEEDBACK,
                relationship=IntentRelationship.UNRELATED,
                confidence=1.0,
                similarity_score=0.0,
                reasoning="Feedback detected"
            )

        # Short clarification answers (1-3 words, not a question)
        if history and len(query.split()) <= 3 and not query.endswith("?"):
            # Check if last message was from assistant asking a question
            if history and history[-1].get("role") == "assistant":
                last_msg = history[-1].get("content", "").lower()
                if "?" in last_msg or any(word in last_msg for word in ["which", "what", "specify"]):
                    return IntentAnalysis(
                        intent_type=IntentType.CLARIFICATION_ANSWER,
                        relationship=IntentRelationship.CONTINUATION,
                        confidence=0.9,
                        similarity_score=0.8,
                        reasoning="Short answer to clarification question"
                    )

        return None

    def _compute_similarity(self, query1: str, query2: str) -> float:
        """
        Compute semantic similarity between two queries using embeddings.

        Args:
            query1: First query
            query2: Second query

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Get embeddings (use cache if available)
            emb1 = self._get_embedding(query1)
            emb2 = self._get_embedding(query2)

            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            return float(similarity)

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            # Fallback to simple keyword overlap
            return self._keyword_similarity(query1, query2)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text (with caching)."""
        # Check cache
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Get embedding from API
        try:
            response = self.embedder_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = np.array(response.data[0].embedding)

            # Cache it
            self._embedding_cache[text] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return random embedding as fallback (not ideal but prevents crash)
            return np.random.rand(1536)  # Default size for text-embedding-3-small

    def _keyword_similarity(self, query1: str, query2: str) -> float:
        """Fallback keyword-based similarity."""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _classify_intent_and_relationship(
        self,
        query: str,
        context: ConversationContext,
        similarity: float,
        history: Optional[List[Dict[str, str]]]
    ) -> Tuple[IntentType, IntentRelationship]:
        """
        Classify intent type and relationship to original question.

        Args:
            query: Current query
            context: Conversation context
            similarity: Semantic similarity score
            history: Conversation history

        Returns:
            Tuple of (IntentType, IntentRelationship)
        """
        query_lower = query.lower()

        # High similarity -> continuation or refinement
        if similarity > 0.8:
            # Check if asking for more details
            if any(word in query_lower for word in ["more", "tell me more", "details", "elaborate"]):
                return IntentType.REFINEMENT, IntentRelationship.CONTINUATION

            # Check if asking a follow-up
            if query.endswith("?") and len(query.split()) > 5:
                return IntentType.FOLLOW_UP, IntentRelationship.CONTINUATION

            # Default to continuation
            return IntentType.CLARIFICATION_ANSWER, IntentRelationship.CONTINUATION

        # Medium similarity -> related or comparison
        elif similarity > 0.5:
            # Check for comparison keywords
            if any(word in query_lower for word in ["what about", "how about", "versus", "vs", "compared to", "difference"]):
                return IntentType.COMPARISON, IntentRelationship.COMPARISON

            # Related topic
            return IntentType.FOLLOW_UP, IntentRelationship.RELATED

        # Low similarity -> likely new topic
        else:
            # Check if it's a completely new question
            question_starters = ["what", "how", "when", "where", "who", "why", "can", "is", "are", "do", "does"]
            if any(query_lower.startswith(starter) for starter in question_starters):
                return IntentType.NEW_TOPIC, IntentRelationship.UNRELATED

            # Might still be clarification answer but very different wording
            return IntentType.CLARIFICATION_ANSWER, IntentRelationship.RELATED

    def _is_topic_shift(self, similarity: float, relationship: IntentRelationship) -> bool:
        """Determine if this represents a topic shift."""
        if relationship == IntentRelationship.UNRELATED:
            return True

        if similarity < 0.4:
            return True

        return False

    def _calculate_confidence(self, similarity: float, intent_type: IntentType) -> float:
        """Calculate confidence score for the intent classification."""
        # Base confidence on similarity and intent type
        if intent_type in [IntentType.GREETING, IntentType.FEEDBACK]:
            return 1.0

        # For semantic intents, confidence correlates with similarity
        if similarity > 0.8:
            return 0.95
        elif similarity > 0.6:
            return 0.85
        elif similarity > 0.4:
            return 0.7
        else:
            return 0.6

    def _generate_reasoning(
        self,
        intent_type: IntentType,
        relationship: IntentRelationship,
        similarity: float
    ) -> str:
        """Generate human-readable reasoning for the classification."""
        reasons = []

        # Similarity
        if similarity > 0.8:
            reasons.append("High semantic similarity")
        elif similarity > 0.5:
            reasons.append("Moderate semantic similarity")
        else:
            reasons.append("Low semantic similarity")

        # Intent type
        reasons.append(f"Intent: {intent_type.value}")

        # Relationship
        reasons.append(f"Relationship: {relationship.value}")

        return "; ".join(reasons)


class MultiIntentDisambiguator:
    """
    Disambiguates queries that might have multiple intents.
    Handles cases like "What about X?" which could be comparison or new question.
    """

    def __init__(self, llm_client, deployment_name: str = None):
        """
        Initialize multi-intent disambiguator.

        Args:
            llm_client: LLM client
            deployment_name: Azure deployment name
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name

    def disambiguate(
        self,
        query: str,
        context: ConversationContext,
        initial_analysis: IntentAnalysis
    ) -> IntentAnalysis:
        """
        Disambiguate ambiguous intents using LLM.

        Args:
            query: Current query
            context: Conversation context
            initial_analysis: Initial intent analysis

        Returns:
            Refined IntentAnalysis
        """
        # Only disambiguate if confidence is low or relationship is unclear
        if initial_analysis.confidence > 0.85:
            return initial_analysis

        # Check for ambiguous patterns
        ambiguous_patterns = ["what about", "how about", "and", "also"]
        if not any(pattern in query.lower() for pattern in ambiguous_patterns):
            return initial_analysis

        # Use LLM to disambiguate
        try:
            refined_analysis = self._disambiguate_with_llm(query, context, initial_analysis)
            return refined_analysis
        except Exception as e:
            logger.error(f"LLM disambiguation error: {e}")
            return initial_analysis

    def _disambiguate_with_llm(
        self,
        query: str,
        context: ConversationContext,
        initial_analysis: IntentAnalysis
    ) -> IntentAnalysis:
        """Use LLM to disambiguate intent."""
        import json

        # Build context
        entities_str = json.dumps(context.get_all_entities())

        prompt = f"""Analyze the intent relationship between the current query and original question.

Original Question: "{context.original_question}"
Topic: {context.primary_topic}
Entities: {entities_str}

Current Query: "{query}"

Initial Analysis:
- Intent Type: {initial_analysis.intent_type.value}
- Relationship: {initial_analysis.relationship.value}
- Similarity: {initial_analysis.similarity_score:.2f}

Is this:
1. CONTINUATION - Same topic, providing clarification or more details
2. RELATED - Related but different aspect of the topic
3. COMPARISON - Comparing to original (e.g., "What about X instead?")
4. UNRELATED - Completely new topic

Respond in JSON format:
{{
    "intent_relationship": "continuation|related|comparison|unrelated",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        try:
            model_param = self.deployment_name if self.deployment_name else "gpt-4o"
            response = self.llm_client.chat.completions.create(
                model=model_param,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Map string to enum
            relationship_map = {
                "continuation": IntentRelationship.CONTINUATION,
                "related": IntentRelationship.RELATED,
                "comparison": IntentRelationship.COMPARISON,
                "unrelated": IntentRelationship.UNRELATED
            }

            relationship = relationship_map.get(
                result["intent_relationship"].lower(),
                initial_analysis.relationship
            )

            # Update analysis
            return IntentAnalysis(
                intent_type=initial_analysis.intent_type,
                relationship=relationship,
                confidence=result.get("confidence", initial_analysis.confidence),
                similarity_score=initial_analysis.similarity_score,
                reasoning=result.get("reasoning", initial_analysis.reasoning),
                topic_shift=relationship == IntentRelationship.UNRELATED
            )

        except Exception as e:
            logger.error(f"LLM disambiguation failed: {e}")
            return initial_analysis
