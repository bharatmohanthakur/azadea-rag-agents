"""
Self-evaluation module for agentic RAG system.
Evaluates answer quality, completeness, and determines if termination is appropriate.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger("SelfEvaluator")


class TerminationReason(Enum):
    """Reasons for termination decision."""
    HIGH_CONFIDENCE = "high_confidence"  # High confidence answer found
    COMPLETE_ANSWER = "complete_answer"  # Answer fully addresses query
    INSUFFICIENT_CONTEXT = "insufficient_context"  # Not enough context available
    PARTIAL_SUFFICIENT = "partial_sufficient"  # Partial info is sufficient
    USER_FRUSTRATION = "user_frustration"  # User wants to proceed
    MAX_ITERATIONS = "max_iterations"  # Reached max attempts
    CANNOT_IMPROVE = "cannot_improve"  # Cannot improve further


@dataclass
class TerminationDecision:
    """Decision on whether to terminate or continue."""
    should_terminate: bool
    reason: str  # TerminationReason value
    confidence_score: float  # 0.0 to 1.0
    completeness_score: float  # 0.0 to 1.0
    grounded_score: float  # 0.0 to 1.0
    recommendations: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "should_terminate": self.should_terminate,
            "reason": self.reason,
            "confidence_score": self.confidence_score,
            "completeness_score": self.completeness_score,
            "grounded_score": self.grounded_score,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }


class SelfEvaluator:
    """Evaluates answers and makes termination decisions."""
    
    def __init__(self, llm_client):
        """
        Initialize self evaluator.
        
        Args:
            llm_client: LLM client for evaluation
        """
        self.llm_client = llm_client
    
    def evaluate_answer_completeness(
        self,
        answer: str,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[float, List[str]]:
        """
        Evaluate if answer fully addresses the query.
        
        Args:
            answer: Generated answer
            query: Original user query
            retrieved_chunks: Retrieved context chunks
        
        Returns:
            Tuple of (completeness_score 0-1, missing_aspects)
        """
        # Extract key aspects from query
        query_lower = query.lower()
        
        # Check for question words that indicate what's needed
        question_indicators = {
            "what": ["definition", "description", "explanation"],
            "how": ["process", "steps", "procedure", "method"],
            "when": ["timing", "date", "schedule"],
            "where": ["location", "place"],
            "who": ["person", "entity", "responsible"],
            "why": ["reason", "cause", "rationale"],
            "how many": ["quantity", "number", "count"],
            "how much": ["amount", "cost", "value"]
        }
        
        # Determine expected aspects
        expected_aspects = []
        for indicator, aspects in question_indicators.items():
            if indicator in query_lower:
                expected_aspects.extend(aspects)
        
        # If no specific indicators, check for common HR query patterns
        if not expected_aspects:
            hr_patterns = {
                "leave": ["duration", "eligibility", "process"],
                "insurance": ["coverage", "benefits", "enrollment"],
                "policy": ["rules", "requirements", "exceptions"],
                "benefit": ["details", "eligibility", "how to claim"]
            }
            for pattern, aspects in hr_patterns.items():
                if pattern in query_lower:
                    expected_aspects.extend(aspects)
        
        # Evaluate answer against expected aspects
        answer_lower = answer.lower()
        addressed_aspects = []
        missing_aspects = []
        
        for aspect in expected_aspects:
            # Simple keyword matching (can be enhanced with embeddings)
            aspect_keywords = {
                "definition": ["is", "means", "refers to", "defined as"],
                "description": ["includes", "contains", "consists", "features"],
                "process": ["step", "process", "procedure", "how to"],
                "duration": ["day", "week", "month", "year", "period"],
                "eligibility": ["eligible", "qualify", "requirement", "criteria"],
                "coverage": ["covers", "includes", "provides", "benefits"],
                "location": ["country", "location", "place", "where"]
            }
            
            keywords = aspect_keywords.get(aspect, [aspect])
            if any(kw in answer_lower for kw in keywords):
                addressed_aspects.append(aspect)
            else:
                missing_aspects.append(aspect)
        
        # Calculate completeness score
        if expected_aspects:
            completeness = len(addressed_aspects) / len(expected_aspects)
        else:
            # If no specific aspects, use length and structure heuristics
            if len(answer) > 100 and "." in answer:
                completeness = 0.7  # Assume reasonable completeness
            elif len(answer) > 50:
                completeness = 0.5
            else:
                completeness = 0.3
        
        return min(completeness, 1.0), missing_aspects
    
    def evaluate_answer_grounding(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate if answer is grounded in retrieved context.
        
        Args:
            answer: Generated answer
            retrieved_chunks: Retrieved context chunks
        
        Returns:
            Grounding score 0.0 to 1.0
        """
        if not retrieved_chunks:
            return 0.0
        
        # Extract text from chunks
        chunk_texts = []
        for chunk in retrieved_chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text_snippet", "") or chunk.get("text", "") or chunk.get("content", "")
                if text:
                    chunk_texts.append(text.lower())
        
        if not chunk_texts:
            return 0.0
        
        # Combine all chunk text
        all_context = " ".join(chunk_texts)
        
        # Extract key phrases from answer (words > 3 chars, not stop words)
        answer_lower = answer.lower()
        answer_words = set(word for word in answer_lower.split() if len(word) > 3)
        
        # Remove common words
        stop_words = {"this", "that", "these", "those", "with", "from", "have", "been", "will", "would"}
        answer_words = answer_words - stop_words
        
        if not answer_words:
            return 0.5  # Neutral if no meaningful words
        
        # Check overlap with context
        context_words = set(word for word in all_context.split() if len(word) > 3)
        context_words = context_words - stop_words
        
        overlap = len(answer_words & context_words)
        overlap_ratio = overlap / len(answer_words) if answer_words else 0.0
        
        return min(overlap_ratio, 1.0)
    
    def evaluate_confidence(
        self,
        answer: str,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        answer_quality: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate overall confidence in the answer.
        
        Args:
            answer: Generated answer
            query: Original query
            retrieved_chunks: Retrieved context
            answer_quality: Optional quality assessment from AnswerQuality module
        
        Returns:
            Confidence score 0.0 to 1.0
        """
        # Start with quality assessment if available
        if answer_quality:
            quality_conf = answer_quality.get("confidence", {}).get("score", 0.5)
        else:
            quality_conf = 0.5
        
        # Evaluate completeness
        completeness, _ = self.evaluate_answer_completeness(answer, query, retrieved_chunks)
        
        # Evaluate grounding
        grounding = self.evaluate_answer_grounding(answer, retrieved_chunks)
        
        # Check for uncertainty indicators in answer
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "unclear", "not available",
            "cannot determine", "no information", "not found", "may vary",
            "depends on", "might be", "possibly", "perhaps"
        ]
        answer_lower = answer.lower()
        has_uncertainty = any(phrase in answer_lower for phrase in uncertainty_phrases)
        
        if has_uncertainty:
            uncertainty_penalty = 0.3
        else:
            uncertainty_penalty = 0.0
        
        # Calculate weighted confidence
        confidence = (
            quality_conf * 0.4 +
            completeness * 0.3 +
            grounding * 0.3 -
            uncertainty_penalty
        )
        
        return max(0.0, min(confidence, 1.0))
    
    def make_termination_decision(
        self,
        answer: str,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        clarification_session: Optional[Any] = None,
        iteration_count: int = 0,
        max_iterations: int = 3,
        answer_quality: Optional[Dict[str, Any]] = None
    ) -> TerminationDecision:
        """
        Make decision on whether to terminate or continue.
        
        Args:
            answer: Generated answer
            query: Original query
            retrieved_chunks: Retrieved context chunks
            clarification_session: Optional active clarification session
            iteration_count: Current iteration number
            max_iterations: Maximum allowed iterations
            answer_quality: Optional quality assessment
        
        Returns:
            TerminationDecision
        """
        # Evaluate metrics
        completeness, missing_aspects = self.evaluate_answer_completeness(answer, query, retrieved_chunks)
        grounding = self.evaluate_answer_grounding(answer, retrieved_chunks)
        confidence = self.evaluate_confidence(answer, query, retrieved_chunks, answer_quality)
        
        recommendations = []
        should_terminate = False
        reason = ""
        
        # Check max iterations
        if iteration_count >= max_iterations:
            should_terminate = True
            reason = TerminationReason.MAX_ITERATIONS.value
            recommendations.append("Reached maximum iterations, returning best available answer")
        
        # High confidence and completeness - terminate
        elif confidence >= 0.8 and completeness >= 0.8 and grounding >= 0.7:
            should_terminate = True
            reason = TerminationReason.HIGH_CONFIDENCE.value
            recommendations.append("High confidence answer with good completeness and grounding")
        
        # Complete answer even if confidence is moderate
        elif completeness >= 0.9 and grounding >= 0.6:
            should_terminate = True
            reason = TerminationReason.COMPLETE_ANSWER.value
            recommendations.append("Answer addresses all aspects of the query")
        
        # Check clarification session status
        elif clarification_session:
            # If user provided comprehensive answer covering multiple questions
            if hasattr(clarification_session, 'user_answers'):
                answers_count = len(clarification_session.user_answers)
                questions_count = len(clarification_session.questions_asked)
                
                # If user answered multiple questions in one response
                if answers_count >= questions_count * 0.7:  # 70% answered
                    if confidence >= 0.6:
                        should_terminate = True
                        reason = TerminationReason.PARTIAL_SUFFICIENT.value
                        recommendations.append("Sufficient information provided, proceeding with answer")
        
        # Low confidence but can't improve - terminate with disclaimer
        elif confidence < 0.5 and iteration_count >= 2:
            should_terminate = True
            reason = TerminationReason.CANNOT_IMPROVE.value
            recommendations.append("Low confidence but cannot improve further - return with disclaimer")
        
        # Insufficient context
        elif len(retrieved_chunks) == 0 or grounding < 0.3:
            should_terminate = False
            reason = TerminationReason.INSUFFICIENT_CONTEXT.value
            recommendations.append("Insufficient context - retrieve more information or ask for clarification")
        
        # Default: continue if confidence/completeness is low
        else:
            should_terminate = False
            reason = "needs_improvement"
            if completeness < 0.7:
                recommendations.append(f"Answer incomplete - missing aspects: {', '.join(missing_aspects[:3])}")
            if confidence < 0.6:
                recommendations.append("Low confidence - consider retrieving more context")
            if grounding < 0.5:
                recommendations.append("Answer may not be well-grounded - verify against sources")
        
        metadata = {
            "iteration_count": iteration_count,
            "chunks_count": len(retrieved_chunks),
            "missing_aspects": missing_aspects[:5],  # Limit to 5
            "answer_length": len(answer),
            "has_clarification_session": clarification_session is not None
        }
        
        return TerminationDecision(
            should_terminate=should_terminate,
            reason=reason,
            confidence_score=confidence,
            completeness_score=completeness,
            grounded_score=grounding,
            recommendations=recommendations,
            metadata=metadata
        )
    
    def evaluate_with_llm(
        self,
        answer: str,
        query: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Use LLM to evaluate answer quality (more sophisticated than heuristics).
        
        Args:
            answer: Generated answer
            query: Original query
            context: Retrieved context
        
        Returns:
            Evaluation dictionary
        """
        prompt = f"""Evaluate the following answer for completeness and accuracy.

User Query: {query}

Retrieved Context:
{context[:2000]}

Generated Answer:
{answer}

Evaluate:
1. Does the answer fully address the user's query? (completeness: 0.0-1.0)
2. Is the answer accurate based on the context? (accuracy: 0.0-1.0)
3. Are there any important details missing? (list missing aspects)
4. Should the system retrieve more information or is this sufficient? (recommendation)

Respond in JSON format:
{{
    "completeness": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "missing_aspects": ["aspect1", "aspect2"],
    "recommendation": "sufficient" | "retrieve_more" | "ask_clarification"
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for evaluation
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            import json
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Fallback to heuristic evaluation
            return {
                "completeness": 0.5,
                "accuracy": 0.5,
                "missing_aspects": [],
                "recommendation": "sufficient"
            }

