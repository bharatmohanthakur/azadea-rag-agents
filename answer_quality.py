"""
Answer quality assessment: grounding verification and confidence scoring.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger("AnswerQuality")


class ConfidenceLevel(Enum):
    """Confidence levels for answers."""
    HIGH = "high"  # Strong evidence in retrieved context
    MEDIUM = "medium"  # Some evidence, may need verification
    LOW = "low"  # Weak or no evidence
    UNCERTAIN = "uncertain"  # Cannot determine from context


class AnswerQuality:
    """Assesses answer quality and confidence."""
    
    @staticmethod
    def calculate_confidence(
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        graphiti_facts: List[Dict[str, Any]],
        answer_mentions_sources: bool = False
    ) -> Tuple[ConfidenceLevel, float, Dict[str, Any]]:
        """
        Calculate confidence score for an answer.
        
        Args:
            answer: Generated answer text
            retrieved_chunks: List of retrieved document chunks with scores
            graphiti_facts: List of retrieved memory facts
            answer_mentions_sources: Whether answer explicitly mentions sources
        
        Returns:
            Tuple of (confidence_level, score_0_to_1, metadata)
        """
        metadata = {
            "chunk_count": len(retrieved_chunks),
            "fact_count": len(graphiti_facts),
            "has_sources": answer_mentions_sources
        }
        
        # No context available
        if not retrieved_chunks and not graphiti_facts:
            return ConfidenceLevel.UNCERTAIN, 0.0, {**metadata, "reason": "No context retrieved"}
        
        # Calculate average retrieval score
        chunk_scores = [c.get("score", 0.0) for c in retrieved_chunks if isinstance(c, dict)]
        avg_chunk_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        
        # Factor in number of sources
        source_diversity = min(len(set(c.get("source", "") for c in retrieved_chunks)), 5) / 5.0
        
        # Check if answer explicitly mentions uncertainty
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "unclear", "not available",
            "cannot determine", "no information", "not found"
        ]
        has_uncertainty_phrase = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        
        # Base score from retrieval quality
        base_score = avg_chunk_score * 0.6 + source_diversity * 0.2
        
        # Boost if answer mentions sources
        if answer_mentions_sources:
            base_score += 0.1
        
        # Penalize if uncertainty phrases present
        if has_uncertainty_phrase:
            base_score *= 0.5
        
        # Boost if we have both chunks and facts
        if retrieved_chunks and graphiti_facts:
            base_score += 0.1
        
        # Normalize to 0-1
        confidence_score = min(max(base_score, 0.0), 1.0)
        
        # Determine level
        if confidence_score >= 0.7:
            level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.4:
            level = ConfidenceLevel.MEDIUM
        elif confidence_score > 0.0:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.UNCERTAIN
        
        metadata.update({
            "avg_chunk_score": round(avg_chunk_score, 3),
            "source_diversity": round(source_diversity, 3),
            "has_uncertainty_phrase": has_uncertainty_phrase,
            "confidence_score": round(confidence_score, 3)
        })
        
        return level, confidence_score, metadata
    
    @staticmethod
    def check_grounding(
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        threshold: float = 0.3
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Check if answer is grounded in retrieved context.
        
        Args:
            answer: Generated answer text
            retrieved_chunks: List of retrieved chunks with text content
            threshold: Minimum similarity threshold for grounding
        
        Returns:
            Tuple of (is_grounded, ungrounded_claims, metadata)
        """
        if not retrieved_chunks:
            return False, [], {"reason": "No context available"}
        
        # Extract key phrases from answer (simple approach)
        answer_lower = answer.lower()
        answer_words = set(word for word in answer_lower.split() if len(word) > 3)
        
        # Extract text from chunks
        chunk_texts = []
        for chunk in retrieved_chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text_snippet", "") or chunk.get("text", "") or chunk.get("content", "")
                if text:
                    chunk_texts.append(text.lower())
        
        if not chunk_texts:
            return False, [], {"reason": "No text content in chunks"}
        
        # Simple keyword overlap check
        all_chunk_text = " ".join(chunk_texts)
        chunk_words = set(word for word in all_chunk_text.split() if len(word) > 3)
        
        # Calculate overlap
        overlap = len(answer_words & chunk_words)
        total_answer_words = len(answer_words)
        
        if total_answer_words == 0:
            overlap_ratio = 0.0
        else:
            overlap_ratio = overlap / total_answer_words
        
        is_grounded = overlap_ratio >= threshold
        
        metadata = {
            "overlap_ratio": round(overlap_ratio, 3),
            "overlap_count": overlap,
            "total_answer_words": total_answer_words,
            "threshold": threshold
        }
        
        # Simple heuristic: if overlap is low, mark as potentially ungrounded
        ungrounded_claims = []
        if not is_grounded:
            # Extract sentences that might be ungrounded
            sentences = answer.split('.')
            for sentence in sentences:
                sentence_words = set(word for word in sentence.lower().split() if len(word) > 3)
                sentence_overlap = len(sentence_words & chunk_words)
                if len(sentence_words) > 0 and sentence_overlap / len(sentence_words) < threshold:
                    ungrounded_claims.append(sentence.strip()[:100])  # Truncate
        
        return is_grounded, ungrounded_claims[:3], metadata  # Limit to 3 claims
    
    @staticmethod
    def assess_answer(
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        graphiti_facts: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive answer quality assessment.
        
        Args:
            answer: Generated answer
            retrieved_chunks: Retrieved document chunks
            graphiti_facts: Retrieved memory facts
            query: Original query (optional)
        
        Returns:
            Assessment dictionary with confidence, grounding, and recommendations
        """
        # Calculate confidence
        confidence_level, confidence_score, conf_metadata = AnswerQuality.calculate_confidence(
            answer, retrieved_chunks, graphiti_facts
        )
        
        # Check grounding
        is_grounded, ungrounded_claims, ground_metadata = AnswerQuality.check_grounding(
            answer, retrieved_chunks
        )
        
        # Overall assessment
        assessment = {
            "confidence": {
                "level": confidence_level.value,
                "score": confidence_score,
                "metadata": conf_metadata
            },
            "grounding": {
                "is_grounded": is_grounded,
                "ungrounded_claims": ungrounded_claims,
                "metadata": ground_metadata
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if confidence_level == ConfidenceLevel.LOW or confidence_level == ConfidenceLevel.UNCERTAIN:
            assessment["recommendations"].append("Consider asking for clarification or providing a more tentative answer")
        
        if not is_grounded:
            assessment["recommendations"].append("Answer may contain information not found in retrieved context")
        
        if ungrounded_claims:
            assessment["recommendations"].append(f"Found {len(ungrounded_claims)} potentially ungrounded claims")
        
        if len(retrieved_chunks) < 2:
            assessment["recommendations"].append("Limited context retrieved - consider expanding search")
        
        return assessment


