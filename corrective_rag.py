"""
Corrective RAG - Evaluate and correct retrieval process.
Based on RAG Techniques: https://github.com/NirDiamant/RAG_Techniques

This technique improves retrieval by:
1. Evaluating initial retrieval results
2. Identifying gaps or irrelevant content
3. Generating refined queries to fill gaps
4. Re-retrieving with improved queries
5. Combining and validating final results
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger("CorrectiveRAG")


class RetrievalQuality(str, Enum):
    """Quality assessment of retrieval results."""
    EXCELLENT = "excellent"  # All relevant information found
    GOOD = "good"  # Most relevant information found
    FAIR = "fair"  # Some relevant information, but gaps exist
    POOR = "poor"  # Little or no relevant information


class RetrievalEvaluation(BaseModel):
    """Evaluation of retrieval results."""
    quality: RetrievalQuality = Field(description="Overall quality assessment")
    relevance_score: float = Field(description="Relevance score (0-1)")
    completeness_score: float = Field(description="Completeness score (0-1)")
    gaps: List[str] = Field(description="Identified information gaps", default_factory=list)
    irrelevant_parts: List[str] = Field(description="Irrelevant content identified", default_factory=list)
    needs_correction: bool = Field(description="Whether retrieval needs correction")
    refined_queries: List[str] = Field(description="Refined queries for re-retrieval", default_factory=list)
    reasoning: str = Field(description="Reasoning for evaluation", default="")


class CorrectiveRAG:
    """Evaluates and corrects retrieval results."""
    
    def __init__(self, llm_client, deployment_name: str):
        """
        Initialize corrective RAG.
        
        Args:
            llm_client: LLM client for evaluation
            deployment_name: Azure deployment name
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_context: str,
        sources: List[Dict[str, Any]]
    ) -> RetrievalEvaluation:
        """
        Evaluate the quality of retrieved results.
        
        Args:
            query: Original user query
            retrieved_context: Retrieved context text
            sources: List of source documents
            
        Returns:
            RetrievalEvaluation object
        """
        if not retrieved_context or not sources:
            return RetrievalEvaluation(
                quality=RetrievalQuality.POOR,
                relevance_score=0.0,
                completeness_score=0.0,
                needs_correction=True,
                refined_queries=[query],
                reasoning="No context retrieved"
            )
        
        prompt = f"""Evaluate the quality of retrieved information for answering the query.

Query: {query}

Retrieved Context:
{retrieved_context[:2000]}  # Limit length

Evaluate:
1. Relevance: How relevant is the retrieved information to the query? (0.0-1.0)
2. Completeness: Does the context fully answer the query, or are there gaps? (0.0-1.0)
3. Gaps: What specific information is missing or unclear?
4. Irrelevant Content: What parts of the context are not relevant to the query?
5. Needs Correction: Should we retrieve more information or refine the query?

Respond in JSON format:
{{
    "relevance_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "quality": "excellent" | "good" | "fair" | "poor",
    "gaps": ["gap 1", "gap 2", ...],
    "irrelevant_parts": ["irrelevant part 1", ...],
    "needs_correction": true/false,
    "refined_queries": ["refined query 1", "refined query 2", ...],
    "reasoning": "brief explanation"
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            quality_str = result.get("quality", "fair").lower()
            quality = RetrievalQuality(quality_str) if quality_str in [q.value for q in RetrievalQuality] else RetrievalQuality.FAIR
            
            return RetrievalEvaluation(
                quality=quality,
                relevance_score=float(result.get("relevance_score", 0.5)),
                completeness_score=float(result.get("completeness_score", 0.5)),
                gaps=result.get("gaps", []),
                irrelevant_parts=result.get("irrelevant_parts", []),
                needs_correction=result.get("needs_correction", False),
                refined_queries=result.get("refined_queries", [query]),
                reasoning=result.get("reasoning", "")
            )
            
        except Exception as e:
            logger.error(f"Error evaluating retrieval: {e}")
            # Fallback: assume fair quality
            return RetrievalEvaluation(
                quality=RetrievalQuality.FAIR,
                relevance_score=0.5,
                completeness_score=0.5,
                needs_correction=False,
                refined_queries=[query],
                reasoning=f"Error during evaluation: {str(e)}"
            )
    
    def should_correct(self, evaluation: RetrievalEvaluation, threshold: float = 0.6) -> bool:
        """
        Determine if retrieval should be corrected.
        Very conservative: only correct if quality is poor, never for good or excellent.
        
        Args:
            evaluation: Retrieval evaluation result
            threshold: Minimum score threshold (below which correction is needed)
            
        Returns:
            True if correction is needed
        """
        # Never correct for excellent or good quality
        if evaluation.quality in [RetrievalQuality.EXCELLENT, RetrievalQuality.GOOD]:
            return False
        
        # Only correct if quality is explicitly poor
        if evaluation.quality == RetrievalQuality.POOR:
            return True
        
        # For fair quality, only correct if both scores are very low (below threshold)
        if evaluation.quality == RetrievalQuality.FAIR:
            if evaluation.relevance_score < threshold and evaluation.completeness_score < threshold:
                return True
        
        # Don't correct for fair quality with decent scores or if only minor gaps exist
        return False
    
    def filter_irrelevant(
        self,
        context: str,
        irrelevant_parts: List[str]
    ) -> str:
        """
        Filter out irrelevant parts from context.
        
        Args:
            context: Original context
            irrelevant_parts: List of irrelevant content to remove
            
        Returns:
            Filtered context
        """
        if not irrelevant_parts:
            return context
        
        filtered = context
        for part in irrelevant_parts:
            # Simple removal (could be enhanced with more sophisticated filtering)
            if part in filtered:
                filtered = filtered.replace(part, "")
        
        # Clean up extra whitespace
        import re
        filtered = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered)
        
        return filtered.strip()

