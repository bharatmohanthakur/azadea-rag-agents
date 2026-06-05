"""
Answer Quality Gate - Multi-stage validation before termination.
Validates completeness, accuracy, relevance, and grounding before returning answer.
"""

import logging
from typing import Dict, List, Any, Optional
from self_evaluator import SelfEvaluator, TerminationDecision
from answer_quality import AnswerQuality

logger = logging.getLogger("AnswerQualityGate")


class AnswerQualityGate:
    """Validates answer quality before termination."""
    
    def __init__(self, self_evaluator: SelfEvaluator):
        """
        Initialize answer quality gate.
        
        Args:
            self_evaluator: SelfEvaluator instance
        """
        self.self_evaluator = self_evaluator
    
    def validate_answer(
        self,
        answer: str,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        graphiti_facts: List[Dict[str, Any]] = None,
        clarification_session: Optional[Any] = None,
        iteration_count: int = 0
    ) -> Dict[str, Any]:
        """
        Multi-stage answer validation.
        
        Args:
            answer: Generated answer
            query: Original query
            retrieved_chunks: Retrieved context chunks
            graphiti_facts: Optional memory facts
            clarification_session: Optional clarification session
            iteration_count: Current iteration
        
        Returns:
            Validation result dictionary
        """
        graphiti_facts = graphiti_facts or []
        
        # Stage 1: Answer Quality Assessment
        quality_assessment = AnswerQuality.assess_answer(
            answer,
            retrieved_chunks,
            graphiti_facts,
            query
        )
        
        # Stage 2: Self-Evaluation
        termination_decision = self.self_evaluator.make_termination_decision(
            answer=answer,
            query=query,
            retrieved_chunks=retrieved_chunks,
            clarification_session=clarification_session,
            iteration_count=iteration_count,
            answer_quality=quality_assessment
        )
        
        # Stage 3: Comprehensive Validation
        validation_result = {
            "quality": quality_assessment,
            "termination": termination_decision.to_dict(),
            "validation_passed": False,
            "can_improve": False,
            "action": "continue",  # "terminate", "continue", "retrieve_more", "ask_clarification"
            "confidence_indicator": None,
            "disclaimer": None
        }
        
        # Decision Logic
        confidence = termination_decision.confidence_score
        completeness = termination_decision.completeness_score
        grounded = termination_decision.grounded_score
        
        # High quality - terminate
        if confidence >= 0.8 and completeness >= 0.8 and grounded >= 0.7:
            validation_result["validation_passed"] = True
            validation_result["action"] = "terminate"
            validation_result["confidence_indicator"] = "high"
            logger.info("✅ Quality gate: PASSED - High quality answer")
        
        # Good quality - terminate with confidence indicator
        elif confidence >= 0.6 and completeness >= 0.6 and grounded >= 0.5:
            validation_result["validation_passed"] = True
            validation_result["action"] = "terminate"
            if confidence >= 0.7:
                validation_result["confidence_indicator"] = "medium-high"
            else:
                validation_result["confidence_indicator"] = "medium"
            logger.info(f"✅ Quality gate: PASSED - Good quality answer (confidence: {confidence:.2f})")
        
        # Moderate quality - can improve
        elif confidence >= 0.4 and completeness >= 0.4:
            if iteration_count < 2:  # Can still improve
                validation_result["can_improve"] = True
                validation_result["action"] = "retrieve_more"
                logger.info("⚠️ Quality gate: MODERATE - Can improve with more context")
            else:
                # Max iterations reached - return with disclaimer
                validation_result["validation_passed"] = True
                validation_result["action"] = "terminate"
                validation_result["confidence_indicator"] = "low-medium"
                validation_result["disclaimer"] = "This answer is based on available information but may not be complete."
                logger.warning("⚠️ Quality gate: MODERATE - Returning with disclaimer")
        
        # Low quality - needs improvement
        else:
            if iteration_count < 2:
                validation_result["can_improve"] = True
                validation_result["action"] = "retrieve_more"
                logger.warning("❌ Quality gate: FAILED - Needs improvement")
            elif completeness < 0.4 and not clarification_session:
                # Low completeness and no clarification - ask for clarification
                validation_result["action"] = "ask_clarification"
                logger.warning("❌ Quality gate: FAILED - Needs clarification")
            else:
                # Can't improve - return with strong disclaimer
                validation_result["validation_passed"] = True
                validation_result["action"] = "terminate"
                validation_result["confidence_indicator"] = "low"
                validation_result["disclaimer"] = (
                    "I found limited information to answer your question. "
                    "The answer below is based on available documents but may not be complete. "
                    "Please consult HR for definitive information."
                )
                logger.error("❌ Quality gate: FAILED - Returning with strong disclaimer")
        
        return validation_result
    
    def should_terminate(self, validation_result: Dict[str, Any]) -> bool:
        """Check if validation result indicates termination."""
        return validation_result.get("action") == "terminate"
    
    def get_improved_answer_prompt(
        self,
        answer: str,
        query: str,
        validation_result: Dict[str, Any]
    ) -> str:
        """
        Generate prompt for improving the answer based on validation.
        
        Args:
            answer: Current answer
            query: Original query
            validation_result: Validation result
        
        Returns:
            Improvement prompt
        """
        quality = validation_result.get("quality", {})
        termination = validation_result.get("termination", {})
        
        missing_aspects = termination.get("metadata", {}).get("missing_aspects", [])
        recommendations = termination.get("recommendations", [])
        
        prompt = f"""The following answer was generated but needs improvement.

Original Query: {query}

Current Answer:
{answer}

Issues Identified:
"""
        
        if missing_aspects:
            prompt += f"- Missing aspects: {', '.join(missing_aspects)}\n"
        
        if recommendations:
            prompt += f"- Recommendations: {'; '.join(recommendations)}\n"
        
        quality_issues = quality.get("recommendations", [])
        if quality_issues:
            prompt += f"- Quality issues: {'; '.join(quality_issues)}\n"
        
        prompt += "\nPlease improve the answer to address these issues. Make it more complete, accurate, and grounded in the context."
        
        return prompt

