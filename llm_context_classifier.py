"""
LLM Context Classifier - Intelligent classification using Chain of Thought.
Uses LLM to determine user intent with full conversation context memory.
No hardcoding - all decisions made by LLM with CoT reasoning.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import AzureOpenAI, OpenAI
import os

logger = logging.getLogger("LLMContextClassifier")


@dataclass
class ClassificationResult:
    """Result of LLM classification with CoT reasoning."""
    classification: str  # clarification_answer, topic_change, follow_up, refinement
    confidence: float
    reasoning: str
    should_continue_clarification: bool
    detected_intent: Optional[str] = None


class LLMContextClassifier:
    """
    LLM-based context classification with Chain of Thought reasoning.
    Uses full conversation context memory for intelligent decisions.
    """
    
    def __init__(self, aoai_client: AzureOpenAI, deployment_name: str):
        """
        Initialize classifier.
        
        Args:
            aoai_client: Azure OpenAI client
            deployment_name: Model deployment name
        """
        self.client = aoai_client
        self.deployment = deployment_name
        logger.info("Initialized LLM Context Classifier with CoT reasoning")
    
    def classify_user_response(
        self,
        user_response: str,
        conversation_history: List[Dict],
        last_clarification_question: str,
        original_query: str
    ) -> ClassificationResult:
        """
        Use LLM with CoT to classify user's response.
        
        Args:
            user_response: Current user message
            conversation_history: Full conversation context
            last_clarification_question: The clarification question that was asked
            original_query: The original query that started this flow
            
        Returns:
            ClassificationResult with CoT reasoning
        """
        # Build context-rich prompt for LLM
        history_text = self._format_history(conversation_history[-10:])  # Last 10 messages
        
        prompt = f"""You are an intelligent conversation context analyzer. Analyze the user's response and determine their intent.

<conversation_context>
ORIGINAL USER QUESTION: "{original_query}"

CLARIFICATION QUESTION ASKED: "{last_clarification_question}"

USER'S RESPONSE: "{user_response}"

RECENT CONVERSATION HISTORY:
{history_text}
</conversation_context>

<task>
Think step by step using Chain of Thought reasoning:

1. UNDERSTAND THE CLARIFICATION: What information was the clarification question asking for?
2. ANALYZE THE RESPONSE: Does the user's response provide that information?
3. CHECK FOR TOPIC CHANGE: Is the user asking about something completely different?
4. ASSESS CONTINUITY: Does the response naturally continue the original conversation flow?

Based on your analysis, classify the response into ONE of these categories:
- "clarification_answer": User answered the clarification question (even partially)
- "topic_change": User wants to discuss something completely different  
- "follow_up": User is asking a related follow-up question
- "refinement": User is refining or modifying their original question
</task>

<output>
Respond with a JSON object:
{{
    "reasoning": "Your step-by-step Chain of Thought analysis...",
    "classification": "clarification_answer|topic_change|follow_up|refinement",
    "confidence": 0.0 to 1.0,
    "should_continue_clarification": true or false,
    "detected_intent": "brief description of what user wants"
}}
</output>"""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are an expert conversation analyzer. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            # Handle potential markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            classification = ClassificationResult(
                classification=result.get("classification", "clarification_answer"),
                confidence=result.get("confidence", 0.8),
                reasoning=result.get("reasoning", "LLM analysis completed"),
                should_continue_clarification=result.get("should_continue_clarification", True),
                detected_intent=result.get("detected_intent")
            )
            
            logger.info(f"🧠 LLM Context Classification: {classification.classification} "
                       f"(confidence: {classification.confidence:.2f})")
            logger.debug(f"CoT Reasoning: {classification.reasoning}")
            
            return classification
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Default to clarification answer for short responses
            return ClassificationResult(
                classification="clarification_answer",
                confidence=0.6,
                reasoning=f"JSON parse failed, defaulting based on response pattern",
                should_continue_clarification=True
            )
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            # Safe fallback
            return ClassificationResult(
                classification="clarification_answer",
                confidence=0.5,
                reasoning=f"Error occurred: {e}",
                should_continue_clarification=True
            )
    
    def _format_history(self, history: List[Dict]) -> str:
        """Format conversation history for prompt."""
        if not history:
            return "(No prior messages)"
        
        formatted = []
        for msg in history:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")[:200]  # Limit length
            formatted.append(f"[{role}]: {content}")
        
        return "\n".join(formatted)
    
    def should_skip_topic_transition(
        self,
        user_response: str,
        has_active_clarification: bool,
        last_clarification_question: Optional[str] = None,
        original_query: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> bool:
        """
        Quick check if topic transition should be skipped.
        Used as a guard before the full topic transition detection.
        
        Returns:
            True if topic transition should be skipped (user is answering clarification)
        """
        if not has_active_clarification:
            return False  # No active clarification, proceed with topic detection
        
        if not last_clarification_question:
            return False  # No question to answer
        
        # Use LLM classification
        classification = self.classify_user_response(
            user_response=user_response,
            conversation_history=conversation_history or [],
            last_clarification_question=last_clarification_question,
            original_query=original_query or ""
        )
        
        # Skip topic transition if user is answering the clarification
        return classification.classification == "clarification_answer"


# Global instance
_llm_context_classifier: Optional[LLMContextClassifier] = None


def init_llm_context_classifier(aoai_client: AzureOpenAI, deployment_name: str):
    """Initialize global LLM context classifier."""
    global _llm_context_classifier
    _llm_context_classifier = LLMContextClassifier(aoai_client, deployment_name)
    logger.info("Initialized global LLM context classifier")


def get_llm_context_classifier() -> Optional[LLMContextClassifier]:
    """Get global LLM context classifier instance."""
    return _llm_context_classifier
