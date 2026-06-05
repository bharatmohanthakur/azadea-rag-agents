"""
LLM Classifier - Zero Hardcoding Approach (Pydantic Edition)
Uses LLM for ALL classification tasks with Chain of Thought reasoning.
Enforces structured outputs using Pydantic models.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple, Type, Union
from enum import Enum
# No OpenAI/Azure SDK needed — using OciAsOpenAI adapter
import hashlib
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

logger = logging.getLogger("LLMClassifier")


class ConfidenceLevel(str, Enum):
    """Confidence levels for answers."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QueryClassificationResult(BaseModel):
    """Comprehensive result from LLM query classification."""
    # Primary classification
    query_type: str = Field(description="Type of query: greeting, question, command, clarification_answer, casual")
    complexity: str = Field(description="Complexity level: simple, moderate, complex")
    confidence: float = Field(description="Classification confidence 0.0-1.0")
    
    # Boolean flags
    is_greeting: bool = Field(default=False, description="Is this a greeting?")
    is_question: bool = Field(default=False, description="Is this a question?")
    is_clarification_answer: bool = Field(default=False, description="Is this answering a clarification question?")
    is_casual: bool = Field(default=False, description="Is this casual conversation?")
    is_frustrated: bool = Field(default=False, description="Is the user showing frustration?")
    
    # Clarification context
    requires_clarification: bool = Field(default=False, description="Does this query need clarification?")
    missing_context: List[str] = Field(default_factory=list, description="List of missing information")
    suggested_assumptions: Dict[str, str] = Field(default_factory=dict, description="Assumptions we can make")
    
    # CoT reasoning
    reasoning: str = Field(default="", description="Step-by-step reasoning for classification")


class UserProfileInfo(BaseModel):
    """Extracted user profile information."""
    role: Optional[str] = Field(default=None, description="Job title or role")
    country: Optional[str] = Field(default=None, description="Country or location")
    department: Optional[str] = Field(default=None, description="Department name")
    brand: Optional[str] = Field(default=None, description="Brand name")
    employment_type: Optional[str] = Field(default=None, description="Employment type (Full-time, etc)")
    confidence: float = Field(default=0.0, description="Extraction confidence")
    reasoning: str = Field(default="", description="Extraction reasoning")


class TopicChangeResult(BaseModel):
    """Result from topic change detection."""
    is_major_change: bool = Field(default=False, description="Is this a major topic change?")
    is_minor_shift: bool = Field(default=False, description="Is this a minor topic shift?")
    similarity: float = Field(default=1.0, description="Semantic similarity to previous topic 0-1")
    new_topic: Optional[str] = Field(default=None, description=" The new detected topic")
    old_topic: Optional[str] = Field(default=None, description="The previous topic")
    should_acknowledge: bool = Field(default=False, description="Should we acknowledge the change?")
    acknowledgment: Optional[str] = Field(default=None, description="Acknowledgment message")
    reasoning: str = Field(default="", description="Reasoning for decision")


class AnswerConfidenceResult(BaseModel):
    """Result from answer confidence assessment."""
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM, description="Overall confidence level")
    confidence_score: float = Field(default=0.5, description="Numerical confidence score 0-1")
    source_quality: str = Field(default="unknown", description="Quality assessment of sources")
    has_sufficient_context: bool = Field(default=True, description="Do we have enough context?")
    missing_info: List[str] = Field(default_factory=list, description="List of missing information")
    suggested_assumptions: Dict[str, str] = Field(default_factory=dict, description="Assumptions made")
    should_show_warning: bool = Field(default=False, description="Should we show a warning to user?")
    warning_message: Optional[str] = Field(default=None, description="Warning message content")
    reasoning: str = Field(default="", description="Confidence assessment reasoning")


class FrustrationResult(BaseModel):
    """Result from frustration detection."""
    is_frustrated: bool = Field(default=False, description="Is user frustrated?")
    confidence: float = Field(default=0.0, description="Confidence in detection")
    reasoning: str = Field(default="", description="Reasoning for detection")


class LLMClassifier:
    """
    LLM-based classifier for all classification tasks.
    Zero hardcoding - uses LLM for all decisions.
    Uses Pydantic + Tools for robust structured output.
    """
    
    def __init__(
        self,
        aoai_client,
        deployment_name: str,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 3600
    ):
        """
        Initialize LLM classifier.
        
        Args:
            aoai_client, client
            deployment_name: Model deployment name
            cache_enabled: Enable caching for repeated queries
            cache_ttl_seconds: Cache time-to-live
        """
        self.client = aoai_client
        self.deployment = deployment_name
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        
        logger.info("Initialized LLM Classifier with Pydantic structured outputs")
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached result if valid."""
        if not self.cache_enabled:
            return None
        
        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return result
            else:
                del self._cache[key]
        return None
    
    def _set_cached(self, key: str, value: Any):
        """Cache a result."""
        if self.cache_enabled:
            self._cache[key] = (value, datetime.now())

    def _call_llm_with_tools(
        self,
        prompt: str,
        result_model: Type[BaseModel],
        temperature: float = 0.1,
        max_tokens: int = 600
    ) -> Any:
        """
        Generic helper to call LLM with tools and parse result into Pydantic model.
        """
        # Convert Pydantic model to function schema
        schema = result_model.model_json_schema()
        function_name = f"return_{result_model.__name__.lower()}"
        
        tool_definition = {
            "type": "function",
            "function": {
                "name": function_name,
                "description": schema.get("description", f"Return structured {result_model.__name__}"),
                "parameters": schema
            }
        }
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are a precise classifier. Call the provided function to return your analysis."},
                    {"role": "user", "content": prompt}
                ],
                tools=[tool_definition],
                tool_choice={"type": "function", "function": {"name": function_name}},
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=10.0
            )
            
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            return result_model.model_validate(function_args)
            
        except Exception as e:
            logger.error(f"LLM tool call failed: {e}")
            raise e

    def classify_query(
        self,
        query: str,
        conversation_context: Optional[List[Dict]] = None,
        active_clarification: bool = False,
        clarification_question: Optional[str] = None,
        original_query: Optional[str] = None
    ) -> QueryClassificationResult:
        """Comprehensive query classification using LLM with structured output."""
        cache_key = self._get_cache_key("classify", query, active_clarification, clarification_question)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        context_str = ""
        if conversation_context:
            context_str = "\n".join([
                f"[{m.get('role', 'unknown').upper()}]: {m.get('content', '')[:150]}"
                for m in conversation_context[-5:]
            ])
        
        clarification_context = ""
        if active_clarification:
            clarification_context = f"""
ACTIVE CLARIFICATION SESSION:
- Original Question: {original_query or 'Unknown'}
- Clarification Asked: {clarification_question or 'Unknown'}
- User's Current Response: {query}
"""
        
        prompt = f"""Analyze the user's input and classify it comprehensively.

<input>
USER QUERY: "{query}"

CONVERSATION CONTEXT:
{context_str or "(No prior context)"}
{clarification_context}
</input>

<task>
Think step by step in the reasoning field:

1. IDENTIFY QUERY TYPE:
   - greeting: User is greeting (hi, hello, hey, good morning, etc.)
   - casual: Non-HR casual message (thanks, ok, bye, etc.)
   - question: User is asking for information
   - command: User is requesting an action
   - clarification_answer: User is answering a clarification question

2. ASSESS COMPLEXITY:
   - simple: Can be answered directly
   - moderate: May need some context
   - complex: Multi-part, needs research

3. CHECK FOR CLARIFICATION:
   - If active_clarification is True and the response seems to answer the question: is_clarification_answer=True
   - Short responses (1-3 words) during clarification are usually answers
   - Unless the response is clearly a new question or topic change

4. DETECT FRUSTRATION:
   - Words like "just", "any", "whatever", "doesn't matter" indicate frustration
   - Impatient or dismissive tone

5. IDENTIFY MISSING CONTEXT:
   - What information would help answer this query?
   - Country, role, employment type, specific policy area?
</task>
"""
        try:
            result = self._call_llm_with_tools(prompt, QueryClassificationResult)
            
            logger.info(f"🧠 LLM Classification: type={result.query_type}, "
                       f"greeting={result.is_greeting}, clarification_answer={result.is_clarification_answer}, "
                       f"complexity={result.complexity}")
            
            self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            # Safe fallback
            return QueryClassificationResult(
                query_type="question",
                complexity="simple",
                confidence=0.5,
                is_question=True,
                reasoning=f"Error occurred: {e}"
            )
    
    def detect_user_profile_info(
        self,
        text: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> UserProfileInfo:
        """Extract user profile information using structured output."""
        cache_key = self._get_cache_key("profile", text)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        history_context = ""
        if conversation_history:
            history_context = "\n".join([
                f"[{m.get('role', 'unknown').upper()}]: {m.get('content', '')[:100]}"
                for m in conversation_history[-5:]
            ])
        
        prompt = f"""Extract user profile information from this text. Be intelligent about understanding context.

<input>
TEXT: "{text}"

CONVERSATION CONTEXT:
{history_context or "(No prior context)"}
</input>

<task>
Extract ANY of the following if mentioned or implied:
- role: Job title/position (Manager, Director, Staff, etc.)
- country: Country or location (Lebanon, UAE, Saudi Arabia, etc.)
- department: Department (HR, IT, Finance, Operations, etc.)
- brand: Company brand (Zara, H&M, etc.)
- employment_type: Full-time, Part-time, Contract, etc.

RULES:
- Infer from city names: "Dubai" → country: "UAE", "Beirut" → country: "Lebanon"
- Understand variations: "I'm a manager" → role: "Manager"
- Be flexible with phrasing
- Only extract if reasonably certain
</task>
"""
        try:
            result = self._call_llm_with_tools(prompt, UserProfileInfo)
            
            extracted = [f for f in ["role", "country", "department", "brand"] 
                        if getattr(result, f)]
            if extracted:
                logger.info(f"👤 LLM Profile Extraction: {', '.join(extracted)}")
            
            self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error in profile extraction: {e}")
            return UserProfileInfo(reasoning=f"Error: {e}")
    
    def detect_topic_change(
        self,
        current_query: str,
        recent_queries: List[str],
        current_topic: Optional[str] = None
    ) -> TopicChangeResult:
        """Detect topic changes using structured output."""
        cache_key = self._get_cache_key("topic", current_query, str(recent_queries[:3]))
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        recent_str = "\n".join([f"- {q}" for q in recent_queries[-3:]]) if recent_queries else "(None)"
        
        prompt = f"""Analyze if the user changed topics in this conversation.

<input>
CURRENT QUERY: "{current_query}"

RECENT QUERIES:
{recent_str}

PREVIOUSLY TRACKED TOPIC: {current_topic or "(None)"}
</input>

<task>
Determine:
1. Is this a MAJOR topic change? (Completely different subject)
2. Is this a MINOR shift? (Related but slightly different)
3. Is this the same topic? (Continuing same discussion)
4. What's the new topic if changed?
5. Should we acknowledge the topic change?
</task>
"""
        try:
            result = self._call_llm_with_tools(prompt, TopicChangeResult)
            
            if result.is_major_change:
                logger.info(f"🔄 LLM Topic Change: {result.old_topic} → {result.new_topic}")
            
            self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error in topic change detection: {e}")
            return TopicChangeResult(reasoning=f"Error: {e}")
    
    def detect_frustration(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Tuple[bool, float, str]:
        """Detect user frustration using structured output."""
        cache_key = self._get_cache_key("frustration", query)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        history_context = ""
        if conversation_history:
            history_context = "\n".join([
                f"[{m.get('role', 'unknown').upper()}]: {m.get('content', '')[:100]}"
                for m in conversation_history[-5:]
            ])
        
        prompt = f"""Analyze if the user is showing frustration in their message.

<input>
USER MESSAGE: "{query}"

CONVERSATION CONTEXT:
{history_context or "(No prior context)"}
</input>

<task>
Look for signals of frustration:
- Impatient language ("just give me", "any", "whatever")
- Dismissive tone ("doesn't matter", "skip", "forget it")
- Repetition or emphasis ("I SAID", "again")
- Short, curt responses in context of long conversation
</task>
"""
        try:
            result = self._call_llm_with_tools(prompt, FrustrationResult)
            
            if result.is_frustrated:
                logger.info(f"😤 LLM Frustration Detected: {result.reasoning[:100]}")
            
            self._set_cached(cache_key, (result.is_frustrated, result.confidence, result.reasoning))
            return (result.is_frustrated, result.confidence, result.reasoning)
            
        except Exception as e:
            logger.error(f"Error in frustration detection: {e}")
            return (False, 0.0, f"Error: {e}")
    
    def assess_answer_confidence(
        self,
        query: str,
        answer: str,
        sources: List[Dict],
        context: str
    ) -> AnswerConfidenceResult:
        """Assess confidence in the generated answer using structured output."""
        source_names = [s.get("source", "Unknown") for s in sources[:5]] if sources else ["No sources"]
        
        prompt = f"""Assess the confidence level of this answer.

<input>
USER QUESTION: "{query}"

ANSWER PROVIDED:
{answer}

SOURCES USED:
{', '.join(source_names)}

CONTEXT USED:
{context}
</input>

<task>
Evaluate:
1. Does the answer directly address the question?
2. Is the answer well-supported by the sources?
3. Is the answer complete, or is it cut off/incomplete?
4. Are sources clearly integrated into the answer text (not just listed at the end)?
5. Is there any missing critical information that the user needs?
6. Should the user be warned about anything?

IMPORTANT: If the answer appears truncated, incomplete, or cut off mid-sentence, mark confidence as LOW and include this in the warning_message.
</task>
"""
        try:
            result = self._call_llm_with_tools(prompt, AnswerConfidenceResult)
            
            logger.info(f"📊 LLM Confidence: {result.confidence_level.value} ({result.confidence_score:.0%})")
            return result
            
        except Exception as e:
            logger.error(f"Error in confidence assessment: {e}")
            return AnswerConfidenceResult(reasoning=f"Error: {e}")
    
    def format_answer_with_confidence(
        self,
        answer: str,
        confidence: AnswerConfidenceResult,
        sources: List[Dict]
    ) -> str:
        """
        Format answer with confidence display and source references.
        
        Args:
            answer: Generated answer
            confidence: Confidence assessment result
            sources: Source documents
            
        Returns:
            Formatted answer with confidence info
        """
        # Get unique source names (top 5 unique sources, sorted by score)
        source_names = []
        if sources:
            # Sort by score (highest first) to prioritize best sources
            sorted_sources = sorted(sources, key=lambda x: x.get("score", 0), reverse=True)
            seen = set()
            for s in sorted_sources[:10]:  # Check top 10 for diversity
                source_name = s.get("source", "Unknown").replace(".md", "").replace("HRD - ", "").strip()
                if source_name and source_name not in seen:
                    source_names.append(source_name)
                    seen.add(source_name)
                    if len(source_names) >= 5:  # Top 5 unique sources
                        break
        if not source_names:
            source_names = ["General Knowledge Base"]
        
        # Build confidence display
        if confidence.confidence_level == ConfidenceLevel.HIGH:
            confidence_emoji = "📊"
            confidence_text = f"HIGH ({confidence.confidence_score:.0%})"
        elif confidence.confidence_level == ConfidenceLevel.MEDIUM:
            confidence_emoji = "📊"
            confidence_text = f"MEDIUM ({confidence.confidence_score:.0%})"
        else:
            confidence_emoji = "⚠️"
            confidence_text = f"LOW ({confidence.confidence_score:.0%})"
        
        # Build footer
        footer_parts = [
            f"\n\n---",
            f"{confidence_emoji} **Confidence:** {confidence_text}",
            f"📚 **Sources:** {', '.join(source_names)}"
        ]
        
        if confidence.should_show_warning and confidence.warning_message:
            # Format warning message more concisely
            warning_msg = confidence.warning_message
            # If warning is too long, truncate it
            if len(warning_msg) > 200:
                warning_msg = warning_msg[:197] + "..."
            footer_parts.append(f"⚠️ **Note:** {warning_msg}")
        
        # Add missing info if available and confidence is low
        if confidence.confidence_level == ConfidenceLevel.LOW and confidence.missing_info:
            missing_list = confidence.missing_info[:3]  # Limit to top 3 missing items
            if missing_list:
                missing_text = ", ".join(missing_list)
                if len(missing_text) > 150:
                    missing_text = missing_text[:147] + "..."
                footer_parts.append(f"📋 **Missing Information:** {missing_text}")
        
        if confidence.confidence_level == ConfidenceLevel.LOW:
            footer_parts.append("💡 **Tip:** Consider contacting HR for verification or requesting more specific information")
        
        footer = "\n".join(footer_parts)
        
        return answer + footer


# Global instance
_llm_classifier: Optional[LLMClassifier] = None


def init_llm_classifier(
    aoai_client,
    deployment_name: str,
    cache_enabled: bool = True
):
    """Initialize global LLM classifier."""
    global _llm_classifier
    _llm_classifier = LLMClassifier(
        aoai_client=aoai_client,
        deployment_name=deployment_name,
        cache_enabled=cache_enabled
    )
    logger.info("Initialized global LLM classifier with zero hardcoding approach")


def get_llm_classifier() -> Optional[LLMClassifier]:
    """Get global LLM classifier instance."""
    return _llm_classifier
