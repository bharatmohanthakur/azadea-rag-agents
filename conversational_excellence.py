"""
Conversational Excellence Module

Makes conversations smooth, natural, flawless, and contextual.
Inspired by best practices from Gemini, Claude, ChatGPT, and Grok.

Key Features:
- Deep context awareness
- Natural response generation
- Tone matching and personality
- Smooth topic transitions
- Proactive assistance
- Graceful error recovery
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
from openai import AzureOpenAI, OpenAI
import json
import re

logger = logging.getLogger(__name__)


class ConversationTone(Enum):
    """User's conversational tone"""
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    FRUSTRATED = "frustrated"
    URGENT = "urgent"
    CURIOUS = "curious"
    GRATEFUL = "grateful"
    CONFUSED = "confused"


class TopicTransition(Enum):
    """Type of topic transition"""
    CONTINUATION = "continuation"  # Same topic, deeper dive
    RELATED = "related"  # Related topic
    PIVOT = "pivot"  # Different but connected
    NEW_TOPIC = "new_topic"  # Completely new topic
    CALLBACK = "callback"  # Returning to earlier topic


@dataclass
class ConversationContext:
    """Rich conversation context"""
    user_id: str
    conversation_history: List[Dict[str, Any]]

    # Current state
    current_topic: Optional[str] = None
    previous_topics: List[str] = field(default_factory=list)
    user_tone: ConversationTone = ConversationTone.PROFESSIONAL

    # Memory
    mentioned_entities: Dict[str, List[str]] = field(default_factory=dict)  # type -> [values]
    asked_questions: List[str] = field(default_factory=list)
    provided_answers: List[str] = field(default_factory=list)

    # Preferences learned
    prefers_details: bool = True
    prefers_examples: bool = False
    response_length_preference: str = "medium"  # short, medium, long

    # Emotional state
    satisfaction_level: int = 5  # 1-10
    needs_encouragement: bool = False


@dataclass
class ConversationEnhancement:
    """Enhanced response with conversational improvements"""
    original_response: str
    enhanced_response: str
    improvements_made: List[str]
    tone_adjustments: List[str]
    context_references: List[str]
    follow_up_suggestions: List[str]


class ConversationalExcellence:
    """
    Makes conversations smooth, natural, and contextual.

    Inspired by:
    - Gemini: Natural, empathetic, warm
    - Claude: Thoughtful, detailed, helpful
    - ChatGPT: Engaging, clear, conversational
    - Grok: Personality, wit, character
    """

    def __init__(
        self,
        llm_client: OpenAI,
        deployment_name: str,
        personality: str = "warm_professional"
    ):
        """
        Initialize conversational excellence.

        Args:
            llm_client: Azure OpenAI client
            deployment_name: Model deployment name
            personality: "warm_professional", "friendly", "formal", "witty"
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name
        self.personality = personality

        # Conversation contexts per user
        self.contexts: Dict[str, ConversationContext] = {}

        logger.info(f"ConversationalExcellence initialized with {personality} personality")

    def get_or_create_context(
        self,
        user_id: str,
        conversation_history: List[Dict[str, Any]]
    ) -> ConversationContext:
        """Get or create conversation context for user."""
        if user_id not in self.contexts:
            self.contexts[user_id] = ConversationContext(
                user_id=user_id,
                conversation_history=conversation_history
            )
        else:
            # Update history
            self.contexts[user_id].conversation_history = conversation_history

        return self.contexts[user_id]

    def detect_user_tone(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]]
    ) -> ConversationTone:
        """
        Detect user's conversational tone from current query and history.

        Inspired by: Gemini's empathy, Claude's understanding
        """
        query_lower = query.lower()

        # Frustrated signals
        frustrated_indicators = [
            "still not", "doesn't work", "not working", "frustrated",
            "annoying", "doesn't make sense", "confused", "don't understand"
        ]
        if any(ind in query_lower for ind in frustrated_indicators):
            return ConversationTone.FRUSTRATED

        # Urgent signals
        urgent_indicators = [
            "urgent", "asap", "immediately", "right now", "emergency",
            "quickly", "fast", "hurry"
        ]
        if any(ind in query_lower for ind in urgent_indicators):
            return ConversationTone.URGENT

        # Grateful signals
        grateful_indicators = [
            "thank", "appreciate", "helpful", "great", "perfect",
            "excellent", "amazing"
        ]
        if any(ind in query_lower for ind in grateful_indicators):
            return ConversationTone.GRATEFUL

        # Confused signals
        confused_indicators = [
            "what do you mean", "don't get it", "unclear", "confusing",
            "not sure", "don't know"
        ]
        if any(ind in query_lower for ind in confused_indicators):
            return ConversationTone.CONFUSED

        # Curious signals
        if query.endswith("?") and len(query.split()) > 5:
            return ConversationTone.CURIOUS

        # Casual signals
        casual_indicators = ["hey", "hi", "btw", "lol", "haha"]
        if any(ind in query_lower for ind in casual_indicators):
            return ConversationTone.CASUAL

        # Default: Professional
        return ConversationTone.PROFESSIONAL

    def detect_topic_transition(
        self,
        current_query: str,
        context: ConversationContext
    ) -> TopicTransition:
        """
        Detect type of topic transition.

        Inspired by: ChatGPT's smooth transitions
        """
        # If no current topic, it's new
        if not context.current_topic:
            return TopicTransition.NEW_TOPIC

        # Check for callback phrases
        callback_phrases = [
            "back to", "earlier you mentioned", "going back",
            "previously", "you said before"
        ]
        if any(phrase in current_query.lower() for phrase in callback_phrases):
            return TopicTransition.CALLBACK

        # Check for continuation signals
        continuation_signals = [
            "also", "additionally", "more about", "tell me more",
            "what else", "any other", "continue"
        ]
        if any(sig in current_query.lower() for sig in continuation_signals):
            return TopicTransition.CONTINUATION

        # Check for pivot signals
        pivot_signals = [
            "but what about", "however", "on the other hand",
            "instead", "rather"
        ]
        if any(sig in current_query.lower() for sig in pivot_signals):
            return TopicTransition.PIVOT

        # Use LLM for semantic similarity
        try:
            current_topic_words = set(context.current_topic.lower().split())
            query_words = set(current_query.lower().split())
            overlap = len(current_topic_words & query_words)

            if overlap >= 2:
                return TopicTransition.RELATED
            elif overlap == 1:
                return TopicTransition.PIVOT
            else:
                return TopicTransition.NEW_TOPIC
        except:
            return TopicTransition.NEW_TOPIC

    def extract_implicit_context(
        self,
        query: str,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Extract implicit context from query based on conversation history.

        Inspired by: Claude's deep understanding, Gemini's context awareness

        Examples:
        - "What about UAE?" → Implicit: Asking about same topic (leave policy) for UAE
        - "And for managers?" → Implicit: Same query but for managers
        - "How much?" → Implicit: Asking about amount/duration from previous topic
        """
        implicit_context = {
            "implied_topic": None,
            "implied_entity": None,
            "reference_to_previous": False,
            "comparison_implied": False
        }

        query_lower = query.lower().strip()

        # Check for pronoun references
        pronouns = ["it", "that", "this", "they", "them"]
        if any(query_lower.startswith(p) for p in pronouns):
            implicit_context["reference_to_previous"] = True
            if context.current_topic:
                implicit_context["implied_topic"] = context.current_topic

        # Check for "and" or "what about" patterns
        continuation_patterns = [
            r"^and (.+)$",
            r"^what about (.+)\??$",
            r"^how about (.+)\??$",
            r"^for (.+)\??$"
        ]

        for pattern in continuation_patterns:
            match = re.match(pattern, query_lower)
            if match:
                implicit_context["reference_to_previous"] = True
                implicit_context["implied_topic"] = context.current_topic
                implicit_context["implied_entity"] = match.group(1)
                break

        # Check for comparison words
        comparison_words = ["difference", "compare", "versus", "vs", "or"]
        if any(word in query_lower for word in comparison_words):
            implicit_context["comparison_implied"] = True

        # Check for short questions that need context
        short_questions = ["how much", "how long", "when", "where", "who"]
        if any(query_lower.startswith(q) for q in short_questions) and len(query.split()) <= 3:
            implicit_context["reference_to_previous"] = True
            implicit_context["implied_topic"] = context.current_topic

        return implicit_context

    def enhance_response(
        self,
        original_response: str,
        user_query: str,
        context: ConversationContext,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationEnhancement:
        """
        Enhance response to be more natural, contextual, and conversational.

        This is the main enhancement function.

        Inspired by:
        - Gemini: Warm, natural language
        - Claude: Thoughtful, well-structured
        - ChatGPT: Engaging, helpful
        - Grok: Personality
        """
        # Detect user tone
        user_tone = self.detect_user_tone(user_query, context.conversation_history)
        context.user_tone = user_tone

        # Detect topic transition
        transition = self.detect_topic_transition(user_query, context)

        # Extract implicit context
        implicit = self.extract_implicit_context(user_query, context)

        # Build enhancement prompt
        enhancement_prompt = self._build_enhancement_prompt(
            original_response=original_response,
            user_query=user_query,
            user_tone=user_tone,
            transition=transition,
            implicit_context=implicit,
            context=context,
            metadata=metadata
        )

        try:
            # Call LLM to enhance
            response = self.llm_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at making AI responses more natural, warm, and contextual while maintaining accuracy."
                    },
                    {
                        "role": "user",
                        "content": enhancement_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback: extract JSON from markdown
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                    result = json.loads(result_text)
                else:
                    # If parsing fails, return original with minimal enhancement
                    result = {
                        "enhanced_response": original_response,
                        "improvements_made": [],
                        "tone_adjustments": [],
                        "context_references": [],
                        "follow_up_suggestions": []
                    }

            enhancement = ConversationEnhancement(
                original_response=original_response,
                enhanced_response=result.get("enhanced_response", original_response),
                improvements_made=result.get("improvements_made", []),
                tone_adjustments=result.get("tone_adjustments", []),
                context_references=result.get("context_references", []),
                follow_up_suggestions=result.get("follow_up_suggestions", [])
            )

            logger.info(f"Response enhanced with {len(enhancement.improvements_made)} improvements")

            return enhancement

        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            # Return original response
            return ConversationEnhancement(
                original_response=original_response,
                enhanced_response=original_response,
                improvements_made=[],
                tone_adjustments=[],
                context_references=[],
                follow_up_suggestions=[]
            )

    def _build_enhancement_prompt(
        self,
        original_response: str,
        user_query: str,
        user_tone: ConversationTone,
        transition: TopicTransition,
        implicit_context: Dict[str, Any],
        context: ConversationContext,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Build detailed prompt for response enhancement."""

        # Build conversation context summary
        recent_history = context.conversation_history[-6:] if context.conversation_history else []
        history_summary = ""
        if recent_history:
            history_summary = "**Recent Conversation:**\n"
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")[:100]
                history_summary += f"- {role.capitalize()}: {content}...\n"

        # Build personality guidelines based on selected personality
        personality_guide = self._get_personality_guidelines()

        # Build tone-specific guidelines
        tone_guide = self._get_tone_guidelines(user_tone)

        # Build transition guidance
        transition_guide = self._get_transition_guidelines(transition)

        prompt = f"""You are enhancing an AI assistant response to be more natural, warm, and contextual.

**Context:**
- User Query: "{user_query}"
- User Tone: {user_tone.value}
- Topic Transition: {transition.value}
- Current Topic: {context.current_topic or "None"}
- Previous Topics: {", ".join(context.previous_topics[-3:]) if context.previous_topics else "None"}

{history_summary}

**Implicit Context Detected:**
{json.dumps(implicit_context, indent=2)}

**Original Response:**
```
{original_response}
```

**Your Task:**
Enhance this response to be more natural, warm, and contextual while maintaining 100% factual accuracy.

{personality_guide}

{tone_guide}

{transition_guide}

**Enhancement Guidelines:**

1. **Natural Language:**
   - Use contractions when appropriate ("you're" vs "you are")
   - Use conversational transitions ("By the way", "Also", "Just to clarify")
   - Avoid robotic phrases like "Based on the provided context"
   - Sound human, not like a documentation bot

2. **Context References:**
   - Reference previous conversation naturally if relevant
   - If user asked about this topic before, acknowledge it
   - Use phrases like "As we discussed", "Following up on", "Building on"

3. **Tone Matching:**
   - Match user's emotional state (urgent → efficient, frustrated → empathetic)
   - Be warmer for grateful users, more careful with confused users
   - Professional but not cold, helpful but not patronizing

4. **Smooth Transitions:**
   - If topic changed, acknowledge it naturally
   - If continuing topic, don't repeat what was already said
   - If returning to earlier topic, reference it

5. **Proactive Helpfulness:**
   - Anticipate follow-up questions
   - Offer related information that might be useful
   - Suggest next steps when appropriate

6. **Response Structure:**
   - Start with a natural acknowledgment or transition
   - Provide the core information clearly
   - End with helpful next steps or offer further help

7. **Length:**
   - Keep it concise but complete
   - Don't add fluff, add value
   - User prefers: {context.response_length_preference} responses

**Output Format (JSON):**
```json
{{
    "enhanced_response": "The enhanced response text here",
    "improvements_made": ["improvement 1", "improvement 2", ...],
    "tone_adjustments": ["adjustment 1", "adjustment 2", ...],
    "context_references": ["reference 1", "reference 2", ...],
    "follow_up_suggestions": ["suggestion 1", "suggestion 2", ...]
}}
```

**Important:**
- Maintain 100% factual accuracy - don't add information not in original
- Keep all specific details, numbers, dates, procedures exactly as stated
- Only improve how the information is communicated, not what is communicated
- Respond ONLY with valid JSON

Enhance the response now:
"""

        return prompt

    def _get_personality_guidelines(self) -> str:
        """Get personality-specific guidelines."""
        personalities = {
            "warm_professional": """
**Personality: Warm Professional HR Assistant**
- Friendly and approachable yet professional
- Genuinely helpful and caring about employee needs
- Use warm language: "I'd be happy to help", "Let me assist you with that"
- Show empathy: "I understand this is important for you"
- Be encouraging: "Great question!", "I'm here to help"
""",
            "friendly": """
**Personality: Friendly HR Buddy**
- Very casual and conversational
- Use more informal language
- Show enthusiasm: "Great question!", "Absolutely!"
- Be encouraging and supportive
""",
            "formal": """
**Personality: Formal HR Professional**
- Professional and courteous
- More structured responses
- Use formal language
- Maintain professional distance
""",
            "witty": """
**Personality: Witty HR Assistant (inspired by Grok)**
- Add subtle wit when appropriate
- Keep it professional but engaging
- Occasional light humor (without being unprofessional)
- Personality without sacrificing helpfulness
"""
        }

        return personalities.get(self.personality, personalities["warm_professional"])

    def _get_tone_guidelines(self, user_tone: ConversationTone) -> str:
        """Get tone-specific response guidelines."""
        tone_guides = {
            ConversationTone.FRUSTRATED: """
**User is Frustrated - Be Extra Empathetic:**
- Acknowledge their frustration: "I understand this has been frustrating"
- Apologize if system caused confusion: "I apologize for any confusion"
- Be extra clear and direct - no fluff
- Offer to help resolve the issue step-by-step
- Show you're on their side
""",
            ConversationTone.URGENT: """
**User Needs Urgent Help:**
- Get straight to the point - no unnecessary intro
- Provide critical information first
- Be efficient and direct
- Highlight time-sensitive details
- Offer quick next steps
""",
            ConversationTone.GRATEFUL: """
**User is Grateful:**
- Accept gratitude warmly: "You're very welcome!"
- Reinforce you're happy to help
- Encourage them to come back with questions
- End on a warm note
""",
            ConversationTone.CONFUSED: """
**User is Confused:**
- Be extra clear and simple
- Break down complex info into steps
- Use examples to illustrate
- Check for understanding: "Does this help clarify?"
- Offer to explain differently
""",
            ConversationTone.CURIOUS: """
**User is Curious/Exploring:**
- Provide thorough information
- Offer context and background
- Suggest related topics they might find interesting
- Be educational and engaging
""",
            ConversationTone.CASUAL: """
**User is Casual:**
- Match their casual tone (while staying professional)
- Be friendly and conversational
- Use natural, flowing language
- Less formal structure
""",
            ConversationTone.PROFESSIONAL: """
**User is Professional:**
- Maintain professional tone
- Structured, clear responses
- Thorough and complete information
- Professional courtesy
"""
        }

        return tone_guides.get(user_tone, tone_guides[ConversationTone.PROFESSIONAL])

    def _get_transition_guidelines(self, transition: TopicTransition) -> str:
        """Get transition-specific guidelines."""
        transition_guides = {
            TopicTransition.CONTINUATION: """
**Topic Continuation:**
- Don't repeat what was already said
- Use phrases like "Additionally", "Also worth noting"
- Build on previous information
- Reference earlier points if relevant
""",
            TopicTransition.RELATED: """
**Related Topic:**
- Acknowledge the connection: "That's related to..."
- Show how topics connect
- Smooth transition phrases
""",
            TopicTransition.PIVOT: """
**Topic Pivot:**
- Acknowledge the shift: "Now looking at..."
- Transition smoothly
- Don't abruptly jump topics
""",
            TopicTransition.NEW_TOPIC: """
**New Topic:**
- Fresh start is OK
- No need to reference previous topics
- Focus on current question
""",
            TopicTransition.CALLBACK: """
**Returning to Earlier Topic:**
- Acknowledge the callback: "Coming back to your earlier question about..."
- Reference what was discussed before
- Show continuity
"""
        }

        return transition_guides.get(transition, "")

    def update_context_from_interaction(
        self,
        user_query: str,
        response: str,
        context: ConversationContext
    ):
        """Update context based on this interaction."""
        # Update current topic (extract from query)
        # Simple extraction - can be enhanced with NER
        words = user_query.lower().split()
        topic_keywords = ["leave", "insurance", "policy", "benefits", "vacation", "maternity", "paternity"]

        for keyword in topic_keywords:
            if keyword in words:
                if context.current_topic and context.current_topic != keyword:
                    context.previous_topics.append(context.current_topic)
                context.current_topic = keyword
                break

        # Track asked questions
        if "?" in user_query:
            context.asked_questions.append(user_query)

        # Track provided answers
        context.provided_answers.append(response[:100])  # Store first 100 chars

        # Update satisfaction (simple heuristic)
        if any(word in user_query.lower() for word in ["thanks", "thank", "great", "perfect"]):
            context.satisfaction_level = min(10, context.satisfaction_level + 1)
        elif any(word in user_query.lower() for word in ["confused", "frustrated", "wrong"]):
            context.satisfaction_level = max(1, context.satisfaction_level - 1)
            context.needs_encouragement = True


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-08-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    excellence = ConversationalExcellence(
        llm_client=client,
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        personality="warm_professional"
    )

    # Test enhancement
    test_history = [
        {"role": "user", "content": "What is maternity leave in Lebanon?"},
        {"role": "assistant", "content": "Maternity leave in Lebanon is 70 days fully paid."}
    ]

    context = excellence.get_or_create_context("test_user", test_history)

    # Test 1: Follow-up question
    original = "Paternity leave in Lebanon is 3 days."
    query = "What about for fathers?"

    enhancement = excellence.enhance_response(original, query, context)

    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"Original: {original}")
    print(f"\nEnhanced: {enhancement.enhanced_response}")
    print(f"\nImprovements: {enhancement.improvements_made}")
    print(f"{'='*80}\n")
