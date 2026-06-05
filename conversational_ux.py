"""
Conversational UX Enhancements
Transforms technical RAG system into delightful, user-friendly experience.
Focuses on visibility, control, empathy, and proactive assistance.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from conversation_context import ConversationContext, Entity
from confidence_and_recovery import ConfidenceScore, ConfidenceLevel

logger = logging.getLogger("ConversationalUX")


class UserMood(Enum):
    """Detected user mood/emotion."""
    NEUTRAL = "neutral"
    CURIOUS = "curious"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    SATISFIED = "satisfied"
    IMPATIENT = "impatient"


class ResponseTone(Enum):
    """Tone for system responses."""
    FORMAL = "formal"
    FRIENDLY = "friendly"
    EMPATHETIC = "empathetic"
    ENCOURAGING = "encouraging"
    DIRECT = "direct"
    CELEBRATORY = "celebratory"


@dataclass
class VisibleContext:
    """User-visible context display."""
    what_i_understand: Dict[str, str]  # Entities understood
    topic: str
    confidence_level: str  # "High", "Medium", "Low"
    progress: Optional[str] = None  # "2 of 3 questions answered"
    missing_info: List[str] = field(default_factory=list)


@dataclass
class ConversationControls:
    """Available user controls."""
    can_skip_clarification: bool = True
    can_restart: bool = True
    can_undo_last: bool = True
    can_provide_more_context: bool = True
    quick_actions: List[str] = field(default_factory=list)  # ["Get examples", "More details"]


@dataclass
class ProactiveAssistance:
    """Proactive help for users."""
    suggested_follow_ups: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    helpful_examples: List[str] = field(default_factory=list)
    quick_tips: Optional[str] = None


@dataclass
class EnhancedResponse:
    """Enhanced response with UX improvements."""
    # Core response
    main_answer: str

    # Visibility
    visible_context: VisibleContext

    # Control
    controls: ConversationControls

    # Proactive assistance
    assistance: ProactiveAssistance

    # Feedback mechanism
    feedback_options: List[str] = field(default_factory=lambda: [
        "👍 This helped", "👎 Not what I needed", "❓ Still confused", "✅ Perfect, thanks"
    ])

    # Visual enhancements
    formatted_sections: Dict[str, Any] = field(default_factory=dict)

    # Emotional tone
    tone: ResponseTone = ResponseTone.FRIENDLY

    # Progress indicator
    conversation_progress: Optional[str] = None


class ContextVisualizer:
    """Makes internal context visible to users."""

    @staticmethod
    def create_visible_context(
        context: ConversationContext,
        confidence_score: Optional[ConfidenceScore] = None
    ) -> VisibleContext:
        """
        Create user-friendly representation of internal context.

        Args:
            context: Internal conversation context
            confidence_score: Optional confidence score

        Returns:
            VisibleContext for display
        """
        # Extract entities in user-friendly format
        entities = context.get_all_entities()
        what_i_understand = {}

        entity_labels = {
            'country': '🌍 Country',
            'position': '👤 Position',
            'policy_type': '📋 Policy Type',
            'duration': '⏱️ Duration',
            'department': '🏢 Department'
        }

        for entity_type, entity_value in entities.items():
            label = entity_labels.get(entity_type, entity_type.replace('_', ' ').title())
            what_i_understand[label] = entity_value

        # Determine confidence level
        confidence_level = "Medium"
        if confidence_score:
            if confidence_score.level == ConfidenceLevel.VERY_HIGH:
                confidence_level = "🟢 Very High"
            elif confidence_score.level == ConfidenceLevel.HIGH:
                confidence_level = "🟢 High"
            elif confidence_score.level == ConfidenceLevel.MEDIUM:
                confidence_level = "🟡 Medium"
            elif confidence_score.level == ConfidenceLevel.LOW:
                confidence_level = "🟠 Low"
            else:
                confidence_level = "🔴 Very Low"

        # Identify missing information
        typical_entities = ['country', 'position', 'policy_type']
        missing = []
        for entity_type in typical_entities:
            if not context.has_entity(entity_type):
                label = entity_labels.get(entity_type, entity_type)
                missing.append(label.split()[-1])  # Just the word, not emoji

        # Calculate progress if in clarification
        progress = None
        if context.turn_count > 1:
            answered = len(entities)
            total = len(typical_entities)
            if answered < total:
                progress = f"📊 {answered} of {total} details provided"

        return VisibleContext(
            what_i_understand=what_i_understand,
            topic=f"💡 {context.primary_topic.title()}" if context.primary_topic else "General HR Question",
            confidence_level=confidence_level,
            progress=progress,
            missing_info=missing
        )


class EmotionalIntelligence:
    """Detects user emotions and adapts tone accordingly."""

    def __init__(self, llm_client=None, deployment_name: str = None):
        """
        Initialize emotional intelligence.

        Args:
            llm_client: Optional LLM for advanced emotion detection
            deployment_name: Azure deployment name
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name

    def detect_mood(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        context: ConversationContext
    ) -> UserMood:
        """
        Detect user's current mood from query and history.

        Args:
            query: Current user query
            conversation_history: Recent history
            context: Conversation context

        Returns:
            Detected UserMood
        """
        query_lower = query.lower()

        # Frustration signals
        frustration_words = [
            "wrong", "not what", "still don't", "confused", "doesn't make sense",
            "not helpful", "what do you mean", "i don't understand", "this is not"
        ]
        if any(word in query_lower for word in frustration_words):
            return UserMood.FRUSTRATED

        # Confusion signals
        confusion_words = [
            "what", "how", "why", "explain", "clarify", "don't get", "not sure"
        ]
        if any(word in query_lower for word in confusion_words) and len(query.split()) > 10:
            return UserMood.CONFUSED

        # Satisfaction signals
        satisfaction_words = [
            "thanks", "thank you", "perfect", "great", "exactly", "that's what", "helpful"
        ]
        if any(word in query_lower for word in satisfaction_words):
            return UserMood.SATISFIED

        # Impatience signals (very short queries after multiple turns)
        if len(query.split()) <= 2 and context.turn_count > 3:
            return UserMood.IMPATIENT

        # Curiosity (questions with multiple clauses)
        if "?" in query and ("also" in query_lower or "and" in query_lower):
            return UserMood.CURIOUS

        return UserMood.NEUTRAL

    def adapt_tone(
        self,
        mood: UserMood,
        confidence_score: Optional[ConfidenceScore] = None
    ) -> ResponseTone:
        """
        Choose appropriate response tone based on mood and confidence.

        Args:
            mood: Detected user mood
            confidence_score: Confidence in answer

        Returns:
            Appropriate ResponseTone
        """
        # Frustrated users need empathy
        if mood == UserMood.FRUSTRATED:
            return ResponseTone.EMPATHETIC

        # Confused users need encouragement
        if mood == UserMood.CONFUSED:
            return ResponseTone.ENCOURAGING

        # Impatient users need direct answers
        if mood == UserMood.IMPATIENT:
            return ResponseTone.DIRECT

        # Satisfied users get celebration
        if mood == UserMood.SATISFIED:
            return ResponseTone.CELEBRATORY

        # High confidence + curious = friendly and detailed
        if mood == UserMood.CURIOUS and confidence_score and confidence_score.overall > 0.8:
            return ResponseTone.FRIENDLY

        # Default: friendly
        return ResponseTone.FRIENDLY

    def generate_tone_prefix(self, tone: ResponseTone, mood: UserMood) -> str:
        """Generate empathetic prefix for response."""
        if tone == ResponseTone.EMPATHETIC:
            return "I understand this can be confusing. Let me help clarify:"
        elif tone == ResponseTone.ENCOURAGING:
            return "Great question! Let me break this down for you:"
        elif tone == ResponseTone.CELEBRATORY:
            return "Excellent! Here's what you need to know:"
        elif tone == ResponseTone.DIRECT:
            return "Here's the direct answer:"
        elif tone == ResponseTone.FRIENDLY:
            return "Happy to help! "
        else:
            return ""


class ProactiveAssistant:
    """Provides proactive suggestions and assistance."""

    def __init__(self, llm_client, deployment_name: str = None):
        """
        Initialize proactive assistant.

        Args:
            llm_client: LLM client
            deployment_name: Azure deployment name
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name

    def generate_assistance(
        self,
        context: ConversationContext,
        answer: str,
        user_mood: UserMood
    ) -> ProactiveAssistance:
        """
        Generate proactive assistance based on context.

        Args:
            context: Conversation context
            answer: Generated answer
            user_mood: User's current mood

        Returns:
            ProactiveAssistance with suggestions
        """
        # Generate follow-up questions
        follow_ups = self._generate_follow_ups(context, answer)

        # Generate related topics
        related = self._generate_related_topics(context)

        # Generate helpful examples if answer is abstract
        examples = []
        if len(answer) > 500 and "example" not in answer.lower():
            examples = [f"Would you like a specific example for {context.primary_topic}?"]

        # Generate quick tip
        tip = self._generate_quick_tip(context, user_mood)

        return ProactiveAssistance(
            suggested_follow_ups=follow_ups,
            related_topics=related,
            helpful_examples=examples,
            quick_tips=tip
        )

    def _generate_follow_ups(
        self,
        context: ConversationContext,
        answer: str
    ) -> List[str]:
        """Generate intelligent follow-up question suggestions."""
        follow_ups = []
        topic = context.primary_topic
        entities = context.get_all_entities()

        # Topic-specific follow-ups
        if "leave" in topic.lower():
            follow_ups.extend([
                "How do I apply for this leave?",
                "What documents are needed?",
                "Can I extend this leave?"
            ])
        elif "insurance" in topic.lower():
            follow_ups.extend([
                "What's the coverage amount?",
                "Who is eligible?",
                "How do I make a claim?"
            ])
        elif "bonus" in topic.lower() or "commission" in topic.lower():
            follow_ups.extend([
                "When is this paid?",
                "How is this calculated?",
                "Are there any conditions?"
            ])

        # Entity-based follow-ups
        if 'country' in entities:
            other_countries = ["UAE", "Lebanon", "Egypt", "Saudi Arabia"]
            current = entities['country']
            others = [c for c in other_countries if c.lower() != current.lower()]
            if others:
                follow_ups.append(f"What about {others[0]}?")

        return follow_ups[:3]  # Limit to 3

    def _generate_related_topics(self, context: ConversationContext) -> List[str]:
        """Generate related topics user might be interested in."""
        related = []
        topic = context.primary_topic

        topic_relations = {
            'maternity leave': ['Paternity leave', 'Parental leave', 'Childcare benefits'],
            'insurance': ['Medical coverage', 'Life insurance', 'Dental insurance'],
            'bonus': ['Commission structure', 'Annual raise', 'Performance review'],
            'vacation': ['Public holidays', 'Sick leave', 'Time-off policies']
        }

        for key, values in topic_relations.items():
            if key in topic.lower():
                related.extend(values)
                break

        return related[:3]

    def _generate_quick_tip(self, context: ConversationContext, mood: UserMood) -> Optional[str]:
        """Generate a helpful quick tip."""
        if mood == UserMood.CONFUSED:
            return "💡 **Tip**: Feel free to ask for examples or specific scenarios!"

        if context.turn_count >= 3:
            return "💡 **Tip**: Type 'restart' to begin a new question anytime."

        if not context.has_entity('country'):
            return "💡 **Tip**: Policies may vary by country - let me know your location for specific info!"

        return None


class ProgressiveDisclosure:
    """Implements progressive disclosure pattern for clarifications."""

    def should_ask_all_at_once(
        self,
        missing_entities: List[str],
        user_mood: UserMood,
        user_profile: Optional[Any] = None
    ) -> bool:
        """
        Decide whether to ask all clarifications at once or step-by-step.

        Args:
            missing_entities: List of missing entities
            user_mood: User's current mood
            user_profile: Optional user profile with preferences

        Returns:
            True if should ask all at once
        """
        # Impatient users prefer all at once
        if user_mood == UserMood.IMPATIENT:
            return True

        # If user profile shows preference for all-at-once
        if user_profile and hasattr(user_profile, 'clarification_style'):
            if 'all_at_once' in user_profile.clarification_style.lower():
                return True

        # If only 2 questions, ask both
        if len(missing_entities) <= 2:
            return True

        # Otherwise, progressive (one at a time)
        return False

    def create_progressive_question(
        self,
        missing_entities: List[str],
        current_step: int,
        context: ConversationContext
    ) -> str:
        """
        Create a single clarification question with progress indicator.

        Args:
            missing_entities: List of missing entities
            current_step: Current step (0-indexed)
            context: Conversation context

        Returns:
            Clarification question with progress
        """
        total_steps = len(missing_entities)
        entity = missing_entities[current_step]

        # Progress indicator
        progress = f"**(Question {current_step + 1} of {total_steps})**"

        # Entity-specific questions
        questions = {
            'country': "Which country are you asking about?",
            'position': "What is your job position or role?",
            'policy_type': "Which specific policy or benefit?",
            'duration': "What time period or duration?",
            'department': "Which department?"
        }

        question = questions.get(entity, f"Could you specify {entity.replace('_', ' ')}?")

        return f"{progress} {question}\n\n*You can skip remaining questions by typing 'skip'*"


class VisualResponseFormatter:
    """Formats responses with visual structure."""

    @staticmethod
    def format_answer_with_sections(
        answer: str,
        context: ConversationContext,
        sources: List[Dict[str, Any]],
        visible_context: VisibleContext
    ) -> Dict[str, Any]:
        """
        Format answer into structured sections.

        Args:
            answer: Raw answer text
            context: Conversation context
            sources: Source documents
            visible_context: Visible context

        Returns:
            Structured response dictionary
        """
        sections = {}

        # Section 1: Context Summary
        sections['context_card'] = {
            'title': '📋 What I Understand',
            'content': visible_context.what_i_understand,
            'type': 'key_value'
        }

        # Section 2: Main Answer
        sections['main_answer'] = {
            'title': f'{visible_context.topic}',
            'content': answer,
            'type': 'text',
            'confidence': visible_context.confidence_level
        }

        # Section 3: Key Points (if answer is long)
        if len(answer) > 300:
            key_points = VisualResponseFormatter._extract_key_points(answer)
            if key_points:
                sections['key_points'] = {
                    'title': '🎯 Key Points',
                    'content': key_points,
                    'type': 'bullet_list'
                }

        # Section 4: Sources
        if sources:
            formatted_sources = [
                f"📄 {src.get('title', 'Document')} (Relevance: {src.get('score', 0):.0%})"
                for src in sources[:3]
            ]
            sections['sources'] = {
                'title': '📚 Sources',
                'content': formatted_sources,
                'type': 'bullet_list'
            }

        return sections

    @staticmethod
    def _extract_key_points(answer: str, max_points: int = 4) -> List[str]:
        """Extract key points from long answer."""
        # Simple extraction: look for sentences with important keywords
        sentences = answer.split('.')
        key_points = []

        important_keywords = ['must', 'required', 'need to', 'should', 'entitled', 'eligible', 'days', 'months', 'policy']

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in important_keywords):
                clean = sentence.strip()
                if clean and len(clean) > 20:
                    key_points.append(clean)
                    if len(key_points) >= max_points:
                        break

        return key_points


class ConversationController:
    """Handles conversation control actions (undo, restart, skip)."""

    def __init__(self, conversation_manager):
        """
        Initialize conversation controller.

        Args:
            conversation_manager: ConversationManager instance
        """
        self.conv_manager = conversation_manager

    def handle_control_action(
        self,
        action: str,
        user_id: str,
        context: ConversationContext
    ) -> Tuple[bool, Optional[str]]:
        """
        Handle user control action.

        Args:
            action: Control action (restart, undo, skip, more)
            user_id: User identifier
            context: Conversation context

        Returns:
            Tuple of (handled: bool, response_message: Optional[str])
        """
        action_lower = action.lower().strip()

        # Restart
        if action_lower in ['restart', 'start over', 'new question']:
            self.conv_manager.clear_history(user_id)
            return True, "🔄 Starting fresh! What would you like to know?"

        # Undo last
        if action_lower in ['undo', 'go back', 'previous']:
            history = self.conv_manager.get_history(user_id)
            if len(history) >= 2:
                # Remove last 2 messages (user + assistant)
                history = history[:-2]
                self.conv_manager.clear_history(user_id)
                for msg in history:
                    self.conv_manager.add_message(
                        user_id,
                        msg['role'],
                        msg['content'],
                        msg.get('metadata', {})
                    )
                return True, "↩️ Undid last exchange. Please continue:"
            else:
                return True, "Nothing to undo - this is the start of our conversation."

        # Skip clarifications
        if action_lower in ['skip', 'skip questions', 'no more questions']:
            return True, "⏭️ Skipping clarifications. I'll do my best with the information I have."

        # More context
        if action_lower in ['more', 'more context', 'tell you more']:
            return True, "📝 Please share any additional details that might help:"

        return False, None

    def create_controls(
        self,
        context: ConversationContext,
        confidence_score: Optional[ConfidenceScore] = None
    ) -> ConversationControls:
        """Create available controls for current state."""
        quick_actions = []

        # Add contextual quick actions
        if confidence_score and confidence_score.overall < 0.7:
            quick_actions.append("📝 Provide more details")

        if context.turn_count > 2:
            quick_actions.append("🔄 Start over")

        if context.turn_count > 1:
            quick_actions.append("↩️ Undo last")

        return ConversationControls(
            can_skip_clarification=context.turn_count > 0,
            can_restart=True,
            can_undo_last=context.turn_count > 0,
            can_provide_more_context=True,
            quick_actions=quick_actions
        )


class UXOrchestrator:
    """Master UX orchestrator - combines all UX enhancements."""

    def __init__(
        self,
        conversation_manager,
        llm_client,
        deployment_name: str = None
    ):
        """
        Initialize UX orchestrator.

        Args:
            conversation_manager: ConversationManager instance
            llm_client: LLM client
            deployment_name: Azure deployment name
        """
        self.conv_manager = conversation_manager

        # Initialize UX components
        self.context_visualizer = ContextVisualizer()
        self.emotional_intelligence = EmotionalIntelligence(llm_client, deployment_name)
        self.proactive_assistant = ProactiveAssistant(llm_client, deployment_name)
        self.progressive_disclosure = ProgressiveDisclosure()
        self.visual_formatter = VisualResponseFormatter()
        self.controller = ConversationController(conversation_manager)

        logger.info("✅ UX Orchestrator initialized")

    def enhance_response(
        self,
        raw_answer: str,
        query: str,
        conversation_history: List[Dict[str, Any]],
        context: ConversationContext,
        confidence_score: Optional[ConfidenceScore] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        user_profile: Optional[Any] = None
    ) -> EnhancedResponse:
        """
        Enhance raw answer with full UX improvements.

        Args:
            raw_answer: Raw answer text
            query: User query
            conversation_history: Conversation history
            context: Conversation context
            confidence_score: Optional confidence score
            sources: Optional source documents
            user_profile: Optional user profile

        Returns:
            EnhancedResponse with all UX enhancements
        """
        # 1. Detect user mood
        mood = self.emotional_intelligence.detect_mood(query, conversation_history, context)

        # 2. Adapt tone
        tone = self.emotional_intelligence.adapt_tone(mood, confidence_score)

        # 3. Create visible context
        visible_context = self.context_visualizer.create_visible_context(context, confidence_score)

        # 4. Generate proactive assistance
        assistance = self.proactive_assistant.generate_assistance(context, raw_answer, mood)

        # 5. Create controls
        controls = self.controller.create_controls(context, confidence_score)

        # 6. Format visually
        formatted_sections = self.visual_formatter.format_answer_with_sections(
            raw_answer,
            context,
            sources or [],
            visible_context
        )

        # 7. Add empathetic prefix
        tone_prefix = self.emotional_intelligence.generate_tone_prefix(tone, mood)
        enhanced_answer = f"{tone_prefix}\n\n{raw_answer}" if tone_prefix else raw_answer

        # 8. Calculate progress
        progress = None
        if context.turn_count > 1:
            progress = f"Turn {context.turn_count}"

        return EnhancedResponse(
            main_answer=enhanced_answer,
            visible_context=visible_context,
            controls=controls,
            assistance=assistance,
            formatted_sections=formatted_sections,
            tone=tone,
            conversation_progress=progress
        )
