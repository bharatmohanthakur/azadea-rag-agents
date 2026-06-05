"""
Conversation Context Tracking System
Tracks entities, topics, constraints, and conversation state for multi-turn conversations.
Best-in-class implementation with explicit entity tracking and topic extraction.
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger("ConversationContext")


class ConversationState(Enum):
    """Conversation states for state machine."""
    INITIAL_QUESTION = "initial"
    CLARIFYING = "clarifying"
    ANSWERING = "answering"
    FOLLOW_UP = "follow_up"
    TOPIC_SWITCH = "topic_switch"
    REFINEMENT = "refinement"
    COMPARISON = "comparison"


class IntentRelationship(Enum):
    """Relationship between current and original intent."""
    CONTINUATION = "continuation"  # Same topic, more details
    RELATED = "related"  # Related topic, new question
    COMPARISON = "comparison"  # Comparing to original
    UNRELATED = "unrelated"  # New topic entirely


@dataclass
class Entity:
    """Represents an extracted entity."""
    type: str  # country, position, policy_type, date, etc.
    value: str
    confidence: float
    turn: int  # Which turn was this extracted from
    source: str  # "user_query", "clarification_answer", "extracted"


@dataclass
class ConversationContext:
    """
    Rich context tracking for conversations.
    Tracks entities, topics, state, and relationships.
    """
    user_id: str
    session_id: str
    original_question: str
    current_state: str  # ConversationState value

    # Entity tracking
    entities: Dict[str, List[Entity]] = field(default_factory=dict)

    # Topic tracking
    primary_topic: Optional[str] = None  # "maternity leave", "insurance", etc.
    sub_topics: List[str] = field(default_factory=list)

    # Constraints and filters
    constraints: List[str] = field(default_factory=list)

    # Confidence scoring
    context_confidence: float = 1.0

    # Turn tracking
    turn_count: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    # Intent relationship
    intent_relationship: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert entities to serializable format
        data['entities'] = {
            k: [asdict(e) for e in v]
            for k, v in self.entities.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Create from dictionary."""
        # Convert entities back to Entity objects
        if 'entities' in data and data['entities']:
            data['entities'] = {
                k: [Entity(**e) for e in v]
                for k, v in data['entities'].items()
            }
        return cls(**data)

    def add_entity(self, entity: Entity):
        """Add an entity to the context."""
        if entity.type not in self.entities:
            self.entities[entity.type] = []

        # Avoid duplicates - check if entity already exists
        existing = [e for e in self.entities[entity.type] if e.value.lower() == entity.value.lower()]
        if not existing:
            self.entities[entity.type].append(entity)
            logger.info(f"Added entity: {entity.type}={entity.value} (confidence={entity.confidence:.2f})")
        else:
            # Update confidence if higher
            if entity.confidence > existing[0].confidence:
                existing[0].confidence = entity.confidence
                logger.info(f"Updated entity confidence: {entity.type}={entity.value} (confidence={entity.confidence:.2f})")

    def get_entity(self, entity_type: str) -> Optional[Entity]:
        """Get the most recent entity of a type."""
        if entity_type in self.entities and self.entities[entity_type]:
            # Return most confident entity
            return max(self.entities[entity_type], key=lambda e: e.confidence)
        return None

    def get_all_entities(self) -> Dict[str, str]:
        """Get all entities as a simple dict."""
        result = {}
        for entity_type, entities_list in self.entities.items():
            if entities_list:
                # Get most confident entity
                best = max(entities_list, key=lambda e: e.confidence)
                result[entity_type] = best.value
        return result

    def has_entity(self, entity_type: str) -> bool:
        """Check if entity type exists."""
        return entity_type in self.entities and len(self.entities[entity_type]) > 0

    def get_combined_query(self) -> str:
        """Generate a combined query with all context."""
        parts = [self.original_question]

        # Add entities
        entities_dict = self.get_all_entities()
        if entities_dict:
            entity_parts = [f"{k}: {v}" for k, v in entities_dict.items()]
            parts.append(f"({', '.join(entity_parts)})")

        # Add constraints
        if self.constraints:
            parts.extend(self.constraints)

        return " ".join(parts)

    def update_state(self, new_state: ConversationState):
        """Update conversation state."""
        old_state = self.current_state
        self.current_state = new_state.value
        self.last_updated = datetime.now().isoformat()
        logger.info(f"State transition: {old_state} -> {new_state.value}")

    def increment_turn(self):
        """Increment turn count."""
        self.turn_count += 1
        self.last_updated = datetime.now().isoformat()


class EntityExtractor:
    """
    Extracts entities from queries using LLM and pattern matching.
    Best-in-class entity extraction with both rule-based and AI-based methods.
    """

    def __init__(self, llm_client, deployment_name: str = None):
        """
        Initialize entity extractor.

        Args:
            llm_client: LLM client for extraction
            deployment_name: Azure deployment name
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name

        # Entity patterns (rule-based fallback)
        self.entity_patterns = {
            'country': [
                'lebanon', 'uae', 'egypt', 'saudi', 'kuwait', 'qatar',
                'jordan', 'bahrain', 'oman', 'united arab emirates',
                'saudi arabia', 'ksa'
            ],
            'position': [
                'manager', 'director', 'employee', 'supervisor', 'executive',
                'staff', 'ceo', 'cfo', 'vp', 'head', 'senior', 'junior',
                'coordinator', 'specialist', 'analyst', 'associate'
            ],
            'policy_type': [
                'leave', 'vacation', 'maternity', 'paternity', 'sick leave',
                'annual leave', 'insurance', 'health', 'dental', 'medical',
                'bonus', 'commission', 'salary', 'compensation', 'relocation',
                'visa', 'benefits', 'retirement', 'pension'
            ],
            'duration': [
                'days?', 'weeks?', 'months?', 'years?',
                r'\d+\s*days?', r'\d+\s*weeks?', r'\d+\s*months?'
            ]
        }

    def extract_entities(
        self,
        query: str,
        turn: int = 0,
        context: Optional[ConversationContext] = None
    ) -> List[Entity]:
        """
        Extract entities from query using both AI and rules.

        Args:
            query: Query text
            turn: Current turn number
            context: Optional existing context for disambiguation

        Returns:
            List of extracted entities
        """
        entities = []

        # Try LLM extraction first (more accurate)
        try:
            llm_entities = self._extract_with_llm(query, context)
            entities.extend(llm_entities)
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")

        # Fallback to pattern matching
        pattern_entities = self._extract_with_patterns(query, turn)

        # Merge entities (avoid duplicates)
        for pe in pattern_entities:
            if not any(e.type == pe.type and e.value.lower() == pe.value.lower() for e in entities):
                entities.append(pe)

        return entities

    def _extract_with_llm(
        self,
        query: str,
        context: Optional[ConversationContext] = None
    ) -> List[Entity]:
        """Extract entities using LLM."""
        # Build context string
        context_str = ""
        if context and context.entities:
            existing_entities = context.get_all_entities()
            context_str = f"\nExisting entities: {json.dumps(existing_entities)}"

        prompt = f"""Extract structured entities from this HR-related query.

Query: "{query}"{context_str}

Extract these entity types:
- country: Country name (e.g., Lebanon, UAE, Egypt)
- position: Job position/role (e.g., Manager, Employee, Director)
- policy_type: Type of policy (e.g., maternity leave, insurance, bonus)
- duration: Time duration (e.g., 90 days, 3 months)
- department: Department name
- other: Any other relevant entities

Respond in JSON format:
{{
    "entities": [
        {{"type": "country", "value": "Lebanon", "confidence": 0.95}},
        {{"type": "position", "value": "Manager", "confidence": 0.90}}
    ]
}}

If no entities found, return {{"entities": []}}"""

        try:
            model_param = self.deployment_name if self.deployment_name else "gpt-4o"
            response = self.llm_client.chat.completions.create(
                model=model_param,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            entities = []

            for entity_data in result.get("entities", []):
                entity = Entity(
                    type=entity_data["type"],
                    value=entity_data["value"],
                    confidence=entity_data.get("confidence", 0.8),
                    turn=0,
                    source="llm_extracted"
                )
                entities.append(entity)

            logger.info(f"LLM extracted {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error(f"LLM entity extraction error: {e}")
            return []

    def _extract_with_patterns(self, query: str, turn: int) -> List[Entity]:
        """Extract entities using pattern matching."""
        entities = []
        query_lower = query.lower()

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                # Simple keyword matching for most
                if entity_type in ['country', 'position', 'policy_type']:
                    if pattern in query_lower:
                        # Extract the actual word
                        match = re.search(r'\b' + pattern + r'\b', query_lower)
                        if match:
                            entity = Entity(
                                type=entity_type,
                                value=match.group().title(),
                                confidence=0.7,  # Lower confidence for pattern matching
                                turn=turn,
                                source="pattern_matched"
                            )
                            entities.append(entity)
                # Regex for duration
                elif entity_type == 'duration':
                    matches = re.findall(pattern, query_lower)
                    for match in matches:
                        entity = Entity(
                            type=entity_type,
                            value=match,
                            confidence=0.8,
                            turn=turn,
                            source="pattern_matched"
                        )
                        entities.append(entity)

        return entities


class TopicExtractor:
    """Extracts primary topic from queries."""

    def __init__(self, llm_client, deployment_name: str = None):
        self.llm_client = llm_client
        self.deployment_name = deployment_name

    def extract_topic(self, query: str, context: Optional[ConversationContext] = None) -> str:
        """
        Extract primary topic from query.

        Args:
            query: Query text
            context: Optional existing context

        Returns:
            Primary topic string
        """
        # Quick pattern matching for common topics
        query_lower = query.lower()
        common_topics = {
            'maternity': 'maternity leave',
            'paternity': 'paternity leave',
            'vacation': 'vacation policy',
            'insurance': 'insurance benefits',
            'bonus': 'bonus compensation',
            'commission': 'commission policy',
            'relocation': 'relocation benefits',
            'visa': 'visa sponsorship',
            'sick': 'sick leave',
            'annual leave': 'annual leave',
        }

        for keyword, topic in common_topics.items():
            if keyword in query_lower:
                return topic

        # Fallback to LLM for complex topics
        try:
            return self._extract_with_llm(query, context)
        except Exception as e:
            logger.error(f"Topic extraction error: {e}")
            return "general HR policy"

    def _extract_with_llm(self, query: str, context: Optional[ConversationContext]) -> str:
        """Extract topic using LLM."""
        context_str = ""
        if context and context.primary_topic:
            context_str = f"\nCurrent topic: {context.primary_topic}"

        prompt = f"""What is the primary topic of this HR query?{context_str}

Query: "{query}"

Respond with just the topic name (2-4 words max). Examples:
- maternity leave
- health insurance
- bonus policy
- relocation benefits

Topic:"""

        try:
            model_param = self.deployment_name if self.deployment_name else "gpt-4o"
            response = self.llm_client.chat.completions.create(
                model=model_param,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )

            topic = response.choices[0].message.content.strip().lower()
            return topic

        except Exception as e:
            logger.error(f"LLM topic extraction error: {e}")
            return "general HR policy"


class ContextManager:
    """
    Manages conversation context lifecycle.
    Integrates entity extraction, topic tracking, and state management.
    """

    def __init__(self, conversation_manager, llm_client, deployment_name: str = None):
        """
        Initialize context manager.

        Args:
            conversation_manager: ConversationManager instance
            llm_client: LLM client
            deployment_name: Azure deployment name
        """
        self.conv_manager = conversation_manager
        self.entity_extractor = EntityExtractor(llm_client, deployment_name)
        self.topic_extractor = TopicExtractor(llm_client, deployment_name)

        # In-memory cache of active contexts
        self._contexts: Dict[str, ConversationContext] = {}

    def get_or_create_context(self, user_id: str, query: str) -> ConversationContext:
        """
        Get existing context or create new one.

        Args:
            user_id: User identifier
            query: Current query

        Returns:
            ConversationContext
        """
        # Try to load existing context
        context = self._load_context(user_id)

        if context:
            # Check if this is a new question (topic switch)
            if self._is_topic_switch(query, context):
                logger.info(f"Topic switch detected for {user_id}, creating new context")
                context = self._create_new_context(user_id, query)
            else:
                # Continue existing context
                context.increment_turn()
        else:
            # Create new context
            context = self._create_new_context(user_id, query)

        # Update context with current query
        self._update_context_from_query(context, query)

        # Cache and save
        self._contexts[user_id] = context
        self._save_context(context)

        return context

    def _create_new_context(self, user_id: str, query: str) -> ConversationContext:
        """Create a new conversation context."""
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        context = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            original_question=query,
            current_state=ConversationState.INITIAL_QUESTION.value,
            turn_count=1
        )

        # Extract initial topic
        context.primary_topic = self.topic_extractor.extract_topic(query)

        logger.info(f"Created new context for {user_id}: topic={context.primary_topic}")
        return context

    def _update_context_from_query(self, context: ConversationContext, query: str):
        """Update context with entities from current query."""
        # Extract entities
        entities = self.entity_extractor.extract_entities(
            query,
            turn=context.turn_count,
            context=context
        )

        # Add entities to context
        for entity in entities:
            context.add_entity(entity)

        # Update topic if needed
        if not context.primary_topic:
            context.primary_topic = self.topic_extractor.extract_topic(query, context)

    def _is_topic_switch(self, query: str, context: ConversationContext) -> bool:
        """Detect if query represents a topic switch."""
        # Extract topic from new query
        new_topic = self.topic_extractor.extract_topic(query, context)

        # Check if significantly different from current topic
        if context.primary_topic:
            # Simple heuristic: if topics share no common words, it's a switch
            current_words = set(context.primary_topic.lower().split())
            new_words = set(new_topic.lower().split())

            common_words = current_words & new_words

            # If less than 30% overlap, it's likely a topic switch
            if len(common_words) / max(len(current_words), len(new_words)) < 0.3:
                logger.info(f"Topic switch detected: '{context.primary_topic}' -> '{new_topic}'")
                return True

        return False

    def _save_context(self, context: ConversationContext):
        """Save context to storage."""
        try:
            key = f"context:{context.user_id}"
            data = json.dumps(context.to_dict(), ensure_ascii=False)

            if self.conv_manager.redis_client:
                from datetime import timedelta
                self.conv_manager.redis_client.setex(
                    key,
                    timedelta(hours=2),
                    data
                )
            else:
                # Fallback to memory
                if not hasattr(self.conv_manager, '_contexts'):
                    self.conv_manager._contexts = {}
                self.conv_manager._contexts[context.user_id] = data
        except Exception as e:
            logger.error(f"Failed to save context: {e}")

    def _load_context(self, user_id: str) -> Optional[ConversationContext]:
        """Load context from storage."""
        # Check cache first
        if user_id in self._contexts:
            return self._contexts[user_id]

        # Load from storage
        try:
            key = f"context:{user_id}"

            if self.conv_manager.redis_client:
                data = self.conv_manager.redis_client.get(key)
                if data:
                    return ConversationContext.from_dict(json.loads(data))
            else:
                # Fallback to memory
                if hasattr(self.conv_manager, '_contexts'):
                    data = self.conv_manager._contexts.get(user_id)
                    if data:
                        return ConversationContext.from_dict(json.loads(data))

            return None
        except Exception as e:
            logger.error(f"Failed to load context: {e}")
            return None

    def clear_context(self, user_id: str):
        """Clear context for user."""
        if user_id in self._contexts:
            del self._contexts[user_id]

        try:
            key = f"context:{user_id}"
            if self.conv_manager.redis_client:
                self.conv_manager.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Failed to clear context: {e}")
