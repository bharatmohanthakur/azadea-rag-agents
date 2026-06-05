"""
Smart Conversation History Compression
Compresses long conversation histories while retaining critical context.
Different from document compression - focuses on conversation flow and context.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from conversation_context import ConversationContext, Entity

logger = logging.getLogger("ConversationCompression")


@dataclass
class CompressedConversation:
    """Result of conversation compression."""
    original_turn_count: int
    compressed_turn_count: int
    original_question: str
    entities_preserved: Dict[str, str]
    topic: str
    recent_turns: List[Dict[str, Any]]
    summary: Optional[str] = None
    compression_ratio: float = 0.0


class ConversationCompressor:
    """
    Compresses conversation history intelligently.
    Always preserves: original question, extracted entities, recent turns.
    Compresses: intermediate exchanges, redundant clarifications.
    """

    def __init__(
        self,
        llm_client,
        deployment_name: str = None,
        keep_recent_turns: int = 3,
        max_turns_before_compression: int = 8
    ):
        """
        Initialize conversation compressor.

        Args:
            llm_client: LLM client for summarization
            deployment_name: Azure deployment name
            keep_recent_turns: Number of recent turns to keep in full
            max_turns_before_compression: Start compressing after this many turns
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name
        self.keep_recent_turns = keep_recent_turns
        self.max_turns_before_compression = max_turns_before_compression

    def should_compress(
        self,
        conversation_history: List[Dict[str, Any]],
        context: Optional[ConversationContext] = None
    ) -> bool:
        """Determine if conversation should be compressed."""
        if not conversation_history:
            return False

        # Count turns (user + assistant = 1 turn)
        turn_count = len(conversation_history) // 2

        # Compress if we exceed threshold
        return turn_count > self.max_turns_before_compression

    def compress_conversation(
        self,
        conversation_history: List[Dict[str, Any]],
        context: ConversationContext
    ) -> CompressedConversation:
        """
        Compress conversation history intelligently.

        Args:
            conversation_history: Full conversation history
            context: Conversation context with entities

        Returns:
            CompressedConversation with essential information
        """
        if not conversation_history:
            return self._empty_compression()

        original_count = len(conversation_history)

        # Extract critical components
        original_question = context.original_question
        entities = context.get_all_entities()
        topic = context.primary_topic

        # Split history into old and recent
        split_point = max(0, len(conversation_history) - (self.keep_recent_turns * 2))
        old_messages = conversation_history[:split_point]
        recent_messages = conversation_history[split_point:]

        # Compress old messages if they exist
        summary = None
        if old_messages:
            summary = self._compress_old_messages(old_messages, context)

        compressed_count = len(recent_messages) + (1 if summary else 0)
        compression_ratio = compressed_count / max(original_count, 1)

        return CompressedConversation(
            original_turn_count=original_count,
            compressed_turn_count=compressed_count,
            original_question=original_question,
            entities_preserved=entities,
            topic=topic,
            recent_turns=recent_messages,
            summary=summary,
            compression_ratio=compression_ratio
        )

    def _compress_old_messages(
        self,
        old_messages: List[Dict[str, Any]],
        context: ConversationContext
    ) -> str:
        """Compress old messages into a summary."""
        # Format old messages
        conversation_text = ""
        for msg in old_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate very long messages
            if len(content) > 300:
                content = content[:300] + "..."
            conversation_text += f"{role.capitalize()}: {content}\n"

        # Extract what was learned
        entities_str = json.dumps(context.get_all_entities())

        prompt = f"""Summarize this conversation exchange in 2-3 sentences.
Focus on:
1. What clarifying questions were asked
2. What information was provided
3. Key entities/details learned

Conversation:
{conversation_text}

Entities learned: {entities_str}
Topic: {context.primary_topic}

Provide a concise summary (2-3 sentences max):"""

        try:
            model_param = self.deployment_name if self.deployment_name else "gpt-4o"
            response = self.llm_client.chat.completions.create(
                model=model_param,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )

            summary = response.choices[0].message.content.strip()
            logger.info(f"Compressed {len(old_messages)} messages into summary: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return self._simple_compression(old_messages, context)

    def _simple_compression(
        self,
        messages: List[Dict[str, Any]],
        context: ConversationContext
    ) -> str:
        """Simple fallback compression without LLM."""
        entities = context.get_all_entities()

        if entities:
            entity_parts = [f"{k}: {v}" for k, v in entities.items()]
            return f"Earlier conversation about {context.primary_topic} - Details provided: {', '.join(entity_parts)}"
        else:
            return f"Earlier conversation about {context.primary_topic}"

    def _empty_compression(self) -> CompressedConversation:
        """Return empty compression result."""
        return CompressedConversation(
            original_turn_count=0,
            compressed_turn_count=0,
            original_question="",
            entities_preserved={},
            topic="",
            recent_turns=[],
            compression_ratio=0.0
        )

    def get_compressed_history_for_llm(
        self,
        compressed: CompressedConversation
    ) -> List[Dict[str, str]]:
        """
        Get compressed history in format suitable for LLM context.

        Args:
            compressed: CompressedConversation result

        Returns:
            List of message dicts for LLM
        """
        messages = []

        # Add summary as system message if exists
        if compressed.summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation context: {compressed.summary}"
            })

        # Add entities as system message
        if compressed.entities_preserved:
            entity_parts = [f"**{k}**: {v}" for k, v in compressed.entities_preserved.items()]
            messages.append({
                "role": "system",
                "content": f"Known details: {', '.join(entity_parts)}"
            })

        # Add recent turns
        for turn in compressed.recent_turns:
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("content", "")
            })

        return messages


class ConversationMemoryRAG:
    """
    Stores conversation context in vector DB for retrieval.
    Enables ultra-long conversations by retrieving relevant past context.
    """

    def __init__(
        self,
        qdrant_client,
        embedder_client,
        collection_name: str = "conversation_memory",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize conversation memory with RAG.

        Args:
            qdrant_client: Qdrant client
            embedder_client: OpenAI client for embeddings
            collection_name: Qdrant collection for conversation memory
            embedding_model: Embedding model name
        """
        self.qdrant_client = qdrant_client
        self.embedder_client = embedder_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure conversation memory collection exists."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                from qdrant_client.models import Distance, VectorParams

                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # text-embedding-3-small dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created conversation memory collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")

    def store_conversation_turn(
        self,
        user_id: str,
        turn_number: int,
        user_query: str,
        assistant_response: str,
        context: ConversationContext,
        session_id: str
    ):
        """
        Store a conversation turn in vector DB for later retrieval.

        Args:
            user_id: User identifier
            turn_number: Turn number in conversation
            user_query: User's query
            assistant_response: Assistant's response
            context: Conversation context
            session_id: Session identifier
        """
        try:
            # Create text to embed (combine query + response + entities)
            entities_str = json.dumps(context.get_all_entities())
            text_to_embed = f"Query: {user_query}\nResponse: {assistant_response}\nEntities: {entities_str}"

            # Get embedding
            embedding = self._get_embedding(text_to_embed)

            # Store in Qdrant
            from qdrant_client.models import PointStruct
            import uuid

            point_id = str(uuid.uuid4())

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "user_id": user_id,
                            "session_id": session_id,
                            "turn_number": turn_number,
                            "user_query": user_query,
                            "assistant_response": assistant_response[:500],  # Truncate long responses
                            "topic": context.primary_topic,
                            "entities": context.get_all_entities(),
                            "timestamp": context.last_updated
                        }
                    )
                ]
            )

            logger.info(f"Stored conversation turn {turn_number} for {user_id}")

        except Exception as e:
            logger.error(f"Failed to store conversation turn: {e}")

    def retrieve_relevant_context(
        self,
        user_id: str,
        current_query: str,
        session_id: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant past conversation context.

        Args:
            user_id: User identifier
            current_query: Current query
            session_id: Current session ID
            top_k: Number of relevant turns to retrieve

        Returns:
            List of relevant past turns
        """
        try:
            # Get embedding for current query
            embedding = self._get_embedding(current_query)

            # Search Qdrant
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                query_filter={
                    "must": [
                        {"key": "user_id", "match": {"value": user_id}}
                    ],
                    "must_not": [
                        {"key": "session_id", "match": {"value": session_id}}  # Exclude current session
                    ]
                },
                limit=top_k
            )

            # Format results
            relevant_context = []
            for hit in results:
                relevant_context.append({
                    "turn_number": hit.payload.get("turn_number"),
                    "query": hit.payload.get("user_query"),
                    "response": hit.payload.get("assistant_response"),
                    "topic": hit.payload.get("topic"),
                    "entities": hit.payload.get("entities"),
                    "relevance_score": hit.score
                })

            logger.info(f"Retrieved {len(relevant_context)} relevant past turns for {user_id}")
            return relevant_context

        except Exception as e:
            logger.error(f"Failed to retrieve relevant context: {e}")
            return []

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        response = self.embedder_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
