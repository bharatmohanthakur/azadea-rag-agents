"""
Persistent conversation management with Redis fallback to in-memory storage.
Maintains conversation history across server restarts.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ConversationManager")

# Try to import Redis, fallback to in-memory if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory storage (conversations will be lost on restart)")

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
CONVERSATION_TTL_DAYS = int(os.getenv("CONVERSATION_TTL_DAYS", "30"))  # 30 days default


class ConversationManager:
    """Manages conversation history with persistent storage."""
    
    def __init__(self):
        self.redis_client = None
        self.memory_fallback: Dict[str, List[Dict[str, Any]]] = {}
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    password=REDIS_PASSWORD,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}. Using in-memory fallback.")
                self.redis_client = None
        else:
            logger.info("Using in-memory conversation storage (Redis not installed)")
    
    def _get_key(self, user_id: str) -> str:
        """Generate Redis key for user conversation."""
        return f"conversation:{user_id}"

    def _get_profile_key(self, user_id: str) -> str:
        return f"user_profile:{user_id}"

    def update_user_profile(self, user_id: str, attrs: Dict[str, Any]) -> None:
        """Merge attrs into the user's profile. 7-day TTL."""
        if not attrs:
            return
        clean = {k: str(v) for k, v in attrs.items() if v is not None and str(v).strip()}
        if not clean:
            return
        try:
            if self.redis_client:
                key = self._get_profile_key(user_id)
                self.redis_client.hset(key, mapping=clean)
                self.redis_client.expire(key, 86400 * 7)
            else:
                self.memory_fallback.setdefault(f"_profile_{user_id}", {}).update(clean)
        except Exception as e:
            logger.error(f"Error updating profile for {user_id}: {e}")

    def get_user_profile(self, user_id: str) -> Dict[str, str]:
        """Return the user's stored profile attributes (empty dict if none)."""
        try:
            if self.redis_client:
                data = self.redis_client.hgetall(self._get_profile_key(user_id))
                return data or {}
            return self.memory_fallback.get(f"_profile_{user_id}", {}).copy()
        except Exception as e:
            logger.error(f"Error reading profile for {user_id}: {e}")
            return {}
    
    def get_history(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.
        
        Args:
            user_id: User identifier
            limit: Optional limit on number of messages to return (most recent)
        
        Returns:
            List of conversation messages
        """
        try:
            if self.redis_client:
                # Get from Redis
                key = self._get_key(user_id)
                data = self.redis_client.get(key)
                if data:
                    history = json.loads(data)
                    if limit:
                        return history[-limit:] if limit > 0 else []
                    return history
            else:
                # Fallback to in-memory
                if user_id in self.memory_fallback:
                    history = self.memory_fallback[user_id]
                    if limit:
                        return history[-limit:] if limit > 0 else []
                    return history
            
            return []
        except Exception as e:
            logger.error(f"Error getting history for {user_id}: {e}")
            # Fallback to in-memory
            if user_id in self.memory_fallback:
                return self.memory_fallback[user_id][-limit:] if limit else self.memory_fallback[user_id]
            return []
    
    def add_message(self, user_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a message to conversation history.
        
        Args:
            user_id: User identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata (e.g., sources, request_id)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        try:
            if self.redis_client:
                # Get existing history
                key = self._get_key(user_id)
                data = self.redis_client.get(key)
                history = json.loads(data) if data else []
                
                # Add new message
                history.append(message)
                
                # Limit history size (keep last 100 messages per user)
                if len(history) > 100:
                    history = history[-100:]
                
                # Save back to Redis with TTL
                self.redis_client.setex(
                    key,
                    timedelta(days=CONVERSATION_TTL_DAYS),
                    json.dumps(history, ensure_ascii=False)
                )
            else:
                # Fallback to in-memory
                if user_id not in self.memory_fallback:
                    self.memory_fallback[user_id] = []
                
                self.memory_fallback[user_id].append(message)
                
                # Limit history size
                if len(self.memory_fallback[user_id]) > 100:
                    self.memory_fallback[user_id] = self.memory_fallback[user_id][-100:]
        except Exception as e:
            logger.error(f"Error adding message for {user_id}: {e}")
            # Fallback to in-memory
            if user_id not in self.memory_fallback:
                self.memory_fallback[user_id] = []
            self.memory_fallback[user_id].append(message)
    
    def clear_history(self, user_id: Optional[str] = None):
        """
        Clear conversation history.
        
        Args:
            user_id: If provided, clear only this user's history. If None, clear all.
        """
        try:
            if self.redis_client:
                if user_id:
                    key = self._get_key(user_id)
                    self.redis_client.delete(key)
                else:
                    # Clear all conversation keys (use pattern matching)
                    keys = self.redis_client.keys("conversation:*")
                    if keys:
                        self.redis_client.delete(*keys)
            else:
                # Fallback to in-memory
                if user_id:
                    if user_id in self.memory_fallback:
                        del self.memory_fallback[user_id]
                else:
                    self.memory_fallback.clear()
        except Exception as e:
            logger.error(f"Error clearing history for {user_id}: {e}")
    
    def get_conversation_summary(self, user_id: str, max_turns: int = 10) -> str:
        """
        Get a summary of recent conversation turns.
        
        Args:
            user_id: User identifier
            max_turns: Maximum number of turns to summarize
        
        Returns:
            Summary string of recent conversation
        """
        history = self.get_history(user_id, limit=max_turns * 2)  # Get messages for max_turns
        
        if not history:
            return ""
        
        summary_parts = []
        for msg in history[-max_turns * 2:]:  # Last max_turns turns
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            summary_parts.append(f"{role.capitalize()}: {content[:200]}")  # Truncate long messages
        
        return "\n".join(summary_parts)
    
    def is_redis_available(self) -> bool:
        """Check if Redis is available and connected."""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def get_conversation_with_summary(self, user_id: str, max_turns: int = 10) -> Dict[str, Any]:
        """
        Get conversation history with optional summarization.

        Args:
            user_id: User identifier
            max_turns: Maximum turns to return

        Returns:
            Dictionary with 'full_history', 'summary', and 'recent_messages'
        """
        full_history = self.get_history(user_id)

        if len(full_history) <= max_turns:
            return {
                "full_history": full_history,
                "summary": None,
                "recent_messages": full_history
            }

        # Split into old and recent
        split_point = len(full_history) - max_turns
        old_messages = full_history[:split_point]
        recent_messages = full_history[split_point:]

        return {
            "full_history": full_history,
            "summary": None,  # Will be set by ConversationSummarizer
            "recent_messages": recent_messages,
            "old_messages": old_messages
        }

    def get_original_question(self, user_id: str, within_last_n: int = 10) -> Optional[str]:
        """
        Get the original user question from recent conversation history.
        Looks for the first substantive user question (not greetings or clarification answers).

        Args:
            user_id: User identifier
            within_last_n: Look within the last N messages

        Returns:
            Original question text or None
        """
        history = self.get_history(user_id, limit=within_last_n)

        if not history:
            return None

        # Define greeting patterns to skip
        greeting_patterns = ["hi", "hello", "hey", "thanks", "thank you", "okay", "ok", "sure", "great", "awesome", "perfect"]

        # Look for the first substantive user question (working backwards from recent)
        for msg in reversed(history):
            if msg.get("role") == "user":
                content = msg.get("content", "").strip().lower()

                # Skip obvious greetings
                is_greeting = any(pattern in content for pattern in greeting_patterns) and len(content.split()) <= 5

                # Skip very short answers (likely clarification responses)
                is_short_answer = len(content.split()) <= 3 and not content.endswith("?")

                if not is_greeting and not is_short_answer:
                    # Check if metadata marks this as the original question
                    metadata = msg.get("metadata", {})
                    if metadata.get("is_original_question", False):
                        return msg.get("content", "")

        # Fallback: return the first non-greeting user message
        for msg in history:
            if msg.get("role") == "user":
                content = msg.get("content", "").strip().lower()
                is_greeting = any(pattern in content for pattern in greeting_patterns) and len(content.split()) <= 5
                if not is_greeting:
                    return msg.get("content", "")

        return None

    def mark_as_original_question(self, user_id: str):
        """
        Mark the most recent user message as the original question.
        This helps track the initial query across multiple clarification turns.

        Args:
            user_id: User identifier
        """
        try:
            history = self.get_history(user_id)
            if not history:
                return

            # Find the most recent user message and mark it
            for i in range(len(history) - 1, -1, -1):
                if history[i].get("role") == "user":
                    history[i]["metadata"] = history[i].get("metadata", {})
                    history[i]["metadata"]["is_original_question"] = True

                    # Save updated history
                    if self.redis_client:
                        key = self._get_key(user_id)
                        self.redis_client.setex(
                            key,
                            timedelta(days=CONVERSATION_TTL_DAYS),
                            json.dumps(history, ensure_ascii=False)
                        )
                    else:
                        self.memory_fallback[user_id] = history

                    logger.info(f"Marked message as original question for user {user_id}")
                    break
        except Exception as e:
            logger.error(f"Error marking original question for {user_id}: {e}")


# Global instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get or create the global conversation manager instance."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager

