"""
Clarification session tracking for maintaining context across multiple conversation turns.
Tracks original queries, clarifying questions, and user answers.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger("ClarificationTracker")


class ClarificationStatus(Enum):
    """Status of a clarification session."""
    AWAITING = "awaiting"  # Waiting for user answers
    COMPLETE = "complete"  # All questions answered, ready to answer
    ABANDONED = "abandoned"  # User asked new question, abandoned clarification


@dataclass
class ClarificationSession:
    """Tracks a clarification session."""
    session_id: str
    user_id: str
    original_query: str
    questions_asked: List[str]
    user_answers: Dict[int, str]  # question_index -> answer
    rag_context: str
    sources: List[Dict[str, Any]]
    status: str  # ClarificationStatus value
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    turn_count: int = 0  # Number of clarification turns (max 3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClarificationSession':
        """Create from dictionary."""
        return cls(**data)
    
    def add_answer(self, question_index: int, answer: str):
        """Add a user answer to a question."""
        self.user_answers[question_index] = answer
        self.turn_count += 1  # Increment turn count
        self.updated_at = datetime.now().isoformat()
        logger.info(f"Added answer to question {question_index}: {answer[:50]} (turn {self.turn_count})")
    
    def has_reached_max_turns(self, max_turns: int = 3) -> bool:
        """Check if clarification has reached maximum turns."""
        return self.turn_count >= max_turns
    
    def is_complete(self) -> bool:
        """Check if all questions have been answered."""
        return len(self.user_answers) >= len(self.questions_asked)
    
    def get_combined_query(self) -> str:
        """
        Combine original query with all answers into a complete query.
        Example: "maternity leave in Lebanon for Manager position"
        """
        if not self.user_answers:
            return self.original_query
        
        # Build answer summary
        answer_parts = []
        for idx, question in enumerate(self.questions_asked):
            if idx in self.user_answers:
                answer_parts.append(f"{question.strip('?')}: {self.user_answers[idx]}")
        
        answers_str = ", ".join(answer_parts)
        combined = f"{self.original_query} ({answers_str})"
        
        return combined
    
    def get_missing_questions(self) -> List[int]:
        """Get indices of questions that haven't been answered yet."""
        answered_indices = set(self.user_answers.keys())
        all_indices = set(range(len(self.questions_asked)))
        return sorted(list(all_indices - answered_indices))


class ClarificationTracker:
    """Manages clarification sessions."""
    
    def __init__(self, conversation_manager):
        """
        Initialize clarification tracker.
        
        Args:
            conversation_manager: ConversationManager instance for storage
        """
        self.conv_manager = conversation_manager
        self.active_sessions: Dict[str, ClarificationSession] = {}  # user_id -> session
    
    def create_session(
        self,
        user_id: str,
        original_query: str,
        questions: List[str],
        rag_context: str,
        sources: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ClarificationSession:
        """
        Create a new clarification session.
        
        Args:
            user_id: User identifier
            original_query: Original user query that needs clarification
            questions: List of clarifying questions
            rag_context: RAG context used to generate questions
            sources: Source documents
            metadata: Optional metadata
        
        Returns:
            Created ClarificationSession
        """
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Carry forward turn_count from any existing session (tracks chain depth)
        prior_turns = 0
        existing = self.active_sessions.get(user_id)
        if existing:
            prior_turns = existing.turn_count + 1
        else:
            # Check storage for recently completed session
            loaded = self._load_session(user_id)
            if loaded:
                prior_turns = loaded.turn_count + 1

        session = ClarificationSession(
            session_id=session_id,
            user_id=user_id,
            original_query=original_query,
            questions_asked=questions,
            user_answers={},
            rag_context=rag_context,
            sources=sources,
            status=ClarificationStatus.AWAITING.value,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata=metadata or {},
            turn_count=prior_turns  # Carry chain depth from prior session
        )
        
        self.active_sessions[user_id] = session
        self._save_session(session)
        
        logger.info(f"Created clarification session for {user_id}: {len(questions)} questions")
        return session
    
    def get_chain_depth(self, user_id: str) -> int:
        """Get current clarification chain depth for a user (how many rounds so far)."""
        existing = self.active_sessions.get(user_id)
        if existing:
            return existing.turn_count
        loaded = self._load_session(user_id)
        if loaded:
            return loaded.turn_count
        return 0

    def get_active_session(self, user_id: str) -> Optional[ClarificationSession]:
        """
        Get active clarification session for a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            Active ClarificationSession or None
        """
        # Check in-memory first
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            # Check if still valid (not too old, not abandoned)
            if session.status == ClarificationStatus.AWAITING.value:
                # Check age (abandon if older than 30 minutes)
                created = datetime.fromisoformat(session.created_at)
                if datetime.now() - created < timedelta(minutes=30):
                    return session
                else:
                    # Session expired
                    session.status = ClarificationStatus.ABANDONED.value
                    self._save_session(session)
                    del self.active_sessions[user_id]
                    return None
        
        # Try to load from conversation manager
        session = self._load_session(user_id)
        if session and session.status == ClarificationStatus.AWAITING.value:
            self.active_sessions[user_id] = session
            return session
        
        return None
    
    def add_answer(self, user_id: str, answer: str, question_index: Optional[int] = None) -> Optional[ClarificationSession]:
        """
        Add an answer to the active clarification session.
        
        Args:
            user_id: User identifier
            answer: User's answer
            question_index: Optional specific question index (if None, uses next unanswered)
        
        Returns:
            Updated ClarificationSession or None if no active session
        """
        session = self.get_active_session(user_id)
        if not session:
            return None
        
        # Determine which question this answers
        if question_index is None:
            missing = session.get_missing_questions()
            if missing:
                question_index = missing[0]  # Answer first unanswered question
            else:
                # All answered, but user sent another answer - might be clarification on answer
                question_index = len(session.questions_asked) - 1
        
        session.add_answer(question_index, answer)
        
        # Check if complete
        if session.is_complete():
            session.status = ClarificationStatus.COMPLETE.value
            logger.info(f"Clarification session complete for {user_id}")
        
        self._save_session(session)
        return session
    
    def complete_session(self, user_id: str):
        """Mark clarification session as complete."""
        session = self.get_active_session(user_id)
        if session:
            session.status = ClarificationStatus.COMPLETE.value
            session.updated_at = datetime.now().isoformat()
            self._save_session(session)
            # Keep in memory briefly, will be cleaned up
    
    def abandon_session(self, user_id: str):
        """Abandon active clarification session (user asked new question)."""
        session = self.get_active_session(user_id)
        if session:
            session.status = ClarificationStatus.ABANDONED.value
            session.updated_at = datetime.now().isoformat()
            self._save_session(session)
            if user_id in self.active_sessions:
                del self.active_sessions[user_id]
            logger.info(f"Abandoned clarification session for {user_id}")
    
    def _save_session(self, session: ClarificationSession):
        """Save session to conversation manager."""
        try:
            key = f"clarification:{session.user_id}"
            data = json.dumps(session.to_dict(), ensure_ascii=False)
            
            if self.conv_manager.redis_client:
                # Save to Redis with 1 hour TTL
                self.conv_manager.redis_client.setex(
                    key,
                    timedelta(hours=1),
                    data
                )
            else:
                # Fallback: store in conversation manager's memory
                if not hasattr(self.conv_manager, '_clarification_sessions'):
                    self.conv_manager._clarification_sessions = {}
                self.conv_manager._clarification_sessions[session.user_id] = data
        except Exception as e:
            logger.error(f"Failed to save clarification session: {e}")
    
    def _load_session(self, user_id: str) -> Optional[ClarificationSession]:
        """Load session from conversation manager."""
        try:
            key = f"clarification:{user_id}"
            
            if self.conv_manager.redis_client:
                data = self.conv_manager.redis_client.get(key)
                if data:
                    return ClarificationSession.from_dict(json.loads(data))
            else:
                # Fallback: load from memory
                if hasattr(self.conv_manager, '_clarification_sessions'):
                    data = self.conv_manager._clarification_sessions.get(user_id)
                    if data:
                        return ClarificationSession.from_dict(json.loads(data))
            
            return None
        except Exception as e:
            logger.error(f"Failed to load clarification session: {e}")
            return None
    
    def is_clarification_response(self, user_id: str, query: str) -> bool:
        """
        Detect if a user query is likely answering a clarifying question.
        Uses LLM classifier with conversation history and clarification context for natural detection.
        
        Args:
            user_id: User identifier
            query: User's query
        
        Returns:
            True if likely a clarification answer
        """
        session = self.get_active_session(user_id)
        if not session:
            return False
        
        if session.status != ClarificationStatus.AWAITING.value:
            return False
        
        # Use LLM classifier for intelligent detection with context
        from llm_classifier import get_llm_classifier
        llm_classifier = get_llm_classifier()
        
        if llm_classifier:
            try:
                # Get conversation history for context
                conversation_history = []
                if hasattr(self, 'conv_manager'):
                    history = self.conv_manager.get_history(user_id, limit=10)
                    conversation_history = [
                        {"role": msg.get("role"), "content": msg.get("content")}
                        for msg in history
                    ]
                
                # Get clarification context
                clarification_question = session.questions_asked[0] if session.questions_asked else None
                original_query = session.original_query if hasattr(session, 'original_query') else None
                
                # Use LLM classifier with full context
                result = llm_classifier.classify_query(
                    query=query,
                    conversation_context=conversation_history,
                    active_clarification=True,
                    clarification_question=clarification_question,
                    original_query=original_query
                )
                
                is_clarification = result.is_clarification_answer
                logger.info(f"🧠 LLM Clarification Detection: {is_clarification} "
                           f"(type={result.query_type}, reasoning: {result.reasoning[:100]})")
                
                return is_clarification
                
            except Exception as e:
                logger.warning(f"LLM classifier failed for clarification detection, using fallback: {e}")
        
        # Fallback to pattern matching if LLM classifier not available or fails
        query_lower = query.lower().strip()
        
        # New question indicators (NOT a clarification answer):
        question_starters = ["what", "how", "when", "where", "who", "why", "can", "is", "are", "do", "does", "will", "would", "should"]
        greeting_words = ["hi", "hello", "hey", "thanks", "thank you", "okay", "ok", "sure"]
        
        # Check if starts with question word or greeting
        first_words = query_lower.split()[:2]
        if any(word in first_words for word in question_starters):
            logger.info(f"Query '{query[:50]}' looks like a new question (starts with question word), not a clarification answer")
            return False
        
        if any(word in first_words for word in greeting_words):
            logger.info(f"Query '{query[:50]}' is a greeting, not a clarification answer")
            return False
        
        # If query is very long (>50 words), it's likely a new question
        if len(query.split()) > 50:
            logger.info(f"Query '{query[:50]}' is too long ({len(query.split())} words), likely a new question")
            return False
        
        # Otherwise, if there's an active session awaiting answers, treat as clarification answer
        return True
    
    def detect_user_frustration(self, user_id: str, query: str) -> bool:
        """
        Detect if user is frustrated or wants to proceed without providing all answers.
        
        Args:
            user_id: User identifier
            query: User's query
        
        Returns:
            True if frustration detected
        """
        session = self.get_active_session(user_id)
        if not session:
            return False
        
        query_lower = query.lower().strip()
        
        # Frustration signals
        frustration_signals = [
            "just tell me", "any", "i don't know", "i don't care", "whatever",
            "doesn't matter", "not important", "skip", "proceed", "continue",
            "just give me", "any is fine", "doesn't matter", "i don't mind"
        ]
        
        return any(signal in query_lower for signal in frustration_signals)
    
    def detect_comprehensive_answer(self, user_id: str, query: str) -> bool:
        """
        Detect if user provided a comprehensive answer covering multiple questions.
        
        Args:
            user_id: User identifier
            query: User's query
        
        Returns:
            True if comprehensive answer detected
        """
        session = self.get_active_session(user_id)
        if not session:
            return False
        
        # Check if query contains multiple pieces of information
        has_multiple_parts = (
            "," in query or 
            " and " in query.lower() or 
            len(query.split()) > 5
        )
        
        return has_multiple_parts

