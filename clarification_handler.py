"""
Optimized clarification handler - eliminates duplicate code and improves maintainability.
Centralizes all clarification logic in one place with clear separation of concerns.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from clarification_tracker import ClarificationTracker, ClarificationSession, ClarificationStatus
from config import get_clarification_config

logger = logging.getLogger("ClarificationHandler")


class ClarificationHandler:
    """
    Handles all clarification logic including:
    - Session management
    - Turn limit enforcement
    - Answer processing
    - Final answer generation
    """

    def __init__(
        self,
        clarification_tracker: ClarificationTracker,
        llm_client,
        deployment_name: str
    ):
        """
        Initialize clarification handler.

        Args:
            clarification_tracker: ClarificationTracker instance
            llm_client: LLM client for generating answers
            deployment_name: Azure deployment name
        """
        self.tracker = clarification_tracker
        self.llm_client = llm_client
        self.deployment_name = deployment_name
        self.config = get_clarification_config()

    def should_force_completion(self, session: ClarificationSession) -> bool:
        """
        Determine if session should be force-completed.

        Args:
            session: Clarification session

        Returns:
            True if should force completion
        """
        # Force completion at max turns
        if session.turn_count >= self.config.max_turns - 1:  # 0-indexed: 0, 1, 2 = 3 turns
            logger.info(f"Force completion: Max turns reached ({session.turn_count + 1}/{self.config.max_turns})")
            return True

        # Check for user frustration
        if session.metadata.get("frustration_detected"):
            logger.info("Force completion: User frustration detected")
            return True

        return False

    def build_clarification_summary(self, session: ClarificationSession) -> str:
        """
        Build a summary of clarification Q&A for context.

        Args:
            session: Clarification session

        Returns:
            Formatted clarification summary
        """
        if not session.user_answers:
            return ""

        summary_parts = []
        for i, question in enumerate(session.questions_asked):
            answer = session.user_answers.get(i, "Not answered")
            summary_parts.append(f"Q{i+1}: {question}\nA: {answer}")

        return "\n".join(summary_parts)

    async def generate_final_answer(
        self,
        session: ClarificationSession,
        context: str,
        sources: List[Dict[str, Any]],
        retrieval_function = None
    ) -> Dict[str, Any]:
        """
        Generate final answer using clarification context.

        Args:
            session: Clarification session
            context: Retrieved context
            sources: Retrieved sources
            retrieval_function: Optional function to retrieve fresh context

        Returns:
            Dictionary with final_answer, sources, awaiting_clarification
        """
        # Get fresh context if retrieval function provided
        if retrieval_function:
            try:
                combined_query = session.get_combined_query() if session.user_answers else session.original_query
                logger.info(f"Retrieving fresh context with combined query: {combined_query[:100]}")
                search_result = await retrieval_function(combined_query, session.user_id)
                context = search_result.get("context", context)
                sources = search_result.get("sources", sources)
            except Exception as e:
                logger.error(f"Error retrieving fresh context: {e}, using existing context")

        # Build clarification summary
        clarification_summary = self.build_clarification_summary(session)

        # Build prompt with CoT
        system_prompt = (
            "You are a helpful HR assistant. Use step-by-step reasoning to provide accurate answers with clarification context.\n\n"
            "**STEP 1 - ANALYSIS:**\n"
            "1. Review the original question - what is being asked?\n"
            "2. Review clarification answers - what additional context was provided?\n"
            "3. Review knowledge base context - what information is available?\n"
            "4. Determine if sufficient information exists to answer\n\n"
            "**STEP 2 - ANSWER GENERATION:**\n"
            "CRITICAL RULES:\n"
            "1. ONLY use information explicitly stated in the provided context\n"
            "2. Do NOT make up, infer, or add information not in the context\n"
            "3. Do NOT use general knowledge or assumptions\n"
            "4. If context is insufficient, state that clearly\n"
            "5. Quote specific details, numbers, dates directly from context\n"
            "6. Incorporate clarification answers into your response naturally"
        )

        user_prompt = f"Original Question: {session.original_query}\n\n"
        if clarification_summary:
            user_prompt += f"Clarification Answers Provided:\n{clarification_summary}\n\n"
        user_prompt += f"Context from Knowledge Base:\n{context}\n\n"
        user_prompt += f"Based STRICTLY on the context above, provide a comprehensive answer to: {session.original_query}\n"
        user_prompt += "If the context does not contain sufficient information, say so explicitly."

        # Generate answer
        try:
            # Use Azure OpenAI client directly
            response = self.llm_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=10000
            )
            answer_text = response.choices[0].message.content

            # Complete session
            self.tracker.complete_session(session.user_id)

            return {
                "final_answer": answer_text,
                "sources": sources,
                "awaiting_clarification": False
            }

        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            # Return error message but complete session
            self.tracker.complete_session(session.user_id)
            return {
                "final_answer": f"I apologize, but I encountered an error while generating the answer: {str(e)}",
                "sources": sources,
                "awaiting_clarification": False
            }

    async def handle_clarification_turn(
        self,
        session: ClarificationSession,
        retrieval_function = None
    ) -> Dict[str, Any]:
        """
        Handle a clarification turn - check if should complete or continue.

        Args:
            session: Clarification session
            retrieval_function: Optional async function for retrieval

        Returns:
            Dictionary with response data
        """
        # Check if should force completion
        if self.should_force_completion(session):
            logger.info(f"Forcing completion for user {session.user_id}")
            return await self.generate_final_answer(
                session=session,
                context=session.rag_context,
                sources=session.sources,
                retrieval_function=retrieval_function
            )

        # Continue clarification - return existing questions
        missing_questions = session.get_missing_questions()
        if missing_questions:
            # Still have unanswered questions
            next_question_idx = missing_questions[0]
            next_question = session.questions_asked[next_question_idx]

            return {
                "final_answer": f"Thank you. {next_question}",
                "sources": session.sources,
                "awaiting_clarification": True,
                "clarifying_questions": [next_question]
            }
        else:
            # All questions answered - generate final answer
            return await self.generate_final_answer(
                session=session,
                context=session.rag_context,
                sources=session.sources,
                retrieval_function=retrieval_function
            )

    def detect_frustration(self, query: str, conversation_history: Optional[List[Dict]] = None) -> bool:
        """
        Detect if user is frustrated and wants to proceed.
        Uses LLM classifier with conversation history for natural, context-aware detection.

        Args:
            query: User's query
            conversation_history: Optional conversation history for context

        Returns:
            True if frustration detected
        """
        # Use LLM classifier for intelligent frustration detection
        from llm_classifier import get_llm_classifier
        llm_classifier = get_llm_classifier()
        
        if llm_classifier:
            try:
                is_frustrated, confidence, reasoning = llm_classifier.detect_frustration(
                    query=query,
                    conversation_history=conversation_history
                )
                
                if is_frustrated:
                    logger.info(f"😤 LLM Frustration Detected: {reasoning[:100]} (confidence: {confidence:.0%})")
                
                return is_frustrated
                
            except Exception as e:
                logger.warning(f"LLM classifier failed for frustration detection, using fallback: {e}")
        
        # Fallback to pattern matching if LLM classifier not available or fails
        query_lower = query.lower().strip()
        return any(signal in query_lower for signal in self.config.frustration_signals)

    def is_clarification_answer(self, user_id: str, query: str) -> bool:
        """
        Determine if query is a clarification answer (not a new question).
        Uses LLM classifier with conversation history and clarification context for natural detection.

        Args:
            user_id: User ID
            query: User's query

        Returns:
            True if likely a clarification answer
        """
        session = self.tracker.get_active_session(user_id)
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
                if hasattr(self.tracker, 'conv_manager'):
                    history = self.tracker.conv_manager.get_history(user_id, limit=10)
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
                logger.info(f"🧠 LLM Clarification Answer Detection: {is_clarification} "
                           f"(type={result.query_type}, reasoning: {result.reasoning[:100]})")
                
                return is_clarification
                
            except Exception as e:
                logger.warning(f"LLM classifier failed for clarification answer detection, using fallback: {e}")
        
        # Fallback to pattern matching if LLM classifier not available or fails
        query_lower = query.lower().strip()

        # New question indicators
        first_words = query_lower.split()[:2]
        if any(word in first_words for word in self.config.new_question_starters):
            logger.info(f"Not a clarification answer: starts with question word")
            return False

        # Greeting indicators
        if any(word in first_words for word in self.config.greeting_patterns) and len(query.split()) <= self.config.max_greeting_words:
            logger.info(f"Not a clarification answer: greeting detected")
            return False

        # Too long (likely a new question)
        if len(query.split()) > self.config.max_answer_length:
            logger.info(f"Not a clarification answer: too long ({len(query.split())} words)")
            return False

        # Otherwise, treat as clarification answer
        return True

    async def process_clarification_answer(
        self,
        user_id: str,
        answer: str,
        retrieval_function = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process a clarification answer and determine next action.

        Args:
            user_id: User ID
            answer: User's answer
            retrieval_function: Optional async function for retrieval

        Returns:
            Response dictionary or None if no active session
        """
        session = self.tracker.get_active_session(user_id)
        if not session:
            logger.warning(f"No active clarification session for {user_id}")
            return None

        # Check for frustration
        # Get conversation history for context-aware frustration detection
        conversation_history = None
        if hasattr(self.tracker, 'conv_manager'):
            history = self.tracker.conv_manager.get_history(user_id, limit=10)
            conversation_history = [
                {"role": msg.get("role"), "content": msg.get("content")}
                for msg in history
            ]
        
        if self.detect_frustration(answer, conversation_history):
            logger.info(f"User frustration detected: {answer[:50]}")
            session.metadata["frustration_detected"] = True
            self.tracker._save_session(session)
            # Force completion
            return await self.generate_final_answer(
                session=session,
                context=session.rag_context,
                sources=session.sources,
                retrieval_function=retrieval_function
            )

        # Add answer to session
        self.tracker.add_answer(user_id, answer)

        # Refresh session after update
        session = self.tracker.get_active_session(user_id)

        # Handle next turn
        return await self.handle_clarification_turn(session, retrieval_function)


# Global instance
_clarification_handler: Optional[ClarificationHandler] = None


def get_clarification_handler() -> Optional[ClarificationHandler]:
    """Get global clarification handler instance."""
    return _clarification_handler


def init_clarification_handler(
    clarification_tracker: ClarificationTracker,
    llm_client,
    deployment_name: str
):
    """
    Initialize global clarification handler.

    Args:
        clarification_tracker: ClarificationTracker instance
        llm_client: LLM client
        deployment_name: Azure deployment name
    """
    global _clarification_handler
    _clarification_handler = ClarificationHandler(
        clarification_tracker=clarification_tracker,
        llm_client=llm_client,
        deployment_name=deployment_name
    )
    logger.info("Initialized clarification handler")
