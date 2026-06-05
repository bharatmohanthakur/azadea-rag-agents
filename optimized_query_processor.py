"""
Optimized query processor with intelligent caching and efficient rewriting.
Replaces scattered query processing logic with a centralized, optimized approach.
"""

import logging
from typing import List, Dict, Any, Optional
from config import get_query_processing_config
from pattern_matcher import get_pattern_matcher
from query_cache import get_query_cache

logger = logging.getLogger("OptimizedQueryProcessor")


class OptimizedQueryProcessor:
    """
    Centralized query processor with:
    - Intelligent caching
    - Efficient query rewriting
    - Pattern-based optimizations
    - History management
    """

    def __init__(self, llm_client, deployment_name: str):
        """
        Initialize query processor.

        Args:
            llm_client: LLM client for query rewriting
            deployment_name: Azure deployment name
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name
        self.config = get_query_processing_config()
        self.pattern_matcher = get_pattern_matcher()
        self.cache = get_query_cache()

    def filter_history_for_rewriting(
        self,
        history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Filter conversation history to keep only relevant messages for rewriting.
        Removes greetings, casual messages, and other noise.

        Args:
            history: Full conversation history

        Returns:
            Filtered history
        """
        if not history:
            return []

        filtered = []

        for msg in history[-self.config.max_history_for_rewrite:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "").strip()

            # Skip empty messages
            if not content:
                continue

            # For user messages, check if it's greeting/casual
            if role == "user":
                if self.pattern_matcher.is_greeting_or_casual(content):
                    # Skip greetings/casual unless it's very recent (last message)
                    if msg != history[-1]:
                        continue

            # Include the message
            filtered.append(msg)

        return filtered

    def needs_rewriting(
        self,
        query: str,
        history: List[Dict[str, str]]
    ) -> bool:
        """
        Determine if query needs rewriting based on history and content.

        Args:
            query: User query
            history: Conversation history

        Returns:
            True if query should be rewritten
        """
        # No history = no need to rewrite
        if not history or len(history) == 0:
            return False

        # Greeting/casual = don't rewrite
        if self.pattern_matcher.is_greeting_or_casual(query):
            return False

        # Check if query has pronouns or references that need resolution
        query_lower = query.lower()
        pronouns = ["it", "they", "them", "this", "that", "these", "those", "he", "she"]
        has_pronouns = any(f" {p} " in f" {query_lower} " for p in pronouns)

        # Check for follow-up indicators
        followup_indicators = ["what about", "how about", "and what", "also", "additionally"]
        is_followup = any(ind in query_lower for ind in followup_indicators)

        # Rewrite if has pronouns or is a follow-up
        needs_rewrite = has_pronouns or is_followup

        if needs_rewrite:
            logger.debug(f"Query needs rewriting (pronouns: {has_pronouns}, followup: {is_followup})")

        return needs_rewrite

    def rewrite_query(
        self,
        query: str,
        history: List[Dict[str, str]],
        user_id: Optional[str] = None
    ) -> str:
        """
        Rewrite query with conversation history for better context.
        Uses caching to avoid redundant LLM calls.

        Args:
            query: User query
            history: Conversation history
            user_id: Optional user ID for context

        Returns:
            Rewritten query (or original if rewriting not needed)
        """
        # Check if rewriting is needed
        if not self.needs_rewriting(query, history):
            logger.debug(f"Query doesn't need rewriting: {query[:50]}")
            return query

        # Filter history
        filtered_history = self.filter_history_for_rewriting(history)

        if not filtered_history:
            logger.debug("No relevant history for rewriting")
            return query

        # Build history context
        history_str = ""
        for msg in filtered_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Limit each message to 200 chars to avoid token bloat
            history_str += f"{role}: {content[:200]}\n"

        # Build rewriting prompt
        prompt = f"""You are an AI assistant. Your task is to rewrite the latest user question into a standalone question using conversation history.

Rules:
1. **Ignore Greetings**: Do NOT include greetings (hi, hello, thanks) in the rewritten query.
2. **Focus on Context**: Use the conversation history to resolve pronouns and add missing context.
3. **Maintain Topic**: Preserve the main topic being discussed.
4. **Resolve Pronouns**: Replace 'it', 'they', 'that', 'this' with their referents from history.
5. **Preserve Clarification Context**: If previous messages show clarifying questions, combine the original query with the answers.
6. **Do Not Hallucinate**: Only use information present in the history.
7. **Be Concise**: Keep the rewritten query focused and clear.

Conversation History (filtered):
{history_str}

Latest User Question: {query}

Standalone Question:"""

        try:
            # Use LLM to rewrite
            response = self.llm_client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.rewrite_temperature,
                max_tokens=self.config.rewrite_max_tokens
            )

            rewritten = response.choices[0].message.content.strip()

            # Clean up quotes
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            if rewritten.startswith("'") and rewritten.endswith("'"):
                rewritten = rewritten[1:-1]

            logger.info(f"Query rewritten: '{query[:30]}...' -> '{rewritten[:30]}...'")
            return rewritten

        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return query

    def process_query(
        self,
        query: str,
        user_id: str,
        history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Process query with intelligent optimizations.

        Args:
            query: User query
            user_id: User ID
            history: Conversation history

        Returns:
            Dictionary with processing results
        """
        # Classify query type
        query_type = self.pattern_matcher.classify_query_type(query)

        # Assess complexity
        complexity = self.pattern_matcher.assess_complexity(query)

        # Rewrite if needed
        rewritten_query = self.rewrite_query(query, history, user_id)

        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "query_type": query_type.query_type.value if query_type.matched else "unknown",
            "complexity": complexity.value,
            "confidence": query_type.confidence if query_type.matched else 0.0
        }


# Global instance
_query_processor: Optional[OptimizedQueryProcessor] = None


def get_query_processor() -> Optional[OptimizedQueryProcessor]:
    """Get global query processor instance."""
    return _query_processor


def init_query_processor(llm_client, deployment_name: str):
    """
    Initialize global query processor.

    Args:
        llm_client: LLM client
        deployment_name: Azure deployment name
    """
    global _query_processor
    _query_processor = OptimizedQueryProcessor(
        llm_client=llm_client,
        deployment_name=deployment_name
    )
    logger.info("Initialized optimized query processor")
