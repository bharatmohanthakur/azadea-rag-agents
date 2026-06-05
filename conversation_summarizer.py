"""
Conversation summarization to compress long histories while preserving key context.
Inspired by ChatGPT's approach to managing long conversations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("ConversationSummarizer")


class ConversationSummarizer:
    """Summarizes conversation history to preserve context in limited token windows."""
    
    def __init__(self, llm_client, deployment_name: str = None, max_turns_before_summarize: int = 10, keep_recent_turns: int = 5):
        """
        Initialize conversation summarizer.
        
        Args:
            llm_client: LLM client for summarization
            deployment_name: Azure deployment name (required for Azure OpenAI)
            max_turns_before_summarize: Number of turns before summarizing
            keep_recent_turns: Number of recent turns to keep in full
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name
        self.max_turns_before_summarize = max_turns_before_summarize
        self.keep_recent_turns = keep_recent_turns
    
    def should_summarize(self, history: List[Dict[str, Any]]) -> bool:
        """Check if history should be summarized."""
        return len(history) > self.max_turns_before_summarize
    
    def summarize_conversation(
        self,
        history: List[Dict[str, Any]],
        preserve_clarification: bool = True
    ) -> Dict[str, Any]:
        """
        Summarize conversation history while preserving key context.
        
        Args:
            history: Full conversation history
            preserve_clarification: Whether to preserve clarification context
        
        Returns:
            Dictionary with 'summary' and 'recent_messages'
        """
        if not self.should_summarize(history):
            return {
                "summary": None,
                "recent_messages": history,
                "full_history": history
            }
        
        # Split into old (to summarize) and recent (to keep)
        split_point = len(history) - self.keep_recent_turns
        old_messages = history[:split_point]
        recent_messages = history[split_point:]
        
        # Extract clarification context if present
        clarification_context = None
        if preserve_clarification:
            clarification_context = self._extract_clarification_context(old_messages)
        
        # Summarize old messages
        summary = self._create_summary(old_messages, clarification_context)
        
        return {
            "summary": summary,
            "recent_messages": recent_messages,
            "full_history": history  # Keep full for reference
        }
    
    def _extract_clarification_context(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract clarification-related context from messages."""
        clarification_info = {
            "original_queries": [],
            "questions_asked": [],
            "answers_given": []
        }
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            metadata = msg.get("metadata", {})
            
            if role == "user":
                # Check if this was part of clarification
                if "clarification" in str(metadata).lower() or "clarifying" in content.lower():
                    clarification_info["original_queries"].append(content)
            
            if role == "assistant":
                # Check if this asked clarifying questions
                if "need a bit more information" in content.lower() or "clarifying" in content.lower():
                    clarification_info["questions_asked"].append(content)
        
        if any(clarification_info.values()):
            return clarification_info
        
        return None
    
    def _create_summary(self, messages: List[Dict[str, Any]], clarification_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a summary of old messages using LLM.
        
        Args:
            messages: Messages to summarize
            clarification_context: Optional clarification context to preserve
        
        Returns:
            Summary string
        """
        # Format messages for summarization
        conversation_text = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            conversation_text += f"{role.capitalize()}: {content}\n\n"
        
        # Build summarization prompt
        prompt = f"""Summarize the following conversation history, preserving:
1. The main topics discussed
2. Key decisions or answers provided
3. Important entities mentioned (countries, positions, policy types, etc.)
4. Any clarification questions and their answers

{"IMPORTANT: Preserve clarification context: " + str(clarification_context) if clarification_context else ""}

Conversation History:
{conversation_text}

Provide a concise summary (2-3 sentences) that captures the essential context:"""

        try:
            # Use deployment name for Azure OpenAI, or model name for OpenAI
            model_param = self.deployment_name if self.deployment_name else "gpt-4o"
            response = self.llm_client.chat.completions.create(
                model=model_param,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            summary = response.choices[0].message.content.strip()
            logger.info(f"Created conversation summary ({len(summary)} chars)")
            return summary
        except Exception as e:
            logger.error(f"Failed to create summary: {e}")
            # Fallback: simple extraction
            return self._simple_summary(messages)
    
    def _simple_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Simple fallback summary extraction."""
        topics = []
        entities = []
        
        for msg in messages:
            content = msg.get("content", "").lower()
            # Extract common entities
            if "lebanon" in content or "uae" in content or "egypt" in content:
                entities.append("Country mentioned")
            if "manager" in content or "employee" in content or "director" in content:
                entities.append("Position mentioned")
            if "leave" in content or "insurance" in content or "benefit" in content:
                topics.append("HR policies")
        
        summary_parts = []
        if topics:
            summary_parts.append(f"Discussed: {', '.join(set(topics))}")
        if entities:
            summary_parts.append(f"Entities: {', '.join(set(entities))}")
        
        return ". ".join(summary_parts) if summary_parts else "Previous conversation about HR policies."
    
    def get_compressed_history(
        self,
        history: List[Dict[str, Any]],
        include_summary: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get compressed history with summary + recent messages.
        
        Args:
            history: Full conversation history
            include_summary: Whether to include summary as a system message
        
        Returns:
            Compressed history list
        """
        result = self.summarize_conversation(history)
        
        compressed = []
        
        # Add summary as context if available
        if include_summary and result["summary"]:
            compressed.append({
                "role": "system",
                "content": f"Previous conversation summary: {result['summary']}"
            })
        
        # Add recent messages
        compressed.extend(result["recent_messages"])
        
        return compressed
    
    def extract_entities(self, history: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Extract key entities from conversation history.
        
        Returns:
            Dictionary of entity types to values
        """
        entities = {
            "countries": [],
            "positions": [],
            "policy_types": [],
            "dates": []
        }
        
        # Common patterns
        country_keywords = ["lebanon", "uae", "egypt", "saudi", "kuwait", "qatar", "jordan"]
        position_keywords = ["manager", "director", "employee", "supervisor", "executive", "staff"]
        policy_keywords = ["leave", "insurance", "benefit", "bonus", "salary", "vacation", "maternity", "paternity"]
        
        for msg in history:
            content = msg.get("content", "").lower()
            
            for country in country_keywords:
                if country in content and country not in entities["countries"]:
                    entities["countries"].append(country.title())
            
            for position in position_keywords:
                if position in content and position not in entities["positions"]:
                    entities["positions"].append(position.title())
            
            for policy in policy_keywords:
                if policy in content and policy not in entities["policy_types"]:
                    entities["policy_types"].append(policy.title())
        
        # Remove empty lists
        return {k: v for k, v in entities.items() if v}


