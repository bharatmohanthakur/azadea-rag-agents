"""
Query Decomposition - Break complex multi-part questions into sub-queries.
Based on RAG Techniques: https://github.com/NirDiamant/RAG_Techniques

This technique improves retrieval for complex questions by:
1. Identifying if a query needs decomposition
2. Breaking it into focused sub-queries
3. Retrieving for each sub-query
4. Synthesizing results into a comprehensive answer
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("QueryDecomposer")


class SubQuery(BaseModel):
    """A decomposed sub-query."""
    query: str = Field(description="The sub-query text")
    intent: str = Field(description="The intent or focus of this sub-query")
    priority: int = Field(description="Priority level (1=highest, 3=lowest)", default=2)


class QueryDecomposition(BaseModel):
    """Result of query decomposition."""
    needs_decomposition: bool = Field(description="Whether the query should be decomposed")
    original_query: str = Field(description="The original query")
    sub_queries: List[SubQuery] = Field(description="List of sub-queries", default_factory=list)
    reasoning: str = Field(description="Reasoning for decomposition decision", default="")


class QueryDecomposer:
    """Decomposes complex queries into simpler sub-queries."""
    
    def __init__(self, llm_client, deployment_name: str):
        """
        Initialize query decomposer.
        
        Args:
            llm_client: LLM client for decomposition
            deployment_name: Azure deployment name
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name
    
    def should_decompose(self, query: str) -> bool:
        """
        Quick heuristic check if query might need decomposition.
        More conservative - only decompose truly complex multi-part queries.
        
        Args:
            query: User query
            
        Returns:
            True if query might need decomposition
        """
        query_lower = query.lower().strip()
        
        # Skip very short queries (likely simple)
        if len(query.split()) < 5:
            return False
        
        # Only decompose if query has clear multiple parts with conjunctions
        # Require at least 2 of these indicators to be present
        strong_indicators = [
            " and ",  # Multiple conditions (but not "and" in phrases like "understand")
            " or ",   # Alternative conditions
            "compare", "comparison", "difference between", "versus", " vs ",  # Comparison queries
            "both",  # Multiple items (but not in phrases like "both of")
        ]
        
        # Count how many strong indicators are present
        indicator_count = sum(1 for indicator in strong_indicators if indicator in query_lower)
        
        # Require at least 2 strong indicators OR a very explicit multi-part structure
        has_multiple_parts = indicator_count >= 2
        
        # Also check for explicit multi-question patterns
        multi_question_patterns = [
            "what are", "list all", "name all", "tell me about",  # List queries
            "how many", "how much",  # Quantitative queries (only if combined with other indicators)
        ]
        
        has_multi_question = any(pattern in query_lower for pattern in multi_question_patterns)
        
        # Only decompose if we have strong evidence of multiple distinct parts
        return has_multiple_parts or (has_multi_question and indicator_count >= 1)
    
    def decompose(self, query: str, context: Optional[str] = None) -> QueryDecomposition:
        """
        Decompose a query into sub-queries if needed.
        
        Args:
            query: Original user query
            context: Optional context from conversation history
            
        Returns:
            QueryDecomposition object
        """
        # Quick check - if no indicators, return as-is (skip LLM call)
        if not self.should_decompose(query):
            logger.debug(f"Query decomposition skipped (simple query): {query[:50]}")
            return QueryDecomposition(
                needs_decomposition=False,
                original_query=query,
                sub_queries=[SubQuery(query=query, intent="main", priority=1)],
                reasoning="Query is simple and doesn't require decomposition"
            )
        
        logger.info(f"Query decomposition check (may need decomposition): {query[:50]}")
        
        # Use LLM to determine if decomposition is needed and generate sub-queries
        prompt = f"""Analyze the following query and determine if it should be decomposed into multiple sub-queries.

IMPORTANT: Only decompose if the query has MULTIPLE DISTINCT, SEPARATE questions or topics that require different retrieval strategies.

A query should be decomposed ONLY if it:
1. Explicitly asks about multiple DISTINCT topics (e.g., "What is X and how does Y work?")
2. Requires comparing different entities (e.g., "Compare X vs Y")
3. Has clear multiple parts with conjunctions AND each part is a separate question
4. Asks for a list of items with DIFFERENT criteria that need separate searches

DO NOT decompose if:
- The query is a single question, even if it mentions multiple things
- The query uses "and" or "or" but is asking one unified question
- The query is simple and can be answered with one search
- The query is about one topic with multiple aspects (these should be handled by the retrieval system)

Query: {query}
{f"Context: {context}" if context else ""}

Respond in JSON format:
{{
    "needs_decomposition": true/false,
    "reasoning": "brief explanation of why decomposition is or isn't needed",
    "sub_queries": [
        {{
            "query": "focused sub-query",
            "intent": "what this sub-query is asking about",
            "priority": 1-3 (1=most important, 3=least important)
        }}
    ]
}}

If needs_decomposition is false, include the original query as a single sub-query with priority 1."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            result_json = json.loads(response.choices[0].message.content)
            
            sub_queries = [
                SubQuery(**sq) for sq in result_json.get("sub_queries", [])
            ]
            
            # If no sub-queries generated, use original query
            if not sub_queries:
                sub_queries = [SubQuery(query=query, intent="main", priority=1)]
            
            return QueryDecomposition(
                needs_decomposition=result_json.get("needs_decomposition", False),
                original_query=query,
                sub_queries=sub_queries,
                reasoning=result_json.get("reasoning", "")
            )
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            # Fallback: return original query as single sub-query
            return QueryDecomposition(
                needs_decomposition=False,
                original_query=query,
                sub_queries=[SubQuery(query=query, intent="main", priority=1)],
                reasoning=f"Error during decomposition: {str(e)}"
            )
    
    def merge_results(self, sub_query_results: List[Dict[str, Any]], original_query: str) -> Dict[str, Any]:
        """
        Merge results from multiple sub-queries into a unified context.
        
        Args:
            sub_query_results: List of results from each sub-query
            original_query: Original user query
            
        Returns:
            Merged context dictionary
        """
        # Combine all sources, deduplicate by content
        all_sources = []
        seen_content = set()
        
        for result in sub_query_results:
            sources = result.get("sources", [])
            for source in sources:
                content_key = source.get("content", "")[:100]  # Use first 100 chars as key
                if content_key not in seen_content:
                    all_sources.append(source)
                    seen_content.add(content_key)
        
        # Combine all context text
        all_context = "\n\n".join([
            result.get("context", "") for result in sub_query_results
        ])
        
        return {
            "sources": all_sources,
            "context": all_context,
            "sub_query_count": len(sub_query_results)
        }

