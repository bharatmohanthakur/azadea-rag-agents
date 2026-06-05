"""
Reranking - Improve retrieval quality by re-ranking documents.
Based on RAG Techniques: https://github.com/NirDiamant/RAG_Techniques

This technique improves retrieval by:
1. Using cross-encoder models for better relevance scoring
2. Re-ranking initial retrieval results
3. Combining multiple ranking signals
4. Selecting top-k most relevant documents
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger("Reranker")


class RankedDocument(BaseModel):
    """A document with ranking information."""
    content: str = Field(description="Document content")
    metadata: Dict[str, Any] = Field(description="Document metadata", default_factory=dict)
    original_score: float = Field(description="Original retrieval score")
    rerank_score: float = Field(description="Reranking score")
    final_score: float = Field(description="Combined final score")
    rank: int = Field(description="Final rank position")


class Reranker:
    """Reranks retrieved documents for better relevance."""
    
    def __init__(
        self,
        llm_client,
        deployment_name: str,
        use_cross_encoder: bool = True,
        top_k: int = 5
    ):
        """
        Initialize reranker.
        
        Args:
            llm_client: LLM client for reranking
            deployment_name: Azure deployment name
            use_cross_encoder: Whether to use cross-encoder style reranking
            top_k: Number of top documents to return
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name
        self.use_cross_encoder = use_cross_encoder
        self.top_k = top_k
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None
    ) -> List[RankedDocument]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: User query
            documents: List of document dictionaries with 'content' and 'metadata'
            original_scores: Optional original retrieval scores
            
        Returns:
            List of ranked documents, sorted by relevance
        """
        if not documents:
            return []
        
        if len(documents) <= self.top_k:
            # If we have fewer documents than top_k, just score them
            return self._score_documents(query, documents, original_scores)
        
        # Score all documents
        scored_docs = self._score_documents(query, documents, original_scores)
        
        # Sort by final score and return top_k
        scored_docs.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign ranks
        for i, doc in enumerate(scored_docs[:self.top_k], 1):
            doc.rank = i
        
        logger.info(f"Reranked {len(documents)} documents, returning top {self.top_k}")
        
        return scored_docs[:self.top_k]
    
    def _score_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None
    ) -> List[RankedDocument]:
        """
        Score documents for relevance to query.
        
        Args:
            query: User query
            documents: List of document dictionaries
            original_scores: Optional original retrieval scores
            
        Returns:
            List of ranked documents with scores
        """
        if self.use_cross_encoder:
            return self._cross_encoder_rerank(query, documents, original_scores)
        else:
            return self._simple_rerank(query, documents, original_scores)
    
    def _cross_encoder_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None
    ) -> List[RankedDocument]:
        """
        Use cross-encoder style reranking (query-document pairs).
        Uses only document numbers to reduce token usage.
        
        Args:
            query: User query
            documents: List of document dictionaries
            original_scores: Optional original retrieval scores
            
        Returns:
            List of ranked documents
        """
        ranked_docs = []
        
        # Process in batches to avoid token limits (max 10 docs per batch)
        batch_size = 10
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_scores = original_scores[i:i + batch_size] if original_scores else None
            
            # Create scoring prompt with document content, metadata, and original scores
            doc_list = []
            for j, doc in enumerate(batch):
                doc_num = i + j + 1
                metadata = doc.get("metadata", {})
                source_file = metadata.get("source_file", "unknown")
                original_score = batch_scores[j] if batch_scores and j < len(batch_scores) else 0.5
                content = doc.get("content", "").strip()
                # Truncate content to 500 chars per document to manage token usage while keeping context
                content_preview = content[:500] + ("..." if len(content) > 500 else "")
                doc_list.append(f"Document {doc_num}:\n  Source: {source_file}\n  Original_Score: {original_score:.3f}\n  Content: {content_preview}")
            
            prompt = f"""Rate the relevance of each document to the query on a scale of 0.0 to 1.0.

Query: {query}

Documents (evaluate based on content, source filename, and original score):
{chr(10).join(doc_list)}

For each document, provide:
1. A relevance score (0.0 = not relevant, 1.0 = highly relevant)
2. A brief reason for the score

Respond in JSON format:
{{
    "scores": [
        {{"document": 1, "score": 0.85, "reason": "brief explanation"}},
        ...
    ]
}}"""

            try:
                response = self.llm_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000,  # Increased to handle content-based reranking with more detailed reasoning
                    response_format={"type": "json_object"}
                )
                
                import json
                import re
                
                # Extract JSON from response, handling potential formatting issues
                response_text = response.choices[0].message.content.strip()
                
                # Try to extract JSON if wrapped in markdown code blocks
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                
                # Parse JSON with better error handling
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError as je:
                    logger.warning(f"JSON parse error, attempting to fix: {je}")
                    # Try to fix common JSON issues
                    # Remove trailing commas
                    response_text = re.sub(r',\s*}', '}', response_text)
                    response_text = re.sub(r',\s*]', ']', response_text)
                    # Fix unclosed strings (basic attempt)
                    response_text = re.sub(r'("reason":\s*"[^"]*?)(?=\s*[,}])', r'\1"', response_text)
                    try:
                        result = json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.error(f"Could not parse JSON after fixes, using fallback")
                        raise
                
                scores = {}
                for item in result.get("scores", []):
                    doc_num = item.get("document", 0)
                    score = float(item.get("score", 0.5))
                    scores[doc_num] = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                
                # Create ranked documents
                for j, doc in enumerate(batch):
                    doc_num = i + j + 1
                    rerank_score = scores.get(doc_num, 0.5)
                    original_score = batch_scores[j] if batch_scores and j < len(batch_scores) else 0.5
                    
                    # Combine scores (weighted average: 60% rerank, 40% original)
                    final_score = 0.6 * rerank_score + 0.4 * original_score
                    
                    ranked_docs.append(RankedDocument(
                        content=doc.get("content", ""),
                        metadata=doc.get("metadata", {}),
                        original_score=original_score,
                        rerank_score=rerank_score,
                        final_score=final_score,
                        rank=0  # Will be set later
                    ))
                    
            except Exception as e:
                logger.error(f"Error in cross-encoder reranking: {e}")
                # Fallback: use original scores or equal scores
                for j, doc in enumerate(batch):
                    original_score = batch_scores[j] if batch_scores and j < len(batch_scores) else 0.5
                    ranked_docs.append(RankedDocument(
                        content=doc.get("content", ""),
                        metadata=doc.get("metadata", {}),
                        original_score=original_score,
                        rerank_score=original_score,
                        final_score=original_score,
                        rank=0
                    ))
        
        return ranked_docs
    
    def _simple_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None
    ) -> List[RankedDocument]:
        """
        Simple reranking using keyword matching and heuristics.
        
        Args:
            query: User query
            documents: List of document dictionaries
            original_scores: Optional original retrieval scores
            
        Returns:
            List of ranked documents
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        ranked_docs = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "").lower()
            
            # Simple keyword overlap score
            content_words = set(content.split())
            overlap = len(query_words & content_words) / len(query_words) if query_words else 0
            
            original_score = original_scores[i] if original_scores and i < len(original_scores) else 0.5
            rerank_score = min(overlap, 1.0)
            
            # Combine scores
            final_score = 0.6 * rerank_score + 0.4 * original_score
            
            ranked_docs.append(RankedDocument(
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                original_score=original_score,
                rerank_score=rerank_score,
                final_score=final_score,
                rank=0
            ))
        
        return ranked_docs

