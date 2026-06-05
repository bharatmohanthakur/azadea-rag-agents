"""
Contextual Compression - Reduce token usage while preserving key information.
Based on RAG Techniques: https://github.com/NirDiamant/RAG_Techniques

This technique compresses retrieved context by:
1. Identifying redundant information
2. Extracting only relevant parts for the query
3. Preserving key facts, numbers, and entities
4. Maintaining coherence for the LLM
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("ContextualCompressor")


class CompressedChunk(BaseModel):
    """A compressed chunk of context."""
    content: str = Field(description="Compressed content")
    original_length: int = Field(description="Original character count")
    compressed_length: int = Field(description="Compressed character count")
    compression_ratio: float = Field(description="Compression ratio (0-1)")
    relevance_score: float = Field(description="Relevance to query (0-1)")


class ContextualCompressor:
    """Compresses retrieved context while preserving key information."""
    
    def __init__(self, llm_client, deployment_name: str, max_compression_ratio: float = 0.5):
        """
        Initialize contextual compressor.
        
        Args:
            llm_client: LLM client for compression
            deployment_name: Azure deployment name
            max_compression_ratio: Maximum compression (0.5 = reduce to 50% of original)
        """
        self.llm_client = llm_client
        self.deployment_name = deployment_name
        self.max_compression_ratio = max_compression_ratio
    
    def should_compress(self, context: str, max_tokens: int = 4000) -> bool:
        """
        Determine if context should be compressed.
        
        Args:
            context: Retrieved context text
            max_tokens: Maximum token limit
            
        Returns:
            True if compression is needed
        """
        # Rough estimate: 1 token ≈ 4 characters
        estimated_tokens = len(context) / 4
        return estimated_tokens > max_tokens
    
    def compress(
        self,
        context: str,
        query: str,
        preserve_entities: bool = True,
        preserve_numbers: bool = True
    ) -> CompressedChunk:
        """
        Compress context while preserving key information.
        
        Args:
            context: Original context to compress
            query: User query for relevance filtering
            preserve_entities: Whether to preserve named entities
            preserve_numbers: Whether to preserve numerical data
            
        Returns:
            CompressedChunk object
        """
        original_length = len(context)
        
        # If context is short, no compression needed
        if not self.should_compress(context):
            return CompressedChunk(
                content=context,
                original_length=original_length,
                compressed_length=original_length,
                compression_ratio=1.0,
                relevance_score=1.0
            )
        
        # Build compression prompt
        preserve_instructions = []
        if preserve_entities:
            preserve_instructions.append("- Preserve all named entities (countries, positions, policy names, etc.)")
        if preserve_numbers:
            preserve_instructions.append("- Preserve all numerical data (dates, amounts, percentages, etc.)")
        
        prompt = f"""Compress the following context while preserving ALL information relevant to the query.

Query: {query}

Context to compress:
{context}

Instructions:
- Remove redundant information and filler words
- Keep only information directly relevant to the query
- Preserve key facts, relationships, and details
- Maintain coherence and readability
{chr(10).join(preserve_instructions)}
- Do NOT add any information not in the original context
- Do NOT change the meaning or facts

Compressed context:"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=int(original_length / 4 * self.max_compression_ratio)  # Limit output tokens
            )
            
            compressed = response.choices[0].message.content.strip()
            compressed_length = len(compressed)
            compression_ratio = compressed_length / original_length if original_length > 0 else 1.0
            
            logger.info(f"Compressed context: {original_length} -> {compressed_length} chars ({compression_ratio:.2%})")
            
            return CompressedChunk(
                content=compressed,
                original_length=original_length,
                compressed_length=compressed_length,
                compression_ratio=compression_ratio,
                relevance_score=1.0  # Could be enhanced with relevance scoring
            )
            
        except Exception as e:
            logger.error(f"Error compressing context: {e}")
            # Fallback: return original context
            return CompressedChunk(
                content=context,
                original_length=original_length,
                compressed_length=original_length,
                compression_ratio=1.0,
                relevance_score=1.0
            )
    
    def compress_multiple(
        self,
        contexts: List[str],
        query: str,
        max_total_length: int = 8000
    ) -> List[CompressedChunk]:
        """
        Compress multiple context chunks, prioritizing most relevant.
        
        Args:
            contexts: List of context chunks
            query: User query
            max_total_length: Maximum total length after compression
            
        Returns:
            List of compressed chunks, sorted by relevance
        """
        compressed_chunks = []
        total_length = 0
        
        # Compress each chunk
        for context in contexts:
            chunk = self.compress(context, query)
            compressed_chunks.append(chunk)
            total_length += chunk.compressed_length
        
        # If still too long, further compress or truncate
        if total_length > max_total_length:
            # Sort by relevance (if we had scores) or compression ratio
            compressed_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Keep chunks until we hit the limit
            selected = []
            current_length = 0
            for chunk in compressed_chunks:
                if current_length + chunk.compressed_length <= max_total_length:
                    selected.append(chunk)
                    current_length += chunk.compressed_length
                else:
                    # Truncate last chunk if needed
                    remaining = max_total_length - current_length
                    if remaining > 100:  # Only if meaningful space left
                        truncated = chunk.content[:remaining] + "..."
                        selected.append(CompressedChunk(
                            content=truncated,
                            original_length=chunk.original_length,
                            compressed_length=len(truncated),
                            compression_ratio=chunk.compression_ratio,
                            relevance_score=chunk.relevance_score
                        ))
                    break
            
            return selected
        
        return compressed_chunks
    
    def merge_compressed(self, chunks: List[CompressedChunk]) -> str:
        """
        Merge compressed chunks into a single context string.
        
        Args:
            chunks: List of compressed chunks
            
        Returns:
            Merged context string
        """
        return "\n\n".join([chunk.content for chunk in chunks])

