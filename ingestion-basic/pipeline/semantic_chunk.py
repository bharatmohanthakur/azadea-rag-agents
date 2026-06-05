"""
Semantic chunker for creating meaningful text chunks with embeddings.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger

from .azure_client import AzureOpenAIClient
from .text_clean import TextCleaner


@dataclass
class TextChunk:
    """Represents a semantic text chunk."""
    chunk_id: str
    doc_id: str
    content: str
    pages: List[int]
    token_count: int
    start_char: int
    end_char: int
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class SemanticChunker:
    """Create semantic chunks from document content."""
    
    def __init__(
        self,
        azure_client: Optional[AzureOpenAIClient] = None,
        target_chunk_size: int = 1000,
        chunk_overlap: int = 150,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.8
    ):
        """
        Initialize semantic chunker.
        
        Args:
            azure_client: Azure OpenAI client for embeddings
            target_chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            max_chunk_size: Maximum chunk size in tokens
            min_chunk_size: Minimum chunk size in tokens
            similarity_threshold: Similarity threshold for semantic boundaries
        """
        self.azure_client = azure_client or AzureOpenAIClient()
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        
        self.text_cleaner = TextCleaner()
        
        logger.info(
            f"Initialized SemanticChunker: target_size={target_chunk_size}, "
            f"overlap={chunk_overlap}, similarity_threshold={similarity_threshold}"
        )
    
    def build_document_content(self, page_records: List[Dict[str, Any]]) -> str:
        """
        Build unified document content stream from page records.
        
        Interleaves text and image descriptions with proper tagging.
        """
        content_parts = []
        
        for record in page_records:
            page_num = record.get("page_number", 0)
            text = (record.get("text") or "").strip()
            image_desc = (record.get("image_description") or "").strip()
            
            # Add page marker
            content_parts.append(f"\n[PAGE_{page_num}]\n")
            
            # Add text content if available
            if text:
                content_parts.append(text)
            
            # Add image description if available
            if image_desc:
                content_parts.append(f"\n[IMAGE_CAPTION] {image_desc}")
            
            # Add page separator
            content_parts.append("\n")
        
        return "".join(content_parts).strip()
    
    def split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences with character positions.
        
        Returns:
            List of (sentence, start_pos, end_pos) tuples
        """
        # Simple sentence splitting - can be enhanced with spaCy/NLTK
        sentence_pattern = re.compile(r'([.!?]+)\s+')
        
        sentences = []
        start_pos = 0
        
        for match in sentence_pattern.finditer(text):
            end_pos = match.end()
            sentence = text[start_pos:end_pos].strip()
            
            if sentence and len(sentence) > 10:  # Skip very short sentences
                sentences.append((sentence, start_pos, end_pos))
            
            start_pos = end_pos
        
        # Don't forget the last sentence if it doesn't end with punctuation
        if start_pos < len(text):
            sentence = text[start_pos:].strip()
            if sentence and len(sentence) > 10:
                sentences.append((sentence, start_pos, len(text)))
        
        return sentences
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.warning(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def find_semantic_boundaries(
        self, 
        sentences: List[Tuple[str, int, int]]
    ) -> List[int]:
        """
        Find semantic boundaries in sentences using embedding similarity.
        
        Returns:
            List of sentence indices where semantic breaks occur
        """
        if len(sentences) < 2:
            return []
        
        # Get embeddings for sentences (batch processing)
        sentence_texts = [sent[0] for sent in sentences]
        
        try:
            embeddings = self.azure_client.get_embeddings(sentence_texts)
            if not embeddings or len(embeddings) != len(sentences):
                logger.warning("Failed to get embeddings for semantic boundary detection")
                return []
        except Exception as e:
            logger.error(f"Error getting embeddings for semantic boundaries: {e}")
            return []
        
        # Find boundaries where similarity drops significantly
        boundaries = []
        
        for i in range(len(embeddings) - 1):
            similarity = self.calculate_similarity(embeddings[i], embeddings[i + 1])
            
            # If similarity drops below threshold, it's a potential boundary
            if similarity < self.similarity_threshold:
                boundaries.append(i + 1)  # Boundary after sentence i
                logger.debug(f"Semantic boundary found at sentence {i + 1} (similarity: {similarity:.3f})")
        
        return boundaries
    
    def create_chunks_from_sentences(
        self,
        sentences: List[Tuple[str, int, int]],
        doc_id: str,
        semantic_boundaries: List[int]
    ) -> List[TextChunk]:
        """Create chunks from sentences respecting semantic boundaries and 2-page limit."""
        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        chunk_counter = 1

        for i, (sentence, start_pos, end_pos) in enumerate(sentences):
            sentence_tokens = self.azure_client.count_tokens(sentence)

            # Check if adding this sentence would exceed max chunk size
            if (current_tokens + sentence_tokens > self.max_chunk_size and
                current_chunk_sentences):

                # Create chunk from current sentences
                chunk = self._create_chunk_from_sentences(
                    current_chunk_sentences, doc_id, chunk_counter
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_counter += 1

                # Start new chunk with overlap
                current_chunk_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, self.chunk_overlap
                )
                current_tokens = sum(
                    self.azure_client.count_tokens(s[0])
                    for s in current_chunk_sentences
                )

            # Check if current chunk already spans more than 2 pages
            if current_chunk_sentences:
                current_pages = self._extract_page_numbers_from_sentences(current_chunk_sentences)
                if len(current_pages) > 2:
                    # Force chunk break if it spans more than 2 pages
                    chunk = self._create_chunk_from_sentences(
                        current_chunk_sentences, doc_id, chunk_counter
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_counter += 1

                    # Start new chunk with minimal overlap to avoid page span issues
                    current_chunk_sentences = []
                    current_tokens = 0

            # Add current sentence
            current_chunk_sentences.append((sentence, start_pos, end_pos))
            current_tokens += sentence_tokens

            # Check current chunk page span after adding sentence
            if current_chunk_sentences:
                current_pages = self._extract_page_numbers_from_sentences(current_chunk_sentences)
                if len(current_pages) > 2:
                    # Remove the sentence that caused the 3-page span
                    current_chunk_sentences.pop()
                    current_tokens -= sentence_tokens

                    # Create chunk with current sentences (should be 2 pages max)
                    if current_chunk_sentences:
                        chunk = self._create_chunk_from_sentences(
                            current_chunk_sentences, doc_id, chunk_counter
                        )
                        if chunk:
                            chunks.append(chunk)
                            chunk_counter += 1

                    # Start new chunk with this sentence
                    current_chunk_sentences = [(sentence, start_pos, end_pos)]
                    current_tokens = sentence_tokens

            # Check if we should break at semantic boundary
            if (i + 1 in semantic_boundaries and
                current_tokens >= self.target_chunk_size and
                len(current_chunk_sentences) > 1):

                # Create chunk at semantic boundary
                chunk = self._create_chunk_from_sentences(
                    current_chunk_sentences, doc_id, chunk_counter
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_counter += 1

                # Start new chunk with overlap
                current_chunk_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, self.chunk_overlap
                )
                current_tokens = sum(
                    self.azure_client.count_tokens(s[0])
                    for s in current_chunk_sentences
                )

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk = self._create_chunk_from_sentences(
                current_chunk_sentences, doc_id, chunk_counter
            )
            if chunk:
                chunks.append(chunk)

        return chunks
    
    def _create_chunk_from_sentences(
        self, 
        sentences: List[Tuple[str, int, int]], 
        doc_id: str, 
        chunk_id: int
    ) -> Optional[TextChunk]:
        """Create a TextChunk from a list of sentences."""
        if not sentences:
            return None
        
        # Combine sentences
        content = " ".join(sent[0] for sent in sentences)
        start_char = sentences[0][1]
        end_char = sentences[-1][2]
        
        # Count tokens
        token_count = self.azure_client.count_tokens(content)
        
        # Skip chunks that are too small
        if token_count < self.min_chunk_size:
            return None
        
        # Extract page numbers from the actual sentences in this chunk
        pages = self._extract_page_numbers_from_sentences(sentences)

        logger.debug(f"Chunk {chunk_id} content preview: {content[:100]}...")
        logger.debug(f"Chunk {chunk_id} extracted pages: {pages}")
    
        chunk = TextChunk(
            chunk_id=f"{doc_id}_chunk_{chunk_id}",
            doc_id=doc_id,
            content=content,
            pages=pages,
            token_count=token_count,
            start_char=start_char,
            end_char=end_char
        )
        
        return chunk
    
    def _get_overlap_sentences(
        self, 
        sentences: List[Tuple[str, int, int]], 
        overlap_tokens: int
    ) -> List[Tuple[str, int, int]]:
        """Get sentences for overlap based on token count."""
        if not sentences or overlap_tokens <= 0:
            return []
        
        # Start from the end and work backwards
        overlap_sentences = []
        current_tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self.azure_client.count_tokens(sentence[0])
            
            if current_tokens + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _extract_page_numbers(self, content: str) -> List[int]:
        """Extract page numbers from content markers."""
        page_pattern = re.compile(r'\[PAGE_(\d+)\]')
        matches = page_pattern.findall(content)
        return [int(match) for match in matches]

    def _extract_page_numbers_from_sentences(self, sentences: List[Tuple[str, int, int]]) -> List[int]:
        """Extract unique page numbers from a list of sentences."""
        page_numbers = set()
        for sentence, _, _ in sentences:
            pages = self._extract_page_numbers(sentence)
            page_numbers.update(pages)
        return sorted(list(page_numbers))
    
    def chunk_document(
        self, 
        page_records: List[Dict[str, Any]], 
        doc_id: str,
        generate_embeddings: bool = False
    ) -> List[TextChunk]:
        """
        Create semantic chunks from document page records.
        
        Args:
            page_records: List of page records with text and image descriptions
            doc_id: Document identifier
            generate_embeddings: Whether to generate embeddings for chunks
            
        Returns:
            List of TextChunk objects
        """
        logger.info(f"Starting semantic chunking for document: {doc_id}")
        
        # Build unified content stream
        content = self.build_document_content(page_records)
        
        if not content.strip():
            logger.warning(f"No content to chunk for document: {doc_id}")
            return []
        
        logger.debug(f"Built content stream: {len(content)} characters")
        
        # Split into sentences
        sentences = self.split_into_sentences(content)
        logger.debug(f"Split into {len(sentences)} sentences")
        
        if not sentences:
            return []
        
        # Find semantic boundaries
        semantic_boundaries = self.find_semantic_boundaries(sentences)
        logger.debug(f"Found {len(semantic_boundaries)} semantic boundaries")
        
        # Create chunks
        chunks = self.create_chunks_from_sentences(sentences, doc_id, semantic_boundaries)
        logger.info(f"Created {len(chunks)} chunks for document: {doc_id}")
        
        # Generate embeddings if requested
        if generate_embeddings and chunks:
            logger.info("Generating embeddings for chunks...")
            self._generate_chunk_embeddings(chunks)
        
        return chunks
    
    def _generate_chunk_embeddings(self, chunks: List[TextChunk]) -> None:
        """Generate embeddings for chunks."""
        try:
            # Extract content for embedding
            contents = [chunk.content for chunk in chunks]
            
            # Get embeddings in batch
            embeddings = self.azure_client.get_embeddings(contents)
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to generate chunk embeddings: {e}")
    
    def chunks_to_dict(self, chunks: List[TextChunk]) -> List[Dict[str, Any]]:
        """Convert TextChunk objects to dictionaries."""
        return [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "pages": chunk.pages,
                "token_count": chunk.token_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "embedding": chunk.embedding,
                "metadata": chunk.metadata or {}
            }
            for chunk in chunks
        ]
    
    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking process."""
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "total_tokens": 0
            }
        
        token_counts = [chunk.token_count for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_tokens": sum(token_counts),
            "chunks_with_embeddings": sum(1 for chunk in chunks if chunk.embedding)
        }
