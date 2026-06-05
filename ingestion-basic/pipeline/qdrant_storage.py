"""
Qdrant storage integration for storing text, image bytes, and metadata.
"""

import base64
import os
import re
import math
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime
import uuid
import hashlib
from collections import defaultdict, Counter

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector
from loguru import logger

from .semantic_chunk import TextChunk


class QdrantStorage:
    """Qdrant client for storing multimodal document data."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "docs_hybrid_azure_backup",  # Unified collection with all data sources
        vector_size: int = 3072,  # text-embedding-3-large dimension
        distance_metric: Distance = Distance.COSINE
    ):
        """
        Initialize Qdrant storage client.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to store documents
            vector_size: Dimension of embedding vectors
            distance_metric: Distance metric for similarity search
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        
        # Initialize client
        try:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Connected to Qdrant at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {host}:{port}: {e}")
            raise
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def generate_point_id(self, identifier: str) -> str:
        """Generate a UUID from a string identifier."""
        # Create a deterministic UUID from the identifier
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace UUID
        return str(uuid.uuid5(namespace, identifier))
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist with hybrid search support."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name} with hybrid search support")

                # Create collection with both dense and sparse vector support
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=self.vector_size,
                            distance=self.distance_metric
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams()
                    }
                )

                logger.info(f"Created collection: {self.collection_name} with dense and sparse vectors")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    def create_sparse_vector(self, text: str, vocab_size: int = 10000) -> SparseVector:
        """
        Create a sparse vector from text using TF-IDF like scoring.
        This creates a simple sparse representation for keyword-based search.

        Args:
            text: Input text
            vocab_size: Maximum vocabulary size

        Returns:
            SparseVector for Qdrant
        """
        # Simple tokenization and preprocessing
        tokens = self._tokenize_text(text)

        if not tokens:
            return SparseVector(indices=[], values=[])

        # Count term frequencies
        term_freq = Counter(tokens)

        # Create simple TF-IDF like scores (simplified BM25)
        indices = []
        values = []

        for i, (term, freq) in enumerate(term_freq.most_common(min(len(term_freq), vocab_size))):
            # Simple scoring: term frequency with some normalization
            score = math.log(1 + freq)  # Log TF scaling
            indices.append(i)  # Use term index as sparse index
            values.append(score)

        return SparseVector(indices=indices, values=values)

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple text tokenization for sparse vectors."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text)
        # Remove stop words and short tokens
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                     'to', 'was', 'will', 'with', 'would', 'but', 'or', 'not', 'no'}
        return [token for token in tokens if len(token) > 2 and token not in stop_words]

    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode image file to base64 string."""
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                return None
            
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                return base64.b64encode(image_bytes).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None
    
    def create_document_metadata(
        self, 
        doc_id: str, 
        pdf_path: str, 
        page_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for a document."""
        metadata = {
            "doc_id": doc_id,
            "pdf_path": pdf_path,
            "pdf_filename": os.path.basename(pdf_path),
            "total_pages": len(page_records),
            "pages_with_text": sum(1 for r in page_records if r.get("text")),
            "pages_with_images": sum(1 for r in page_records if r.get("image_path")),
            "ingestion_timestamp": datetime.now().isoformat(),
            "file_size": os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
        }
        
        # Add text statistics
        total_chars = sum(len(r.get("text", "")) for r in page_records)
        total_words = sum(len(r.get("text", "").split()) for r in page_records)
        
        metadata.update({
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chars_per_page": total_chars / len(page_records) if page_records else 0,
            "avg_words_per_page": total_words / len(page_records) if page_records else 0
        })
        
        return metadata
    
    def store_page_record(
        self, 
        page_record: Dict[str, Any], 
        doc_metadata: Dict[str, Any],
        point_id: Optional[str] = None
    ) -> bool:
        """
        Store a single page record in Qdrant.
        
        Args:
            page_record: Page record with text, image info, etc.
            doc_metadata: Document-level metadata
            point_id: Optional custom point ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc_id = page_record.get("doc_id")
            page_number = page_record.get("page_number")
            
            if not point_id:
                point_id = self.generate_point_id(f"{doc_id}_page_{page_number}")
            
            # Prepare payload
            payload = {
                "type": "page",
                "doc_id": doc_id,
                "page_number": page_number,
                "text": page_record.get("text", ""),
                "image_description": page_record.get("image_description", ""),
                "is_image_dominant": page_record.get("is_image_dominant", False),
                "visible_char_count": page_record.get("visible_char_count", 0),
                "has_images": page_record.get("has_images", False),
                "doc_metadata": doc_metadata
            }
            
            # Add image data if available
            image_path = page_record.get("image_path")
            if image_path:
                image_base64 = self.encode_image_to_base64(image_path)
                if image_base64:
                    payload["image_base64"] = image_base64
                    payload["image_path"] = image_path
                    
                    # Add image metadata
                    try:
                        image_stat = os.stat(image_path)
                        payload["image_size_bytes"] = image_stat.st_size
                    except:
                        pass
            
            # Prepare vectors for both dense and sparse search
            vectors = {}

            # Dense vector (zero vector for pages since no embeddings)
            vectors["dense"] = [0.0] * self.vector_size

            # Sparse vector (keyword search from page content)
            sparse_vector = self.create_sparse_vector(page_record.get("text", ""))
            if sparse_vector.indices:  # Only add if we have terms
                vectors["sparse"] = sparse_vector

            # Create point
            point = PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload
            )
            
            # Upsert point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Stored page record: {point_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store page record {page_record.get('page_number', '?')}: {e}")
            return False
    
    def store_chunk(
        self, 
        chunk: TextChunk, 
        doc_metadata: Dict[str, Any],
        related_images: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Store a text chunk in Qdrant.
        
        Args:
            chunk: TextChunk object
            doc_metadata: Document-level metadata
            related_images: Related image data for pages covered by this chunk
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare payload
            payload = {
                "type": "chunk",
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "pages": chunk.pages,
                "token_count": chunk.token_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "doc_metadata": doc_metadata
            }
            
            # Add chunk metadata if available
            if chunk.metadata:
                payload["chunk_metadata"] = chunk.metadata
            
            # Add related images
            if related_images:
                payload["related_images"] = related_images
            
            # Prepare vectors for both dense and sparse search
            vectors = {}

            # Dense vector (semantic search)
            if chunk.embedding:
                vectors["dense"] = chunk.embedding
            else:
                vectors["dense"] = [0.0] * self.vector_size
                logger.warning(f"No embedding available for chunk {chunk.chunk_id}, using zero vector")

            # Sparse vector (keyword search) - add to the same vectors dict
            sparse_vector = self.create_sparse_vector(chunk.content)
            if sparse_vector.indices:  # Only add if we have terms
                vectors["sparse"] = sparse_vector

            # Create point with UUID
            point_id = self.generate_point_id(chunk.chunk_id)
            point = PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload
            )
            
            # Upsert point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Stored chunk: {chunk.chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunk {chunk.chunk_id}: {e}")
            return False
    
    def store_document(
        self, 
        doc_id: str,
        pdf_path: str,
        page_records: List[Dict[str, Any]],
        chunks: List[TextChunk]
    ) -> bool:
        """
        Store complete document data in Qdrant.
        
        Args:
            doc_id: Document identifier
            pdf_path: Path to original PDF file
            page_records: List of page records
            chunks: List of text chunks
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Storing document in Qdrant: {doc_id}")
        
        try:
            # Create document metadata
            doc_metadata = self.create_document_metadata(doc_id, pdf_path, page_records)
            
            # Store page records
            pages_stored = 0
            for page_record in page_records:
                if self.store_page_record(page_record, doc_metadata):
                    pages_stored += 1
            
            logger.info(f"Stored {pages_stored}/{len(page_records)} page records")
            
            # Prepare related images mapping for chunks
            page_images = {
                record["page_number"]: {
                    "image_path": record.get("image_path"),
                    "image_description": record.get("image_description"),
                    "image_base64": self.encode_image_to_base64(record["image_path"]) 
                                  if record.get("image_path") else None
                }
                for record in page_records 
                if record.get("image_path")
            }
            
            # Store chunks
            chunks_stored = 0
            for chunk in chunks:
                # Find related images for this chunk's pages (limit to max 2 images per chunk)
                related_images = []
                max_images_per_chunk = 2  # Limit images per chunk to prevent overwhelming AI

                logger.debug(f"Chunk {chunk.chunk_id} spans pages: {chunk.pages}")
                logger.debug(f"Available image pages: {list(page_images.keys())}")

                for page_num in chunk.pages:
                    if page_num in page_images and page_images[page_num]["image_base64"]:
                        if len(related_images) < max_images_per_chunk:
                            related_images.append(page_images[page_num])
                            logger.debug(f"Added image from page {page_num} to chunk {chunk.chunk_id}")

                logger.debug(f"Chunk {chunk.chunk_id} has {len(related_images)} related images")

                if self.store_chunk(chunk, doc_metadata, related_images):
                    chunks_stored += 1
            
            logger.info(f"Stored {chunks_stored}/{len(chunks)} chunks")
            
            # Store document summary
            self._store_document_summary(doc_id, doc_metadata, len(page_records), len(chunks))
            
            logger.info(f"Successfully stored document {doc_id} in Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store document {doc_id}: {e}")
            return False
    
    def _store_document_summary(
        self, 
        doc_id: str, 
        doc_metadata: Dict[str, Any], 
        page_count: int, 
        chunk_count: int
    ):
        """Store document summary record."""
        try:
            payload = {
                "type": "document_summary",
                "doc_id": doc_id,
                "page_count": page_count,
                "chunk_count": chunk_count,
                **doc_metadata
            }
            
            vectors = {"dense": [0.0] * self.vector_size}  # Zero vector for summary

            point_id = self.generate_point_id(f"{doc_id}_summary")
            point = PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Stored document summary: {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to store document summary {doc_id}: {e}")
    
    def search_similar_chunks(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.1,
        doc_id_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            doc_id_filter: Optional filter by document ID
            
        Returns:
            List of similar chunks with scores
        """
        try:
            # Prepare filter
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="chunk")
                    )
                ]
            )
            
            if doc_id_filter:
                query_filter.must.append(
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id_filter)
                    )
                )
            
            # Search using query_points (new API)
            query_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using="dense",
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            formatted_results = []
            for point in query_result.points:
                formatted_results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                })
            
            logger.debug(f"Found {len(formatted_results)} similar chunks")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            return []

    def bm25_keyword_search(
        self,
        query: str,
        limit: int = 10,
        doc_id_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search using Qdrant's native sparse vector search.

        Args:
            query: Search query string
            limit: Maximum number of results
            doc_id_filter: Optional filter by document ID

        Returns:
            List of chunks with BM25 scores
        """
        try:
            # Create sparse vector from query using the same logic as during ingestion
            sparse_query = self.create_sparse_vector(query)

            if not sparse_query.indices:
                logger.debug("No terms found in query for sparse search")
                return []

            # Create filter for chunks only (and optional doc_id)
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="chunk")
                    )
                ]
            )

            if doc_id_filter:
                query_filter.must.append(
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id_filter)
                    )
                )

            # Use Qdrant's native sparse vector search
            query_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=sparse_query,
                        using="sparse",
                        filter=query_filter,
                        limit=limit * 2  # Get more candidates for better results
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),  # Use Reciprocal Rank Fusion
                limit=limit,
                with_payload=True
            )

            # Format results to match expected structure
            formatted_results = []
            for point in query_result.points:
                formatted_results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                })

            logger.debug(f"BM25 search found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to perform BM25 search: {e}")
            return []

    def _preprocess_query(self, query: str) -> List[str]:
        """Preprocess query by tokenizing and normalizing."""
        # Convert to lowercase
        query = query.lower()
        # Split into words and remove punctuation
        terms = re.findall(r'\b\w+\b', query)
        # Remove common stop words (basic list)
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                     'to', 'was', 'will', 'with', 'would'}
        return [term for term in terms if term not in stop_words and len(term) > 1]

    def _calculate_bm25_scores(
        self,
        chunks: List[Any],
        query_terms: List[str],
        k1: float,
        b: float
    ) -> Dict[str, float]:
        """Calculate BM25 scores for all chunks."""
        if not chunks or not query_terms:
            return {}

        # Calculate document lengths and term frequencies
        doc_lengths = {}
        term_doc_freq = defaultdict(int)  # Document frequency for each term
        term_doc_counts = defaultdict(lambda: defaultdict(int))  # Term frequency per document

        for chunk in chunks:
            content = chunk.payload.get('content', '').lower()
            doc_length = len(content.split())
            doc_lengths[chunk.id] = doc_length

            # Count term frequencies in this document
            content_terms = re.findall(r'\b\w+\b', content)
            term_counts = Counter(content_terms)

            for term in query_terms:
                if term in term_counts:
                    term_doc_counts[chunk.id][term] = term_counts[term]
                    term_doc_freq[term] += 1

        # Calculate average document length
        if doc_lengths:
            avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)
        else:
            avg_doc_length = 0

        # Total number of documents
        N = len(chunks)

        # Calculate BM25 scores
        bm25_scores = {}

        for chunk in chunks:
            doc_id = chunk.id
            score = 0.0

            for term in query_terms:
                if term in term_doc_counts[doc_id]:
                    # Term frequency in document
                    tf = term_doc_counts[doc_id][term]

                    # Document frequency
                    df = term_doc_freq[term]

                    if df > 0:
                        # IDF calculation
                        idf = math.log((N - df + 0.5) / (df + 0.5))

                        # BM25 term score
                        doc_length = doc_lengths[doc_id]
                        term_score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length))))
                        score += term_score

            if score > 0:
                bm25_scores[doc_id] = score

        return bm25_scores

    def hybrid_search_qdrant(
        self,
        query: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.1,
        doc_id_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Qdrant's Query API with Reciprocal Rank Fusion.

        This method uses Qdrant's built-in hybrid search capabilities:
        - Dense vectors for semantic search
        - Sparse vectors for keyword-based search
        - Reciprocal Rank Fusion to combine results

        Args:
            query: Text query for sparse vector search
            query_vector: Embedding vector for dense search
            limit: Maximum number of results
            score_threshold: Minimum combined score threshold
            doc_id_filter: Optional filter by document ID

        Returns:
            List of results with combined scores
        """
        try:
            # Create sparse vector from query for keyword search
            sparse_query = self.create_sparse_vector(query)

            # Prepare filter
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="type",
                        match=models.MatchValue(value="chunk")
                    )
                ]
            )

            if doc_id_filter:
                query_filter.must.append(
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id_filter)
                    )
                )

            # Use Qdrant's Query API with Reciprocal Rank Fusion
            if sparse_query.indices:  # If we have sparse terms
                query_result = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        # Prefetch with dense vectors
                        models.Prefetch(
                            query=query_vector,
                            using="dense",
                            limit=limit * 2,
                            filter=query_filter
                        ),
                        # Prefetch with sparse vectors
                        models.Prefetch(
                            query=sparse_query,
                            using="sparse",
                            limit=limit * 2,
                            filter=query_filter
                        )
                    ],
                    # Combine with Reciprocal Rank Fusion
                    query=models.FusionQuery(
                        fusion=models.Fusion.RRF
                    ),
                    limit=limit,
                    with_payload=True
                )
            else:
                # Fallback to dense search only if no sparse terms
                query_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using="dense",
                    filter=query_filter,
                    limit=limit,
                    with_payload=True
                )

            # Format results
            formatted_results = []
            for point in query_result.points:
                formatted_results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                })

            logger.debug(f"Qdrant hybrid search found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to perform Qdrant hybrid search: {e}")
            # Fallback to regular vector search
            return self.search_similar_chunks(
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                doc_id_filter=doc_id_filter
            )
    
    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document information and statistics."""
        try:
            # Get document summary
            summary_id = self.generate_point_id(f"{doc_id}_summary")
            summary_result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[summary_id]
            )
            
            if not summary_result:
                logger.warning(f"Document summary not found: {doc_id}")
                return None
            
            summary_data = summary_result[0].payload
            
            # Get chunk count
            chunk_filter = models.Filter(
                must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value="chunk")),
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                ]
            )
            
            chunk_count = self.client.count(
                collection_name=self.collection_name,
                count_filter=chunk_filter
            ).count
            
            # Get page count
            page_filter = models.Filter(
                must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value="page")),
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                ]
            )
            
            page_count = self.client.count(
                collection_name=self.collection_name,
                count_filter=page_filter
            ).count
            
            return {
                "doc_id": doc_id,
                "summary": summary_data,
                "actual_chunk_count": chunk_count,
                "actual_page_count": page_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get document info for {doc_id}: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete all data for a document."""
        try:
            # Delete by doc_id filter
            delete_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id)
                    )
                ]
            )
            
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=delete_filter)
            )
            
            logger.info(f"Deleted document {doc_id} from Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance.value
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
