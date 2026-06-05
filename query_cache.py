"""
Intelligent query cache with semantic similarity matching.
Caches query results to avoid redundant searches and LLM calls.
"""

import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger("QueryCache")


@dataclass
class CachedQueryResult:
    """Cached query result with metadata."""
    query: str
    query_embedding: List[float]
    result: Dict[str, Any]
    user_id: str
    timestamp: float
    hit_count: int = 0
    last_access: float = None

    def __post_init__(self):
        if self.last_access is None:
            self.last_access = self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedQueryResult":
        """Create from dictionary."""
        return cls(**data)


class QueryCache:
    """
    Intelligent query cache with semantic similarity matching.
    Uses cosine similarity to match similar queries even if wording differs.
    """

    def __init__(
        self,
        embedding_function,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        similarity_threshold: float = 0.95
    ):
        """
        Initialize query cache.

        Args:
            embedding_function: Function to embed queries (callable that takes string, returns List[float])
            ttl_seconds: Time-to-live for cache entries (default 1 hour)
            max_size: Maximum number of cached entries
            similarity_threshold: Minimum cosine similarity for cache hit (0.95 = very similar)
        """
        self.embedding_function = embedding_function
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

        # In-memory cache: query_hash -> CachedQueryResult
        self._cache: Dict[str, CachedQueryResult] = {}

        # Embeddings index for fast similarity search: List[(query_hash, embedding)]
        self._embeddings_index: List[Tuple[str, np.ndarray]] = []

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_queries": 0
        }

    def _compute_hash(self, query: str, user_id: str) -> str:
        """Compute hash for query and user_id."""
        key = f"{user_id}:{query.lower().strip()}"
        return hashlib.sha256(key.encode()).hexdigest()

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _find_similar_query(
        self,
        query_embedding: np.ndarray,
        user_id: str
    ) -> Optional[str]:
        """
        Find a cached query with similar embedding.

        Args:
            query_embedding: Embedding of the query
            user_id: User ID (only match within same user)

        Returns:
            Query hash if similar query found, None otherwise
        """
        if not self._embeddings_index:
            return None

        best_match = None
        best_similarity = 0.0

        for query_hash, cached_embedding in self._embeddings_index:
            # Only match queries from same user
            cached_entry = self._cache.get(query_hash)
            if not cached_entry or cached_entry.user_id != user_id:
                continue

            similarity = self._cosine_similarity(query_embedding, cached_embedding)

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = query_hash

        if best_match:
            logger.info(f"Found similar cached query with similarity {best_similarity:.3f}")

        return best_match

    def _evict_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_hashes = []

        for query_hash, entry in self._cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_hashes.append(query_hash)

        for query_hash in expired_hashes:
            del self._cache[query_hash]
            # Remove from embeddings index
            self._embeddings_index = [
                (h, e) for h, e in self._embeddings_index if h != query_hash
            ]
            self.stats["evictions"] += 1

        if expired_hashes:
            logger.info(f"Evicted {len(expired_hashes)} expired cache entries")

    def _evict_lru(self):
        """Evict least recently used entry if cache is full."""
        if len(self._cache) < self.max_size:
            return

        # Find LRU entry
        lru_hash = None
        lru_access_time = float('inf')

        for query_hash, entry in self._cache.items():
            if entry.last_access < lru_access_time:
                lru_access_time = entry.last_access
                lru_hash = query_hash

        if lru_hash:
            del self._cache[lru_hash]
            self._embeddings_index = [
                (h, e) for h, e in self._embeddings_index if h != lru_hash
            ]
            self.stats["evictions"] += 1
            logger.debug(f"Evicted LRU cache entry: {lru_hash}")

    def get(
        self,
        query: str,
        user_id: str,
        use_similarity: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result for query.

        Args:
            query: User query
            user_id: User ID
            use_similarity: Use semantic similarity matching (default True)

        Returns:
            Cached result if found, None otherwise
        """
        self.stats["total_queries"] += 1

        # Clean expired entries periodically
        if self.stats["total_queries"] % 100 == 0:
            self._evict_expired()

        query_normalized = query.lower().strip()

        # Try exact match first
        query_hash = self._compute_hash(query_normalized, user_id)
        if query_hash in self._cache:
            entry = self._cache[query_hash]
            entry.hit_count += 1
            entry.last_access = time.time()
            self.stats["hits"] += 1
            logger.info(f"Cache HIT (exact): {query[:50]}... (hit count: {entry.hit_count})")
            return entry.result

        # Try similarity matching if enabled
        if use_similarity:
            try:
                # Embed query
                query_embedding = self.embedding_function(query_normalized)
                query_embedding_np = np.array(query_embedding)

                # Find similar query
                similar_hash = self._find_similar_query(query_embedding_np, user_id)

                if similar_hash and similar_hash in self._cache:
                    entry = self._cache[similar_hash]
                    entry.hit_count += 1
                    entry.last_access = time.time()
                    self.stats["hits"] += 1
                    logger.info(f"Cache HIT (similar): {query[:50]}... matched with {entry.query[:50]}...")
                    return entry.result

            except Exception as e:
                logger.error(f"Error in similarity matching: {e}")

        self.stats["misses"] += 1
        logger.debug(f"Cache MISS: {query[:50]}...")
        return None

    def set(
        self,
        query: str,
        user_id: str,
        result: Dict[str, Any]
    ):
        """
        Cache a query result.

        Args:
            query: User query
            user_id: User ID
            result: Query result to cache
        """
        query_normalized = query.lower().strip()
        query_hash = self._compute_hash(query_normalized, user_id)

        try:
            # Embed query
            query_embedding = self.embedding_function(query_normalized)
            query_embedding_np = np.array(query_embedding)

            # Evict if necessary
            self._evict_lru()

            # Create cache entry
            entry = CachedQueryResult(
                query=query_normalized,
                query_embedding=query_embedding,
                result=result,
                user_id=user_id,
                timestamp=time.time()
            )

            # Store in cache
            self._cache[query_hash] = entry

            # Add to embeddings index
            self._embeddings_index.append((query_hash, query_embedding_np))

            logger.debug(f"Cached query: {query[:50]}...")

        except Exception as e:
            logger.error(f"Error caching query: {e}")

    def clear(self, user_id: Optional[str] = None):
        """
        Clear cache.

        Args:
            user_id: If provided, clear only this user's cache
        """
        if user_id:
            # Clear specific user's cache
            hashes_to_remove = [
                h for h, e in self._cache.items() if e.user_id == user_id
            ]
            for query_hash in hashes_to_remove:
                del self._cache[query_hash]
            self._embeddings_index = [
                (h, e) for h, e in self._embeddings_index if h not in hashes_to_remove
            ]
            logger.info(f"Cleared cache for user {user_id}")
        else:
            # Clear all
            self._cache.clear()
            self._embeddings_index.clear()
            logger.info("Cleared entire cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_queries = self.stats["total_queries"]
        hits = self.stats["hits"]
        hit_rate = (hits / total_queries * 100) if total_queries > 0 else 0.0

        return {
            "total_queries": total_queries,
            "hits": hits,
            "misses": self.stats["misses"],
            "hit_rate_percent": round(hit_rate, 2),
            "cached_entries": len(self._cache),
            "evictions": self.stats["evictions"]
        }


# Global cache instance (will be initialized with embedding function)
_query_cache: Optional[QueryCache] = None


def get_query_cache() -> Optional[QueryCache]:
    """Get global query cache instance."""
    return _query_cache


def init_query_cache(
    embedding_function,
    ttl_seconds: int = 3600,
    max_size: int = 1000,
    similarity_threshold: float = 0.95
):
    """
    Initialize global query cache.

    Args:
        embedding_function: Function to embed queries
        ttl_seconds: Cache TTL
        max_size: Maximum cache size
        similarity_threshold: Similarity threshold for matching
    """
    global _query_cache
    _query_cache = QueryCache(
        embedding_function=embedding_function,
        ttl_seconds=ttl_seconds,
        max_size=max_size,
        similarity_threshold=similarity_threshold
    )
    logger.info(f"Initialized query cache (TTL: {ttl_seconds}s, max size: {max_size}, threshold: {similarity_threshold})")
