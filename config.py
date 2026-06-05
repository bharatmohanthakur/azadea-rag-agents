"""
Centralized configuration management for the conversational RAG system.
All tunable parameters are defined here to avoid hardcoding throughout the codebase.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ClarificationConfig:
    """Configuration for clarification flow."""
    max_turns: int = 3  # Maximum clarification turns before forcing completion
    session_timeout_minutes: int = 30  # Session expiration timeout
    max_questions_per_session: int = 4  # Maximum questions to ask in one turn
    min_answer_length: int = 2  # Minimum answer length in words
    max_answer_length: int = 100  # Maximum answer length in words
    frustration_threshold: float = 0.7  # Confidence threshold for detecting frustration

    # Patterns for detecting clarification responses (not new questions)
    new_question_starters: List[str] = field(default_factory=lambda: [
        "what", "how", "when", "where", "who", "why",
        "can", "is", "are", "do", "does", "will", "would", "should"
    ])

    greeting_patterns: List[str] = field(default_factory=lambda: [
        "hi", "hello", "hey", "thanks", "thank you",
        "okay", "ok", "sure", "great", "awesome", "perfect"
    ])

    frustration_signals: List[str] = field(default_factory=lambda: [
        "just tell me", "any", "i don't know", "i don't care", "whatever",
        "doesn't matter", "not important", "skip", "proceed", "continue",
        "just give me", "any is fine", "i don't mind"
    ])


@dataclass
class QueryProcessingConfig:
    """Configuration for query processing and routing."""
    max_history_messages: int = 20  # Maximum messages to keep in history
    max_history_for_rewrite: int = 10  # Maximum history for query rewriting
    rewrite_temperature: float = 0.0  # LLM temperature for query rewriting
    rewrite_max_tokens: int = 200  # Max tokens for rewritten query

    # Query complexity thresholds
    simple_query_max_words: int = 15  # Queries under this are likely simple
    complex_query_min_indicators: int = 2  # Minimum indicators for complex query

    # Greeting detection
    max_greeting_words: int = 5  # Maximum words for a greeting
    greeting_confidence_threshold: float = 0.8  # Confidence for LLM greeting detection


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval."""
    # Qdrant search limits
    dense_prefetch_limit: int = 15  # Dense vector prefetch limit
    sparse_prefetch_limit: int = 15  # Sparse vector prefetch limit
    fusion_limit: int = 7  # Final fusion limit

    # Reranking
    rerank_enabled: bool = True  # Enable reranking
    rerank_top_k: int = 7  # Top K after reranking

    # Contextual compression
    compression_enabled: bool = True  # Enable compression
    compression_threshold_chars: int = 3000  # Compress if context > this
    max_compressed_chars: int = 2000  # Maximum compressed context size

    # Corrective RAG
    corrective_rag_enabled: bool = True  # Enable corrective RAG
    max_correction_depth: int = 1  # Maximum correction recursion depth
    correction_timeout_seconds: float = 10.0  # Timeout for re-retrieval
    max_additional_sources: int = 3  # Max sources from re-retrieval

    # Graphiti memory
    graphiti_num_results: int = 5  # Number of facts to retrieve

    # Images
    max_images_per_chunk: int = 2  # Max images per chunk
    max_total_images: int = 3  # Max total images in response

    # Filename scoring
    filename_boost_factor: float = 0.3  # Boost score for filename match


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""
    default_temperature: float = 0.0  # Default temperature
    default_max_tokens: int = 1500  # Default max tokens
    streaming_enabled: bool = True  # Enable streaming responses

    # Timeout configuration
    llm_timeout_seconds: float = 30.0  # Timeout for LLM calls
    search_timeout_seconds: float = 30.0  # Timeout for search operations

    # Retry configuration
    max_retries: int = 3  # Maximum retries for failed calls
    initial_retry_delay: float = 1.0  # Initial retry delay in seconds
    max_retry_delay: float = 10.0  # Maximum retry delay


@dataclass
class QualityGateConfig:
    """Configuration for answer quality gate."""
    # Confidence thresholds
    high_confidence_threshold: float = 0.8  # High confidence
    medium_confidence_threshold: float = 0.6  # Medium confidence
    low_confidence_threshold: float = 0.4  # Low confidence

    # Completeness thresholds
    high_completeness_threshold: float = 0.8  # High completeness
    medium_completeness_threshold: float = 0.6  # Medium completeness
    low_completeness_threshold: float = 0.4  # Low completeness

    # Grounding thresholds
    high_grounding_threshold: float = 0.7  # High grounding
    medium_grounding_threshold: float = 0.5  # Medium grounding

    # Iteration limits
    max_improvement_iterations: int = 2  # Max iterations for improvement

    # Disclaimers
    enable_disclaimers: bool = True  # Add disclaimers for low quality


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True  # Enable caching
    ttl_seconds: int = 3600  # Cache TTL (1 hour default)
    max_cache_size: int = 1000  # Maximum cache entries

    # Query similarity threshold for cache hits
    similarity_threshold: float = 0.95  # Cosine similarity for cache hit


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    enable_parallel_search: bool = True  # Parallel Qdrant + Graphiti search
    enable_connection_pooling: bool = True  # Connection pooling
    max_concurrent_requests: int = 10  # Max concurrent requests

    # Rate limiting
    rate_limit_enabled: bool = False  # Enable rate limiting
    rate_limit_requests_per_minute: int = 60  # Requests per minute


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"  # Logging level
    log_to_file: bool = True  # Log to file
    log_file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_file_backup_count: int = 5  # Keep 5 backup files

    # Request tracking
    enable_request_tracking: bool = True  # Track request IDs
    log_request_details: bool = True  # Log detailed request info


@dataclass
class RAGSystemConfig:
    """Main configuration for the RAG system."""
    clarification: ClarificationConfig = field(default_factory=ClarificationConfig)
    query_processing: QueryProcessingConfig = field(default_factory=QueryProcessingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "clarification": self.clarification.__dict__,
            "query_processing": self.query_processing.__dict__,
            "retrieval": self.retrieval.__dict__,
            "llm": self.llm.__dict__,
            "quality_gate": self.quality_gate.__dict__,
            "cache": self.cache.__dict__,
            "performance": self.performance.__dict__,
            "logging": self.logging.__dict__
        }

    @classmethod
    def from_env(cls) -> "RAGSystemConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Override from environment if present
        if os.getenv("CLARIFICATION_MAX_TURNS"):
            config.clarification.max_turns = int(os.getenv("CLARIFICATION_MAX_TURNS"))

        if os.getenv("MAX_HISTORY_MESSAGES"):
            config.query_processing.max_history_messages = int(os.getenv("MAX_HISTORY_MESSAGES"))

        if os.getenv("RERANKING_ENABLED"):
            config.retrieval.rerank_enabled = os.getenv("RERANKING_ENABLED").lower() == "true"

        if os.getenv("CORRECTIVE_RAG_ENABLED"):
            config.retrieval.corrective_rag_enabled = os.getenv("CORRECTIVE_RAG_ENABLED").lower() == "true"

        if os.getenv("CACHE_ENABLED"):
            config.cache.enabled = os.getenv("CACHE_ENABLED").lower() == "true"

        if os.getenv("CACHE_TTL_SECONDS"):
            config.cache.ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS"))

        return config


# Global configuration instance
_config: RAGSystemConfig = None


def get_config() -> RAGSystemConfig:
    """Get or create global configuration instance."""
    global _config
    if _config is None:
        _config = RAGSystemConfig.from_env()
    return _config


def reload_config():
    """Reload configuration from environment."""
    global _config
    _config = RAGSystemConfig.from_env()


# Helper functions for backward compatibility
def get_clarification_config() -> ClarificationConfig:
    """Get clarification configuration."""
    return get_config().clarification


def get_retrieval_config() -> RetrievalConfig:
    """Get retrieval configuration."""
    return get_config().retrieval


def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return get_config().llm


def get_quality_gate_config() -> QualityGateConfig:
    """Get quality gate configuration."""
    return get_config().quality_gate


def get_query_processing_config() -> QueryProcessingConfig:
    """Get query processing configuration."""
    return get_config().query_processing
