"""
Intelligent pattern matching for query classification.
Replaces hardcoded patterns with configurable, maintainable patterns.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger("PatternMatcher")


class QueryType(Enum):
    """Query type classification."""
    GREETING = "greeting"
    THANKS = "thanks"
    CASUAL = "casual"
    QUESTION = "question"
    STATEMENT = "statement"
    COMMAND = "command"


class QueryComplexity(Enum):
    """Query complexity classification."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class PatternMatch:
    """Result of pattern matching."""
    matched: bool
    query_type: Optional[QueryType] = None
    confidence: float = 0.0  # 0.0 to 1.0
    matched_patterns: List[str] = None

    def __post_init__(self):
        if self.matched_patterns is None:
            self.matched_patterns = []


class PatternMatcher:
    """
    Intelligent pattern matcher for query classification.
    Uses regex patterns and heuristics for fast, accurate classification.
    """

    def __init__(self):
        """Initialize pattern matcher with default patterns."""
        # Note: Patterns are hardcoded here for now to avoid circular dependencies
        # These can be made configurable via environment variables later

        # Greeting patterns
        self.greeting_patterns = {
            "basic": [
                r"^(hi|hello|hey|good\s+(morning|afternoon|evening|day))$",
                r"^(hi|hello|hey)\s+(there|everyone|all)$"
            ],
            "formal": [
                r"^greetings$",
                r"^good\s+to\s+see\s+you$"
            ]
        }

        # Thanks patterns
        self.thanks_patterns = [
            r"^(thanks?|thank\s+you|thx|ty)$",
            r"^(thanks?|thank\s+you)\s+(so\s+)?much$",
            r"^(appreciate|appreciated)\s+it$"
        ]

        # Casual acknowledgment patterns
        self.casual_patterns = [
            r"^(ok|okay|sure|alright|got\s+it|understood)$",
            r"^(great|awesome|perfect|excellent|wonderful)$",
            r"^(yes|yep|yeah|nope|no)$"
        ]

        # Question indicators (for detecting new questions vs. answers)
        self.question_starters = [
            r"^(what|how|when|where|who|why|which)\s+",
            r"^(can|could|would|should|will|do|does|did|is|are|was|were)\s+",
            r"^(tell\s+me|show\s+me|give\s+me|explain|describe)\s+"
        ]

        # Command patterns
        self.command_patterns = [
            r"^(list|show|display|tell|explain|describe|summarize|compare)\s+",
            r"^(find|search|look\s+for|get|fetch)\s+"
        ]

        # Complexity indicators
        self.complexity_indicators = {
            "simple": [
                r"^what\s+is\s+",
                r"^how\s+do\s+i\s+",
                r"^when\s+",
                r"^where\s+"
            ],
            "moderate": [
                r"\s+and\s+",
                r"\s+or\s+",
                r"compare|comparison"
            ],
            "complex": [
                r"compare\s+.+\s+(with|vs|versus|to|against)",
                r"\s+and\s+.+\s+and\s+",  # Multiple "and" conjunctions
                r"(all|every|each)\s+.+\s+in\s+",  # Aggregation queries
                r"(difference|differences)\s+between"
            ]
        }

    def _match_patterns(
        self,
        text: str,
        patterns: List[str],
        case_sensitive: bool = False
    ) -> Optional[str]:
        """
        Match text against a list of regex patterns.

        Args:
            text: Text to match
            patterns: List of regex patterns
            case_sensitive: Whether to match case-sensitively

        Returns:
            First matched pattern or None
        """
        flags = 0 if case_sensitive else re.IGNORECASE

        for pattern in patterns:
            if re.search(pattern, text, flags):
                return pattern

        return None

    def is_greeting(self, query: str) -> PatternMatch:
        """
        Check if query is a greeting.

        Args:
            query: User query

        Returns:
            PatternMatch result
        """
        query = query.strip()

        # Check basic greetings
        for category, patterns in self.greeting_patterns.items():
            matched = self._match_patterns(query, patterns)
            if matched:
                # Calculate confidence based on length (shorter = more likely greeting)
                word_count = len(query.split())
                confidence = max(0.5, min(1.0, 1.0 - (word_count - 1) * 0.1))

                return PatternMatch(
                    matched=True,
                    query_type=QueryType.GREETING,
                    confidence=confidence,
                    matched_patterns=[matched]
                )

        return PatternMatch(matched=False)

    def is_thanks(self, query: str) -> PatternMatch:
        """
        Check if query is expressing gratitude.

        Args:
            query: User query

        Returns:
            PatternMatch result
        """
        query = query.strip()

        matched = self._match_patterns(query, self.thanks_patterns)
        if matched:
            return PatternMatch(
                matched=True,
                query_type=QueryType.THANKS,
                confidence=0.95,
                matched_patterns=[matched]
            )

        return PatternMatch(matched=False)

    def is_casual(self, query: str) -> PatternMatch:
        """
        Check if query is a casual acknowledgment.

        Args:
            query: User query

        Returns:
            PatternMatch result
        """
        query = query.strip()

        matched = self._match_patterns(query, self.casual_patterns)
        if matched:
            return PatternMatch(
                matched=True,
                query_type=QueryType.CASUAL,
                confidence=0.9,
                matched_patterns=[matched]
            )

        return PatternMatch(matched=False)

    def is_question(self, query: str) -> PatternMatch:
        """
        Check if query is a question.

        Args:
            query: User query

        Returns:
            PatternMatch result
        """
        query = query.strip()

        # Check question patterns
        matched_patterns = []
        for pattern in self.question_starters:
            if re.search(pattern, query, re.IGNORECASE):
                matched_patterns.append(pattern)

        if matched_patterns:
            # Higher confidence if ends with "?"
            confidence = 0.9 if query.endswith("?") else 0.7

            return PatternMatch(
                matched=True,
                query_type=QueryType.QUESTION,
                confidence=confidence,
                matched_patterns=matched_patterns
            )

        # Check if ends with "?"
        if query.endswith("?"):
            return PatternMatch(
                matched=True,
                query_type=QueryType.QUESTION,
                confidence=0.6,
                matched_patterns=["ends_with_question_mark"]
            )

        return PatternMatch(matched=False)

    def is_command(self, query: str) -> PatternMatch:
        """
        Check if query is a command.

        Args:
            query: User query

        Returns:
            PatternMatch result
        """
        query = query.strip()

        matched = self._match_patterns(query, self.command_patterns)
        if matched:
            return PatternMatch(
                matched=True,
                query_type=QueryType.COMMAND,
                confidence=0.8,
                matched_patterns=[matched]
            )

        return PatternMatch(matched=False)

    def classify_query_type(self, query: str) -> PatternMatch:
        """
        Classify query into one of the defined types.

        Args:
            query: User query

        Returns:
            PatternMatch with best classification
        """
        # Check in order of priority
        checks = [
            self.is_greeting,
            self.is_thanks,
            self.is_casual,
            self.is_question,
            self.is_command
        ]

        for check in checks:
            result = check(query)
            if result.matched and result.confidence >= 0.7:
                logger.debug(f"Query classified as {result.query_type.value} (confidence: {result.confidence})")
                return result

        # Default: statement
        return PatternMatch(
            matched=True,
            query_type=QueryType.STATEMENT,
            confidence=0.5,
            matched_patterns=[]
        )

    def assess_complexity(self, query: str) -> QueryComplexity:
        """
        Assess query complexity.

        Args:
            query: User query

        Returns:
            QueryComplexity classification
        """
        query_lower = query.lower()
        word_count = len(query.split())

        # Check complex patterns first
        for pattern in self.complexity_indicators["complex"]:
            if re.search(pattern, query_lower):
                logger.debug(f"Query classified as COMPLEX (pattern: {pattern})")
                return QueryComplexity.COMPLEX

        # Check moderate patterns
        moderate_matches = 0
        for pattern in self.complexity_indicators["moderate"]:
            if re.search(pattern, query_lower):
                moderate_matches += 1

        if moderate_matches >= 2 or (moderate_matches >= 1 and word_count > 15):
            logger.debug(f"Query classified as MODERATE (matches: {moderate_matches}, words: {word_count})")
            return QueryComplexity.MODERATE

        # Check simple patterns
        simple_query_max_words = 15  # Default threshold
        for pattern in self.complexity_indicators["simple"]:
            if re.search(pattern, query_lower):
                if word_count <= simple_query_max_words:
                    logger.debug(f"Query classified as SIMPLE (pattern: {pattern}, words: {word_count})")
                    return QueryComplexity.SIMPLE

        # Default: simple if short, moderate if longer
        if word_count <= simple_query_max_words:
            return QueryComplexity.SIMPLE
        else:
            return QueryComplexity.MODERATE

    def is_greeting_or_casual(self, query: str) -> bool:
        """
        Quick check if query is greeting or casual (backward compatibility).

        Args:
            query: User query

        Returns:
            True if greeting or casual
        """
        result = self.classify_query_type(query)
        return result.query_type in [QueryType.GREETING, QueryType.THANKS, QueryType.CASUAL]

    def starts_with_question_word(self, query: str) -> bool:
        """
        Check if query starts with a question word.

        Args:
            query: User query

        Returns:
            True if starts with question word
        """
        query_lower = query.lower().strip()
        first_words = query_lower.split()[:2]

        # Default question starters (can be made configurable later)
        question_words = ["what", "how", "when", "where", "who", "why",
                         "can", "is", "are", "do", "does", "will", "would", "should"]
        return any(word in first_words for word in question_words)


# Global instance
_pattern_matcher: Optional[PatternMatcher] = None


def get_pattern_matcher() -> PatternMatcher:
    """Get or create global pattern matcher instance."""
    global _pattern_matcher
    if _pattern_matcher is None:
        _pattern_matcher = PatternMatcher()
    return _pattern_matcher


# Backward compatibility functions
def is_greeting_or_casual(query: str) -> bool:
    """Check if query is greeting or casual (backward compatibility)."""
    return get_pattern_matcher().is_greeting_or_casual(query)
