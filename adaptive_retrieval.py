"""
Adaptive retrieval control for agentic RAG system.
Implements value-based stopping and iterative retrieval with confidence checks.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("AdaptiveRetrieval")


class AdaptiveRetriever:
    """Manages adaptive retrieval with value-based stopping."""
    
    def __init__(self, retrieval_function, confidence_evaluator=None):
        """
        Initialize adaptive retriever.
        
        Args:
            retrieval_function: Function to call for retrieval (async)
            confidence_evaluator: Optional function to evaluate confidence
        """
        self.retrieval_function = retrieval_function
        self.confidence_evaluator = confidence_evaluator
    
    def calculate_retrieval_value(
        self,
        current_results: List[Dict[str, Any]],
        previous_results: List[Dict[str, Any]],
        query: str
    ) -> float:
        """
        Calculate the value added by new retrieval results.
        
        Args:
            current_results: Current retrieval results
            previous_results: Previous retrieval results
            query: Search query
        
        Returns:
            Value score 0.0 to 1.0 (higher = more value)
        """
        if not previous_results:
            # First retrieval - assume high value
            return 1.0
        
        if not current_results:
            return 0.0
        
        # Extract unique sources from both
        prev_sources = set(r.get("source", "") for r in previous_results if isinstance(r, dict))
        curr_sources = set(r.get("source", "") for r in current_results if isinstance(r, dict))
        
        # New sources found
        new_sources = curr_sources - prev_sources
        source_diversity_value = len(new_sources) / max(len(curr_sources), 1)
        
        # Check score improvements
        if previous_results:
            prev_avg_score = np.mean([r.get("score", 0) for r in previous_results if isinstance(r, dict)])
        else:
            prev_avg_score = 0
        
        curr_avg_score = np.mean([r.get("score", 0) for r in current_results if isinstance(r, dict)])
        score_improvement = max(0, curr_avg_score - prev_avg_score)
        
        # Check content diversity (simple keyword overlap)
        prev_texts = " ".join([r.get("text_snippet", "")[:200] for r in previous_results if isinstance(r, dict)])
        curr_texts = " ".join([r.get("text_snippet", "")[:200] for r in current_results if isinstance(r, dict)])
        
        # Calculate unique content
        prev_words = set(prev_texts.lower().split())
        curr_words = set(curr_texts.lower().split())
        new_words = curr_words - prev_words
        content_diversity = len(new_words) / max(len(curr_words), 1) if curr_words else 0
        
        # Combined value score
        value = (
            source_diversity_value * 0.4 +
            min(score_improvement * 2, 1.0) * 0.3 +  # Scale score improvement
            min(content_diversity, 1.0) * 0.3
        )
        
        return min(value, 1.0)
    
    def evaluate_retrieval_confidence(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> float:
        """
        Evaluate confidence in retrieval results.
        
        Args:
            results: Retrieval results
            query: Search query
        
        Returns:
            Confidence score 0.0 to 1.0
        """
        if not results:
            return 0.0
        
        # Average relevance score
        scores = [r.get("score", 0) for r in results if isinstance(r, dict)]
        if scores:
            avg_score = np.mean(scores)
            max_score = max(scores)
        else:
            return 0.0
        
        # Number of results (more results = potentially more confidence)
        result_count_score = min(len(results) / 5.0, 1.0)  # Normalize to 5 results
        
        # Source diversity
        sources = set(r.get("source", "") for r in results if isinstance(r, dict))
        source_diversity = min(len(sources) / 3.0, 1.0)  # Normalize to 3 sources
        
        # Combined confidence
        confidence = (
            avg_score * 0.5 +
            max_score * 0.2 +
            result_count_score * 0.15 +
            source_diversity * 0.15
        )
        
        return min(confidence, 1.0)
    
    async def adaptive_retrieve(
        self,
        query: str,
        user_id: str,
        initial_limit: int = 3,
        max_iterations: int = 3,
        confidence_threshold: float = 0.8,
        value_threshold: float = 0.2
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform adaptive retrieval with value-based stopping.
        
        Args:
            query: Search query
            user_id: User identifier
            initial_limit: Initial number of results to retrieve
            max_iterations: Maximum retrieval iterations
            confidence_threshold: Stop if confidence exceeds this
            value_threshold: Stop if value added is below this
        
        Returns:
            Tuple of (results, metadata)
        """
        all_results = []
        previous_results = []
        metadata = {
            "iterations": 0,
            "stopped_early": False,
            "stopping_reason": "",
            "final_confidence": 0.0,
            "total_results": 0
        }
        
        for iteration in range(max_iterations):
            metadata["iterations"] = iteration + 1
            
            # Calculate limit for this iteration
            current_limit = initial_limit + (iteration * 2)  # Increase by 2 each time
            
            # Retrieve results
            try:
                search_result = await self.retrieval_function(query, user_id)
                current_results = search_result.get("sources", [])
                
                if not current_results:
                    logger.warning(f"No results retrieved in iteration {iteration + 1}")
                    break
                
                # Combine with previous results (deduplicate by source)
                seen_sources = set()
                for result in all_results:
                    if isinstance(result, dict):
                        seen_sources.add(result.get("source", ""))
                
                # Add new results
                for result in current_results:
                    if isinstance(result, dict):
                        source = result.get("source", "")
                        if source not in seen_sources:
                            all_results.append(result)
                            seen_sources.add(source)
                
                # Evaluate confidence
                confidence = self.evaluate_retrieval_confidence(all_results, query)
                metadata["final_confidence"] = confidence
                
                # Check confidence threshold
                if confidence >= confidence_threshold:
                    metadata["stopped_early"] = True
                    metadata["stopping_reason"] = f"high_confidence_{confidence:.2f}"
                    logger.info(f"Stopped retrieval early: high confidence ({confidence:.2f})")
                    break
                
                # Calculate value added
                if iteration > 0:  # Skip first iteration
                    value = self.calculate_retrieval_value(all_results, previous_results, query)
                    
                    # Check value threshold
                    if value < value_threshold:
                        metadata["stopped_early"] = True
                        metadata["stopping_reason"] = f"low_value_{value:.2f}"
                        logger.info(f"Stopped retrieval early: low value added ({value:.2f})")
                        # Revert to previous results if they were better
                        if len(previous_results) > 0 and confidence < 0.6:
                            all_results = previous_results
                        break
                
                # Store current results as previous for next iteration
                previous_results = all_results.copy()
                
                # If we have enough results, consider stopping
                if len(all_results) >= 10:  # Enough results collected
                    metadata["stopped_early"] = True
                    metadata["stopping_reason"] = "sufficient_results"
                    logger.info(f"Stopped retrieval: sufficient results ({len(all_results)})")
                    break
                
            except Exception as e:
                logger.error(f"Retrieval iteration {iteration + 1} failed: {e}")
                break
        
        metadata["total_results"] = len(all_results)
        
        # If no results after all iterations, return empty
        if not all_results:
            logger.warning("No results retrieved after all iterations")
            return [], metadata
        
        # Sort by score (descending)
        all_results.sort(key=lambda x: x.get("score", 0) if isinstance(x, dict) else 0, reverse=True)
        
        # Return top results (limit to reasonable number)
        final_results = all_results[:10]
        
        return final_results, metadata

