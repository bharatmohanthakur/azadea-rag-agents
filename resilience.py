"""
Resilience utilities: retry logic, circuit breakers, and graceful degradation.
"""

import asyncio
import logging
import time
from typing import Callable, Any, Optional, TypeVar, List
from functools import wraps
from enum import Enum

logger = logging.getLogger("Resilience")

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Simple circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from function
        """
        if self.state == CircuitState.OPEN:
            if time.time() - (self.last_failure_time or 0) > self.recovery_timeout:
                # Try to recover
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker: Attempting recovery (half-open)")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            # Success - reset failure count
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                logger.info("Circuit breaker: Service recovered, closing circuit")
            self.failure_count = 0
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker: OPEN after {self.failure_count} failures")
            
            raise
    
    async def acall(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Async version of call."""
        if self.state == CircuitState.OPEN:
            if time.time() - (self.last_failure_time or 0) > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker: Attempting recovery (half-open)")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                logger.info("Circuit breaker: Service recovered, closing circuit")
            self.failure_count = 0
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker: OPEN after {self.failure_count} failures")
            
            raise


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function(exception, attempt_number)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                        time.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(f"All {max_retries} retries failed")
            
            raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                        await asyncio.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(f"All {max_retries} retries failed")
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def with_timeout(timeout_seconds: float):
    """
    Decorator to add timeout to async functions.
    
    Args:
        timeout_seconds: Timeout in seconds
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {timeout_seconds}s")
                raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        return wrapper
    return decorator


# Global circuit breakers for different services
_qdrant_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
_graphiti_circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
_llm_circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)


def get_qdrant_circuit() -> CircuitBreaker:
    """Get circuit breaker for Qdrant."""
    return _qdrant_circuit


def get_graphiti_circuit() -> CircuitBreaker:
    """Get circuit breaker for Graphiti."""
    return _graphiti_circuit


def get_llm_circuit() -> CircuitBreaker:
    """Get circuit breaker for LLM."""
    return _llm_circuit


