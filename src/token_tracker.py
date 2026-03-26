"""
Token Tracker for POIROT System
================================

Centralized token counting for all LLM calls in POIROT.
Extracts token usage from Gemini API responses via LangChain's usage_metadata.

Usage:
    from src.token_tracker import TokenTracker, extract_tokens_from_response
    
    # Global tracker
    tracker = TokenTracker()
    
    response = llm.invoke(messages)
    tracker.add_from_response(response)
    
    print(f"Total tokens: {tracker.total_tokens}")
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import threading


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0


class TokenTracker:
    """
    Thread-safe token tracker for accumulating usage across multiple LLM calls.
    
    Attributes:
        input_tokens: Total input tokens across all calls
        output_tokens: Total output tokens across all calls
        total_tokens: Total tokens (input + output) across all calls
        call_count: Number of LLM calls tracked
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_tokens: int = 0
        self.cache_read_tokens: int = 0
        self.reasoning_tokens: int = 0
        self.call_count: int = 0
    
    def add(self, usage: TokenUsage) -> None:
        """Add token usage from a single call."""
        with self._lock:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.total_tokens += usage.total_tokens
            self.cache_read_tokens += usage.cache_read_tokens
            self.reasoning_tokens += usage.reasoning_tokens
            self.call_count += 1
    
    def add_from_response(self, response: Any) -> TokenUsage:
        """
        Extract token usage from a LangChain LLM response and add to tracker.
        
        Works with:
        - ChatGoogleGenerativeAI responses (Gemini)
        - ChatOpenAI responses (OpenAI/LM Studio)
        
        Args:
            response: LangChain AIMessage or similar response object
            
        Returns:
            TokenUsage object with extracted tokens
        """
        usage = extract_tokens_from_response(response)
        self.add(usage)
        return usage
    
    def reset(self) -> None:
        """Reset all counters to zero."""
        with self._lock:
            self.input_tokens = 0
            self.output_tokens = 0
            self.total_tokens = 0
            self.cache_read_tokens = 0
            self.reasoning_tokens = 0
            self.call_count = 0
    
    def get_summary(self) -> Dict[str, int]:
        """Get a dictionary summary of token usage."""
        with self._lock:
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "call_count": self.call_count
            }
    
    def __repr__(self) -> str:
        return (
            f"TokenTracker(total={self.total_tokens}, "
            f"input={self.input_tokens}, output={self.output_tokens}, "
            f"calls={self.call_count})"
        )


def extract_tokens_from_response(response: Any) -> TokenUsage:
    """
    Extract token usage from a LangChain LLM response.
    
    Supports:
    - Gemini API via ChatGoogleGenerativeAI (usage_metadata attribute)
    - OpenAI API via ChatOpenAI (usage_metadata attribute)
    - LM Studio via ChatOpenAI (may or may not have usage info)
    
    Args:
        response: LangChain AIMessage or similar response object
        
    Returns:
        TokenUsage dataclass with extracted values (zeros if not available)
    """
    usage = TokenUsage()
    
    # Try to get usage_metadata (LangChain standard)
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        metadata = response.usage_metadata
        
        # Handle dict-like access
        if isinstance(metadata, dict):
            usage.input_tokens = metadata.get('input_tokens', 0)
            usage.output_tokens = metadata.get('output_tokens', 0)
            usage.total_tokens = metadata.get('total_tokens', 0)
            
            # Gemini-specific details
            input_details = metadata.get('input_token_details', {})
            if input_details:
                usage.cache_read_tokens = input_details.get('cache_read', 0)
            
            output_details = metadata.get('output_token_details', {})
            if output_details:
                usage.reasoning_tokens = output_details.get('reasoning', 0)
        
        # Handle object-like access (some LangChain versions)
        elif hasattr(metadata, 'input_tokens'):
            usage.input_tokens = getattr(metadata, 'input_tokens', 0)
            usage.output_tokens = getattr(metadata, 'output_tokens', 0)
            usage.total_tokens = getattr(metadata, 'total_tokens', 0)
    
    # Fallback: try response_metadata (older format)
    elif hasattr(response, 'response_metadata') and response.response_metadata:
        metadata = response.response_metadata
        
        if isinstance(metadata, dict):
            # Some providers put usage in response_metadata
            usage_data = metadata.get('usage', metadata.get('usage_metadata', {}))
            if usage_data:
                usage.input_tokens = usage_data.get('prompt_tokens', 
                                     usage_data.get('input_tokens', 0))
                usage.output_tokens = usage_data.get('completion_tokens',
                                      usage_data.get('output_tokens', 0))
                usage.total_tokens = usage_data.get('total_tokens', 
                                     usage.input_tokens + usage.output_tokens)
    
    # Ensure total_tokens is set even if only input/output were provided
    if usage.total_tokens == 0 and (usage.input_tokens > 0 or usage.output_tokens > 0):
        usage.total_tokens = usage.input_tokens + usage.output_tokens
    
    return usage


# Global tracker instance (optional convenience)
_global_tracker: Optional[TokenTracker] = None


def get_global_tracker() -> TokenTracker:
    """Get or create the global token tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TokenTracker()
    return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global token tracker."""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset()
