"""
LLM Factory for POIROT System
=============================

Centralized factory for creating LLM instances that support:
- Google Gemini API (cloud-based)
- DeepSeek API (cloud-based, OpenAI-compatible)
- LM Studio (local, OpenAI-compatible API)
- Ollama (local, OpenAI-compatible API)

This allows POIROT to run with different providers based on user preference,
avoiding API rate limits and enabling offline operation when needed.

Usage:
    from .llm_factory import LLMFactory
    
    # For Gemini (default)
    llm = LLMFactory.create_chat_llm(model_name="gemini-2.5-flash")
    
    # For DeepSeek
    llm = LLMFactory.create_chat_llm(
        model_name="deepseek-chat",
        provider="deepseek"
    )
    
    # For LM Studio (local)
    llm = LLMFactory.create_chat_llm(
        model_name="gpt-oss-20b",
        use_local=True
    )
    
    # For Ollama (local)
    llm = LLMFactory.create_chat_llm(
        model_name="deepseek-r1:32b-opt",
        provider="ollama"
    )
"""

import os
from typing import Optional, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class LLMFactory:
    """
    Factory class for creating LLM instances.
    
    Supports five backends:
    - Gemini API: Uses langchain_google_genai.ChatGoogleGenerativeAI
    - OpenAI API: Uses langchain_openai.ChatOpenAI
    - DeepSeek API: Uses langchain_openai.ChatOpenAI with DeepSeek endpoint
    - LM Studio: Uses langchain_openai.ChatOpenAI with local endpoint
    - Ollama: Uses langchain_openai.ChatOpenAI with Ollama local endpoint
    """
    
    # Default LM Studio configuration
    DEFAULT_LOCAL_BASE_URL = "http://localhost:1234/v1"
    DEFAULT_LOCAL_MODEL = "gpt-oss-20b"
    DEFAULT_LOCAL_API_KEY = "lm-studio"  # LM Studio doesn't require a real key
    
    # DeepSeek configuration
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
    
    # Ollama configuration
    DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
    DEFAULT_OLLAMA_MODEL = "deepseek-r1:32b-opt"  # Optimized DeepSeek R1 32B
    FALLBACK_OLLAMA_MODEL = "gpt-oss:20b"  # Alternative OSS model
    DEFAULT_OLLAMA_API_KEY = "ollama"  # Ollama doesn't require a real key
    
    @classmethod
    def create_chat_llm(
        cls,
        model_name: str = "gemini-2.5-flash",
        use_local: bool = False,
        local_model_name: Optional[str] = None,
        provider: str = "gemini",  # "gemini", "deepseek", "local", or "ollama"
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        num_ctx: int = 131072,  # Ollama context window (128K). Only used when provider="ollama"
        **kwargs
    ) -> Any:
        """
        Create a Chat LLM instance.

        Args:
            model_name: Model name for the provider
            use_local: If True, use LM Studio (legacy support, same as provider="local")
            local_model_name: Model name for LM Studio (default: gpt-oss-20b)
            provider: LLM provider - "gemini", "openai", "deepseek", "local", or "ollama"
            temperature: Temperature for generation (default: 0)
            max_tokens: Maximum tokens for generation (optional)
            base_url: Custom base URL (optional)
            num_ctx: Context window size for Ollama models (default: 131072 = 128K tokens).
                     Ignored for non-Ollama providers. Ollama's built-in default is only 2048,
                     so always set this explicitly for POIROT workloads.
            **kwargs: Additional arguments passed to the LLM constructor

        Returns:
            ChatGoogleGenerativeAI or ChatOpenAI instance
        """
        # Legacy support: use_local=True is equivalent to provider="local"
        if use_local:
            provider = "local"
        
        if provider == "local":
            # Use LM Studio via OpenAI-compatible API
            actual_model = local_model_name or cls.DEFAULT_LOCAL_MODEL
            actual_base_url = base_url or cls.DEFAULT_LOCAL_BASE_URL
            
            llm_kwargs = {
                "model": actual_model,
                "base_url": actual_base_url,
                "api_key": cls.DEFAULT_LOCAL_API_KEY,
                "temperature": temperature,
            }
            
            if max_tokens:
                llm_kwargs["max_tokens"] = max_tokens
                
            llm_kwargs.update(kwargs)
            
            return ChatOpenAI(**llm_kwargs)
        
        elif provider == "deepseek":
            # Use DeepSeek API via OpenAI-compatible endpoint
            # Always use DeepSeek model - ignore any Gemini model names from DB
            if model_name.startswith("gemini") or model_name.startswith("models/gemini"):
                actual_model = cls.DEFAULT_DEEPSEEK_MODEL
            else:
                actual_model = model_name  # Allow explicit DeepSeek model override
            actual_base_url = base_url or cls.DEEPSEEK_BASE_URL
            
            # Get DeepSeek API key from environment
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not found in environment variables. Please set it in your .env file.")
            
            llm_kwargs = {
                "model": actual_model,
                "base_url": actual_base_url,
                "api_key": api_key,
                "temperature": temperature,
            }
            
            if max_tokens:
                llm_kwargs["max_tokens"] = max_tokens
                
            llm_kwargs.update(kwargs)
            
            return ChatOpenAI(**llm_kwargs)
        
        elif provider == "ollama":
            # Use Ollama via OpenAI-compatible API
            # Override Gemini model names from DB with Ollama model
            if model_name.startswith("gemini") or model_name.startswith("models/gemini"):
                raw_model = local_model_name or cls.DEFAULT_OLLAMA_MODEL
            else:
                raw_model = model_name  # Allow explicit Ollama model override
            # Auto-convert LM Studio default name to Ollama format (dash → colon)
            if raw_model == cls.DEFAULT_LOCAL_MODEL:
                actual_model = cls.FALLBACK_OLLAMA_MODEL
            else:
                actual_model = raw_model
            actual_base_url = base_url or cls.DEFAULT_OLLAMA_BASE_URL

            # num_ctx is passed via extra_body so the OpenAI-compatible client
            # includes it as a raw field in the HTTP request body.  Ollama's
            # /v1/chat/completions endpoint forwards the "options" dict to the
            # underlying llama.cpp runtime, which respects num_ctx at load time.
            # extra_body must be a top-level kwarg (not inside model_kwargs) —
            # newer langchain_openai versions require explicit top-level params.
            llm_kwargs = {
                "model": actual_model,
                "base_url": actual_base_url,
                "api_key": cls.DEFAULT_OLLAMA_API_KEY,
                "temperature": temperature,
                "extra_body": {"options": {"num_ctx": num_ctx}},
            }

            if max_tokens:
                llm_kwargs["max_tokens"] = max_tokens

            llm_kwargs.update(kwargs)

            return ChatOpenAI(**llm_kwargs)
        
        elif provider == "openai":
            # Use OpenAI API
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

            llm_kwargs = {
                "model": model_name,
                "temperature": temperature,
                "api_key": api_key,
            }

            if max_tokens:
                llm_kwargs["max_tokens"] = max_tokens

            llm_kwargs.update(kwargs)

            return ChatOpenAI(**llm_kwargs)

        else:
            # Default: Use Gemini API
            llm_kwargs = {
                "model": model_name,
                "temperature": temperature,
            }

            if max_tokens:
                llm_kwargs["max_tokens"] = max_tokens

            llm_kwargs.update(kwargs)

            return ChatGoogleGenerativeAI(**llm_kwargs)
    
    @classmethod
    def create_chat_llm_with_tools(
        cls,
        tools: List[Any],
        model_name: str = "gemini-2.5-flash",
        use_local: bool = False,
        local_model_name: Optional[str] = None,
        provider: str = "gemini",
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        num_ctx: int = 131072,
        **kwargs
    ) -> Any:
        """
        Create a Chat LLM instance with tools bound.

        Args:
            tools: List of LangChain tools to bind
            model_name: Model name for the provider
            use_local: If True, use LM Studio instead of Gemini
            local_model_name: Model name for LM Studio (default: gpt-oss-20b)
            provider: LLM provider - "gemini", "openai", "deepseek", "local", or "ollama"
            temperature: Temperature for generation (default: 0)
            max_tokens: Maximum tokens for generation (optional)
            base_url: Custom base URL
            num_ctx: Context window size for Ollama (default: 131072 = 128K). Ignored for other providers.
            **kwargs: Additional arguments passed to the LLM constructor

        Returns:
            LLM instance with tools bound
        """
        llm = cls.create_chat_llm(
            model_name=model_name,
            use_local=use_local,
            local_model_name=local_model_name,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            num_ctx=num_ctx,
            **kwargs
        )
        
        return llm.bind_tools(tools)
    
    @classmethod
    def get_model_info(
        cls, 
        use_local: bool, 
        model_name: str, 
        local_model_name: Optional[str] = None,
        provider: str = "gemini"
    ) -> str:
        """
        Get a descriptive string about the model being used.
        
        Args:
            use_local: Whether using local LLM
            model_name: Model name
            local_model_name: Local model name
            provider: LLM provider
            
        Returns:
            Descriptive string like "gemini-2.5-flash (API)" or "deepseek-chat (DeepSeek API)"
        """
        if use_local:
            provider = "local"
            
        if provider == "local":
            actual_model = local_model_name or cls.DEFAULT_LOCAL_MODEL
            return f"{actual_model} (LM Studio - Local)"
        elif provider == "openai":
            return f"{model_name} (OpenAI API)"
        elif provider == "deepseek":
            # Show actual DeepSeek model, not the Gemini model from DB
            if model_name.startswith("gemini") or model_name.startswith("models/gemini"):
                actual_model = cls.DEFAULT_DEEPSEEK_MODEL
            else:
                actual_model = model_name
            return f"{actual_model} (DeepSeek API)"
        elif provider == "ollama":
            if model_name.startswith("gemini") or model_name.startswith("models/gemini"):
                raw_model = local_model_name or cls.DEFAULT_OLLAMA_MODEL
            else:
                raw_model = model_name
            # Auto-convert LM Studio default name to Ollama format (dash → colon)
            if raw_model == cls.DEFAULT_LOCAL_MODEL:
                actual_model = cls.FALLBACK_OLLAMA_MODEL
            else:
                actual_model = raw_model
            return f"{actual_model} (Ollama - Local)"
        else:
            return f"{model_name} (Gemini API)"
