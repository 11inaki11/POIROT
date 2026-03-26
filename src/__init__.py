"""
POIROT: Automated Forensic Analysis for Multi-Agent AI Systems
===============================================================

Public API entry point. The primary interface is `run_poirot()`.

Example usage::

    import poirot

    results = poirot.run_poirot(
        database_path="my_system.db",
        system_name="StockTradingBot",
        system_description="A multi-agent stock trading system...",
        provider="gemini",
        api_key="YOUR_API_KEY",
    )

Version: 1.1.0
"""

import os
from typing import Any, Dict, List, Optional

from .poirot_agent import POIROTAgent
from .poirot_pipeline import POIROTPipeline

__all__ = ["run_poirot", "POIROTAgent", "POIROTPipeline"]
__version__ = "1.1.0"

# ---------------------------------------------------------------------------
# Provider → default model mapping
# ---------------------------------------------------------------------------
_PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "gemini":   "gemini-2.5-pro",
    "deepseek": "deepseek-chat",
    "local":    "gpt-oss-20b",
    "ollama":   "gpt-oss:20b",
}


def run_poirot(
    # ── Mandatory ──────────────────────────────────────────────────────────
    database_path: str,
    system_name: str,
    system_description: str,
    provider: str = "gemini",
    api_key: Optional[str] = None,
    # ── Optional — basic ───────────────────────────────────────────────────
    session_id: Optional[str] = None,
    output_dir: str = "poirot_results",
    model: Optional[str] = None,
    ignore_list: Optional[List[str]] = None,
    # ── Optional — message context ─────────────────────────────────────────
    include_tool_calls: bool = False,
    include_broadcast_messages: bool = False,
    # ── Optional — context window ──────────────────────────────────────────
    ollama_num_ctx: int = 131_072,
    token_budget: int = 95_000,
    # ── Optional — phase 2 ─────────────────────────────────────────────────
    max_agent_messages: int = 8,
    # ── Optional — retries / delay ─────────────────────────────────────────
    api_call_delay: float = 0.0,
    max_llm_retries: int = 5,
    retry_delay_503: int = 30,
    retry_delay_429: int = 60,
) -> Dict[str, Any]:
    """Run a complete POIROT forensic analysis on a multi-agent session.

    Args:
        database_path: Path to the SQLite database containing session data.
        system_name: Short name of the multi-agent system (e.g. "StockTradingBot").
        system_description: Textual description of the system architecture and agents.
        provider: LLM provider — "gemini", "deepseek", "local", or "ollama".
        api_key: API key for the provider. Not required for "local" or "ollama".
        session_id: Specific session to analyze. Defaults to the most recent session.
        output_dir: Directory where results and intermediate files are written.
        model: Override the default model for the selected provider.
        ignore_list: Component names to exclude from error vector analysis.
        include_tool_calls: Include tool call messages in agent context windows.
        include_broadcast_messages: Include broadcast messages (sent to all agents).
        ollama_num_ctx: Context window size for Ollama models (tokens).
        token_budget: Maximum tokens for agent context windows in Phase 2.
        max_agent_messages: Maximum LLM calls per agent in Phase 2.
        api_call_delay: Seconds to wait between LLM calls (rate limiting).
        max_llm_retries: Maximum retry attempts on transient API errors.
        retry_delay_503: Seconds to wait after a 503 UNAVAILABLE error.
        retry_delay_429: Seconds to wait after a 429 RESOURCE_EXHAUSTED error.

    Returns:
        Dictionary with full analysis results including votes and hazard vector.

    Raises:
        ValueError: If an unknown provider is specified.
    """
    valid_providers = set(_PROVIDER_DEFAULT_MODELS)
    if provider not in valid_providers:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Valid options: {sorted(valid_providers)}"
        )

    # Set API key in the correct environment variable
    if api_key:
        if provider == "deepseek":
            os.environ["DEEPSEEK_API_KEY"] = api_key
        else:
            os.environ["GOOGLE_API_KEY"] = api_key

    resolved_model = model if model is not None else _PROVIDER_DEFAULT_MODELS[provider]
    use_local_llm = provider == "local"

    pipeline = POIROTPipeline(
        system_name=system_name,
        system_description=system_description,
        database_path=database_path,
        output_dir=output_dir,
        ignore_list=ignore_list,
        llm_model=resolved_model,
        session_id=session_id,
        include_tool_calls=include_tool_calls,
        include_broadcast_messages=include_broadcast_messages,
        api_call_delay=api_call_delay,
        use_local_llm=use_local_llm,
        local_model_name=resolved_model if use_local_llm else None,
        llm_provider=provider,
        ollama_num_ctx=ollama_num_ctx,
        token_budget=token_budget,
        max_agent_messages=max_agent_messages,
        max_llm_retries=max_llm_retries,
        retry_delay_503=retry_delay_503,
        retry_delay_429=retry_delay_429,
    )

    return pipeline.run_full_analysis(specific_session_id=session_id)
