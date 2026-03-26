"""
POIROT: Automated Forensic Analysis for Multi-Agent AI Systems
===============================================================

Public API entry point. Two entry points are available:

- ``run_poirot()``              — analyze a session stored in a SQLite database
- ``run_poirot_from_agents()``  — analyze live LangChain agent objects directly

Example (database)::

    import poirot

    results = poirot.run_poirot(
        database_path="my_system.db",
        system_name="StockTradingBot",
        system_description="A multi-agent stock trading system...",
        provider="gemini",
        api_key="YOUR_API_KEY",
    )

Example (LangChain agents)::

    from poirot import run_poirot_from_agents, LangChainAgentAdapter

    results = run_poirot_from_agents(
        agents=[
            LangChainAgentAdapter(agent=planner, messages=planner_result["messages"]),
            LangChainAgentAdapter(agent=executor, messages=executor_result["messages"]),
        ],
        system_name="MySystem",
        system_description="...",
        provider="gemini",
        model="gemini-2.5-pro",
        api_key="YOUR_KEY",
    )

Version: 1.1.0
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .poirot_agent import POIROTAgent
from .poirot_pipeline import POIROTPipeline
from .adapters import LangChainAgentAdapter

__all__ = [
    "run_poirot",
    "run_poirot_from_agents",
    "LangChainAgentAdapter",
    "POIROTAgent",
    "POIROTPipeline",
]
__version__ = "1.1.0"

# ---------------------------------------------------------------------------
# Provider → default model mapping
# ---------------------------------------------------------------------------
_PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "gemini":   "gemini-2.5-pro",
    "openai":   "gpt-4o",
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
    full_context: bool = False,
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
        full_context: If True, each agent receives all messages from all agents in
            the session, not just the ones they sent or received. Messages from other
            agents are presented as HumanMessage with a "From [agent_id]:" header.
            Warning: enabling this can cause context overflow in long sessions.
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
        elif provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
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
        full_context=full_context,
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


def run_poirot_from_agents(
    # ── Mandatory ──────────────────────────────────────────────────────────
    agents: List["LangChainAgentAdapter"],
    system_name: str,
    system_description: str,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    # ── Optional — basic ───────────────────────────────────────────────────
    output_dir: str = "poirot_results",
    ignore_list: Optional[List[str]] = None,
    # ── Optional — message context ─────────────────────────────────────────
    include_tool_calls: bool = False,
    include_broadcast_messages: bool = False,
    full_context: bool = False,
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
    """Run a complete POIROT forensic analysis on live LangChain agent objects.

    This entry point does not require a SQLite database. Pass the compiled
    LangGraph agent objects and their message histories directly.

    Args:
        agents: List of ``LangChainAgentAdapter`` instances (one per agent).
        system_name: Short name of the multi-agent system.
        system_description: Textual description of the system architecture.
        provider: LLM provider — ``"gemini"``, ``"openai"``, ``"deepseek"``,
            ``"local"``, or ``"ollama"``. **Required — no default.**
        model: Model identifier for the chosen provider (e.g. ``"gpt-4o"``,
            ``"gemini-2.5-pro"``). **Required — no default.**
        api_key: API key for the provider. Not required for ``"local"`` or
            ``"ollama"``.
        output_dir: Directory where results and intermediate files are written.
        ignore_list: Component names to exclude from error vector analysis.
        include_tool_calls: Include tool call messages in agent context windows.
        include_broadcast_messages: Include broadcast messages (sent to all agents).
        full_context: If True, each agent receives all messages from all agents
            in the session. Warning: can cause context overflow in long sessions.
        ollama_num_ctx: Context window size for Ollama models (tokens).
        token_budget: Maximum tokens for agent context windows in Phase 2.
        max_agent_messages: Maximum LLM calls per agent in Phase 2.
        api_call_delay: Seconds to wait between LLM calls (rate limiting).
        max_llm_retries: Maximum retry attempts on transient API errors.
        retry_delay_503: Seconds to wait after a 503 UNAVAILABLE error.
        retry_delay_429: Seconds to wait after a 429 RESOURCE_EXHAUSTED error.

    Returns:
        Dictionary with full analysis results including ``votes`` and
        ``phase0_error_space``.

    Raises:
        ValueError: If an unknown provider is specified.
    """
    from .adapters import build_session_data
    from .phase1_protocol import execute_phase1_analysis
    from .phase2_protocol import execute_phase2_analysis
    from .llm_factory import LLMFactory

    # Validate provider
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
        elif provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            os.environ["GOOGLE_API_KEY"] = api_key

    use_local_llm = provider in ("local", "ollama")
    local_model_name = model if use_local_llm else None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build session data from adapters
    processed_messages, historical_messages, agents_configs = build_session_data(
        agents, system_name=system_name
    )

    # Add LLM instances to each agent config (required by Phase 1)
    for agent_data in agents_configs.values():
        agent_data["llm"] = LLMFactory.create_chat_llm(
            model_name=model,
            provider=provider,
            use_local=use_local_llm,
            local_model_name=local_model_name,
            temperature=0,
            num_ctx=ollama_num_ctx,
        )

    # Phase 0: Build error vector space
    poirot_agent = POIROTAgent(
        model=model,
        vectors_to_ignore=ignore_list,
        use_local_llm=use_local_llm,
        local_model_name=local_model_name,
        llm_provider=provider,
    )
    error_space = poirot_agent.analyze_system(system_description, ignore_list=ignore_list)

    # Phase 1: Individual analysis (no agent factory — agentless mode)
    phase1_result = execute_phase1_analysis(
        agents=agents_configs,
        processed_messages=processed_messages,
        agent_factory=None,
        error_space=error_space,
        vectors_to_ignore=ignore_list,
        output_dir=output_path,
        session_name=system_name,
        communication_tool_names=set(),
        include_tool_calls=include_tool_calls,
        include_broadcast_messages=include_broadcast_messages,
        full_context=full_context,
        use_local_llm=use_local_llm,
        local_model_name=local_model_name,
        llm_provider=provider,
        max_llm_retries=max_llm_retries,
        retry_delay_503=retry_delay_503,
        retry_delay_429=retry_delay_429,
        api_call_delay=api_call_delay,
    )

    phase1_reports = (
        phase1_result.get("reports", {})
        if isinstance(phase1_result, dict)
        else phase1_result
    )

    # Phase 2: Peer consultation (bypass DB reads via override params)
    phase2_result = execute_phase2_analysis(
        session_id="langchain_session",
        db_path=None,
        phase1_reports=phase1_reports,
        vectors_to_ignore=ignore_list,
        error_space=error_space,
        model_name=model,
        output_dir=output_path / "phase2",
        include_tool_calls=include_tool_calls,
        include_broadcast_messages=include_broadcast_messages,
        full_context=full_context,
        api_call_delay=api_call_delay,
        use_local_llm=use_local_llm,
        local_model_name=local_model_name,
        llm_provider=provider,
        max_agent_messages=max_agent_messages,
        max_llm_retries=max_llm_retries,
        retry_delay_503=retry_delay_503,
        retry_delay_429=retry_delay_429,
        token_budget=token_budget,
        historical_messages_override=historical_messages,
        agents_data_override=agents_configs,
    )

    return {
        "system_name": system_name,
        "phase0_error_space": error_space,
        "phase1_protocol": phase1_reports,
        "phase2_protocol": phase2_result,
        "votes": phase2_result.get("votes", {}),
    }
