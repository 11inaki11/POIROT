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

__version__ = "0.1.0"

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
    output_dir: Optional[str] = None,
    model: Optional[str] = None,
    ignore_list: Optional[List[str]] = None,
    # ── Optional — output / verbosity ──────────────────────────────────────
    verbose: bool = True,
    debug: bool = False,
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
            If ``None`` (default) no files are saved.
        verbose: If ``False``, suppress all protocol progress output to stdout.
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
        output_dir=output_dir or "poirot_results",
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
        verbose=verbose,
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
    output_dir: Optional[str] = None,
    ignore_list: Optional[List[str]] = None,
    # ── Optional — output / verbosity ──────────────────────────────────────
    verbose: bool = True,
    debug: bool = False,
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
        output_dir: Directory where result files are written.
            If ``None`` (default) no files are saved.
        verbose: If ``False``, suppress all protocol progress output to stdout.
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
        Dictionary with the following keys:

        ``system_name``
            Name of the analyzed system.
        ``error_space``
            List of error dimensions defined in Phase 0 — each with ``id``,
            ``name``, ``type``, and ``description``.
        ``consensus``
            Aggregated verdict: ``faulty_component`` (name), ``fault_vector``
            (binary list), ``confidence_pct``, ``is_tie``,
            ``tied_components``.
        ``agent_reports``
            Per-agent dict: ``name``, ``vote`` (binary list),
            ``vote_description``, ``justification``.
        ``details``
            Raw phase outputs for advanced inspection.

    Raises:
        ValueError: If an unknown provider is specified.
    """
    from .adapters import build_session_data
    from .phase1_protocol import execute_phase1_analysis
    from .phase2_protocol import execute_phase2_analysis

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

    # Only create output directory when the caller requests it
    output_path: Optional[Path] = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Build session data from adapters.
    # Each agent's compiled graph is stored in agents_configs[id]["compiled_agent"]
    # and will be used directly in Phase 1 and Phase 2 — no new LLM instances are
    # created for the agents themselves.
    processed_messages, historical_messages, agents_configs = build_session_data(
        agents, system_name=system_name, include_tool_calls=include_tool_calls
    )

    # Phase 0: Build error vector space
    poirot_agent = POIROTAgent(
        model=model,
        vectors_to_ignore=ignore_list,
        use_local_llm=use_local_llm,
        local_model_name=local_model_name,
        llm_provider=provider,
    )
    error_space = poirot_agent.analyze_system(
        system_description, ignore_list=ignore_list, verbose=verbose
    )

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
        verbose=verbose,
        debug=debug,
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
        output_dir=output_path,
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
        verbose=verbose,
        debug=debug,
    )

    # ── Build user-friendly result ───────────────────────────────────────────
    voting_results = phase2_result.get("voting_results", {})
    winning = voting_results.get("winning_location", {})
    voting_summary = voting_results.get("voting_summary", {})
    tied_locations = voting_results.get("tied_locations") or []
    is_tie = voting_summary.get("is_tie", False)
    tied_names = [t["name"] for t in tied_locations] if is_tie else []

    return {
        "system_name": system_name,
        # Error dimensions defined in Phase 0
        "error_space": error_space.get("error_regions", []),
        # Aggregated verdict
        "consensus": {
            # When tied, lists all tied component names joined by " / "
            "faulty_component": " / ".join(tied_names) if is_tie else winning.get("name", "unknown"),
            "fault_vector": winning.get("vector", []),
            "confidence_pct": winning.get("percentage", 0.0),
            "is_tie": is_tie,
            # Empty list when no tie; full list of tied names when tied
            "tied_components": tied_names,
        },
        # Per-agent final votes from Phase 2
        "agent_reports": {
            agent_id: {
                "name": data.get("agent_name", agent_id),
                "vote": data.get("location", []),
                "vote_description": data.get("hazard_vector", ""),
                "justification": data.get("justification", ""),
            }
            for agent_id, data in phase2_result.get("votes", {}).items()
        },
        # Raw phase outputs for advanced inspection
        "details": {
            "phase0_error_space": error_space,
            "phase1_reports": phase1_reports,
            "phase2_voting": voting_results,
        },
    }
