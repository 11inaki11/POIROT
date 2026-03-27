"""
POIROT LangChain Adapter
========================

Converts live LangChain agent objects and their message histories into the
internal POIROT representation, allowing `run_poirot_from_agents()` to analyze
sessions without a SQLite database.

Supported agent types:
- `create_react_agent` (LangGraph prebuilt) — fully supported
- Any agent whose message history is a List[BaseMessage]

Usage::

    from poirot.adapters import LangChainAgentAdapter
    import poirot

    planner_result = planner.invoke({"messages": [HumanMessage("...")]})

    results = poirot.run_poirot_from_agents(
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
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# ---------------------------------------------------------------------------
# Internal import — AgentConfig dataclass
# ---------------------------------------------------------------------------
from src.agent_factory import AgentConfig, ProcessedMessage


# ---------------------------------------------------------------------------
# LangChainAgentAdapter
# ---------------------------------------------------------------------------

@dataclass
class LangChainAgentAdapter:
    """
    Wraps a compiled LangChain agent and its message history for POIROT analysis.

    The compiled agent is invoked directly in Phase 1. For Phase 2, POIROT builds
    a new agent using the same LLM (specified via provider/model) plus the agent's
    original tools and POIROT communication tools.

    Attributes:
        agent:     Compiled LangGraph graph (e.g. from create_react_agent).
        messages:  Message history produced by the agent during the session.
        agent_id:  Unique identifier. Derived from agent_name if not given.
        agent_name: Human-readable name. Auto-generated ("Agent 1", ...) if not given.
        agent_type: Agent category string (default: "agent").
        tools:     Tool list. Extracted from agent automatically if not given.
        provider:  LLM provider for this agent (overrides the global default in
                   run_poirot_from_agents). E.g. "gemini", "openai", "deepseek".
        model:     Model name for this agent (overrides the global default).
        api_key:   API key for this agent's provider (overrides the global default).
    """
    agent: Any
    messages: List[BaseMessage]
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    agent_type: str = "agent"
    tools: Optional[List[Any]] = field(default=None)
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool extraction
# ---------------------------------------------------------------------------

def _extract_tools_from_agent(agent: Any) -> List[Any]:
    """Best-effort extraction of tools from a compiled LangGraph graph."""
    try:
        tools_node = agent.nodes.get("tools")
        if tools_node and hasattr(tools_node, "tools_by_name"):
            return list(tools_node.tools_by_name.values())
    except Exception:
        pass
    return []


def _tool_to_dict(tool: Any) -> Dict[str, Any]:
    """Convert a LangChain tool object to the dict format AgentConfig expects."""
    return {
        "name": getattr(tool, "name", str(tool)),
        "description": getattr(tool, "description", ""),
        "schema": {},
        "code": "",
    }


# ---------------------------------------------------------------------------
# Name / ID resolution
# ---------------------------------------------------------------------------

def _to_snake_case(name: str) -> str:
    """Convert a human-readable name to a snake_case identifier."""
    return re.sub(r"\s+", "_", name.strip()).lower()


def _resolve_names_and_ids(adapters: List[LangChainAgentAdapter]) -> List[LangChainAgentAdapter]:
    """
    Fill in missing agent_name and agent_id values, handling duplicates.

    Resolution order:
    1. agent_name=None  → "Agent {i+1}"
    2. Duplicate names  → append _1, _2, ...
    3. agent_id=None    → snake_case(agent_name)
    4. Duplicate ids    → append _1, _2, ...
    """
    # Step 1: assign default names
    for i, adapter in enumerate(adapters):
        if adapter.agent_name is None:
            adapter.agent_name = f"Agent {i + 1}"

    # Step 2: deduplicate names
    name_counts: Dict[str, int] = {}
    for adapter in adapters:
        name_counts[adapter.agent_name] = name_counts.get(adapter.agent_name, 0) + 1

    name_seen: Dict[str, int] = {}
    for adapter in adapters:
        if name_counts[adapter.agent_name] > 1:
            name_seen[adapter.agent_name] = name_seen.get(adapter.agent_name, 0) + 1
            adapter.agent_name = f"{adapter.agent_name}_{name_seen[adapter.agent_name]}"

    # Step 3: derive IDs from names
    for adapter in adapters:
        if adapter.agent_id is None:
            adapter.agent_id = _to_snake_case(adapter.agent_name)

    # Step 4: deduplicate IDs
    id_counts: Dict[str, int] = {}
    for adapter in adapters:
        id_counts[adapter.agent_id] = id_counts.get(adapter.agent_id, 0) + 1

    id_seen: Dict[str, int] = {}
    for adapter in adapters:
        if id_counts[adapter.agent_id] > 1:
            id_seen[adapter.agent_id] = id_seen.get(adapter.agent_id, 0) + 1
            adapter.agent_id = f"{adapter.agent_id}_{id_seen[adapter.agent_id]}"

    return adapters


# ---------------------------------------------------------------------------
# Message routing inference
# ---------------------------------------------------------------------------

def _build_ai_sender_map(adapters: List[LangChainAgentAdapter]) -> Dict[int, str]:
    """
    First pass: map content_hash → agent_id for every AIMessage found across
    all adapter message lists.  Used in the second pass to infer the sender of
    HumanMessages whose content matches a known AIMessage.
    """
    sender_map: Dict[int, str] = {}
    for adapter in adapters:
        for msg in adapter.messages:
            if isinstance(msg, AIMessage) and msg.content:
                sender_map[hash(msg.content)] = adapter.agent_id
    return sender_map


def _infer_sender(msg: BaseMessage, this_agent_id: str, ai_sender_map: Dict[int, str]) -> str:
    """Return the best-guess sender agent_id for a message."""
    # Priority 1: explicit name field (LangGraph multi-agent style)
    if hasattr(msg, "name") and msg.name:
        return msg.name
    # Priority 2: content-hash cross-reference
    if msg.content:
        inferred = ai_sender_map.get(hash(msg.content))
        if inferred and inferred != this_agent_id:
            return inferred
    # Fallback
    return "external"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_session_data(
    adapters: List[LangChainAgentAdapter],
    system_name: str = "langchain_system",
) -> Tuple[List[ProcessedMessage], List[BaseMessage], Dict[str, Any]]:
    """
    Convert a list of LangChainAgentAdapters into POIROT's internal session
    representation.

    Returns:
        processed_messages: List[ProcessedMessage] for Phase 1.
        historical_messages: List[BaseMessage] (with metadata) for Phase 2.
        agents_configs: Dict[agent_id → agent data dict] for Phase 2 override.
    """
    adapters = _resolve_names_and_ids(adapters)
    ai_sender_map = _build_ai_sender_map(adapters)

    processed_messages: List[ProcessedMessage] = []
    historical_messages: List[BaseMessage] = []
    seen_historical: set = set()   # (content_hash, from_node, to_node)
    agents_configs: Dict[str, Any] = {}

    for adapter in adapters:
        agent_id = adapter.agent_id
        agent_name = adapter.agent_name

        # Resolve tools
        raw_tools = adapter.tools if adapter.tools is not None else _extract_tools_from_agent(adapter.agent)
        tools_dicts = [_tool_to_dict(t) for t in raw_tools]

        # Build AgentConfig
        config = AgentConfig(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=adapter.agent_type,
            system_name=system_name,
            system_prompt="",       # system prompt is internal to the compiled agent
            llm_model="",           # LLM is internal to the compiled agent
            temperature=0.0,
            max_tokens=8000,
            tools=tools_dicts,
            can_communicate_with=[a.agent_id for a in adapters if a.agent_id != agent_id],
        )

        agents_configs[agent_id] = {
            "name": agent_name,
            "config": config,
            "system_prompt": "",            # internal to the compiled agent
            "compiled_agent": adapter.agent,# the original compiled graph
            "provider": adapter.provider,   # per-agent LLM override (None = use global)
            "model": adapter.model,         # per-agent model override (None = use global)
            "api_key": adapter.api_key,     # per-agent API key override (None = use global)
            "tools": raw_tools,
            "tools_dict": {getattr(t, "name", str(t)): t for t in raw_tools},
            "communication_tools": [],
            "can_communicate_with": config.can_communicate_with,
        }

        # Process each message
        for msg in adapter.messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if not content or not content.strip():
                continue

            # Determine routing
            if isinstance(msg, AIMessage):
                from_agent = agent_id
                to_agent = "unknown"
                msg_type = "ai"
                is_tool_call = bool(getattr(msg, "tool_calls", None))
                tool_name = msg.tool_calls[0].get("name") if is_tool_call and msg.tool_calls else None

            elif isinstance(msg, ToolMessage):
                from_agent = agent_id
                to_agent = agent_id
                msg_type = "tool"
                is_tool_call = False
                tool_name = getattr(msg, "name", None)

            elif isinstance(msg, SystemMessage):
                from_agent = "system"
                to_agent = agent_id
                msg_type = "system"
                is_tool_call = False
                tool_name = None

            else:
                # HumanMessage — infer sender
                from_agent = _infer_sender(msg, agent_id, ai_sender_map)
                to_agent = agent_id
                msg_type = "human"
                is_tool_call = False
                tool_name = None

            # Build ProcessedMessage for Phase 1
            pm = ProcessedMessage(
                message=msg,
                from_agent=from_agent,
                to_agent=to_agent,
                msg_type=msg_type,
                is_tool_call=is_tool_call,
                tool_name=tool_name,
                is_communication=False,
            )
            processed_messages.append(pm)

            # Build historical message for Phase 2 (deduplicated)
            dedup_key = (hash(content), from_agent, to_agent)
            if dedup_key not in seen_historical:
                seen_historical.add(dedup_key)
                # Clone with metadata
                if isinstance(msg, AIMessage):
                    hist_msg = HumanMessage(content=content)   # will be re-typed by filter_messages_for_agent
                elif isinstance(msg, SystemMessage):
                    hist_msg = SystemMessage(content=content)
                else:
                    hist_msg = HumanMessage(content=content)

                if not hasattr(hist_msg, "additional_kwargs"):
                    hist_msg.additional_kwargs = {}
                hist_msg.additional_kwargs["metadata"] = {
                    "from_node": from_agent,
                    "to_node": to_agent,
                    "timestamp": "",
                }
                historical_messages.append(hist_msg)

    return processed_messages, historical_messages, agents_configs
