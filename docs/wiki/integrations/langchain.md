# LangChain Integration

POIROT can analyze LangChain/LangGraph agents directly, without requiring a SQLite database. Pass your compiled agent objects and their message histories to `run_poirot_from_agents()`.

The agents participate in all POIROT phases themselves — POIROT does not reconstruct them or create substitute LLMs. In Phase 1 each agent is invoked directly with its own compiled graph. In Phase 2 each agent's LLM is used to build a new graph that adds POIROT peer-consultation tools on top of the agent's original toolset.

---

## Supported agent types

- `create_react_agent` (LangGraph prebuilt) — fully supported
- Any agent whose message history is a `List[BaseMessage]`

---

## Minimal example

```python
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from poirot import run_poirot_from_agents, LangChainAgentAdapter

# Build and run your agents normally
planner  = create_react_agent(llm, tools=planner_tools)
executor = create_react_agent(llm, tools=executor_tools)

planner_result  = planner.invoke({"messages": [HumanMessage("Analyze AAPL")]})
executor_result = executor.invoke({"messages": planner_result["messages"]})

# Analyze with POIROT
results = run_poirot_from_agents(
    agents=[
        LangChainAgentAdapter(agent=planner,  messages=planner_result["messages"]),
        LangChainAgentAdapter(agent=executor, messages=executor_result["messages"]),
    ],
    system_name="StockTradingBot",
    system_description="A two-agent system: planner decomposes tasks, executor runs them.",
    provider="gemini",
    model="gemini-2.5-pro",
    api_key="YOUR_KEY",
)

print(results["votes"])
```

---

## Provider and model

`provider` and `model` in `run_poirot_from_agents()` serve two purposes:

1. **Phase 0** — POIROT uses this LLM to build the error vector space from your system description.
2. **Phase 2 default** — agents that do not declare their own provider/model use this as their LLM for peer consultation.

They have **no defaults** and must always be specified explicitly.

---

## Per-agent provider and model

By default all agents use the global `provider`/`model`. If your system is heterogeneous — different agents run on different models — you can override per agent:

```python
LangChainAgentAdapter(
    agent=my_agent,
    messages=result["messages"],
    provider="openai",      # this agent uses GPT-4o in Phase 2
    model="gpt-4o",
    api_key="sk-...",       # optional if already set in env
)
```

Agents without a per-agent override fall back to the global `provider`/`model` from `run_poirot_from_agents()`.

---

## Agent names and IDs

`agent_name` and `agent_id` are optional. If omitted:

- Names are auto-generated: `"Agent 1"`, `"Agent 2"`, etc.
- IDs are derived from names: `"agent_1"`, `"agent_2"`, etc.
- Duplicate names are disambiguated with `_1`, `_2` suffixes.

Provide them explicitly for meaningful labels in the analysis output:

```python
LangChainAgentAdapter(
    agent=planner,
    messages=planner_result["messages"],
    agent_name="Planner",
    agent_id="planner",
)
```

---

## Tool extraction

POIROT automatically extracts tools from compiled LangGraph graphs. This is best-effort — it reads the internal `"tools"` node of the graph. If extraction fails, the tool list defaults to empty, which does not affect the analysis quality.

You can supply tools explicitly to guarantee correct extraction:

```python
LangChainAgentAdapter(
    agent=my_agent,
    messages=result["messages"],
    tools=my_tools,   # explicit list, skips auto-extraction
)
```

---

## Full parameter reference

See [API Reference → LangChainAgentAdapter](../api-reference.md#langchainagentadapter) for all parameters.

---

## Context overflow warning

If `full_context=True`, every agent receives **all** messages from all agents. In long sessions this can exceed the model's context window. Use with care.
