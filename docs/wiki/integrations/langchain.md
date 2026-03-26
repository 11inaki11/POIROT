# LangChain Integration

POIROT can analyze LangChain/LangGraph agents directly, without requiring a SQLite database. Pass your compiled agent objects and their message histories to `run_poirot_from_agents()`.

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
planner = create_react_agent(llm, tools=planner_tools)
executor = create_react_agent(llm, tools=executor_tools)

planner_result = planner.invoke({"messages": [HumanMessage("Analyze AAPL")]})
executor_result = executor.invoke({"messages": planner_result["messages"]})

# Analyze with POIROT
results = run_poirot_from_agents(
    agents=[
        LangChainAgentAdapter(agent=planner, messages=planner_result["messages"]),
        LangChainAgentAdapter(agent=executor, messages=executor_result["messages"]),
    ],
    system_name="StockTradingBot",
    system_description="A two-agent system where a planner decomposes tasks and an executor runs them.",
    provider="gemini",
    model="gemini-2.5-pro",
    api_key="YOUR_KEY",
)

print(results["votes"])
```

---

## Provider and model are required

Unlike `run_poirot()`, the `provider` and `model` parameters have **no defaults** in `run_poirot_from_agents()`. You must specify them explicitly:

```python
run_poirot_from_agents(
    agents=[...],
    system_name="...",
    system_description="...",
    provider="openai",   # required
    model="gpt-4o",      # required
)
```

See [Providers](../providers.md) for all supported providers and their model names.

---

## Agent names and IDs

`agent_name` and `agent_id` are optional. If omitted:

- Names are auto-generated: `"Agent 1"`, `"Agent 2"`, etc.
- IDs are derived from names: `"agent_1"`, `"agent_2"`, etc.
- Duplicate names are disambiguated with `_1`, `_2` suffixes.

Provide them explicitly when you want meaningful labels in the analysis output:

```python
LangChainAgentAdapter(
    agent=planner,
    messages=planner_result["messages"],
    agent_name="Planner",
    agent_id="planner",
    system_prompt="You are a planning agent...",
)
```

---

## Tool extraction

POIROT automatically extracts tools from compiled LangGraph graphs. This is best-effort — it reads the internal `"tools"` node of the graph. If extraction fails, the tool list defaults to empty, which does not affect the analysis.

You can also supply tools manually:

```python
LangChainAgentAdapter(
    agent=my_agent,
    messages=result["messages"],
    tools=my_tools,   # explicit list, skips auto-extraction
)
```

---

## Full parameter reference

See [API Reference](../api-reference.md#run_poirot_from_agents) for all parameters.

---

## Context overflow warning

If `full_context=True`, every agent receives **all** messages from all agents. In long sessions this can exceed the model's context window. Use with care.
