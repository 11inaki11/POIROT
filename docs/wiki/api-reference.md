# API Reference

- `run_poirot()` — analyze a session stored in a SQLite database
- `run_poirot_from_agents()` — analyze live LangChain agent objects directly

---

## `run_poirot()`

```python
import poirot

results = poirot.run_poirot(
    database_path,
    system_name,
    system_description,
    provider="gemini",
    api_key=None,
    ...
)
```

Returns a `dict` with the full analysis results including agent votes and the final hazard vector.

---

### Mandatory parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `database_path` | `str` | Path to the SQLite database containing session data |
| `system_name` | `str` | Short name for your system, e.g. `"StockTradingBot"` |
| `system_description` | `str` | Description of your system architecture. List the agents, their roles, and how they communicate. The more detail, the better the analysis. |
| `provider` | `str` | LLM provider: `"gemini"`, `"openai"`, `"deepseek"`, `"ollama"`, or `"local"`. Default: `"gemini"` |
| `api_key` | `str` | API key for the provider. Not required for `"ollama"` or `"local"`. |

---

### Optional parameters — basic

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `session_id` | `str` | `None` | Specific session to analyze. If `None`, POIROT uses the most recent session in the database. |
| `output_dir` | `str \| None` | `None` | Directory where result files are written. Created automatically if it does not exist. If `None`, no files are saved. |
| `verbose` | `bool` | `True` | If `False`, suppress all protocol progress output to the terminal. |
| `debug` | `bool` | `False` | If `True`, saves intermediate context files (LLM inputs per agent per call) to a `debug/` subfolder inside `output_dir`. Has no effect when `output_dir=None`. |
| `model` | `str` | `None` | Override the default model for the selected provider. See [LLM Providers](providers.md) for defaults. |
| `ignore_list` | `list[str]` | `None` | Component names to exclude from the error vector analysis. Useful for known non-issues. |

---

### Optional parameters — message context

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `include_tool_calls` | `bool` | `False` | Include tool call and tool result messages in agent context windows during Phase 1 and 2. |
| `include_broadcast_messages` | `bool` | `False` | Include messages sent to all agents in context windows. |
| `full_context` | `bool` | `False` | If `True`, each agent receives all messages from all agents in the session, not just the ones they sent or received. Messages from other agents are presented as `HumanMessage` with a `From [agent_id]:` header. See warning below. |

> **Warning — `full_context=True` and context overflow**
>
> In the default mode each agent only sees the messages it was directly involved in. With `full_context=True`, every agent sees the entire session history. In systems with many agents or long sessions this can multiply the token count per agent by N (number of agents), easily exceeding the model's context window. Use with caution.

---

### Optional parameters — context window

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `ollama_num_ctx` | `int` | `131072` | Context window size in tokens for Ollama models. Only used when `provider="ollama"`. |
| `token_budget` | `int` | `95000` | Maximum tokens allowed in agent context windows during Phase 2. |

---

### Optional parameters — Phase 2

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `max_agent_messages` | `int` | `8` | Maximum number of LLM calls per agent during Phase 2 peer consultation. |

---

### Optional parameters — retries and rate limiting

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `api_call_delay` | `float` | `0.0` | Seconds to wait between LLM calls. Use for rate-limited free-tier APIs. |
| `max_llm_retries` | `int` | `5` | Maximum retry attempts on transient API errors. |
| `retry_delay_503` | `int` | `30` | Seconds to wait after a 503 Service Unavailable error. |
| `retry_delay_429` | `int` | `60` | Seconds to wait after a 429 Resource Exhausted (quota) error. |

---

### Full example

```python
import poirot

results = poirot.run_poirot(
    database_path="my_system.db",
    system_name="StockTradingBot",
    system_description="""
        A 3-agent trading system:
        - AnalystAgent: analyzes market data and produces a buy/sell recommendation
        - RiskAgent: evaluates the risk of the recommendation
        - ExecutorAgent: executes the final trade decision
    """,
    provider="gemini",
    api_key="YOUR_KEY",
    session_id="trading_session_042",
    output_dir="analysis_output",
    model="gemini-2.5-pro",
    ignore_list=["memory limitations applied for testing"],
    include_tool_calls=True,
    token_budget=80_000,
    max_agent_messages=10,
    api_call_delay=1.0,
    max_llm_retries=3,
)
```

---

## `run_poirot_from_agents()`

Analyze live LangChain/LangGraph agent objects directly — no database required. See the [LangChain integration guide](integrations/langchain.md) for a full explanation of how agents participate in the analysis.

```python
from poirot import run_poirot_from_agents, LangChainAgentAdapter

results = run_poirot_from_agents(
    agents=[...],
    system_name="MySystem",
    system_description="...",
    provider="gemini",
    model="gemini-2.5-pro",
    api_key="YOUR_KEY",
)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `agents` | `list[LangChainAgentAdapter]` | required | Agent objects and their message histories |
| `system_name` | `str` | required | Short name for your system |
| `system_description` | `str` | required | Description of your system architecture |
| `provider` | `str` | required | Default LLM provider for Phase 0 and agents without a per-agent override |
| `model` | `str` | required | Default model name. No default — must be explicit |
| `api_key` | `str` | `None` | Default API key. Not required for `"ollama"` or `"local"` |
| `output_dir` | `str \| None` | `None` | Directory for result files. If `None`, no files are saved. |
| `verbose` | `bool` | `True` | If `False`, suppress all protocol progress output to the terminal. |
| `debug` | `bool` | `False` | If `True`, saves intermediate context files to a `debug/` subfolder inside `output_dir`. |
| `ignore_list` | `list[str]` | `None` | Component names to exclude from error vector analysis |
| `include_tool_calls` | `bool` | `False` | Include tool call messages in context windows |
| `include_broadcast_messages` | `bool` | `False` | Include broadcast messages in context windows |
| `full_context` | `bool` | `False` | Each agent sees all session messages (see context overflow warning above) |
| `ollama_num_ctx` | `int` | `131072` | Context window size for Ollama models |
| `token_budget` | `int` | `95000` | Max tokens in Phase 2 context windows |
| `max_agent_messages` | `int` | `8` | Max LLM calls per agent in Phase 2 |
| `api_call_delay` | `float` | `0.0` | Seconds between LLM calls |
| `max_llm_retries` | `int` | `5` | Max retries on transient errors |
| `retry_delay_503` | `int` | `30` | Wait seconds after a 503 error |
| `retry_delay_429` | `int` | `60` | Wait seconds after a 429 error |

---

## Reading the results

Both entry points return a `dict`. The structure is the same for `run_poirot_from_agents()`; `run_poirot()` follows the same top-level keys.

```python
results = run_poirot_from_agents(...)

# ── Top-level keys ────────────────────────────────────────────────────────────
results["system_name"]       # str  — name you passed in
results["error_space"]       # list — error dimensions from Phase 0
results["consensus"]         # dict — aggregated verdict
results["agent_reports"]     # dict — per-agent final votes
results["details"]           # dict — raw phase outputs for advanced inspection
```

### `error_space`

List of dicts, one per error dimension defined in Phase 0.

```python
for dim in results["error_space"]:
    print(dim["id"])          # "x1", "x2", ...
    print(dim["name"])        # "DiagnosisAgent omits medication info"
    print(dim["type"])        # "agent" | "software" | "hardware" | ...
    print(dim["description"]) # Explanation of the failure mode
```

### `consensus`

The weighted vote aggregation — the main verdict.

```python
c = results["consensus"]
print(c["faulty_component"])  # "DiagnosisAgent" — single winner, or "AgentA / AgentB" when tied
print(c["fault_vector"])      # [1, 0] — binary position in the error space
print(c["confidence_pct"])    # 72.4  — percentage of total vote weight
print(c["is_tie"])            # True when two or more components share the top score
print(c["tied_components"])   # [] when no tie, ["AgentA", "AgentB"] when is_tie=True
```

When `is_tie=True`, `faulty_component` is a `/`-joined string of all tied names and `tied_components` holds each one individually. Always check `is_tie` before acting on `faulty_component`:

```python
c = results["consensus"]
if c["is_tie"]:
    print(f"TIE between: {', '.join(c['tied_components'])}")
else:
    print(f"Faulty component: {c['faulty_component']}")
```

### `agent_reports`

One entry per agent, keyed by `agent_id`.

```python
for agent_id, report in results["agent_reports"].items():
    print(report["name"])             # "DiagnosisAgent"
    print(report["vote"])             # [1, 0] — the binary vector this agent voted
    print(report["vote_description"]) # Short description of the hazard
    print(report["justification"])    # Full reasoning from the agent
```

### `details`

Raw outputs from each phase, useful for debugging or custom analysis.

```python
results["details"]["phase0_error_space"]  # Full Phase 0 output dict
results["details"]["phase1_reports"]      # Per-agent Phase 1 self-assessment
results["details"]["phase2_voting"]       # Full voting analysis (weights, rankings)
```

### Minimal example

```python
results = run_poirot_from_agents(
    agents=[...],
    system_name="EmergencyDiagnosisSystem",
    system_description="...",
    provider="gemini",
    model="gemini-2.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

c = results["consensus"]
if c["is_tie"]:
    print(f"TIE between: {', '.join(c['tied_components'])}")
else:
    print(f"Faulty component: {c['faulty_component']}")
print(f"Confidence: {c['confidence_pct']:.1f}%")

for agent_id, report in results["agent_reports"].items():
    print(f"\n{report['name']} voted: {report['vote']}")
    print(f"  Justification: {report['justification'][:120]}...")
```

---

## Output files

When `output_dir` is set, POIROT writes the following files:

```text
<output_dir>/
├── summary.txt        # Human-readable verdict: result, faulty component, confidence, per-agent votes
├── votes.json         # Machine-readable votes: one entry per agent with vector and justification
└── phase1/
    ├── AgentName.txt  # Phase 1 self-assessment report for each agent
    └── ...
```

If `debug=True` is also set, an additional `debug/` folder is created:

```text
<output_dir>/
└── debug/
    ├── phase1_context_AgentName.json   # Exact context passed to each agent in Phase 1
    ├── agent_id_call_1.txt             # Context passed to each agent in Phase 2, per LLM call
    └── ...
```

---

## `LangChainAgentAdapter`

Wraps a compiled LangChain agent and its message history for use with `run_poirot_from_agents()`.

```python
from poirot import LangChainAgentAdapter

LangChainAgentAdapter(
    agent=my_compiled_agent,
    messages=result["messages"],
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `agent` | `CompiledGraph` | required | Compiled LangGraph graph (e.g. from `create_react_agent`) |
| `messages` | `list[BaseMessage]` | required | Message history produced by the agent during the session |
| `agent_id` | `str` | `None` | Unique identifier. Derived from `agent_name` if not given |
| `agent_name` | `str` | `None` | Human-readable name. Auto-generated (`"Agent 1"`, ...) if not given |
| `agent_type` | `str` | `"agent"` | Category string, used in output labels |
| `tools` | `list` | `None` | Tool list. Auto-extracted from the compiled graph if not provided |
| `provider` | `str` | `None` | Per-agent LLM provider override. Falls back to `run_poirot_from_agents()` global |
| `model` | `str` | `None` | Per-agent model override. Falls back to global |
| `api_key` | `str` | `None` | Per-agent API key override. Falls back to global |
