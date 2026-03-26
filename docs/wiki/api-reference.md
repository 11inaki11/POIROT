# API Reference — `run_poirot()`

```python
poirot.run_poirot(
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

## Mandatory parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `database_path` | `str` | Path to the SQLite database containing session data |
| `system_name` | `str` | Short name for your system, e.g. `"StockTradingBot"` |
| `system_description` | `str` | Description of your system architecture. List the agents, their roles, and how they communicate. The more detail, the better the analysis. |
| `provider` | `str` | LLM provider: `"gemini"`, `"deepseek"`, `"ollama"`, or `"local"`. Default: `"gemini"` |
| `api_key` | `str` | API key for the provider. Not required for `"ollama"` or `"local"`. |

---

## Optional parameters — basic

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str` | `None` | Specific session to analyze. If `None`, POIROT uses the most recent session in the database. |
| `output_dir` | `str` | `"poirot_results"` | Directory where result files are written. Created automatically if it does not exist. |
| `model` | `str` | `None` | Override the default model for the selected provider. See [LLM Providers](providers.md) for defaults. |
| `ignore_list` | `list[str]` | `None` | Component names to exclude from the error vector analysis. Useful for known non-issues. |

---

## Optional parameters — message context

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `include_tool_calls` | `bool` | `False` | Include tool call and tool result messages in agent context windows during Phase 1 and 2. |
| `include_broadcast_messages` | `bool` | `False` | Include messages sent to all agents (`to_agent_id = NULL`) in context windows. |
| `full_context` | `bool` | `False` | If `True`, each agent receives **all messages from all agents** in the session, not just the ones they sent or received. Messages from other agents are presented as `HumanMessage` with a `From [agent_id]:` header. See warning below. |

> **Warning — `full_context=True` and context overflow**
>
> In the default mode (`full_context=False`), each agent only sees the messages it was directly involved in. With `full_context=True`, every agent sees the entire session history. In systems with many agents or long sessions, this can multiply the token count per agent by N (number of agents), easily exceeding the model's context window. Use with caution and consider lowering `token_budget` accordingly.

---

## Optional parameters — context window

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ollama_num_ctx` | `int` | `131072` | Context window size in tokens for Ollama models. Only used when `provider="ollama"`. |
| `token_budget` | `int` | `95000` | Maximum tokens allowed in agent context windows during Phase 2. Messages are truncated if this limit is exceeded. |

---

## Optional parameters — Phase 2

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_agent_messages` | `int` | `8` | Maximum number of LLM calls per agent during the Phase 2 peer consultation graph. |

---

## Optional parameters — retries and rate limiting

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_call_delay` | `float` | `0.0` | Seconds to wait between LLM calls. Use this for aggressive rate limiting on free-tier APIs. |
| `max_llm_retries` | `int` | `5` | Maximum retry attempts when a transient API error occurs. |
| `retry_delay_503` | `int` | `30` | Seconds to wait after a `503 Service Unavailable` error before retrying. |
| `retry_delay_429` | `int` | `60` | Seconds to wait after a `429 Resource Exhausted` (quota) error before retrying. |

---

## Example — all parameters

```python
import poirot

results = poirot.run_poirot(
    # Mandatory
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

    # Basic
    session_id="trading_session_042",
    output_dir="analysis_output",
    model="gemini-2.5-pro",
    ignore_list=["memory limitations applied for testing"],

    # Message context
    include_tool_calls=True,
    include_broadcast_messages=False,

    # Context window
    token_budget=80_000,

    # Phase 2
    max_agent_messages=10,

    # Retries
    api_call_delay=1.0,
    max_llm_retries=3,
    retry_delay_503=30,
    retry_delay_429=60,
)
```
