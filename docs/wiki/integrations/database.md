# Integration — SQLite Database

The SQLite database integration is the most flexible way to use POIROT. Any system — regardless of language or framework — can write session data to a SQLite file and hand it to POIROT for analysis.

---

## When to use this integration

- Your agents are not built with a specific Python framework
- You want full control over what data POIROT receives
- Your system is in production and you want to log sessions for later analysis
- Your system is not written in Python

---

## Setup

Copy `templates/poirot_schema.sql` from the repository and initialize your database:

```python
import sqlite3

conn = sqlite3.connect("my_system.db")
with open("poirot_schema.sql") as f:
    conn.executescript(f.read())
conn.close()
```

---

## Tables

### `sessions`

One row per execution of your system.

**Required columns:**

| Column | Type | Example |
| ------ | ---- | ------- |
| `session_id` | TEXT (PK) | `"run_001"` |
| `system_name` | TEXT | `"StockTradingBot"` |
| `session_number` | INTEGER | `1` |
| `created_at` | TEXT (ISO 8601) | `"2025-01-20T10:30:00.000000"` |

**Optional columns:** `updated_at`, `session_notes`, `input_context`, `final_output`, `execution_time_seconds`, `custom_metadata` (JSON string).

```python
from datetime import datetime

cursor.execute("""
    INSERT INTO sessions (session_id, system_name, session_number, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?)
""", ("run_001", "StockTradingBot", 1, datetime.now().isoformat(), datetime.now().isoformat()))
```

---

### `agents`

One row per agent. **Treat entries as immutable once referenced by a session.** If you update an agent's configuration, create a new row with a new `agent_id` — never overwrite an existing one. This preserves the historical record for past sessions.

**Required columns:**

| Column | Type | Example |
| ------ | ---- | ------- |
| `agent_id` | TEXT (PK) | `"portfolio_manager"` |
| `agent_name` | TEXT | `"Portfolio Manager"` |
| `system_name` | TEXT | `"StockTradingBot"` |

**Recommended optional columns:** `agent_type`, `system_prompt`, `llm_model`, `temperature`, `max_tokens`, `has_tools` (0/1), `can_communicate_with` (JSON array of agent IDs).

```python
cursor.execute("""
    INSERT INTO agents (agent_id, agent_name, system_name, system_prompt, llm_model, has_tools)
    VALUES (?, ?, ?, ?, ?, ?)
""", (
    "portfolio_manager",
    "Portfolio Manager",
    "StockTradingBot",
    "You are an expert portfolio manager...",
    "gemini-2.5-pro",
    1
))
```

---

### `messages`

The most important table. One row per message exchanged in the session.

**Required columns:**

| Column | Type | Example |
| ------ | ---- | ------- |
| `message_id` | TEXT (PK) | `"msg_001"` |
| `session_id` | TEXT (FK) | `"run_001"` |
| `from_agent_id` | TEXT | `"portfolio_manager"` |
| `to_agent_id` | TEXT | `"risk_manager"` (NULL = broadcast) |
| `message_type` | TEXT | `"ai"`, `"human"`, `"tool"` |
| `content` | TEXT | `"I recommend selling AAPL"` |
| `timestamp` | TEXT (ISO 8601) | `"2025-01-20T10:31:45.678901"` |
| `sequence_number` | INTEGER | `1`, `2`, `3`, ... |

**Message types:** `"human"` (user input or prompt), `"ai"` (agent response), `"tool"` (tool call or result), `"system"` (system notification).

**Optional columns:** `input_tokens`, `output_tokens`, `total_tokens`, `is_tool_call` (0/1), `tool_name`, `tool_input` (JSON), `tool_output` (JSON), `custom_data` (JSON).

```python
from datetime import datetime

cursor.execute("""
    INSERT INTO messages (
        message_id, session_id, from_agent_id, to_agent_id,
        message_type, content, timestamp, sequence_number
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (
    "msg_001", "run_001",
    "portfolio_manager", "risk_manager",
    "ai",
    "AAPL shows a bearish trend. Requesting risk assessment.",
    datetime.now().isoformat(),
    1
))
```

---

### `agent_tools` (optional)

One row per tool available to an agent. Only needed if your agents use tools.

**Required columns:** `agent_id` (FK) + `tool_name` — together they form the primary key.

**Optional columns:** `tool_description`, `tool_schema` (JSON), `tool_code` (valid Python with docstring), `is_enabled` (0/1).

---

## Key rules

1. **`sequence_number` must be strictly sequential** within each session. POIROT relies on message order.
2. **Timestamps must be ISO 8601** with microseconds. In Python: `datetime.now().isoformat()`.
3. **JSON fields** (`can_communicate_with`, `tool_input`, etc.) must use standard JSON with double quotes.
4. **Agent entries are immutable** once a session references them. Add a new `agent_id` for updated configurations.

---

## Best practices

- **Track tokens.** Populate `input_tokens`, `output_tokens`, and `total_tokens` whenever the LLM exposes usage metadata. POIROT can use this to analyse efficiency.
- **Log tool calls fully.** Set `is_tool_call=1` and populate `tool_name`, `tool_input`, and `tool_output` for every tool invocation. This reveals agent reasoning patterns.
- **Populate `can_communicate_with`.** Listing the allowed recipients for each agent helps POIROT understand the system's communication graph before reading the messages.
- **Use `custom_metadata` / `custom_data`.** JSON extension fields are the right place for system-specific information (confidence scores, market conditions, patient metadata) rather than adding custom columns.
- **Agent IDs: lowercase with underscores.** Example: `"portfolio_manager"`, `"risk_agent"`. No spaces, no special characters.
- **One session = one execution.** A session represents a single end-to-end run of the system. Do not mix multiple runs into the same `session_id`.

---

## Complete setup example

```python
import sqlite3
from datetime import datetime

# Initialize database from the schema template
conn = sqlite3.connect("my_system.db")
with open("templates/poirot_schema.sql") as f:
    conn.executescript(f.read())

cursor = conn.cursor()
now = datetime.now().isoformat()

# Session
cursor.execute(
    "INSERT INTO sessions (session_id, system_name, session_number, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
    ("run_001", "StockTradingBot", 1, now, now),
)

# Agents
cursor.executemany(
    "INSERT INTO agents (agent_id, agent_name, system_name, agent_type, system_prompt, llm_model, has_tools) VALUES (?, ?, ?, ?, ?, ?, ?)",
    [
        ("analyst",  "Market Analyst",  "StockTradingBot", "analyst",  "You analyse market data.", "gemini-2.5-pro", 1),
        ("risk",     "Risk Manager",    "StockTradingBot", "evaluator","You evaluate risk.",       "gemini-2.5-pro", 0),
        ("executor", "Trade Executor",  "StockTradingBot", "executor", "You execute trades.",      "gemini-2.5-pro", 0),
    ],
)

# Messages
messages = [
    ("msg_001", "run_001", "system",   "analyst",  "human", "Analyse AAPL and recommend action.", now, 1),
    ("msg_002", "run_001", "analyst",  "risk",     "ai",    "AAPL shows a bearish trend. Requesting risk assessment.", now, 2),
    ("msg_003", "run_001", "risk",     "executor", "ai",    "Risk approved. Proceed with SELL 50 shares.", now, 3),
    ("msg_004", "run_001", "executor", None,       "ai",    "Executed: SELL 50 AAPL at market price.", now, 4),
]
cursor.executemany(
    "INSERT INTO messages (message_id, session_id, from_agent_id, to_agent_id, message_type, content, timestamp, sequence_number) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    messages,
)

conn.commit()
conn.close()
```

For the full schema reference, data format requirements, and integration examples see [`docs/DATABASE_SCHEMA.md`](../../../DATABASE_SCHEMA.md).

---

## Running the analysis

Once the database is populated:

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
    api_key="YOUR_API_KEY",
)
```

See [API Reference](../api-reference.md) for all available parameters.
