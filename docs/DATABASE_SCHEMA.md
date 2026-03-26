# POIROT Generic Database Schema - Developer Guide

**Version:** 1.0  
**Last Updated:** January 2025  
**Target Audience:** Developers integrating multi-agent systems with POIROT

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Data Integrity & Versioning](#data-integrity--versioning)
3. [Database Architecture](#database-architecture)
4. [Core Tables](#core-tables)
   - [sessions](#sessions)
   - [agents](#agents)
   - [agent_tools](#agent_tools)
   - [messages](#messages)
5. [Data Format Requirements](#data-format-requirements)
6. [Integration Examples](#integration-examples)
7. [Best Practices](#best-practices)
8. [FAQ](#faq)

---

## 🛡️ Data Integrity & Versioning

### ⚠️ IMPORTANT: Handling Agent Changes

POIROT performs forensic analysis on past sessions. For this analysis to be accurate, the `agents` table must reflect the **exact state** of the agent at the time of the session.

If you modify an agent's configuration (System Prompt, Temperature, Model, etc.) for a **new** session, you **MUST NOT** overwrite the existing entry in the `agents` table if it was used in previous sessions. Doing so would corrupt the historical record.

**Correct Procedure for Updates:**
1. Create a **new entry** in the `agents` table with a new `agent_id` (e.g., `agent_v2`).
2. Use this new `agent_id` for the new session.
3. Keep the old `agent_id` record unchanged so old sessions can still be analyzed correctly.

**Rule of Thumb:** `agents` table entries should be treated as **immutable** once a session references them.

---

## 🏗️ Database Architecture

---

## Overview

POIROT is a **Protocol for Observing, Investigating, and Reasoning Over Threats** in multi-agent systems. It analyzes agent conversations, trust evolution, and error detection capabilities to identify weaknesses in your AI system.

### What POIROT Needs

To analyze your multi-agent system, POIROT requires:

1. **Session execution data**: What happened during each run?
2. **Agent configurations**: What agents exist and what are their roles?
3. **Message history**: What did agents say to each other?
4. **Tool usage** (optional): What external functions did agents use?

**Note:** POIROT will automatically generate error vectors and analysis results. You do not need to define error dimensions or ground truth labels in the database structure; POIROT handles this internally based on the provided session data.

### How to Provide Data

You provide this data by creating an SQLite database following the schema defined in `schema.sql`. POIROT reads this database and performs its analysis autonomously.

---

## Database Architecture

### Design Principles

1. **Generic**: Works with ANY multi-agent system (medical, trading, robotics, customer service, etc.)
2. **Flexible**: Core tables are required; optional tables add functionality
3. **Extensible**: JSON fields allow system-specific metadata
4. **Standards-compliant**: Uses ISO 8601 timestamps, standard SQL types

### Database Structure

```
┌──────────────┐
│   sessions   │ ← Top level: Each execution session
└──────┬───────┘
       │
       ├──────┐
       │      │
┌──────▼───┐  │
│ messages │◄─┤ ← Core data: All agent communications
└──────────┘  │
              │
       ┌──────▼───────┐
       │    agents    │ ← Agent definitions
       └──────┬───────┘
              │
       ┌──────▼───────┐
       │ agent_tools  │ ← Agent capabilities
       └──────────────┘

Optional:
┌────────────────┐     ┌─────────────────────┐
│ error_vectors  │     │ ground_truth_labels │
│ (POIROT ONLY)  │     │    (POIROT ONLY)    │
└────────────────┘     └─────────────────────┘
```

---

## Core Tables (User Provided)

### `sessions`

Represents one complete execution of your multi-agent system.

#### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `session_id` | TEXT (PK) | Unique identifier | `"trading_session_001"` |
| `system_name` | TEXT | Name of your system | `"StockTradingBot"` |
| `session_number` | INTEGER | Sequential number | `1`, `2`, `3`, ... |
| `created_at` | TEXT | ISO 8601 timestamp | `"2025-01-20T10:30:00.000000"` |
| `updated_at` | TEXT | Last update timestamp | `"2025-01-20T10:45:23.123456"` |

#### Optional Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `session_notes` | TEXT | Brief description | `"Market crash scenario"` |
| `input_context` | TEXT | Initial problem | `"Analyze AAPL volatility"` |
| `final_output` | TEXT | System's decision | `"SELL 100 shares"` |
| `execution_time_seconds` | REAL | Duration | `23.45` |
| `has_ground_truth` | INTEGER | 0 or 1 | `1` |
| `ground_truth_data` | TEXT | JSON string | `"[1,0,0,1]"` |
| `custom_metadata` | TEXT | JSON object | `'{"market_condition":"volatile"}'` |

#### Example Insertion

```sql
INSERT INTO sessions (
    session_id, system_name, session_number, 
    created_at, updated_at, session_notes
) VALUES (
    'trading_session_001',
    'StockTradingBot',
    1,
    '2025-01-20T10:30:00.000000',
    '2025-01-20T10:30:00.000000',
    'Test run with historical data'
);
```

---

### `agents`

Defines the AI agents in your system.

#### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `agent_id` | TEXT (PK) | Unique identifier | `"portfolio_manager"` |
| `agent_name` | TEXT | Human-readable name | `"Portfolio Manager"` |
| `system_name` | TEXT | System this belongs to | `"StockTradingBot"` |

#### Recommended Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `agent_type` | TEXT | Category | `"decision_maker"`, `"analyst"` |
| `system_prompt` | TEXT | Role definition | `"You are a portfolio manager..."` |
| `llm_model` | TEXT | Model used | `"gemini-2.5-pro"`, `"gpt-4o"` |
| `temperature` | REAL | LLM temperature | `0.7` |
| `max_tokens` | INTEGER | Token limit | `8000` |
| `has_tools` | INTEGER | 0 or 1 | `1` |
| `can_communicate_with` | TEXT | JSON array of agent IDs | `'["market_analyst","risk_manager"]'` |

#### Example Insertion

```sql
INSERT INTO agents (
    agent_id, agent_name, agent_type, system_name,
    system_prompt, llm_model, temperature, has_tools
) VALUES (
    'portfolio_manager',
    'Portfolio Manager',
    'decision_maker',
    'StockTradingBot',
    'You are an expert portfolio manager. Analyze market data and make trading decisions.',
    'gemini-2.5-pro',
    0.5,
    1
);
```

---

### `agent_tools`

Defines tools/functions available to each agent.

#### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `agent_id` | TEXT (PK) | Agent reference | `"portfolio_manager"` |
| `tool_name` | TEXT (PK) | Tool identifier | `"get_stock_price"` |

#### Optional Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `tool_description` | TEXT | What it does | `"Fetches current stock price"` |
| `tool_schema` | TEXT | JSON schema | `'{"ticker":{"type":"string"}}'` |
| `tool_code` | TEXT | **MUST** be valid, executable Python code with docstrings | `"def get_stock_price(ticker):\n    \"\"\"Fetches price.\"\"\"\n    return api.get(ticker)"` |
| `is_enabled` | INTEGER | 0 or 1 | `1` |

#### Example Insertion

```sql
INSERT INTO agent_tools (
    agent_id, tool_name, tool_description, tool_code, is_enabled
) VALUES (
    'portfolio_manager',
    'get_stock_price',
    'Retrieves real-time stock price from market API',
    'def get_stock_price(ticker: str) -> float:
    """
    Retrieves the current stock price for a given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").

    Returns:
        float: The current stock price.
    """
    # In a real scenario, this would call an external API
    import random
    return 150.0 + random.uniform(-5, 5)',
    1
);
```

---

### `messages`

**This is the most important table.** It stores all agent communications.

#### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `message_id` | TEXT (PK) | Unique identifier | `"msg_001"` |
| `session_id` | TEXT (FK) | Session reference | `"trading_session_001"` |
| `from_agent_id` | TEXT | Sender agent | `"portfolio_manager"` |
| `to_agent_id` | TEXT | Recipient (NULL=broadcast) | `"risk_manager"` |
| `message_type` | TEXT | Type of message | `"ai"`, `"human"`, `"tool"` |
| `content` | TEXT | Message content | `"I recommend selling AAPL"` |
| `timestamp` | TEXT | ISO 8601 timestamp | `"2025-01-20T10:31:45.678901"` |
| `sequence_number` | INTEGER | Order in session | `1`, `2`, `3`, ... |

#### Message Types

- `"human"`: User input or system prompt
- `"ai"`: Agent-generated response
- `"system"`: System notifications
- `"tool"`: Tool call or result
- `"function"`: Function invocation

#### Optional Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `input_tokens` | INTEGER | Tokens consumed | `150` |
| `output_tokens` | INTEGER | Tokens generated | `200` |
| `total_tokens` | INTEGER | Total usage | `350` |
| `is_tool_call` | INTEGER | 0 or 1 | `1` |
| `tool_name` | TEXT | Tool used | `"get_stock_price"` |
| `tool_input` | TEXT | JSON parameters | `'{"ticker":"AAPL"}'` |
| `tool_output` | TEXT | JSON result | `'{"price":150.25}'` |
| `custom_data` | TEXT | JSON metadata | `'{"confidence":0.95}'` |

#### Example Insertions

**Human message:**
```sql
INSERT INTO messages (
    message_id, session_id, from_agent_id, to_agent_id,
    message_type, content, timestamp, sequence_number
) VALUES (
    'msg_001',
    'trading_session_001',
    'system',
    'portfolio_manager',
    'human',
    'Analyze AAPL stock performance and recommend action',
    '2025-01-20T10:30:00.000000',
    1
);
```

**AI response:**
```sql
INSERT INTO messages (
    message_id, session_id, from_agent_id, to_agent_id,
    message_type, content, timestamp, sequence_number,
    input_tokens, output_tokens, total_tokens
) VALUES (
    'msg_002',
    'trading_session_001',
    'portfolio_manager',
    'risk_manager',
    'ai',
    'AAPL shows bearish trend. Requesting risk assessment before deciding.',
    '2025-01-20T10:30:15.123456',
    2,
    250,
    180,
    430
);
```

**Tool call:**
```sql
INSERT INTO messages (
    message_id, session_id, from_agent_id, to_agent_id,
    message_type, content, timestamp, sequence_number,
    is_tool_call, tool_name, tool_input, tool_output
) VALUES (
    'msg_003',
    'trading_session_001',
    'portfolio_manager',
    NULL,
    'tool',
    'Fetching AAPL price',
    '2025-01-20T10:30:10.000000',
    3,
    1,
    'get_stock_price',
    '{"ticker":"AAPL"}',
    '{"price":150.25,"change":-2.15}'
);
```

---

## POIROT Internal Results

POIROT generates all error vector analysis internally during Phases 0–2. Results are saved
as JSON files in the output directory — no additional database tables are required.

---

## Data Format Requirements

### 1. Session IDs
- **Format:** Any unique string
- **Recommendation:** `"{system}_{session_num}_{timestamp}"` or similar
- **Example:** `"trading_session_001"`, `"medical_case_2025_01_20_001"`

### 2. Agent IDs
- **Format:** Alphanumeric with underscores (no spaces)
- **Recommendation:** Lowercase with descriptive names
- **Example:** `"portfolio_manager"`, `"diagnosis_doctor"`, `"qa_agent"`

### 3. Message IDs
- **Format:** Any unique string
- **Recommendation:** Sequential with prefix
- **Example:** `"msg_001"`, `"msg_002"`, or UUIDs

### 4. Timestamps
- **Format:** ISO 8601 with microseconds
- **Format string:** `"YYYY-MM-DDTHH:MM:SS.ffffff"`
- **Example:** `"2025-01-20T10:30:45.123456"`
- **Python:** `datetime.now().isoformat()`

### 5. JSON Fields
All JSON fields should be **valid JSON strings**:

```python
# Correct
custom_metadata = '{"key": "value", "number": 123}'

# Incorrect
custom_metadata = "{'key': 'value'}"  # Single quotes invalid
```

### 6. Error Vectors
- **Format:** JSON array of integers (0 or 1)
- **Example:** `"[1,0,0,1,1,0]"`
- **Length:** Must match number of dimensions in `error_vectors` table

---

## Integration Examples

### Example 1: Simple Trading Bot

```python
import sqlite3
from datetime import datetime

# Create database
conn = sqlite3.connect('trading_bot.db')
cursor = conn.cursor()

# Execute schema
with open('schema.sql', 'r') as f:
    cursor.executescript(f.read())

# Add session
session_id = "trading_001"
cursor.execute("""
    INSERT INTO sessions (
        session_id, system_name, session_number, 
        created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?)
""", (
    session_id, "TradingBot", 1,
    datetime.now().isoformat(),
    datetime.now().isoformat()
))

# Add agents
agents = [
    ("portfolio_mgr", "Portfolio Manager", "TradingBot"),
    ("risk_mgr", "Risk Manager", "TradingBot")
]
cursor.executemany("""
    INSERT INTO agents (agent_id, agent_name, system_name)
    VALUES (?, ?, ?)
""", agents)

# Add messages
messages = [
    ("msg_1", session_id, "system", "portfolio_mgr", "human", 
     "Analyze AAPL", datetime.now().isoformat(), 1),
    ("msg_2", session_id, "portfolio_mgr", "risk_mgr", "ai",
     "Recommend sell", datetime.now().isoformat(), 2)
]
cursor.executemany("""
    INSERT INTO messages (
        message_id, session_id, from_agent_id, to_agent_id,
        message_type, content, timestamp, sequence_number
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", messages)

conn.commit()
conn.close()
```

### Example 2: Medical Diagnosis System

```python
import sqlite3
from datetime import datetime
import json

conn = sqlite3.connect('medical_system.db')
cursor = conn.cursor()

# Execute schema
with open('schema.sql', 'r') as f:
    cursor.executescript(f.read())

# Add session
session_id = "case_001"
cursor.execute("""
    INSERT INTO sessions (
        session_id, system_name, session_number,
        created_at, updated_at, has_ground_truth
    ) VALUES (?, ?, ?, ?, ?, ?)
""", (
    session_id, "MedicalAI", 1,
    datetime.now().isoformat(),
    datetime.now().isoformat(),
    1
))

# Note: error_vectors and ground_truth_labels are NOT inserted here.
# They will be generated by POIROT during analysis.

conn.commit()
conn.close()
```

---

## Best Practices

### 1. Message Ordering
Always increment `sequence_number` sequentially within each session. POIROT relies on message order for analysis.

### 2. Token Tracking
Track `input_tokens`, `output_tokens`, and `total_tokens` whenever possible. This helps POIROT analyze efficiency.

### 3. Tool Usage
Set `is_tool_call=1` and populate `tool_name`, `tool_input`, `tool_output` for tool invocations. This reveals agent reasoning patterns.

### 4. Agent Communication Graph
Populate `can_communicate_with` in the `agents` table to help POIROT understand system architecture.

### 5. Custom Metadata
Use `custom_metadata` and `custom_data` fields for system-specific information that might be useful later.

### 6. Ground Truth
If available, ALWAYS provide ground truth labels. This dramatically improves POIROT's analysis capabilities.

### 7. Error Vectors
Define error vectors that align with your system's goals. Examples:
- **Medical:** Diagnosis error, treatment error, communication error
- **Trading:** Price prediction error, risk assessment error, execution error
- **Customer Service:** Understanding error, solution error, escalation error

---

## FAQ

### Q1: Do I need to use all tables?

**A:** No. Required tables are:
- `sessions`
- `agents`
- `messages`

Optional but recommended:

- `agent_tools` (required if your agents use tools)

---

### Q2: Can I add custom columns to tables?

**A:** We recommend using the `custom_metadata` and `custom_data` JSON fields instead of modifying the schema. This ensures compatibility with future POIROT versions.

---

### Q3: What if my agents don't use LLMs?

**A:** That's fine. Leave `llm_model`, `temperature`, etc. as NULL or empty. POIROT focuses on agent behavior, not implementation.

---

### Q4: How do I handle multi-turn conversations?

**A:** Use `sequence_number` to order messages. Each message-response pair gets sequential numbers:
- User asks (sequence 1)
- Agent responds (sequence 2)
- Agent asks another agent (sequence 3)
- Other agent responds (sequence 4)
- ...

---

### Q5: What if my system has 50+ agents?

**A:** No problem. POIROT scales to any number of agents. Just add them all to the `agents` table with unique `agent_id` values.

---

### Q6: Can POIROT analyze real-time systems?

**A:** POIROT analyzes completed sessions stored in the database. For real-time systems, log messages as they occur, then run POIROT analysis after session completion.

---

### Q7: What if my error vector changes between sessions?

**A:** POIROT handles error vector generation dynamically. You do not need to manage this manually.

---

### Q8: How do I represent agent disagreements?

**A:** This is captured naturally in the `messages` table. If Agent A says "sell" and Agent B says "buy", both messages appear with their `from_agent_id` values. POIROT detects these conflicts automatically.

---

### Q9: Can I use this with non-Python systems?

**A:** Yes! Any system that can write to SQLite can use POIROT. The database format is language-agnostic.

---

### Q10: Where do I put the database file?

**A:** Name it `{system_name}_poirot.db` and place it in a dedicated folder. Example:
```
my_project/
├── trading_bot_poirot.db  ← Your data
├── poirot_analysis/        ← POIROT output (auto-generated)
└── ...
```

---

## Next Steps

1. ✅ **Read this guide thoroughly**
2. ✅ **Execute `schema.sql` to create your database**
3. ✅ **Populate tables with your system's data**
4. ✅ **Run POIROT analysis** (see main POIROT documentation)
5. ✅ **Review results and iterate**

---

## Support

For questions or issues:
- Check POIROT main documentation
- Review example databases in `examples/`
- Open an issue on the POIROT GitHub repository

---

**Happy analyzing! 🔍**
