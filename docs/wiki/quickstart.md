# Quickstart

## 1. Install

```bash
pip install poirot-framework
```

---

## 2. Prepare your database

You need a SQLite database populated with your session data. See [Database Schema](database-schema.md) for the full guide.

The easiest way to get the schema is to copy `templates/poirot_schema.sql` from the repository and run it:

```python
import sqlite3

conn = sqlite3.connect("my_system.db")
with open("poirot_schema.sql") as f:
    conn.executescript(f.read())
conn.close()
```

---

## 3. Run an analysis

```python
import poirot

results = poirot.run_poirot(
    database_path="my_system.db",
    system_name="MyAgentSystem",
    system_description="""
        A multi-agent system with 3 agents:
        - PlannerAgent: decomposes the task into subtasks
        - ExecutorAgent: executes each subtask
        - ReviewerAgent: validates the final output
    """,
    provider="gemini",
    api_key="YOUR_API_KEY",
)
```

That is all that is required. POIROT will print progress to the terminal and save detailed results to `poirot_results/` (configurable via `output_dir`).

---

## 4. Use the launcher script (alternative)

If you prefer a script-based workflow, copy `run_poirot.py` from the repository, edit the configuration section at the top, and run:

```bash
python run_poirot.py --provider gemini
python run_poirot.py --provider deepseek
python run_poirot.py --provider ollama
python run_poirot.py --provider local     # LM Studio
```

The launcher detects all sessions in the database and lets you select one interactively.

---

## 5. Using a .env file (recommended)

Avoid hardcoding API keys. Create a `.env` file:

```
GOOGLE_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
```

Then load it before calling `run_poirot()`:

```python
from dotenv import load_dotenv
import os
import poirot

load_dotenv()

results = poirot.run_poirot(
    database_path="my_system.db",
    system_name="MyAgentSystem",
    system_description="...",
    provider="gemini",
    api_key=os.getenv("GOOGLE_API_KEY"),
)
```
