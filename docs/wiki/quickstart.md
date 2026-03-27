# Quickstart

## 1. Install

```bash
pip install poirot-framework
```

---

## 2. Choose how to provide your session data

POIROT needs the conversation history of the session you want to analyze. How you provide it depends on your system:

- **SQLite database** — the most flexible option, works with any system. See [Integrations → Database](integrations/database.md) for setup.
- **LangChain agents** — pass your compiled agents directly. See [Integrations → LangChain](integrations/langchain.md).

The quickstart below uses the SQLite database integration.

---

## 3. Run an analysis

```python
import poirot

results = poirot.run_poirot(
    database_path="my_system.db",
    system_name="MyAgentSystem",
    system_description="""
        A multi-agent system with 3 agents:
        - PlannerAgent: decomposes the user request into subtasks
        - ExecutorAgent: executes each subtask using external tools
        - ReviewerAgent: validates the final output before delivery
        Communication flow: User → Planner → Executor → Reviewer → User
    """,
    provider="gemini",
    api_key="YOUR_API_KEY",
)
```

POIROT prints progress to the terminal and writes detailed results to `poirot_results/` (configurable via `output_dir`).

---

## 4. Use a .env file (recommended)

Never hardcode API keys. Create a `.env` file:

```env
GOOGLE_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
```

Then:

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

---

## 5. Use the launcher script

For a script-based workflow, copy `run_poirot.py` from the repository, edit the configuration block at the top, and run:

```bash
python run_poirot.py --provider gemini
python run_poirot.py --provider deepseek
python run_poirot.py --provider ollama
python run_poirot.py --provider local    # LM Studio
```

The script lists all sessions in the database and lets you pick one interactively.
