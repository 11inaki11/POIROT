# POIROT Framework — Wiki

**POIROT** (Protocol for Observing, Investigating, and Reasoning Over Threats) is a Python library that performs automated forensic analysis on multi-agent AI systems. Given a session stored in a SQLite database, POIROT identifies which agent or component is responsible for a failure.

---

## How it works

POIROT runs in 4 phases:

| Phase | Name | What it does |
|-------|------|-------------|
| 0 | Error Vector Space | An LLM builds the space of possible error sources for your specific system |
| 1 | Individual Analysis | Each agent reviews its own message history and self-assesses |
| 2 | Peer Consultation | Agents consult each other via a LangGraph multi-agent graph |
| 3 | Weighted Voting | Votes are aggregated using a weighted Hamming-distance formula |

All you need to provide is a SQLite database with your session data and a description of your system. POIROT handles everything else.

---

## Pages

- [Quickstart](quickstart.md) — Install and run your first analysis
- [API Reference](api-reference.md) — All `run_poirot()` parameters
- [Database Schema](database-schema.md) — How to populate your database
- [LLM Providers](providers.md) — Gemini, DeepSeek, Ollama, LM Studio
