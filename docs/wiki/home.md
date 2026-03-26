# POIROT

**POIROT** is a forensic debugging algorithm for multi-agent AI systems.

When a multi-agent system produces a wrong, dangerous, or unexpected output, the hardest question is: **which agent caused it?** In a system with 5 agents that all communicated with each other, tracing the root cause manually is slow, error-prone, and often impossible.

POIROT automates that investigation. You give it a description of your system and the conversation history of a session. It returns a structured forensic report identifying which component is responsible for the failure.

---

## What POIROT does

POIROT treats a failed session as a crime scene. It:

1. Maps all possible error sources in your system into an **error vector space**
2. Has each agent independently **self-assess** their own behavior
3. Runs a **peer consultation** where agents interrogate each other
4. Produces a **weighted consensus vote** identifying the faulty component

The result is an explainable, auditable forensic report — not a black-box prediction.

---

## How to feed data to POIROT

POIROT needs two things about the session it is analyzing:

- **System description** — who the agents are, their roles, and how they communicate
- **Session messages** — the conversation history of the failed run

How you provide that data depends on how your system is built. POIROT supports multiple integration methods:

| Integration | When to use |
| ----------- | ----------- |
| [SQLite database](integrations/database.md) | Any system, any language, maximum control |
| LangChain agents | *(coming soon)* |

---

## Pages

- [How it works](how-it-works.md) — The 4 phases in depth
- [Quickstart](quickstart.md) — Run your first analysis in minutes
- [API Reference](api-reference.md) — All `run_poirot()` parameters
- [LLM Providers](providers.md) — Gemini, DeepSeek, Ollama, LM Studio
- [Integrations](integrations/database.md) — How to connect your system
