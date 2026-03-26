# How POIROT works

POIROT runs in 4 sequential phases. Each phase builds on the previous one. The only inputs you provide are a description of your system and the session data — POIROT drives the rest autonomously.

---

## Phase 0 — Error Vector Space

Before analyzing any session, POIROT needs to understand what can go wrong in your specific system.

An LLM reads your `system_description` and produces a list of all possible error sources: agent misbehaviors, communication failures, tool misuse, reasoning errors, etc. Each error source becomes one dimension of a binary **error vector space**.

This phase is cached — it only runs once per system. If you analyze multiple sessions of the same system, the error space is reused.

**Output:** A set of N error dimensions specific to your system.

---

## Phase 1 — Individual Analysis (Self-Assessment)

Each agent that participated in the session analyzes its own behavior independently, without communicating with other agents.

POIROT reconstructs what each agent saw during the session — the messages they sent and received — and presents them as context. The agent then produces a structured self-assessment report answering: *"Based on what I did and what I observed, where do I think the error lies?"*

By default, each agent only sees its own messages. With `full_context=True`, each agent sees all messages from all agents in the session (useful for systems where full observability is desirable, but can cause context overflow in long sessions).

**Output:** One structured report per agent.

---

## Phase 2 — Peer Consultation

Agents don't always have full information from Phase 1. In Phase 2, they can interrogate each other.

POIROT runs a LangGraph multi-agent graph where each agent can send questions to other agents, receive their responses, and update their position. This models the real forensic process: investigators talk to witnesses, cross-check stories, and converge on a conclusion.

Each agent ends Phase 2 by casting a **vote**: a binary vector over the error space indicating which dimensions they believe are active in this session.

**Output:** One vote vector per agent.

---

## Phase 3 — Weighted Consensus

Not all agent votes are equally reliable. An agent that was directly involved in the failure is a better witness than one that observed it from a distance.

POIROT aggregates votes using a weighted formula:

```
w_i = β + 0.5 × (1 - d_H(position_i, vote_i) / N)
```

Where:
- `β = 0.25` is a baseline weight guaranteed to every agent
- `d_H` is the Hamming distance between the agent's own self-assessed position and their final vote
- `N` is the number of error dimensions

Agents whose vote is consistent with their own position (low `d_H`) get higher weight. The weighted sum produces the **final hazard vector** — a ranked identification of the error sources in the session.

**Output:** Final hazard vector + full audit trail of votes and weights.
