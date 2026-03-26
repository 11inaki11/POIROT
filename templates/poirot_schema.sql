-- =============================================================================
-- POIROT Framework — SQLite Database Schema
-- =============================================================================
-- 4 core tables: sessions, agents, agent_tools, messages
-- Copy this file to your project and populate it with your system's data.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- sessions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    system_name         TEXT NOT NULL,
    session_number      INTEGER,
    created_at          TEXT NOT NULL,   -- ISO-8601 timestamp
    updated_at          TEXT,
    session_notes       TEXT,
    input_context       TEXT,
    final_output        TEXT,
    execution_time_seconds REAL,
    custom_metadata     TEXT            -- JSON string for arbitrary key/value pairs
);

-- ---------------------------------------------------------------------------
-- agents
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agents (
    agent_id            TEXT PRIMARY KEY,
    agent_name          TEXT NOT NULL,
    system_name         TEXT NOT NULL,
    agent_type          TEXT,           -- e.g. "planner", "executor", "critic"
    system_prompt       TEXT,
    llm_model           TEXT,
    temperature         REAL,
    max_tokens          INTEGER,
    has_tools           INTEGER NOT NULL DEFAULT 0,  -- 0 = false, 1 = true
    can_communicate_with TEXT            -- JSON array of agent_ids
);

-- ---------------------------------------------------------------------------
-- agent_tools
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_tools (
    agent_id            TEXT NOT NULL REFERENCES agents(agent_id),
    tool_name           TEXT NOT NULL,
    tool_description    TEXT,
    tool_schema         TEXT,           -- JSON schema string
    tool_code           TEXT,
    is_enabled          INTEGER NOT NULL DEFAULT 1,  -- 0 = false, 1 = true
    PRIMARY KEY (agent_id, tool_name)
);

-- ---------------------------------------------------------------------------
-- messages
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    message_id          TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(session_id),
    from_agent_id       TEXT NOT NULL,
    to_agent_id         TEXT NOT NULL,
    message_type        TEXT NOT NULL,  -- e.g. "text", "tool_call", "tool_result"
    content             TEXT,
    timestamp           TEXT NOT NULL,  -- ISO-8601 timestamp
    sequence_number     INTEGER NOT NULL,
    -- Optional token usage columns
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    total_tokens        INTEGER,
    -- Optional tool call columns
    is_tool_call        INTEGER DEFAULT 0,
    tool_name           TEXT,
    tool_input          TEXT,           -- JSON string
    tool_output         TEXT,           -- JSON string
    -- Optional arbitrary metadata
    custom_data         TEXT            -- JSON string
);

-- ---------------------------------------------------------------------------
-- Indexes for common query patterns
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_messages_session_seq
    ON messages (session_id, sequence_number);

CREATE INDEX IF NOT EXISTS idx_messages_agents
    ON messages (from_agent_id, to_agent_id);
