# POIROT Phase 2: Peer Consultation Protocol

**Status:** ✅ Implemented  
**Module:** `src/phase2_protocol.py`  
**Last Updated:** December 2024

---

## 📋 Overview

**Phase 2** implements the peer consultation stage of the POIROT protocol, where agents engage in collaborative analysis to identify hazard vectors. Unlike Phase 1 (individual analysis), agents can now communicate with each other to validate observations, gather additional perspectives, and reach more informed conclusions.

### Key Features

- 🤝 **Peer-to-Peer Communication**: Dynamic tool generation based on `can_communicate_with` relationships
- 🌐 **POIROT Server Mediation**: Central server prevents message flooding and ensures orderly consultation
- 🎯 **State-Based Tracking**: Consultation flags and call counts prevent infinite loops
- 📊 **Vote Collection**: JSON-formatted votes with hazard_vector, location (binary), and justification
- 🔄 **Retry Logic**: Handles empty responses with automatic retries
- 👥 **Non-Participant Support**: Agents who didn't participate in original session can still contribute

---

## 🏗️ Architecture

### Component Hierarchy

```
Phase2State (TypedDict)
    ↓
CommunicationToolFactory
    ↓
create_agent_nodes() → (call_llm, take_action, tools_dict)
    ↓
create_poirot_server_node() → poirot_server
    ↓
LangGraph StateGraph
    ↓
execute_phase2_analysis() → votes
```

### State Management

Each agent has dedicated state flags:

- `{agent_id}_consulted`: Boolean, True if agent used communication tools
- `{agent_id}_final_report`: Boolean, True if agent provided JSON vote
- `{agent_id}_call_count`: Integer, number of LLM invocations for this agent
- `{agent_id}_hazard_vector`: String, identified hazard vector name
- `{agent_id}_hazard_location`: List[int], binary location vector
- `{agent_id}_justification`: String, detailed explanation of vote

### Message Flow

```
Agent LLM Call
    ↓
[has tool calls?] → Yes → Execute Tools → [pending messages?] → Yes → POIROT Server
    ↓                          ↓                                       ↓
    No                         No                                 Mediate Communication
    ↓                          ↓                                       ↓
[has final report?] → Yes → Next Agent                         Route Response Back
    ↓
    No → Request Final Report → Call LLM Again
```

---

## 🔧 Implementation Details

### 1. Communication Tool Generation

Tools are created dynamically based on agent relationships:

```python
tool_factory = CommunicationToolFactory()

for agent_id, agent_data in agents_data.items():
    tools = []
    for target_id in agent_data['can_communicate_with']:
        tool = tool_factory.create_tool(target_id, agents_data[target_id]['name'])
        tools.append(tool)
```

**Tool Behavior:**
- Sets flag: `tool_factory.message_flags[target_id] = True`
- Stores message: `tool_factory.pending_messages[target_id] = message`
- Returns confirmation: `"Message sent to {target_name}: {message}"`

### 2. Agent Node Creation

Each agent gets three functions:

#### `call_llm(state: Phase2State) -> Phase2State`

**Responsibilities:**
1. Filter messages where agent participated (via metadata)
2. Clean messages for Gemini API compatibility
3. Build context in strict order:
   - System prompt
   - Historical messages
   - Protocol message (first call only)
   - Phase 1 reports (first call only)
   - Vectors to ignore (first call only)
   - Final report request (if already consulted)
4. Invoke LLM with tools
5. Update state flags based on response

**Key Logic:**
```python
if not consulted:
    # First call: add protocol, phase1, ignore list
    messages.append(HumanMessage(content=protocol))
    messages.append(SystemMessage(content=phase1_context))
    messages.append(HumanMessage(content=ignore_msg))

if consulted:
    # Already consulted: request final report
    messages.append(HumanMessage(content=final_request))
```

#### `take_action(state: Phase2State) -> Phase2State`

**Responsibilities:**
1. Extract tool calls from LLM response
2. Execute each tool (sets flags in tool_factory)
3. Create ToolMessage results
4. Return results to agent

**Tool Execution:**
```python
for tool_call in tool_calls:
    tool_input = tool_call['args'].get('message', '')
    result = tools_dict[tool_call['name']].invoke(tool_input)
    tool_msg = ToolMessage(tool_call_id=tool_call['id'], content=str(result))
    results.append(tool_msg)
```

### 3. POIROT Server Mediation

The POIROT server handles all inter-agent communication:

**Process:**
1. Check `tool_factory.has_pending_messages()`
2. For each pending message:
   - Filter target agent's historical context
   - Build consultation query with protocol context
   - Invoke target agent's LLM (no tools, just answer)
   - Retry up to 3 times if response is empty
   - Return response to original sender
3. Clear all pending flags

**Retry Logic (prevents empty responses):**
```python
MAX_RETRIES = 3
retry_count = 0

while retry_count < MAX_RETRIES:
    response = consultation_llm.invoke(context)
    
    if response.content.strip():
        break  # Valid response
    else:
        retry_count += 1
        context.append(HumanMessage(
            content="CRITICAL: Your previous response was empty. You MUST provide a substantive answer."
        ))
```

### 4. Routing Functions

Each agent has two routing functions:

#### `route_after_llm(state) -> str`

Returns one of:
- `"tools"`: Agent made tool calls → go to `tools_{agent_id}`
- `"next_agent"`: Agent provided final report → go to next agent (or END)
- `"continue"`: No tools or report → call LLM again to request final report

#### `route_after_tools(state) -> str`

Returns one of:
- `"poirot_server"`: Pending communications → mediate via POIROT server
- `"next_agent"`: Final report complete → go to next agent (or END)
- `"continue"`: No pending messages, no final report → return to LLM

### 5. LangGraph Construction

```python
graph = StateGraph(Phase2State)

# Add nodes for each agent
for agent_id in agent_order:
    graph.add_node(f"llm_{agent_id}", call_llm)
    graph.add_node(f"tools_{agent_id}", take_action)

# Add POIROT server
graph.add_node("poirot_server", poirot_server)

# Set entry point
graph.set_entry_point(f"llm_{agent_order[0]}")

# Add conditional edges
for i, agent_id in enumerate(agent_order):
    next_node = f"llm_{agent_order[i+1]}" if i < len(agent_order) - 1 else END
    
    graph.add_conditional_edges(
        f"llm_{agent_id}",
        route_after_llm,
        {"tools": f"tools_{agent_id}", "next_agent": next_node, "continue": f"llm_{agent_id}"}
    )
    
    graph.add_conditional_edges(
        f"tools_{agent_id}",
        route_after_tools,
        {"poirot_server": "poirot_server", "next_agent": next_node, "continue": f"llm_{agent_id}"}
    )

# Compile
compiled_graph = graph.compile()
```

---

## 📊 Output Format

Phase 2 generates three output files:

### 1. `phase2_votes.json`

JSON file with all agent votes:

```json
{
  "agent_1": {
    "agent_name": "Market Analysis Agent",
    "hazard_vector": "Incorrect risk assessment threshold",
    "location": [0, 1, 0, 0],
    "justification": "After consulting with Risk Agent, I identified that the threshold was set too high..."
  },
  "agent_2": {
    "agent_name": "Risk Assessment Agent",
    "hazard_vector": "Data synchronization delay",
    "location": [1, 0, 0, 0],
    "justification": "Portfolio Manager confirmed they received stale data due to sync delay..."
  }
}
```

### 2. `phase2_state.json`

Complete state snapshot:

```json
{
  "session_id": 42,
  "votes": { ... },
  "agent_tracking": {
    "agent_1": {
      "consulted": true,
      "final_report": true,
      "call_count": 3
    },
    "agent_2": {
      "consulted": true,
      "final_report": true,
      "call_count": 2
    }
  }
}
```

### 3. `phase2_summary.txt`

Human-readable report:

```
═══════════════════════════════════════════════════════════════════════════
POIROT PHASE 2: PEER CONSULTATION SUMMARY
═══════════════════════════════════════════════════════════════════════════

Session ID: 42
Database: database/example_trading_system.db
Agents Consulted: 4

────────────────────────────────────────────────────────────────────────────
HAZARD VECTOR VOTES
────────────────────────────────────────────────────────────────────────────

Agent: Market Analysis Agent (agent_1)
Hazard Vector: Incorrect risk assessment threshold
Location: [0, 1, 0, 0]
Justification:
After consulting with Risk Agent, I identified that the threshold was set too high...

────────────────────────────────────────────────────────────────────────────

Agent: Risk Assessment Agent (agent_2)
Hazard Vector: Data synchronization delay
Location: [1, 0, 0, 0]
Justification:
Portfolio Manager confirmed they received stale data due to sync delay...

...
```

---

## 🚀 Usage

### Basic Usage

```python
from poirot_pipeline import POIROTPipeline

# Initialize pipeline
pipeline = POIROTPipeline(
    system_name="TradingBot",
    system_description="...",
    database_path="trading.db"
)

# Run prerequisite phases
pipeline.run_phase0_error_space()
pipeline.run_phase1_agent_factory()
pipeline.run_phase1_protocol()

# Run Phase 2
result = pipeline.run_phase2_protocol()

print(f"Votes collected: {result['num_agents']}")
for agent_id, vote in result['votes'].items():
    print(f"{vote['agent_name']}: {vote['hazard_vector']}")
```

### Advanced Usage with Custom Configuration

```python
from pathlib import Path
from phase2_protocol import execute_phase2_analysis

# Direct Phase 2 execution with custom parameters
result = execute_phase2_analysis(
    session_id=42,
    db_path=Path("my_system.db"),
    phase1_reports=phase1_data,  # Dict[agent_id, report_text]
    vectors_to_ignore=[
        "Known behavior: System has 5 agents by design",
        "Market volatility is expected"
    ],
    error_space=error_space_json,
    model_name="gemini-2.0-flash-exp",
    output_dir=Path("results/phase2/")
)

votes = result['votes']
state = result['state']
output_files = result['output_files']
```

---

## 🔍 Debugging

### Enable Verbose Logging

Phase 2 prints detailed execution logs:

```
═══════════════════════════════════════════════════════════════════════════
🤖 AGENT: Market Analysis Agent (agent_1)
═══════════════════════════════════════════════════════════════════════════
Total messages: 150
Filtered for agent_1: 42
Consulted: False, Call count: 1

────────────────────────────────────────────────────────────────────────────
RESPONSE from Market Analysis Agent:
────────────────────────────────────────────────────────────────────────────
I need to consult with the Risk Agent to validate my observation about...
────────────────────────────────────────────────────────────────────────────

  ✅ Market Analysis Agent consulted with peers (tool calls detected)

🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧
MARKET ANALYSIS AGENT EXECUTING TOOLS
🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧

📌 Tool 1/1: talk_to_agent_2
   💬 Message: Can you confirm if you noticed any issues with the risk threshold during the trading session?
   ✅ Message sent to Risk Assessment Agent: Can you confirm...

✅ All tools executed.
🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧

═══════════════════════════════════════════════════════════════════════════
🌐 POIROT SERVER - MEDIATING COMMUNICATIONS
═══════════════════════════════════════════════════════════════════════════

💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬
CONSULTATION TO: Risk Assessment Agent (agent_2)
💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬💬

  📊 Context for agent_2: 38 messages

📥 QUERY:
────────────────────────────────────────────────────────────────────────────
Can you confirm if you noticed any issues with the risk threshold during the trading session?
────────────────────────────────────────────────────────────────────────────

📤 RESPONSE from Risk Assessment Agent:
────────────────────────────────────────────────────────────────────────────
Yes, I observed that the threshold was set to 0.15 when market conditions suggested it should have been 0.10...
────────────────────────────────────────────────────────────────────────────

✅ All pending consultations processed.
═══════════════════════════════════════════════════════════════════════════
```

### Common Issues

**1. Empty Responses from Agents**
- **Symptom**: Agent returns empty string or whitespace
- **Cause**: LLM context too large or confusing
- **Solution**: Retry logic automatically handles this (up to 3 attempts)

**2. Infinite Tool Calls**
- **Symptom**: Agent keeps calling tools without providing final report
- **Cause**: Agent not understanding when to stop
- **Solution**: `consulted` flag triggers final report request automatically

**3. Vote Extraction Fails**
- **Symptom**: `"Error extracting vote"` in hazard_vector
- **Cause**: Agent didn't format JSON correctly
- **Solution**: Check agent's response, improve prompt if needed

**4. Non-Participants Have No Context**
- **Symptom**: Agent says "I have no information"
- **Cause**: Agent wasn't in original session
- **Solution**: Phase 2 provides clinical summary and special protocol for non-participants

---

## 🧪 Testing

### Unit Tests (Coming Soon)

```python
# tests/test_phase2_protocol.py

def test_communication_tool_generation():
    """Test that tools are created correctly based on can_communicate_with."""
    pass

def test_message_filtering():
    """Test that agents only see messages where they participated."""
    pass

def test_poirot_server_mediation():
    """Test that server routes messages correctly."""
    pass

def test_vote_extraction():
    """Test JSON vote parsing from agent responses."""
    pass
```

### Integration Test

```bash
cd examples/
python run_phase2_example.py
```

Expected output:
- No errors during execution
- 4 votes collected (one per agent)
- 3 files created in `POIROT_output/phase2/session_1/`

---

## 📚 Related Documentation

- [Phase 1B: Individual Analysis Protocol](PHASE1_PROTOCOL.md)
- [Phase 0: Error Space Construction](PHASE0_ERROR_SPACE.md)
- [Database Schema](DATABASE_SCHEMA.md)
- [POIROT Agent Documentation](POIROT_AGENT.md)

---

## 🔮 Future Enhancements

### Planned Features

1. **Adaptive Consultation Limits**
   - Currently unlimited consultations (controlled by LLM)
   - Add max_consultations parameter to prevent excessive back-and-forth

2. **Group Discussions**
   - Allow multi-agent "round table" discussions
   - All agents in conversation see each other's messages

3. **Consultation History Visualization**
   - Graph showing who talked to whom and when
   - Highlight key information exchanges

4. **Confidence Scores**
   - Agents can express confidence in their votes (0-1 scale)
   - Use for weighted voting in Phase 3

5. **Evidence Tracking**
   - Link justifications to specific messages from consultation
   - Enable "proof" trail for votes

---

## 🤔 Design Decisions

### Why Central POIROT Server?

**Alternatives Considered:**
1. Direct agent-to-agent messaging
2. Broadcast all messages to all agents
3. Hierarchical communication tree

**Decision:** Central server mediation

**Rationale:**
- Prevents message flooding (agents can't spam each other)
- Enables empty response retry logic
- Allows monitoring of all communications
- Simplifies routing logic
- Matches reference implementation behavior

### Why Mandatory Consultation?

Agents **must** consult before voting (enforced by protocol message).

**Rationale:**
- Phase 1 is for individual analysis; Phase 2 is for collaboration
- Without consultation, Phase 2 adds no value over Phase 1
- Ensures diverse perspectives are considered
- Reduces individual bias

### Why Binary Location Vector?

Location must be binary array (e.g., `[0, 1, 0, 0]`), not text.

**Rationale:**
- Enables programmatic analysis in Phase 3
- Allows vote aggregation (majority voting per position)
- Compatible with error space from Phase 0
- Structured format prevents ambiguity

---

**Last Updated:** December 2024  
**Contributors:** CORTEX Team  
**Questions?** Open an issue on GitHub
