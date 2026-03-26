# POIROT-SW Architecture

**Version:** 1.0  
**Last Updated:** January 2025  
**Status:** Design Phase

---

## 📐 System Architecture Overview

POIROT-SW follows a **modular pipeline architecture** with 6 independent phases. Each phase can be developed, tested, and deployed independently.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEVELOPER'S SYSTEM                            │
│                  (Any Multi-Agent Framework)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Execution Data
                            ▼
                   ┌────────────────┐
                   │   SQLite DB    │
                   │  (schema.sql)  │
                   └────────┬───────┘
                            │
         ┌──────────────────┴──────────────────┐
         │         POIROT-SW PIPELINE          │
         └──────────────────┬──────────────────┘
                            │
                            │
    ┌───────────────────────┴───────────────────────┐
    │                                               │
    │  Phase 1: Agent Factory                      │
    │  ├─ Load agent configurations                │
    │  ├─ Create LangChain agents with tools       │
    │  └─ Validate agent setup                     │
    │                                               │
    └───────────────────────┬───────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────────┐
    │  Phase 2: Communication System                │
    │  ├─ Build agent communication graph           │
    │  ├─ Setup message routing                     │
    │  └─ Enable inter-agent messaging              │
    └───────────────────────┬───────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────────┐
    │  Phase 3: Graph Builder                       │
    │  ├─ Define state schema dynamically           │
    │  ├─ Create StateGraph nodes/edges             │
    │  └─ Compile executable workflow               │
    └───────────────────────┬───────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────────┐
    │  Phase 4: Execution & Logging                 │
    │  ├─ Run multi-agent workflow                  │
    │  ├─ Capture all messages and tool calls       │
    │  └─ Store execution data back to database     │
    └───────────────────────┬───────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────────┐
    │  Phase 5: Analysis Engine                     │
    │  ├─ Load session data from database           │
    │  ├─ Run POIROT protocol (Phase 1 & 2)        │
    │  ├─ Calculate error vectors and trust         │
    │  └─ Generate consensus and accuracy metrics   │
    └───────────────────────┬───────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────────┐
    │  Phase 6: Visualization                       │
    │  ├─ Generate hypercube visualizations         │
    │  ├─ Plot trust evolution graphs               │
    │  ├─ Create CSV reports                        │
    │  └─ Build HTML dashboard                      │
    └───────────────────────┬───────────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │    Results     │
                   │  (PNG, CSV,    │
                   │   HTML, JSON)  │
                   └────────────────┘
```

---

## 🔧 Phase 1: Agent Factory

### Purpose
Load agent definitions from database and instantiate them as executable LangChain agents.

### Input
- SQLite database with `agents` and `agent_tools` tables

### Process
1. Query `agents` table
2. For each agent:
   - Load system prompt
   - Configure LLM (model, temperature, max_tokens)
   - Query `agent_tools` table for this agent
   - Create tool instances
   - Bind tools to agent
3. Validate agent setup

### Output
- Dictionary of `agent_id` → `configured_agent` mappings
- Agent metadata structure

### Key Components

```python
class AgentFactory:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.agents = {}
    
    def load_agents(self) -> Dict[str, Any]:
        """Load all agents from database."""
        pass
    
    def create_agent(self, agent_config: dict) -> Any:
        """Create single agent with tools."""
        pass
    
    def validate_agent(self, agent_id: str) -> bool:
        """Validate agent configuration."""
        pass
```

### Design Decisions

**Tool Creation:**
- Tools defined in database as JSON schemas
- Convert JSON schema → Python function dynamically
- Support both built-in tools (LangChain) and custom tools

**LLM Selection:**
- Support multiple providers: Gemini, OpenAI, Anthropic, local models
- Use `llm_model` field to determine provider
- Handle authentication via environment variables

---

## 🔗 Phase 2: Communication System

### Purpose
Enable agents to send messages to specific other agents based on communication graph.

### Input
- Agent dictionary from Phase 1
- `can_communicate_with` field from `agents` table

### Process
1. Parse `can_communicate_with` JSON for each agent
2. Build directed communication graph
3. Create message routing infrastructure
4. Setup message state channels

### Output
- Communication graph: `agent_id` → `[allowed_recipients]`
- Message router function
- State channels for each agent

### Key Components

```python
class CommunicationSystem:
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.comm_graph = {}
    
    def build_graph(self) -> Dict[str, List[str]]:
        """Build communication graph from agent configs."""
        pass
    
    def can_communicate(self, from_id: str, to_id: str) -> bool:
        """Check if from_agent can send to to_agent."""
        pass
    
    def route_message(self, from_id: str, to_id: str, message: str):
        """Route message from one agent to another."""
        pass
```

### Design Decisions

**Message Format:**
- Use LangChain message objects (HumanMessage, AIMessage)
- Include metadata: timestamp, sequence_number, from/to agents
- Support broadcast messages (to_agent_id=None)

**State Management:**
- Each agent gets dedicated state channel
- Global state channel for system-wide messages
- Checkpoint state at each step

---

## 🏗️ Phase 3: Graph Builder

### Purpose
Dynamically construct a LangGraph StateGraph based on agents and communication rules.

### Input
- Agent dictionary from Phase 1
- Communication graph from Phase 2

### Process
1. Define state schema dynamically based on number of agents
2. Create node for each agent
3. Add edges based on communication graph
4. Add conditional edges for routing
5. Compile StateGraph

### Output
- Compiled StateGraph ready for execution

### Key Components

```python
class GraphBuilder:
    def __init__(self, agents: Dict, comm_system: CommunicationSystem):
        self.agents = agents
        self.comm_system = comm_system
        self.graph = None
    
    def build_state_schema(self) -> TypedDict:
        """Create state schema with channels for all agents."""
        pass
    
    def add_agent_nodes(self):
        """Add node for each agent."""
        pass
    
    def add_edges(self):
        """Add edges based on communication graph."""
        pass
    
    def compile(self) -> StateGraph:
        """Compile and return executable graph."""
        pass
```

### Design Decisions

**State Schema:**
```python
State = TypedDict(
    "State",
    {
        "messages": Annotated[list, add_messages],  # Global messages
        f"{agent_id}_messages": Annotated[list, add_messages],  # Per-agent
        # ... one channel per agent
        "session_id": str,
        "execution_metadata": dict
    }
)
```

**Node Creation:**
- Each node = agent invocation function
- Node reads from its channel, processes, writes to target channel(s)
- Automatic tool calling handled by LangChain

**Routing:**
- Use conditional edges for dynamic routing
- Route based on agent's target in message metadata
- Support sequential, parallel, and conditional flows

---

## ⚙️ Phase 4: Execution & Logging

### Purpose
Execute the multi-agent workflow and log all data for analysis.

### Input
- Compiled StateGraph from Phase 3
- Initial problem statement
- Session metadata

### Process
1. Create session in database
2. Initialize state with input
3. Execute graph with checkpointing
4. Intercept and log every message
5. Track token usage via callbacks
6. Store all data in `messages` table

### Output
- Populated `sessions` table with execution metadata
- Populated `messages` table with full conversation
- Updated `checkpoints` and `writes` tables (if applicable)

### Key Components

```python
class ExecutionEngine:
    def __init__(self, graph: StateGraph, db_path: str):
        self.graph = graph
        self.db_path = db_path
        self.session_id = None
    
    def create_session(self, system_name: str, input_context: str):
        """Create session record in database."""
        pass
    
    def execute(self, initial_input: str) -> Dict:
        """Run graph and log all messages."""
        pass
    
    def log_message(self, message: BaseMessage, metadata: dict):
        """Store message in database."""
        pass
```

### Design Decisions

**Token Tracking:**
- Use LangChain callbacks to capture token usage
- Store input/output/total tokens per message
- Support models with and without usage metadata

**Tool Logging:**
- Set `is_tool_call=1` for tool messages
- Store tool_name, tool_input, tool_output as JSON
- Link tool message to agent that called it

**Checkpointing:**
- Use SqliteSaver for LangGraph checkpoints
- Store checkpoints in same database
- Enable session recovery and debugging

---

## 🧠 Phase 5: Analysis Engine

### Purpose
Analyze completed sessions to detect errors, track trust, and calculate metrics.

### Input
- Session data from database (session_id)
- Ground truth labels (if available)
- Error vector definitions

### Process
1. Load session messages in sequence order
2. **POIROT Phase 1**: Individual agent error detection
   - Each agent analyzes conversation independently
   - Generates error vector hypothesis
3. **POIROT Phase 2**: Consensus formation
   - Agents vote on error vector
   - Calculate consensus using trust-weighted voting
4. Calculate trust evolution:
   - Compare agent predictions to ground truth
   - Update trust scores using learning rate
5. Calculate accuracy metrics:
   - Consensus accuracy per error dimension
   - Per-agent accuracy (exact, partial, missed)
6. Store results

### Output
- Error vectors per agent and consensus
- Trust evolution data
- Accuracy metrics
- JSON results file

### Key Components

```python
class AnalysisEngine:
    def __init__(self, db_path: str, trust_config: str = "balanced"):
        self.db_path = db_path
        self.trust_system = AgentTrustSystem(trust_config)
        self.results = {}
    
    def analyze_session(self, session_id: str) -> Dict:
        """Run full POIROT analysis on session."""
        pass
    
    def poirot_phase1(self, session_id: str) -> Dict:
        """Individual error detection."""
        pass
    
    def poirot_phase2(self, agent_vectors: Dict) -> List[int]:
        """Trust-weighted consensus."""
        pass
    
    def calculate_trust_evolution(self, session_id: str):
        """Track trust changes over session."""
        pass
```

### Design Decisions

**Trust System:**
- Maintain trust scores per agent
- Update based on agreement with ground truth (not consensus!)
- Support multiple trust configurations (conservative, balanced, aggressive)

**Error Vector Format:**
- Binary vector [0,1,0,1,...] where 1 = error
- Length = number of error dimensions
- Each dimension maps to specific failure mode

**Consensus Algorithm:**
- Trust-weighted voting: `vote_weight = agent_trust * vote_value`
- Threshold: If weighted sum > N/2, dimension = 1 (error)
- Handle ties: Default to 0 (no error) for conservative approach

---

## 📊 Phase 6: Visualization

### Purpose
Generate visual and textual reports of analysis results.

### Input
- Analysis results from Phase 5
- Session metadata
- Multiple sessions for batch analysis

### Process
1. Load results for one or more sessions
2. Generate visualizations:
   - Hypercube projection (MDS) of error vectors
   - Agent trust evolution over time
   - Consensus accuracy per error dimension
   - Per-agent performance metrics
3. Create CSV reports
4. Build HTML dashboard

### Output
- PNG images (hypercube, trust graphs, etc.)
- CSV files (experiment_results.csv)
- HTML dashboard (index.html)
- Summary statistics (printed to terminal)

### Key Components

```python
class Visualizer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
    
    def plot_hypercube(self, error_vectors: List, labels: List):
        """Generate MDS projection of error space."""
        pass
    
    def plot_trust_evolution(self, trust_data: Dict):
        """Plot trust changes over sessions."""
        pass
    
    def generate_csv_report(self, results: List[Dict]):
        """Create CSV with all metrics."""
        pass
    
    def build_dashboard(self, results: List[Dict]):
        """Create HTML dashboard."""
        pass
```

### Design Decisions

**Color Scheme:**
- Use HSV color space for unique agent colors
- Dynamic color generation based on number of agents
- Consistent colors across all visualizations

**Per-Configuration Visualizations:**
- Generate separate folders for each trust config
- Global visualizations show all sessions
- Per-config visualizations filter by trust setting

**Statistics Output:**
- Print detailed stats to terminal for quick review
- Include consensus accuracy per ground truth
- Show per-agent exact/partial accuracy

---

## 🗃️ Data Flow

### End-to-End Example

```
Developer System (Trading Bot)
↓
Generates SQLite DB with 3 agents, 1 session, 10 messages
↓
POIROT Phase 1: Agent Factory
  - Loads 3 agents from DB
  - Creates Portfolio Manager, Risk Assessor, Market Analyst
↓
POIROT Phase 2: Communication System
  - Builds graph: PM ↔ RA, PM ↔ MA
↓
POIROT Phase 3: Graph Builder
  - Creates StateGraph with 3 nodes, 4 edges
↓
POIROT Phase 4: Execution (OPTIONAL if data already exists)
  - Runs graph with input "Analyze AAPL"
  - Logs 10 messages to DB
↓
POIROT Phase 5: Analysis Engine
  - Loads 10 messages
  - POIROT Phase 1: Each agent predicts error vector
  - POIROT Phase 2: Consensus = [0,1,1]
  - Compares to ground truth [0,1,1] → 100% accuracy
  - Updates trust scores
↓
POIROT Phase 6: Visualization
  - Plots error hypercube (1 point)
  - Plots trust evolution (3 lines)
  - Generates CSV report
  - Creates HTML dashboard
↓
Output: figures/ folder with PNG images + experiment_results.csv
```

---

## 🎨 Technology Stack

| Layer | Technology |
|-------|------------|
| **Database** | SQLite 3 |
| **Agent Framework** | LangChain, LangGraph |
| **LLM Providers** | Gemini, OpenAI, Anthropic, Ollama |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Dimensionality Reduction** | Scikit-learn (MDS) |
| **Logging** | Python logging module |
| **Testing** | pytest, unittest |
| **Documentation** | Markdown, Sphinx (planned) |

---

## 🚦 Execution Modes

### Mode 1: Full Pipeline
Run all 6 phases in sequence. Used when:
- You want POIROT to execute your multi-agent system
- You're testing system behavior
- You need end-to-end analysis

```python
poirot = POIROTFullPipeline(db_path="my_system.db")
results = poirot.run(input_problem="Analyze AAPL stock")
```

### Mode 2: Analysis Only
Skip execution (Phases 1-4), run analysis on existing data. Used when:
- Data already exists in database
- You're analyzing past executions
- You're running batch analysis

```python
analyzer = AnalysisEngine(db_path="my_system.db")
results = analyzer.analyze_session(session_id="trading_001")
visualizer = Visualizer(results_dir="output/")
visualizer.plot_all(results)
```

### Mode 3: Visualization Only
Generate visualizations from existing analysis results. Used when:
- Results JSON already exists
- You want different visualization styles
- You're creating custom reports

```python
visualizer = Visualizer(results_dir="output/")
visualizer.load_results("results.json")
visualizer.plot_hypercube()
visualizer.plot_trust_evolution()
visualizer.generate_csv_report()
```

---

## 🔌 Integration Patterns

### Pattern 1: Offline Integration
1. Developer runs their multi-agent system
2. System writes execution data to SQLite
3. Developer runs POIROT analysis separately
4. **Pros**: No coupling, works with any framework
5. **Cons**: Requires manual database creation

### Pattern 2: Wrapper Integration
1. Developer wraps their agents with POIROT execution engine
2. POIROT runs system and logs automatically
3. Analysis runs immediately after execution
4. **Pros**: Automated logging, integrated workflow
5. **Cons**: Requires LangChain/LangGraph compatibility

### Pattern 3: Real-time Integration (Future)
1. POIROT monitors system via message bus
2. Analysis runs in background during execution
3. Real-time dashboard shows trust evolution
4. **Pros**: Immediate feedback, live monitoring
5. **Cons**: Higher complexity, resource overhead

---

## 🛡️ Error Handling

### Database Errors
- Missing tables → Check schema, re-run schema.sql
- Missing required fields → Validate with schema checker
- Corrupted data → Rollback to last checkpoint

### Execution Errors
- Agent creation fails → Check LLM API keys, model availability
- Tool call fails → Log error, continue with partial data
- Timeout → Configurable timeout per agent, graceful shutdown

### Analysis Errors
- Missing ground truth → Skip accuracy calculation, show warning
- Invalid error vector → Validate length matches dimensions
- Trust calculation fails → Use default trust values

---

## 📈 Scalability Considerations

### Number of Agents
- **Current**: Tested with up to 8 agents (CORTEX)
- **Target**: Support 100+ agents
- **Strategy**: Optimize graph construction, parallel agent invocation

### Number of Messages
- **Current**: Analyzed sessions with 500+ messages
- **Target**: Handle 10,000+ messages per session
- **Strategy**: Batch processing, incremental analysis

### Number of Sessions
- **Current**: Batch analysis of 100+ sessions
- **Target**: Analyze 10,000+ sessions
- **Strategy**: Parallel processing, result caching

---

## 🔒 Security & Privacy

### Data Privacy
- All data stored locally in SQLite
- No external data transmission (except LLM API calls)
- Support for local LLMs (Ollama) for sensitive data

### API Key Management
- Use environment variables for LLM API keys
- Never store keys in database
- Support for key rotation

### Access Control
- File-based access control via OS permissions
- No network exposure required
- Optional encryption for database at rest

---

## 🧪 Testing Strategy

### Unit Tests
- Test each phase independently
- Mock database interactions
- Validate data transformations

### Integration Tests
- Test phase-to-phase data flow
- Use example database
- Verify end-to-end results

### Regression Tests
- Compare results against CORTEX baseline
- Ensure backward compatibility
- Validate against known good outputs

---

## 📅 Roadmap

### Version 1.0 (Current)
- [x] Database schema design
- [x] Documentation
- [ ] Core implementation (Phases 1-6)
- [ ] Example integrations

### Version 1.1 (Q2 2025)
- [ ] Web dashboard
- [ ] Multi-session batch analysis
- [ ] AutoGen integration
- [ ] CrewAI integration

### Version 2.0 (Q3 2025)
- [ ] Real-time monitoring mode
- [ ] Error pattern detection with ML
- [ ] Automated recommendations
- [ ] Distributed execution

---

## 📚 References

- **LangChain Documentation**: https://python.langchain.com/
- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **Multi-Agent Systems**: Russell & Norvig, "Artificial Intelligence: A Modern Approach"
- **Trust in AI**: "Trust and Transparency in AI Systems" (Various papers)

---

**Document Status**: ✅ Complete  
**Last Review**: January 2025  
**Next Review**: After Phase 1 implementation
