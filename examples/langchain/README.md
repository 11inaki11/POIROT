# LangChain Examples

Each example is a self-contained multi-agent system with a pre-simulated
conversation that contains an intentional bug. Run any example to let POIROT
identify which agent is responsible.

## Examples

| File | Domain | Agents | Bug |
| ---- | ------ | ------ | --- |
| `01_medical_diagnosis.py` | Emergency medicine | DiagnosisAgent, TreatmentAgent | DiagnosisAgent omits current medications from handoff; TreatmentAgent activates cath lab with contrast without holding Metformin |
| `02_stock_trading.py` | Algorithmic trading | MarketAnalystAgent, RiskManagerAgent, TradeExecutorAgent | RiskManagerAgent approves a trade despite a tool warning that portfolio sector concentration would hit 67% (limit: 40%) |
| `03_web_development.py` | Frontend development | DesignerAgent, DeveloperAgent, QAAgent | DesignerAgent uses a non-compliant contrast color; QAAgent skips the accessibility audit before approving for production |

## Setup

```bash
# Install the library (from the repo root)
pip install -e .

# Create a .env file with your API key
echo "GOOGLE_API_KEY=your_key_here" > .env
```

## Run

```bash
python examples/langchain/01_medical_diagnosis.py
python examples/langchain/02_stock_trading.py
python examples/langchain/03_web_development.py
```

Results are written to `poirot_results/<domain>/` and printed to the terminal.

## Structure of each example

1. **Tool definitions** — `@tool`-decorated functions that simulate real external systems
2. **Agent creation** — `create_react_agent(llm, tools=...)` for each agent
3. **Simulated conversation** — pre-built `List[BaseMessage]` representing a real session
4. **POIROT call** — `run_poirot_from_agents()` with one `LangChainAgentAdapter` per agent

The pre-built conversations mean you only need an API key for the POIROT analysis
itself — the agents do not make LLM calls when the examples run.
