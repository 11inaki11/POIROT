"""Regression tests for the LangChain adapter mode, the Ollama routing and the paper prompts.

No LLM is called: compiled agents are replaced by stubs that capture their input.
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from poirot.adapters import LangChainAgentAdapter, build_session_data
from poirot.llm_factory import LLMFactory
from poirot.phase1_protocol import POIROT_PHASE1_MESSAGE, execute_phase1_analysis
from poirot.phase2_protocol import (
    clean_historical_messages,
    create_phase2_protocol_message,
    filter_messages_for_agent,
    format_phase1_observations,
)
from poirot.poirot_agent import POIROT_SYSTEM_PROMPT

RAW = '{"knee_flexion_peak_deg": 18.2}'
REPORT = "Knee flexion peak 54 deg, on target."
MONITOR_PROMPT = "You are GaitMonitor. Summarise gait batches faithfully."


class StubAgent:
    """Stands in for a compiled LangGraph agent: records what Phase 1 sends it."""

    def __init__(self):
        self.received = None

    def invoke(self, payload):
        self.received = payload["messages"]
        return {"messages": [AIMessage(content=json.dumps({
            "self_evaluation": {"role_fulfilled": True, "anomalies_detected": False,
                                "description": "ok", "evidence": "N/A"},
            "peer_observations": [], "suspected_agents": []}))]}


def make_adapters(include_system_message=False):
    monitor_msgs = [
        HumanMessage(content="Analyse batch 42.", name="scheduler"),
        AIMessage(content="", tool_calls=[{"name": "read_gait_batch", "args": {},
                                           "id": "c1", "type": "tool_call"}]),
        ToolMessage(content=RAW, tool_call_id="c1", name="read_gait_batch"),
        AIMessage(content=REPORT),
    ]
    if include_system_message:
        monitor_msgs.insert(0, SystemMessage(content=MONITOR_PROMPT))
    clinical_msgs = [
        HumanMessage(content=REPORT, name="gait_monitor"),
        AIMessage(content="Increasing speed by 20%."),
    ]
    monitor, clinical = StubAgent(), StubAgent()
    adapters = [
        LangChainAgentAdapter(agent=monitor, messages=monitor_msgs, agent_id="gait_monitor",
                              agent_name="GaitMonitor", tools=[],
                              system_prompt=None if include_system_message else MONITOR_PROMPT),
        LangChainAgentAdapter(agent=clinical, messages=clinical_msgs, agent_id="clinical_agent",
                              agent_name="ClinicalAgent", tools=[], system_prompt="You are ClinicalAgent."),
    ]
    return adapters, monitor, clinical


# -- Ollama routing ---------------------------------------------------------------

def test_ollama_provider_is_not_rerouted_to_lm_studio():
    llm = LLMFactory.create_chat_llm(model_name="gpt-oss:20b", use_local=True,
                                     local_model_name="gpt-oss:20b", provider="ollama")
    assert "11434" in str(llm.openai_api_base)
    assert llm.model_name == "gpt-oss:20b"


def test_use_local_still_means_lm_studio():
    llm = LLMFactory.create_chat_llm(use_local=True, provider="gemini")
    assert "1234" in str(llm.openai_api_base)


# -- Adapter routing metadata -----------------------------------------------------

def test_adapter_messages_carry_routing_metadata():
    adapters, _, _ = make_adapters()
    processed, _, _ = build_session_data(adapters, include_tool_calls=True)
    for pm in processed:
        meta = pm.message.additional_kwargs["metadata"]
        assert (meta["from_node"], meta["to_node"]) == (pm.from_agent, pm.to_agent)

    report = [pm for pm in processed if pm.message.content == REPORT]
    assert len(report) == 1, "A's output and B's copy of it must be one message"
    assert (report[0].from_agent, report[0].to_agent) == ("gait_monitor", "clinical_agent")

    tool = [pm for pm in processed if pm.msg_type == "tool"][0]
    assert (tool.from_agent, tool.to_agent) == ("tool_read_gait_batch", "gait_monitor")


def test_adapter_does_not_mutate_user_messages():
    adapters, _, _ = make_adapters()
    build_session_data(adapters)
    assert all("metadata" not in m.additional_kwargs for a in adapters for m in a.messages)


# -- Phase 1: full session, own messages as AIMessage -----------------------------

def test_phase1_agent_reviews_full_session_with_correct_roles():
    adapters, monitor, clinical = make_adapters()
    processed, _, configs = build_session_data(adapters, include_tool_calls=True)
    execute_phase1_analysis(agents=configs, processed_messages=processed, agent_factory=None,
                            error_space=None, vectors_to_ignore=None,
                            include_tool_calls=True, verbose=False)

    session = monitor.received[:-2]   # last two are the protocol + Phase 1 task
    own = [m for m in session if isinstance(m, AIMessage)]
    assert any(REPORT in m.content for m in own), "own report must be an AIMessage"
    assert any(RAW in m.content and isinstance(m, HumanMessage) for m in session), "own tool result"
    assert any("Increasing speed" in m.content and isinstance(m, HumanMessage) for m in session), \
        "peer messages must be visible (full session)"
    assert all("From  to :" not in m.content for m in session)

    clinical_session = clinical.received[:-2]
    assert any("Increasing speed" in m.content and isinstance(m, AIMessage) for m in clinical_session)
    assert not any(RAW in m.content for m in clinical_session), "peers' tool results stay private"
    assert monitor.received[-1].content == POIROT_PHASE1_MESSAGE


# -- Phase 2: tool results kept, role prompt kept ---------------------------------

def test_phase2_history_keeps_tool_results_for_their_agent():
    adapters, _, _ = make_adapters()
    _, history, _ = build_session_data(adapters, include_tool_calls=True)
    ctx = clean_historical_messages(filter_messages_for_agent(history, "gait_monitor"),
                                    include_tool_calls=True, current_agent_id="gait_monitor")
    assert any(RAW in m.content for m in ctx)


def test_phase2_history_respects_include_tool_calls():
    adapters, _, _ = make_adapters()
    _, history, _ = build_session_data(adapters, include_tool_calls=False)
    assert not any(RAW in m.content for m in history)


def test_system_prompt_from_field_or_leading_system_message():
    adapters, _, _ = make_adapters()
    _, _, configs = build_session_data(adapters)
    assert configs["gait_monitor"]["system_prompt"] == MONITOR_PROMPT

    adapters, _, _ = make_adapters(include_system_message=True)
    processed, _, configs = build_session_data(adapters)
    assert configs["gait_monitor"]["system_prompt"] == MONITOR_PROMPT
    assert not any(pm.message.content == MONITOR_PROMPT for pm in processed)


# -- Paper prompts ----------------------------------------------------------------

def test_paper_prompts_are_in_place():
    assert "Error signature:" in POIROT_SYSTEM_PROMPT
    assert "PART A" in POIROT_PHASE1_MESSAGE and "PART B" in POIROT_PHASE1_MESSAGE
    assert '"suspected_agents"' in POIROT_PHASE1_MESSAGE
    space = {"error_regions": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]}
    p2 = create_phase2_protocol_message(space)
    assert "VOTE THE ROOT CAUSE ONLY" in p2 and "WHAT YOU ARE LOOKING FOR" in p2

    report = json.dumps({"self_evaluation": {"anomalies_detected": True, "description": "lied"},
                         "peer_observations": [], "suspected_agents": ["gait_monitor"]})
    summary = format_phase1_observations({"gait_monitor": report})
    assert "ANOMALY DETECTED" in summary and "gait_monitor" in summary
