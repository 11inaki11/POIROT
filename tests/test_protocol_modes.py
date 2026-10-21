"""Protocol modes C0-C3: what each agent sees before its vote, and which phases run.

No LLM is called: the chat models used for the vote and the compiled agents used
in Phase 1 are replaced by stubs that capture their input.
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import poirot
import poirot.phase1_protocol as phase1_protocol
import poirot.phase2_protocol as phase2_protocol
from poirot.phase2_protocol import (
    PHASE2_OVERRIDE,
    build_direct_vote_context,
    create_protocol_message_without_consultation,
    execute_phase2_analysis,
    resolve_mode,
)

from test_adapter_and_protocol import REPORT, make_adapters

SPACE = {"system_name": "S", "error_regions": [
    {"id": "gait_monitor", "name": "GaitMonitor", "type": "agent", "description": "d"},
    {"id": "clinical_agent", "name": "ClinicalAgent", "type": "agent", "description": "d"},
    {"id": "exoskeleton", "name": "Exoskeleton", "type": "hardware", "description": "d"},
], "error_vector_example": [0, 0, 0]}
VOTE = {"hazard_vector": "GaitMonitor misreported the batch", "location": [1, 0, 0],
        "justification": "Raw 18.2 deg vs reported 54 deg."}


class StubChatModel:
    """Records every vote request; fails if tools are bound to it."""

    def __init__(self, log):
        self.log = log

    def bind_tools(self, tools):
        raise AssertionError("the vote in modes C0-C2 must not bind tools")

    def invoke(self, messages):
        self.log.append(messages)
        return AIMessage(content="```json\n" + json.dumps(VOTE) + "\n```")


@pytest.fixture
def vote_calls(monkeypatch):
    log = []
    monkeypatch.setattr(phase2_protocol, "create_agent_llm", lambda *a, **k: StubChatModel(log))
    return log


class StubConstructor:
    def __init__(self, **kwargs):
        pass

    def analyze_system(self, *args, **kwargs):
        return SPACE


def run_from_agents(monkeypatch, mode, **kwargs):
    monkeypatch.setattr(poirot, "POIROTAgent", StubConstructor)
    adapters, monitor, clinical = make_adapters()
    result = poirot.run_poirot_from_agents(
        agents=adapters, system_name="S", system_description="desc",
        provider="openai", model="stub", verbose=False, mode=mode, **kwargs)
    return result, monitor, clinical


# -- mode names -------------------------------------------------------------------

def test_default_mode_is_c1():
    assert resolve_mode(None) == "C1"
    assert poirot.DEFAULT_MODE == "C1"


@pytest.mark.parametrize("name,code", [("c0", "C0"), ("vote_only", "C0"), ("self_assessment", "C1"),
                                       ("shared_reports", "C2"), ("full", "C3"), ("C3", "C3")])
def test_mode_aliases(name, code):
    assert resolve_mode(name) == code


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown POIROT mode"):
        resolve_mode("C4")


# -- vote context per mode ------------------------------------------------------------

HISTORY = [HumanMessage(content="From gait_monitor to clinical_agent:\n" + REPORT)]


def test_c0_votes_from_the_session_only():
    msgs = build_direct_vote_context("C0", "clinical_agent", "ROLE", HISTORY, SPACE,
                                     phase1_reports={"clinical_agent": "my report"})
    assert msgs[0].content == "ROLE" + PHASE2_OVERRIDE
    assert msgs[1:-1] == HISTORY
    assert "review of the session above" in msgs[-1].content
    assert not any("my report" in m.content for m in msgs)


def test_c1_votes_right_after_its_own_report():
    ctx = [SystemMessage(content="ROLE"), HumanMessage(content="session"), HumanMessage(content="phase 1 prompt")]
    msgs = build_direct_vote_context("C1", "clinical_agent", "ROLE", HISTORY, SPACE,
                                     phase1_reports={"clinical_agent": "my report", "gait_monitor": "other"},
                                     phase1_contexts={"clinical_agent": ctx})
    assert msgs[:3] == ctx
    assert isinstance(msgs[3], AIMessage) and msgs[3].content == "my report"
    assert "individual analysis above" in msgs[-1].content
    assert not any("other" == m.content for m in msgs)


def test_c1_adds_the_role_prompt_when_the_phase1_context_has_none():
    ctx = [HumanMessage(content="session")]
    msgs = build_direct_vote_context("C1", "a", "ROLE", HISTORY, SPACE,
                                     phase1_reports={"a": "r"}, phase1_contexts={"a": ctx})
    assert isinstance(msgs[0], SystemMessage) and msgs[0].content == "ROLE"


def test_c1_without_phase1_report_falls_back_to_the_session():
    msgs = build_direct_vote_context("C1", "a", "ROLE", HISTORY, SPACE, phase1_reports={}, phase1_contexts={})
    assert msgs[0].content == "ROLE" + PHASE2_OVERRIDE and msgs[1:-1] == HISTORY


def test_c2_shares_every_report_without_consultation():
    msgs = build_direct_vote_context("C2", "a", "ROLE", HISTORY, SPACE,
                                     phase1_reports={"a": "report A", "b": "report B"},
                                     vectors_to_ignore=["network latency"])
    text = "\n".join(m.content for m in msgs)
    assert "report A" in text and "report B" in text
    assert "network latency" in text
    assert "talk_to" not in text and "Begin your consultation" not in text
    assert "Phase 1 observations of all agents" in msgs[-1].content


def test_protocol_without_consultation_keeps_the_voting_guidance():
    txt = create_protocol_message_without_consultation(SPACE)
    assert "VOTE THE ROOT CAUSE ONLY" in txt
    assert "provide your final hazard vector identification" in txt
    for instruction in ("talk_to", "communication tools", "DO consult", "DON'T skip consultation",
                        "Begin your consultation", "After consultation, your response"):
        assert instruction not in txt


# -- execution --------------------------------------------------------------------

def test_direct_vote_produces_the_usual_outputs(vote_calls, tmp_path):
    agents = {"gait_monitor": {"name": "GaitMonitor", "system_prompt": "A"},
              "clinical_agent": {"name": "ClinicalAgent", "system_prompt": "B"}}
    result = execute_phase2_analysis(
        session_id="s", db_path=None, error_space=SPACE, output_dir=tmp_path, verbose=False,
        historical_messages_override=[], agents_data_override=agents, mode="C0")
    assert len(vote_calls) == 2
    assert result["votes"]["clinical_agent"]["location"] == [1, 0, 0]
    assert result["voting_results"]["winning_location"]["name"] == "GaitMonitor"
    assert (tmp_path / "votes.json").exists() and (tmp_path / "summary.txt").exists()


def test_parallel_votes_keep_agent_order(vote_calls):
    agents = {f"a{i}": {"name": f"A{i}", "system_prompt": "p"} for i in range(4)}
    space = {"system_name": "S", "error_regions": [
        {"id": a, "name": a, "type": "agent", "description": "d"} for a in agents], "error_vector_example": [0] * 4}
    result = execute_phase2_analysis(
        session_id="s", db_path=None, error_space=space, verbose=False, historical_messages_override=[],
        agents_data_override=agents, mode="C0", max_parallel_votes=4)
    assert list(result["votes"]) == list(agents) and len(vote_calls) == 4


def test_c0_skips_phase1(monkeypatch, vote_calls):
    monkeypatch.setattr(phase1_protocol, "execute_phase1_analysis",
                        lambda **k: pytest.fail("Phase 1 must not run in mode C0"))
    result, monitor, clinical = run_from_agents(monkeypatch, "C0")
    assert result["mode"] == "C0"
    assert monitor.received is None and clinical.received is None
    assert result["consensus"]["faulty_component"] == "GaitMonitor"


def test_c1_end_to_end_votes_after_phase1(monkeypatch, vote_calls):
    result, monitor, clinical = run_from_agents(monkeypatch, "C1")
    assert result["mode"] == "C1"
    assert monitor.received is not None and clinical.received is not None
    assert len(vote_calls) == 2
    for ctx in vote_calls:
        assert isinstance(ctx[0], SystemMessage)
        assert isinstance(ctx[-2], AIMessage) and "self_evaluation" in ctx[-2].content
    assert result["agent_reports"]["clinical_agent"]["vote"] == [1, 0, 0]


def test_default_call_runs_c1(monkeypatch, vote_calls):
    monkeypatch.setattr(poirot, "POIROTAgent", StubConstructor)
    adapters, _, _ = make_adapters()
    result = poirot.run_poirot_from_agents(agents=adapters, system_name="S", system_description="d",
                                           provider="openai", model="stub", verbose=False)
    assert result["mode"] == "C1"


def test_c3_still_runs_the_consultation_graph(monkeypatch):
    log = []

    class ToolAwareStub(StubChatModel):
        def bind_tools(self, tools):
            log.append(("tools", sorted(t.name for t in tools)))
            return self

    monkeypatch.setattr(phase2_protocol, "create_agent_llm", lambda *a, **k: ToolAwareStub(log))
    monkeypatch.setenv("OPENAI_API_KEY", "test")  # the POIROT server model is built, never called
    agents = {"gait_monitor": {"name": "GaitMonitor", "system_prompt": "A", "can_communicate_with": ["clinical_agent"]},
              "clinical_agent": {"name": "ClinicalAgent", "system_prompt": "B", "can_communicate_with": ["gait_monitor"]}}
    result = execute_phase2_analysis(
        session_id="s", db_path=None, error_space=SPACE, verbose=False, llm_provider="openai",
        model_name="stub", historical_messages_override=[], agents_data_override=agents, mode="C3")
    assert any(entry[0] == "tools" for entry in log if isinstance(entry, tuple))
    assert result["votes"]["gait_monitor"]["location"] == [1, 0, 0]
