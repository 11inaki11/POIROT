"""Phase I must give every voting agent a region whose id is its agent_id."""
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from poirot.poirot_agent import (
    HazardSpaceMismatchError,
    POIROTAgent,
    find_agent_region_problems,
)
from poirot.poirot_pipeline import POIROTPipeline
from poirot.voting_system import find_agent_position, weighted_voting_analysis

AGENTS = [{"id": "gait_monitor", "name": "GaitMonitor"},
          {"id": "clinical_agent", "name": "ClinicalAgent"}]


def space(*ids):
    return {"system_name": "S", "error_regions": [
        {"id": i, "name": i.title(), "type": "agent", "description": "d"} for i in ids
    ] + [{"id": "exoskeleton", "name": "Exoskeleton", "type": "hardware", "description": "d"}],
        "error_vector_example": [0] * (len(ids) + 1)}


class StubLLM:
    def __init__(self, *answers):
        self.answers = [json.dumps(a) for a in answers]
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.answers[len(self.calls) - 1])


def constructor(*answers):
    agent = POIROTAgent(model="stub")
    agent.llm = StubLLM(*answers)
    return agent


def test_prompt_lists_the_required_agent_ids():
    agent = constructor(space("gait_monitor", "clinical_agent"))
    agent.analyze_system("desc", agents=AGENTS, verbose=False)
    instruction = agent.llm.calls[0][-1]
    assert isinstance(instruction, HumanMessage)
    assert '"gait_monitor"' in instruction.content and '"clinical_agent"' in instruction.content


def test_valid_space_needs_a_single_call():
    agent = constructor(space("gait_monitor", "clinical_agent"))
    result = agent.analyze_system("desc", agents=AGENTS, verbose=False)
    assert [r["id"] for r in result["error_regions"]][:2] == ["gait_monitor", "clinical_agent"]
    assert len(agent.llm.calls) == 1


def test_renamed_agent_gets_one_correction():
    agent = constructor(space("gait_monitoring_agent", "clinical_agent"),
                        space("gait_monitor", "clinical_agent"))
    result = agent.analyze_system("desc", agents=AGENTS, verbose=False)
    assert "gait_monitor" in [r["id"] for r in result["error_regions"]]
    assert len(agent.llm.calls) == 2
    assert "gait_monitor" in agent.llm.calls[1][-1].content


def test_still_wrong_after_correction_raises():
    bad = space("gait_monitoring_agent", "clinical_decision_maker")
    with pytest.raises(HazardSpaceMismatchError, match="gait_monitor"):
        constructor(bad, bad).analyze_system("desc", agents=AGENTS, verbose=False)


def test_without_agents_the_space_is_not_validated():
    agent = constructor(space("anything"))
    assert "error_regions" in agent.analyze_system("desc", verbose=False)


def test_duplicate_agent_region_is_a_problem():
    s = space("gait_monitor", "gait_monitor", "clinical_agent")
    assert any("more than one" in p for p in find_agent_region_problems(s, ["gait_monitor", "clinical_agent"]))


def test_voting_uses_exact_ids_only():
    regions = [{"id": "physio_data_stream", "name": "Physio Data Stream"},
               {"id": "physio", "name": "Physio"}]
    assert find_agent_position("physio", regions) == [0, 1]
    with pytest.raises(ValueError, match="no region"):
        find_agent_position("clinical_agent", regions)
    with pytest.raises(ValueError):
        weighted_voting_analysis(
            [{"agent_name": "clinical_agent", "location": [1, 0], "hazard_vector": "x"}],
            {"error_regions": regions})


def test_database_mode_requires_the_session_agents(tmp_path):
    db = Path(__file__).resolve().parent.parent / "examples" / "database" / "medical_diagnosis.db"
    pipeline = POIROTPipeline(system_name="MedicalDiagnosis", system_description="desc",
                              database_path=str(db), output_dir=str(tmp_path), verbose=False)
    ids = [a["id"] for a in pipeline._get_voting_agents()]
    assert sorted(ids) == ["diagnosis_agent", "treatment_agent"]


def test_json_surrounded_by_text_is_parsed():
    agent = POIROTAgent(model="stub")
    agent.llm = StubLLM()
    agent.llm.answers = ["Here is the space:\n" + json.dumps(space("gait_monitor", "clinical_agent")) + "\nDone."]
    result = agent.analyze_system("desc", agents=AGENTS, verbose=False)
    assert "error" not in result
