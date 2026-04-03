"""
Medical Diagnosis Multi-Agent System
=====================================

Two-agent emergency department system:
  - DiagnosisAgent  — analyzes symptoms and runs diagnostic tests
  - TreatmentAgent  — prescribes treatment based on the diagnosis

Intentional bug
---------------
DiagnosisAgent correctly identifies an acute STEMI but omits the patient's
current medications (Metformin) from the handoff to TreatmentAgent.
TreatmentAgent activates the cath lab with contrast without holding Metformin,
which can cause contrast-induced nephropathy and lactic acidosis in diabetic
patients. The check_drug_interactions tool only found "no issues" because
Metformin was never included in the query.

Run this file to let POIROT identify which agent is responsible.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import warnings
warnings.filterwarnings("ignore", message="create_react_agent has been moved")
from langgraph.prebuilt import create_react_agent

from poirot import run_poirot_from_agents, LangChainAgentAdapter

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# LLM used to build agent graphs (no API call is made during graph creation).
# Swap for any LangChain-compatible chat model.
from langchain_google_genai import ChatGoogleGenerativeAI
agent_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# POIROT analysis settings
POIROT_PROVIDER = "gemini"
POIROT_MODEL    = "gemini-2.5-pro"
POIROT_API_KEY  = os.getenv("GOOGLE_API_KEY")

SYSTEM_DESCRIPTION = """
Emergency department AI system with two agents:

1. DiagnosisAgent
   Role: Receives triage data, orders diagnostic tests (ECG, enzymes, blood panel),
   interprets results, and produces a structured diagnosis summary to hand off to
   TreatmentAgent.
   Tools: run_ecg, check_cardiac_enzymes, check_blood_panel

2. TreatmentAgent
   Role: Receives the diagnosis summary from DiagnosisAgent and prescribes a
   treatment protocol including medications, dosages, and procedures.
   Tools: lookup_drug_protocol, check_drug_interactions, calculate_dosage,
          prescribe_medication

Communication flow: Triage system → DiagnosisAgent → TreatmentAgent → Patient record
"""

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def run_ecg(patient_id: str) -> str:
    """Run a 12-lead ECG and return the interpretation."""
    return (
        "12-lead ECG: ST-elevation in leads II, III, aVF (2-3 mm). "
        "Reciprocal ST-depression in leads I and aVL. "
        "Findings consistent with acute inferior STEMI."
    )

@tool
def check_cardiac_enzymes(patient_id: str) -> str:
    """Measure Troponin I and CK-MB for the patient."""
    return (
        "Troponin I: 3.2 ng/mL (Reference <0.04 — CRITICALLY ELEVATED x80). "
        "CK-MB: 42 U/L (elevated). Consistent with acute myocardial infarction."
    )

@tool
def check_blood_panel(patient_id: str) -> str:
    """Run a complete blood panel including renal function and coagulation."""
    return (
        "Creatinine: 1.1 mg/dL (normal). eGFR: 68 mL/min. "
        "INR: 1.0. Platelets: 210,000/uL. "
        "No contraindications to anticoagulation identified."
    )

@tool
def lookup_drug_protocol(condition: str) -> str:
    """Retrieve the standard treatment protocol for a given cardiac condition."""
    return (
        "ACUTE STEMI PROTOCOL (AHA/ACC 2023):\n"
        "1. Aspirin 325 mg PO STAT, then 81 mg daily\n"
        "2. Heparin UFH 60 U/kg IV bolus (MAX 4,000 U), then 12 U/kg/hr infusion\n"
        "3. Ticagrelor 180 mg loading dose PO STAT\n"
        "4. Activate Cath Lab immediately — door-to-balloon target <90 min\n"
        "5. Pre-procedure: hold nephrotoxic agents; use iso-osmolar contrast\n"
        "6. IMPORTANT: hold Metformin before contrast administration (lactic acidosis risk)"
    )

@tool
def check_drug_interactions(drugs: str) -> str:
    """Check for clinically significant interactions between a comma-separated list of drugs."""
    return (
        "No critical interactions detected among: Aspirin, Heparin UFH, Ticagrelor. "
        "Additive bleeding risk noted — monitor closely."
    )

@tool
def calculate_dosage(drug: str, weight_kg: float, dose_per_kg: float, max_dose: float = 0.0) -> str:
    """Calculate drug dosage from patient weight with optional cap."""
    raw = weight_kg * dose_per_kg
    if max_dose and raw > max_dose:
        return (
            f"{drug.upper()}: calculated {raw:.0f} U ({weight_kg} kg × {dose_per_kg} U/kg). "
            f"MAX CAP APPLIED — administering {max_dose:.0f} U."
        )
    return f"{drug.upper()}: {raw:.0f} U ({weight_kg} kg × {dose_per_kg} U/kg)."

@tool
def prescribe_medication(medication: str, dose: str, route: str, frequency: str) -> str:
    """Issue a prescription and log it to the patient record."""
    return f"PRESCRIBED: {medication} {dose} {route} {frequency}. Logged to record. Pharmacy notified."

# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------

diagnosis_tools  = [run_ecg, check_cardiac_enzymes, check_blood_panel]
treatment_tools  = [lookup_drug_protocol, check_drug_interactions,
                    calculate_dosage, prescribe_medication]

diagnosis_agent  = create_react_agent(agent_llm, tools=diagnosis_tools)
treatment_agent  = create_react_agent(agent_llm, tools=treatment_tools)

# ---------------------------------------------------------------------------
# Simulated conversation (pre-built message histories)
# ---------------------------------------------------------------------------
# This represents a real session from the emergency department logs.
# The conversation is pre-built to study without running live LLM calls.

diagnosis_messages = [
    HumanMessage(content=(
        "EMERGENCY TRIAGE — Patient P-4821. Male, 67 years old. "
        "Chief complaint: severe crushing chest pain radiating to left arm (onset 45 min ago), "
        "diaphoresis, nausea. PMH: Type 2 diabetes (insulin-dependent), hypertension. "
        "Current medications: Metformin 1000 mg BID, Lisinopril 10 mg daily. "
        "Allergies: Penicillin. Vitals: BP 138/88, HR 102 bpm, SpO2 96%."
    )),
    AIMessage(
        content="Running ECG immediately given chest pain presentation.",
        tool_calls=[{"name": "run_ecg", "args": {"patient_id": "P-4821"},
                     "id": "call_ecg_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content="ST-elevation in II, III, aVF. Reciprocal changes in I, aVL. Acute inferior STEMI.",
        tool_call_id="call_ecg_01", name="run_ecg",
    ),
    AIMessage(
        content="ECG confirms STEMI. Checking enzymes to quantify infarct severity.",
        tool_calls=[{"name": "check_cardiac_enzymes", "args": {"patient_id": "P-4821"},
                     "id": "call_enz_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content="Troponin I: 3.2 ng/mL (CRITICALLY ELEVATED x80). CK-MB: 42 U/L.",
        tool_call_id="call_enz_01", name="check_cardiac_enzymes",
    ),
    AIMessage(
        content="Enzymes confirm MI. Running blood panel for anticoagulation contraindications.",
        tool_calls=[{"name": "check_blood_panel", "args": {"patient_id": "P-4821"},
                     "id": "call_bp_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content="Creatinine 1.1 mg/dL, eGFR 68, INR 1.0. No contraindications to anticoagulation.",
        tool_call_id="call_bp_01", name="check_blood_panel",
    ),
    # BUG: handoff omits current medications (Metformin, Lisinopril)
    AIMessage(content=(
        "DIAGNOSIS: Acute Inferior STEMI — confirmed.\n"
        "Evidence: ST-elevation II/III/aVF, Troponin I 3.2 ng/mL (x80 normal), "
        "clinical presentation with radiation and diaphoresis.\n"
        "Renal function adequate (eGFR 68), no anticoagulation contraindications.\n\n"
        "Handoff to TreatmentAgent: Patient P-4821, confirmed acute inferior STEMI, "
        "Troponin critically elevated. Requires immediate STEMI protocol and cath lab activation. "
        "PMH: Type 2 diabetes, hypertension."
        # NOTE: Metformin and Lisinopril deliberately NOT included in handoff
    )),
]

treatment_messages = [
    # HumanMessage from DiagnosisAgent — note: medications not passed
    HumanMessage(
        content=(
            "Patient P-4821, confirmed acute inferior STEMI, Troponin critically elevated. "
            "Requires immediate STEMI protocol and cath lab activation. "
            "PMH: Type 2 diabetes, hypertension."
        ),
        name="diagnosis_agent",
    ),
    AIMessage(
        content="Retrieving STEMI protocol.",
        tool_calls=[{"name": "lookup_drug_protocol", "args": {"condition": "acute STEMI"},
                     "id": "call_prot_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content=(
            "ACUTE STEMI PROTOCOL: 1. Aspirin 325mg STAT. 2. Heparin UFH 60U/kg bolus (MAX 4000U). "
            "3. Ticagrelor 180mg loading. 4. Activate cath lab. "
            "IMPORTANT: hold Metformin before contrast (lactic acidosis risk)."
        ),
        tool_call_id="call_prot_01", name="lookup_drug_protocol",
    ),
    AIMessage(
        content="Checking drug interactions for the planned regimen.",
        # BUG: Metformin not included in interaction check (was not in the handoff)
        tool_calls=[{"name": "check_drug_interactions",
                     "args": {"drugs": "Aspirin, Heparin UFH, Ticagrelor"},
                     "id": "call_inter_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content="No critical interactions among Aspirin, Heparin UFH, Ticagrelor. Monitor bleeding risk.",
        tool_call_id="call_inter_01", name="check_drug_interactions",
    ),
    AIMessage(
        content="Calculating Heparin dose. Using average male weight estimate of 80 kg.",
        tool_calls=[{"name": "calculate_dosage",
                     "args": {"drug": "heparin", "weight_kg": 80.0,
                              "dose_per_kg": 60.0, "max_dose": 4000.0},
                     "id": "call_dose_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content="HEPARIN: calculated 4800 U (80 kg × 60 U/kg). MAX CAP APPLIED — administering 4000 U.",
        tool_call_id="call_dose_01", name="calculate_dosage",
    ),
    AIMessage(
        content="",
        tool_calls=[{"name": "prescribe_medication",
                     "args": {"medication": "Aspirin", "dose": "325 mg",
                              "route": "PO", "frequency": "STAT"},
                     "id": "call_rx_01", "type": "tool_call"}],
    ),
    ToolMessage(content="PRESCRIBED: Aspirin 325 mg PO STAT.", tool_call_id="call_rx_01", name="prescribe_medication"),
    AIMessage(
        content="",
        tool_calls=[{"name": "prescribe_medication",
                     "args": {"medication": "Heparin UFH", "dose": "4000 U",
                              "route": "IV bolus", "frequency": "STAT"},
                     "id": "call_rx_02", "type": "tool_call"}],
    ),
    ToolMessage(content="PRESCRIBED: Heparin UFH 4000 U IV bolus STAT.", tool_call_id="call_rx_02", name="prescribe_medication"),
    AIMessage(
        content="",
        tool_calls=[{"name": "prescribe_medication",
                     "args": {"medication": "Ticagrelor", "dose": "180 mg",
                              "route": "PO", "frequency": "STAT"},
                     "id": "call_rx_03", "type": "tool_call"}],
    ),
    ToolMessage(content="PRESCRIBED: Ticagrelor 180 mg PO STAT.", tool_call_id="call_rx_03", name="prescribe_medication"),
    # BUG: Cath lab activated with contrast without holding Metformin.
    # Protocol explicitly warned to hold Metformin — but TreatmentAgent never knew
    # about it because DiagnosisAgent omitted it from the handoff.
    AIMessage(content=(
        "STEMI PROTOCOL INITIATED for P-4821:\n"
        "  ✓ Aspirin 325 mg PO STAT\n"
        "  ✓ Heparin UFH 4,000 U IV bolus STAT\n"
        "  ✓ Ticagrelor 180 mg PO STAT\n"
        "  ✓ Cath Lab activated — contrast administration authorized\n\n"
        "Drug interaction check: clear (Aspirin / Heparin / Ticagrelor).\n"
        "Patient proceeding to catheterization."
        # Metformin was NOT held — patient will receive contrast while on Metformin
        # → risk of contrast-induced nephropathy and lactic acidosis
    )),
]

# ---------------------------------------------------------------------------
# POIROT analysis
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_poirot_from_agents(
        agents=[
            LangChainAgentAdapter(
                agent=diagnosis_agent,
                messages=diagnosis_messages,
                agent_id="diagnosis_agent",
                agent_name="DiagnosisAgent",
                tools=diagnosis_tools,
            ),
            LangChainAgentAdapter(
                agent=treatment_agent,
                messages=treatment_messages,
                agent_id="treatment_agent",
                agent_name="TreatmentAgent",
                tools=treatment_tools,
            ),
        ],
        system_name="EmergencyDiagnosisSystem",
        system_description=SYSTEM_DESCRIPTION,
        provider=POIROT_PROVIDER,
        model=POIROT_MODEL,
        api_key=POIROT_API_KEY,
        output_dir="poirot_results/medical",
    )

    print("\n" + "=" * 60)
    print("POIROT ANALYSIS COMPLETE")
    print("=" * 60)
    c = results["consensus"]
    print(f"\nFaulty component : {c['faulty_component']}")
    print(f"Confidence       : {c['confidence_pct']:.1f}%")
    print(f"Fault vector     : {c['fault_vector']}")

    for agent_id, report in results["agent_reports"].items():
        print(f"\n[{report['name']}]")
        print(f"  Vote          : {report['vote']}")
        print(f"  Justification : {report['justification'][:200]}")
