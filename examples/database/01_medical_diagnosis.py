"""
Medical Diagnosis Multi-Agent System — Database Integration
============================================================

Same scenario as examples/langchain/01_medical_diagnosis.py, but session data
is stored in a SQLite database and analyzed via run_poirot().

Two-agent emergency department system:
  - diagnosis_agent  — analyzes symptoms and runs diagnostic tests
  - treatment_agent  — prescribes treatment based on the diagnosis

Intentional bug
---------------
DiagnosisAgent correctly identifies an acute STEMI but omits the patient's
current medications (Metformin) from the handoff to TreatmentAgent.
TreatmentAgent activates the cath lab with contrast without holding Metformin,
which can cause contrast-induced nephropathy and lactic acidosis in diabetic
patients.

Run this file to let POIROT identify which agent is responsible.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import poirot
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).parent
REPO_ROOT    = SCRIPT_DIR.parent.parent
SCHEMA_PATH  = REPO_ROOT / "templates" / "poirot_schema.sql"
DB_PATH      = SCRIPT_DIR / "medical_diagnosis.db"

# ---------------------------------------------------------------------------
# POIROT analysis settings
# ---------------------------------------------------------------------------

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

Communication flow: Triage system -> DiagnosisAgent -> TreatmentAgent -> Patient record
"""

# ---------------------------------------------------------------------------
# Build database
# ---------------------------------------------------------------------------

def build_database():
    # Initialize schema from template (as documented in the wiki)
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    cursor = conn.cursor()
    now = datetime.now().isoformat()
    session_id = "medical_session_001"

    # Session
    cursor.execute(
        "INSERT INTO sessions (session_id, system_name, session_number, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, "EmergencyDiagnosisSystem", 1, now, now),
    )

    # Agents
    cursor.executemany(
        "INSERT INTO agents (agent_id, agent_name, system_name, agent_type, system_prompt, llm_model, has_tools, can_communicate_with) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "diagnosis_agent",
                "DiagnosisAgent",
                "EmergencyDiagnosisSystem",
                "agent",
                "You are an emergency diagnosis AI. Analyze patient triage data, run diagnostic tests, and produce a structured diagnosis summary.",
                "gemini-2.5-flash",
                1,
                '["treatment_agent"]',
            ),
            (
                "treatment_agent",
                "TreatmentAgent",
                "EmergencyDiagnosisSystem",
                "agent",
                "You are a treatment AI. Receive diagnosis summaries and prescribe appropriate treatment protocols.",
                "gemini-2.5-flash",
                1,
                '[]',
            ),
        ],
    )

    # Agent tools
    cursor.executemany(
        "INSERT INTO agent_tools (agent_id, tool_name, tool_description) VALUES (?, ?, ?)",
        [
            ("diagnosis_agent", "run_ecg",               "Run a 12-lead ECG and return the interpretation."),
            ("diagnosis_agent", "check_cardiac_enzymes", "Measure Troponin I and CK-MB for the patient."),
            ("diagnosis_agent", "check_blood_panel",     "Run a complete blood panel including renal function and coagulation."),
            ("treatment_agent", "lookup_drug_protocol",  "Retrieve the standard treatment protocol for a given cardiac condition."),
            ("treatment_agent", "check_drug_interactions","Check for clinically significant interactions between a list of drugs."),
            ("treatment_agent", "calculate_dosage",      "Calculate drug dosage from patient weight with optional cap."),
            ("treatment_agent", "prescribe_medication",  "Issue a prescription and log it to the patient record."),
        ],
    )

    # Messages — strictly following wiki column names and message types
    seq = 0
    def msg(message_id, from_agent_id, to_agent_id, message_type, content,
            is_tool_call=0, tool_name=None, tool_input=None, tool_output=None):
        nonlocal seq
        seq += 1
        cursor.execute(
            """INSERT INTO messages
               (message_id, session_id, from_agent_id, to_agent_id,
                message_type, content, timestamp, sequence_number,
                is_tool_call, tool_name, tool_input, tool_output)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (message_id, session_id, from_agent_id, to_agent_id,
             message_type, content, now, seq,
             is_tool_call, tool_name, tool_input, tool_output),
        )

    # Triage input arrives as a human message to DiagnosisAgent
    msg("msg_001", "system", "diagnosis_agent", "human",
        "EMERGENCY TRIAGE - Patient P-4821. Male, 67 years old. "
        "Chief complaint: severe crushing chest pain radiating to left arm (onset 45 min ago), "
        "diaphoresis, nausea. PMH: Type 2 diabetes (insulin-dependent), hypertension. "
        "Current medications: Metformin 1000 mg BID, Lisinopril 10 mg daily. "
        "Allergies: Penicillin. Vitals: BP 138/88, HR 102 bpm, SpO2 96%.")

    msg("msg_002", "diagnosis_agent", "diagnosis_agent", "ai",
        "Running ECG immediately given chest pain presentation.")

    msg("msg_003", "diagnosis_agent", "diagnosis_agent", "tool",
        "ST-elevation in II, III, aVF. Reciprocal changes in I, aVL. Acute inferior STEMI.",
        is_tool_call=1, tool_name="run_ecg",
        tool_input='{"patient_id": "P-4821"}',
        tool_output='"ST-elevation in leads II, III, aVF (2-3 mm). Findings consistent with acute inferior STEMI."')

    msg("msg_004", "diagnosis_agent", "diagnosis_agent", "ai",
        "ECG confirms STEMI. Checking enzymes to quantify infarct severity.")

    msg("msg_005", "diagnosis_agent", "diagnosis_agent", "tool",
        "Troponin I: 3.2 ng/mL (CRITICALLY ELEVATED x80). CK-MB: 42 U/L.",
        is_tool_call=1, tool_name="check_cardiac_enzymes",
        tool_input='{"patient_id": "P-4821"}',
        tool_output='"Troponin I: 3.2 ng/mL (Reference <0.04 - CRITICALLY ELEVATED x80). CK-MB: 42 U/L."')

    msg("msg_006", "diagnosis_agent", "diagnosis_agent", "ai",
        "Enzymes confirm MI. Running blood panel for anticoagulation contraindications.")

    msg("msg_007", "diagnosis_agent", "diagnosis_agent", "tool",
        "Creatinine 1.1 mg/dL, eGFR 68, INR 1.0. No contraindications to anticoagulation.",
        is_tool_call=1, tool_name="check_blood_panel",
        tool_input='{"patient_id": "P-4821"}',
        tool_output='"Creatinine: 1.1 mg/dL (normal). eGFR: 68 mL/min. INR: 1.0. No contraindications to anticoagulation."')

    # BUG: handoff omits current medications (Metformin, Lisinopril)
    msg("msg_008", "diagnosis_agent", "treatment_agent", "ai",
        "DIAGNOSIS: Acute Inferior STEMI - confirmed.\n"
        "Evidence: ST-elevation II/III/aVF, Troponin I 3.2 ng/mL (x80 normal), "
        "clinical presentation with radiation and diaphoresis.\n"
        "Renal function adequate (eGFR 68), no anticoagulation contraindications.\n\n"
        "Handoff to TreatmentAgent: Patient P-4821, confirmed acute inferior STEMI, "
        "Troponin critically elevated. Requires immediate STEMI protocol and cath lab activation. "
        "PMH: Type 2 diabetes, hypertension.")
    # NOTE: Metformin and Lisinopril deliberately NOT included in handoff

    msg("msg_009", "treatment_agent", "treatment_agent", "ai",
        "Retrieving STEMI protocol.")

    msg("msg_010", "treatment_agent", "treatment_agent", "tool",
        "ACUTE STEMI PROTOCOL retrieved.",
        is_tool_call=1, tool_name="lookup_drug_protocol",
        tool_input='{"condition": "acute STEMI"}',
        tool_output='"ACUTE STEMI PROTOCOL (AHA/ACC 2023): 1. Aspirin 325mg STAT. '
                    '2. Heparin UFH 60U/kg bolus (MAX 4000U). 3. Ticagrelor 180mg loading. '
                    '4. Activate cath lab. IMPORTANT: hold Metformin before contrast (lactic acidosis risk)."')

    # BUG: Metformin not included in interaction check (was not in the handoff)
    msg("msg_011", "treatment_agent", "treatment_agent", "ai",
        "Checking drug interactions for the planned regimen.")

    msg("msg_012", "treatment_agent", "treatment_agent", "tool",
        "No critical interactions among Aspirin, Heparin UFH, Ticagrelor. Monitor bleeding risk.",
        is_tool_call=1, tool_name="check_drug_interactions",
        tool_input='{"drugs": "Aspirin, Heparin UFH, Ticagrelor"}',
        tool_output='"No critical interactions detected among: Aspirin, Heparin UFH, Ticagrelor."')

    msg("msg_013", "treatment_agent", "treatment_agent", "ai",
        "Calculating Heparin dose. Using average male weight estimate of 80 kg.")

    msg("msg_014", "treatment_agent", "treatment_agent", "tool",
        "HEPARIN: calculated 4800 U (80 kg x 60 U/kg). MAX CAP APPLIED - administering 4000 U.",
        is_tool_call=1, tool_name="calculate_dosage",
        tool_input='{"drug": "heparin", "weight_kg": 80.0, "dose_per_kg": 60.0, "max_dose": 4000.0}',
        tool_output='"HEPARIN: calculated 4800 U. MAX CAP APPLIED - administering 4000 U."')

    msg("msg_015", "treatment_agent", "treatment_agent", "tool",
        "PRESCRIBED: Aspirin 325 mg PO STAT.",
        is_tool_call=1, tool_name="prescribe_medication",
        tool_input='{"medication": "Aspirin", "dose": "325 mg", "route": "PO", "frequency": "STAT"}',
        tool_output='"PRESCRIBED: Aspirin 325 mg PO STAT. Logged to record."')

    msg("msg_016", "treatment_agent", "treatment_agent", "tool",
        "PRESCRIBED: Heparin UFH 4000 U IV bolus STAT.",
        is_tool_call=1, tool_name="prescribe_medication",
        tool_input='{"medication": "Heparin UFH", "dose": "4000 U", "route": "IV bolus", "frequency": "STAT"}',
        tool_output='"PRESCRIBED: Heparin UFH 4000 U IV bolus STAT. Logged to record."')

    msg("msg_017", "treatment_agent", "treatment_agent", "tool",
        "PRESCRIBED: Ticagrelor 180 mg PO STAT.",
        is_tool_call=1, tool_name="prescribe_medication",
        tool_input='{"medication": "Ticagrelor", "dose": "180 mg", "route": "PO", "frequency": "STAT"}',
        tool_output='"PRESCRIBED: Ticagrelor 180 mg PO STAT. Logged to record."')

    # BUG: Cath lab activated with contrast without holding Metformin.
    msg("msg_018", "treatment_agent", "treatment_agent", "ai",
        "STEMI PROTOCOL INITIATED for P-4821:\n"
        "  OK Aspirin 325 mg PO STAT\n"
        "  OK Heparin UFH 4,000 U IV bolus STAT\n"
        "  OK Ticagrelor 180 mg PO STAT\n"
        "  OK Cath Lab activated - contrast administration authorized\n\n"
        "Drug interaction check: clear (Aspirin / Heparin / Ticagrelor).\n"
        "Patient proceeding to catheterization.")
    # Metformin was NOT held - patient will receive contrast while on Metformin

    conn.commit()
    conn.close()
    print(f"Database created: {DB_PATH}")


# ---------------------------------------------------------------------------
# POIROT analysis
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_database()

    results = poirot.run_poirot(
        database_path=str(DB_PATH),
        system_name="EmergencyDiagnosisSystem",
        system_description=SYSTEM_DESCRIPTION,
        provider=POIROT_PROVIDER,
        api_key=POIROT_API_KEY,
        model=POIROT_MODEL,
    )

    print("\n" + "=" * 60)
    print("POIROT ANALYSIS COMPLETE")
    print("=" * 60)
    c = results["consensus"]
    if c["is_tie"]:
        print(f"\nResult           : TIE")
        print(f"Tied components  : {', '.join(c['tied_components'])}")
    else:
        print(f"\nResult           : CONSENSUS")
        print(f"Faulty component : {c['faulty_component']}")
    print(f"Confidence       : {c['confidence_pct']:.1f}%")
    print(f"Fault vector     : {c['fault_vector']}")

    for agent_id, report in results["agent_reports"].items():
        print(f"\n[{report['name']}]")
        print(f"  Vote          : {report['vote']}")
        print(f"  Justification : {report['justification'][:200]}")
