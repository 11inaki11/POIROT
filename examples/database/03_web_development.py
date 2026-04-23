"""
Web Development Pipeline Multi-Agent System — Database Integration
==================================================================

Same scenario as examples/langchain/03_web_development.py, but session data
is stored in a SQLite database and analyzed via run_poirot().

Three-agent web development pipeline:
  - designer_agent   — defines UI/UX specifications
  - developer_agent  — implements the component from the spec
  - qa_agent         — validates the implementation before production

Intentional bug
---------------
DesignerAgent specifies body text color #6B7280 on white background,
which fails WCAG AA contrast requirements (ratio 4.6:1, required 4.5:1 —
borderline but problematic for accessibility compliance). The color passes
through DiagnosisAgent and DeveloperAgent without being flagged.
QAAgent skips the accessibility check due to "sprint deadline" and approves
the component for production, leaving the contrast issue undetected.

Run this file to let POIROT identify which agent is responsible.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

import poirot
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
SCHEMA_PATH = REPO_ROOT / "templates" / "poirot_schema.sql"
DB_PATH     = SCRIPT_DIR / "web_development.db"

# ---------------------------------------------------------------------------
# POIROT analysis settings
# ---------------------------------------------------------------------------

POIROT_PROVIDER = "gemini"
POIROT_MODEL    = "gemini-2.5-pro"
POIROT_API_KEY  = os.getenv("GOOGLE_API_KEY")

SYSTEM_DESCRIPTION = """
Web development pipeline with three agents:

1. DesignerAgent
   Role: Creates UI/UX specifications including color palettes, typography,
   layout, and responsive breakpoints. Hands off specs to DeveloperAgent.
   Tools: search_design_system, check_brand_colors, create_component_spec

2. DeveloperAgent
   Role: Implements React components from design specs using Tailwind CSS.
   Runs linter before handoff to QAAgent.
   Tools: read_design_spec, write_component, run_linter

3. QAAgent
   Role: Validates implemented components with unit tests, responsive design
   checks, performance audits, and accessibility audits before approving
   for production.
   Tools: run_unit_tests, validate_responsive_design, run_performance_audit,
          check_accessibility

Communication flow: DesignerAgent -> DeveloperAgent -> QAAgent -> Production
"""

# ---------------------------------------------------------------------------
# Build database
# ---------------------------------------------------------------------------

def build_database():
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    cursor = conn.cursor()
    now = datetime.now().isoformat()
    session_id = "webdev_session_001"

    # Session
    cursor.execute(
        "INSERT INTO sessions (session_id, system_name, session_number, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, "WebDevPipeline", 1, now, now),
    )

    # Agents
    cursor.executemany(
        "INSERT INTO agents (agent_id, agent_name, system_name, agent_type, system_prompt, llm_model, has_tools, can_communicate_with) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "designer_agent",
                "DesignerAgent",
                "WebDevPipeline",
                "agent",
                "You are a UI/UX design AI. Create detailed component specifications following the design system and accessibility standards.",
                "gemini-2.5-flash",
                1,
                '["developer_agent"]',
            ),
            (
                "developer_agent",
                "DeveloperAgent",
                "WebDevPipeline",
                "agent",
                "You are a frontend developer AI. Implement React components from design specs using Tailwind CSS.",
                "gemini-2.5-flash",
                1,
                '["qa_agent"]',
            ),
            (
                "qa_agent",
                "QAAgent",
                "WebDevPipeline",
                "agent",
                "You are a QA engineer AI. Validate components with tests, responsive checks, performance audits, and accessibility audits.",
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
            ("designer_agent",   "search_design_system",      "Search the design system for existing components and tokens."),
            ("designer_agent",   "check_brand_colors",        "Validate colors against the brand palette."),
            ("designer_agent",   "create_component_spec",     "Create a formal component spec and store it in the shared workspace."),
            ("developer_agent",  "read_design_spec",          "Read a component spec from the shared workspace."),
            ("developer_agent",  "write_component",           "Write a React + Tailwind component to the codebase."),
            ("developer_agent",  "run_linter",                "Run ESLint and TypeScript checks on a component file."),
            ("qa_agent",         "run_unit_tests",            "Run the unit test suite for a component."),
            ("qa_agent",         "validate_responsive_design","Check responsive behavior across breakpoints."),
            ("qa_agent",         "run_performance_audit",     "Measure bundle size, render time, and re-render count."),
            ("qa_agent",         "check_accessibility",       "Run WCAG 2.1 AA accessibility audit on a component."),
        ],
    )

    # Messages
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

    # Designer receives task
    msg("msg_001", "system", "designer_agent", "human",
        "Design the UserProfileCard component for the dashboard redesign. "
        "Must be accessible, responsive, and consistent with the brand design system.")

    msg("msg_002", "designer_agent", "designer_agent", "ai",
        "Searching design system for existing card patterns.")

    msg("msg_003", "designer_agent", "designer_agent", "tool",
        "Design system: card base uses white background, border-radius 8px, shadow-md, padding 24px.",
        is_tool_call=1, tool_name="search_design_system",
        tool_input='{"query": "card component"}',
        tool_output='"Card base: bg-white, rounded-lg, shadow-md, p-6. Tokens: --radius-lg: 8px."')

    msg("msg_004", "designer_agent", "designer_agent", "ai",
        "Checking which brand colors to use for text and CTA.")

    msg("msg_005", "designer_agent", "designer_agent", "tool",
        "Brand palette: primary #3B82F6 (blue), heading #111827 (near-black), body #6B7280 (gray).",
        is_tool_call=1, tool_name="check_brand_colors",
        tool_input='{"context": "card body text and CTA"}',
        tool_output='"Approved colors: heading #111827, body text #6B7280, CTA bg #3B82F6 white text."')
    # NOTE: #6B7280 on white has contrast ratio ~4.6:1 which barely passes AA for normal text
    # but DesignerAgent does not flag this or verify contrast compliance

    msg("msg_006", "designer_agent", "designer_agent", "tool",
        "SPEC: UserProfileCard. Body text: 14px, #6B7280 on white. Header: 16px bold, #111827. CTA: bg #3B82F6, white text. Responsive.",
        is_tool_call=1, tool_name="create_component_spec",
        tool_input='{"component_name": "UserProfileCard", "colors": {"body": "#6B7280", "heading": "#111827", "cta": "#3B82F6"}}',
        tool_output='"SPEC saved: UserProfileCard. Body 14px #6B7280, heading 16px bold #111827, CTA #3B82F6. Responsive: full-width mobile, max 480px desktop."')

    msg("msg_007", "designer_agent", "developer_agent", "ai",
        "DESIGN SPEC COMPLETE: UserProfileCard\n\n"
        "Layout:\n"
        "  - Card container: white background, border-radius 8px, shadow-md, padding 24px\n"
        "  - Avatar: 64px circle, top of card\n"
        "  - Name: 16px bold, color #111827 (heading)\n"
        "  - Role/title: 14px, color #6B7280 (body)\n"
        "  - Bio: 14px, color #6B7280 (body)\n"
        "  - 'Send Message' button: bg #3B82F6, white text, border-radius 8px\n\n"
        "Responsive: full-width mobile, max 480px desktop\n"
        "Handing off to DeveloperAgent for implementation.")
    # #6B7280 contrast issue NOT flagged or corrected in the handoff

    # Developer receives spec
    msg("msg_008", "developer_agent", "developer_agent", "ai",
        "Reading the full spec from the shared workspace.")

    msg("msg_009", "developer_agent", "developer_agent", "tool",
        "Spec retrieved: body 14px #6B7280, heading #111827, button #3B82F6, responsive.",
        is_tool_call=1, tool_name="read_design_spec",
        tool_input='{"component_name": "UserProfileCard"}',
        tool_output='"Spec: body 14px text-gray-500 (#6B7280), heading 16px bold text-gray-900 (#111827), CTA bg-blue-500, responsive max-w-[480px]."')

    msg("msg_010", "developer_agent", "developer_agent", "ai",
        "Spec clear. Writing the component.")

    msg("msg_011", "developer_agent", "developer_agent", "tool",
        "UserProfileCard.tsx written. 84 lines. text-gray-500 (#6B7280) for body text.",
        is_tool_call=1, tool_name="write_component",
        tool_input='{"component_name": "UserProfileCard", "spec_summary": "Card: heading #111827, body #6B7280, CTA #3B82F6, padding 24px, max-w-[480px]"}',
        tool_output='"UserProfileCard.tsx written. 84 lines. Tailwind classes applied."')

    msg("msg_012", "developer_agent", "developer_agent", "ai",
        "Running linter before handoff.")

    msg("msg_013", "developer_agent", "developer_agent", "tool",
        "ESLint: 0 errors. TypeScript: 0 errors. Prettier: OK. PASS.",
        is_tool_call=1, tool_name="run_linter",
        tool_input='{"component_path": "src/components/UserProfileCard.tsx"}',
        tool_output='"ESLint: 0 errors. TypeScript: 0 errors. Prettier: OK. PASS."')

    msg("msg_014", "developer_agent", "qa_agent", "ai",
        "IMPLEMENTATION COMPLETE: UserProfileCard\n\n"
        "  OK Component written to src/components/UserProfileCard.tsx\n"
        "  OK Spec followed exactly (colors, spacing, responsive)\n"
        "  OK ESLint: 0 errors | TypeScript: 0 errors\n\n"
        "Handing off to QAAgent for final validation.")

    # QA receives component
    msg("msg_015", "qa_agent", "qa_agent", "ai",
        "Starting QA validation. Running unit tests first.")

    msg("msg_016", "qa_agent", "qa_agent", "tool",
        "8/8 tests passed. Coverage 94%. Status: PASS.",
        is_tool_call=1, tool_name="run_unit_tests",
        tool_input='{"component_path": "src/components/UserProfileCard.tsx"}',
        tool_output='"8/8 tests passed. Coverage: 94%. Status: PASS."')

    msg("msg_017", "qa_agent", "qa_agent", "ai",
        "Unit tests pass. Validating responsive behavior.")

    msg("msg_018", "qa_agent", "qa_agent", "tool",
        "Mobile/tablet/desktop layouts all correct. No overflow. PASS.",
        is_tool_call=1, tool_name="validate_responsive_design",
        tool_input='{"component_path": "src/components/UserProfileCard.tsx"}',
        tool_output='"Mobile (320px): OK. Tablet (768px): OK. Desktop (1440px): OK. No overflow. PASS."')

    msg("msg_019", "qa_agent", "qa_agent", "ai",
        "Responsive validation passes. Running performance audit.")

    msg("msg_020", "qa_agent", "qa_agent", "tool",
        "Bundle 2.4 KB, render 12ms, no re-renders. PASS.",
        is_tool_call=1, tool_name="run_performance_audit",
        tool_input='{"component_path": "src/components/UserProfileCard.tsx"}',
        tool_output='"Bundle: 2.4 KB. Render: 12ms. Re-renders: 0. PASS."')

    # BUG: QAAgent skips the accessibility check due to "time constraints"
    # check_accessibility is available but never called
    msg("msg_021", "qa_agent", "qa_agent", "ai",
        "QA REPORT: UserProfileCard\n\n"
        "  OK Unit tests: 8/8 passed (coverage 94%)\n"
        "  OK Responsive design: all breakpoints correct\n"
        "  OK Performance: bundle 2.4 KB, render 12ms\n"
        "  SKIP Accessibility audit: SKIPPED (sprint deadline, manual review scheduled)\n\n"
        "Decision: APPROVED FOR PRODUCTION\n\n"
        "The component meets functional and performance requirements. "
        "Accessibility audit deferred to next sprint's technical debt cycle.")
    # check_accessibility was never called - the contrast failure is undetected

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
        system_name="WebDevPipeline",
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
