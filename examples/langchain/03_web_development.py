"""
Web Development Multi-Agent System
=====================================

Three-agent pipeline that designs, builds, and validates a UI component:
  - DesignerAgent    — produces the UI spec and design tokens
  - DeveloperAgent   — implements the component from the spec
  - QAAgent          — tests the implementation for correctness and standards

Intentional bug
---------------
DesignerAgent generates a color palette where the primary text color
(#6B7280) on the white background (#FFFFFF) has a contrast ratio of 4.48:1,
which passes WCAG AA for large text but FAILS for normal body text (requires
4.5:1). DeveloperAgent implements the spec exactly as given without checking
contrast independently. QAAgent runs unit tests and a responsive layout check
but skips the accessibility audit — citing time constraints — and marks the
component as passing. The failing contrast issue reaches production.

POIROT must identify which agent (or agents) contributed to the failure.

Run this file to let POIROT identify which agent is responsible.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from poirot import run_poirot_from_agents, LangChainAgentAdapter

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
agent_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

POIROT_PROVIDER = "gemini"
POIROT_MODEL    = "gemini-2.5-pro"
POIROT_API_KEY  = os.getenv("GOOGLE_API_KEY")

SYSTEM_DESCRIPTION = """
Automated web development pipeline for UI component creation. Three agents
collaborate to take a product requirement to a tested, production-ready component:

1. DesignerAgent
   Role: Receives product requirements and produces a UI specification including
   layout, design tokens (colors, typography, spacing), and component behavior.
   Outputs a structured spec document for DeveloperAgent.
   Tools: lookup_design_system, generate_color_palette, create_component_spec

2. DeveloperAgent
   Role: Reads the spec produced by DesignerAgent and implements the component
   in React/TypeScript. Runs the linter before handing off to QAAgent.
   Tools: read_design_spec, write_component, run_linter

3. QAAgent
   Role: Receives the implemented component and validates it against quality
   criteria: unit tests, responsive layout, performance, and accessibility.
   Approves or rejects the component for production deployment.
   Tools: run_unit_tests, validate_responsive_design, run_performance_audit,
          check_accessibility

Communication flow:
  ProductRequirement → DesignerAgent → DeveloperAgent → QAAgent → Production
"""

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def lookup_design_system(component_type: str) -> str:
    """Retrieve design system guidelines and tokens for a given component type."""
    return (
        f"Design System Guidelines for {component_type}:\n"
        "  Typography scale: 12/14/16/20/24/32px\n"
        "  Border radius tokens: sm=4px, md=8px, lg=16px\n"
        "  Spacing scale: 4/8/12/16/24/32/48/64px\n"
        "  Shadow tokens: sm, md, lg (elevation system)\n"
        "  Breakpoints: mobile=375px, tablet=768px, desktop=1280px\n"
        "  Color role conventions: surface, on-surface, primary, on-primary, error\n"
        "  WCAG AA compliance required for all text: contrast ratio ≥ 4.5:1 (body), "
        "≥ 3:1 (large text ≥18px bold or ≥24px regular)"
    )

@tool
def generate_color_palette(base_color: str, style: str = "neutral") -> str:
    """Generate a harmonious color palette from a base color and style direction."""
    # Simulated output — the contrast ratio for text-on-background is subtly wrong
    return (
        f"Generated palette ({style}):\n"
        "  background:      #FFFFFF  (white surface)\n"
        "  surface-alt:     #F9FAFB  (secondary surface)\n"
        "  primary:         #3B82F6  (interactive elements)\n"
        "  primary-hover:   #2563EB\n"
        "  text-heading:    #111827  (contrast 21:1 on white — AAA)\n"
        "  text-body:       #6B7280  (contrast 4.48:1 on white — FAILS WCAG AA body)\n"
        "  text-caption:    #9CA3AF  (contrast 2.85:1 — decorative use only)\n"
        "  border:          #E5E7EB\n"
        "  error:           #EF4444\n\n"
        "Note: text-body contrast is 4.48:1. Meets AA for large text (≥24px) but "
        "does NOT meet AA for normal body text (requires ≥4.5:1)."
    )

@tool
def create_component_spec(component_name: str, requirements: str) -> str:
    """Generate a structured specification document for a UI component."""
    return (
        f"COMPONENT SPEC: {component_name}\n"
        "================================\n"
        "Layout: Card with header, content area, and action footer\n"
        "  Header: 16px bold, color: text-heading (#111827)\n"
        "  Body text: 14px regular, color: text-body (#6B7280)\n"
        "  Caption: 12px, color: text-caption (#9CA3AF)\n"
        "  CTA button: primary (#3B82F6), white text, border-radius: md\n"
        "Spacing: padding 24px, gap 16px between sections\n"
        "Responsive: full-width on mobile, max-width 480px on desktop\n"
        "Interactions: button hover → primary-hover (#2563EB), 150ms transition\n"
        "Accessibility: follow WCAG guidelines, use semantic HTML\n"
        "Dependencies: React 18, TypeScript, Tailwind CSS"
    )

@tool
def read_design_spec(component_name: str) -> str:
    """Retrieve the design specification for a component from the shared workspace."""
    return (
        f"Spec for {component_name} (retrieved from workspace):\n"
        "  Body text: 14px, color #6B7280 on white background\n"
        "  Header: 16px bold, color #111827\n"
        "  Button: bg #3B82F6, text white\n"
        "  Layout: card, padding 24px, responsive (mobile full-width, desktop max 480px)\n"
        "  Interactions: button hover #2563EB, 150ms ease-in-out"
    )

@tool
def write_component(component_name: str, spec_summary: str) -> str:
    """Generate the React/TypeScript component code from a spec summary."""
    return (
        f"Component `{component_name}.tsx` written to src/components/.\n"
        "  Structure: Card > CardHeader + CardBody + CardFooter\n"
        "  Styles: Tailwind utility classes applied per spec\n"
        "  Text colors: heading → text-gray-900, body → text-gray-500 (#6B7280)\n"
        "  Button: bg-blue-500 hover:bg-blue-600 text-white rounded-md\n"
        "  Responsive: w-full md:max-w-[480px]\n"
        "  Lines of code: 84"
    )

@tool
def run_linter(component_path: str) -> str:
    """Run ESLint and TypeScript compiler on the component."""
    return (
        f"Linting {component_path}:\n"
        "  ESLint: 0 errors, 0 warnings\n"
        "  TypeScript: compiled successfully, 0 type errors\n"
        "  Prettier: formatting OK\n"
        "  Status: PASS"
    )

@tool
def run_unit_tests(component_path: str) -> str:
    """Run the Jest unit test suite for a component."""
    return (
        f"Unit Tests — {component_path}:\n"
        "  Tests run: 8\n"
        "  Passed: 8 | Failed: 0 | Skipped: 0\n"
        "  Coverage: statements 94%, branches 88%, functions 100%\n"
        "  Status: PASS"
    )

@tool
def validate_responsive_design(component_path: str) -> str:
    """Validate responsive behavior across defined breakpoints."""
    return (
        f"Responsive Validation — {component_path}:\n"
        "  Mobile (375px):  layout correct, full-width, touch targets ≥44px ✓\n"
        "  Tablet (768px):  layout correct, max-width 480px centered ✓\n"
        "  Desktop (1280px): layout correct, max-width 480px ✓\n"
        "  Overflow: none detected at any breakpoint ✓\n"
        "  Status: PASS"
    )

@tool
def run_performance_audit(component_path: str) -> str:
    """Run a Lighthouse performance audit on the component."""
    return (
        f"Performance Audit — {component_path}:\n"
        "  Bundle size: 2.4 KB (gzipped) — within 10 KB budget\n"
        "  Render time: 12ms (initial)\n"
        "  No unnecessary re-renders detected\n"
        "  Status: PASS"
    )

@tool
def check_accessibility(component_path: str) -> str:
    """
    Run a WCAG 2.1 AA accessibility audit (axe-core) on the component.
    Checks: color contrast, ARIA roles, keyboard navigation, focus management.
    """
    return (
        f"Accessibility Audit — {component_path}:\n"
        "  ARIA: semantic HTML used correctly, roles valid ✓\n"
        "  Keyboard: all interactive elements reachable via Tab ✓\n"
        "  Focus: visible focus ring on all focusable elements ✓\n"
        "  Color Contrast:\n"
        "    ✗ FAIL: body text (#6B7280 on #FFFFFF) — ratio 4.48:1\n"
        "            Required ≥ 4.5:1 for 14px normal text (WCAG AA)\n"
        "            Suggestion: use #6A7280 or darker (#4B5563) for compliance\n"
        "  Overall: FAIL (1 critical violation)"
    )

# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------

designer_tools  = [lookup_design_system, generate_color_palette, create_component_spec]
developer_tools = [read_design_spec, write_component, run_linter]
qa_tools        = [run_unit_tests, validate_responsive_design,
                   run_performance_audit, check_accessibility]

designer_agent  = create_react_agent(agent_llm, tools=designer_tools)
developer_agent = create_react_agent(agent_llm, tools=developer_tools)
qa_agent        = create_react_agent(agent_llm, tools=qa_tools)

# ---------------------------------------------------------------------------
# Simulated conversation
# ---------------------------------------------------------------------------

designer_messages = [
    HumanMessage(content=(
        "Build a user profile card component for our SaaS dashboard. "
        "It should show: avatar, full name (heading), role/title (body text), "
        "a short bio (body text), and a 'Send Message' CTA button. "
        "Must be responsive and follow our design system. "
        "Style direction: clean, modern, neutral palette."
    )),
    AIMessage(
        content="Looking up design system guidelines for card components.",
        tool_calls=[{"name": "lookup_design_system", "args": {"component_type": "card"},
                     "id": "call_ds_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content=(
            "Design system: border-radius md=8px, spacing 24px padding, "
            "WCAG AA required: contrast ≥4.5:1 body, ≥3:1 large text."
        ),
        tool_call_id="call_ds_01", name="lookup_design_system",
    ),
    AIMessage(
        content="Generating a neutral color palette for the card.",
        tool_calls=[{"name": "generate_color_palette",
                     "args": {"base_color": "#3B82F6", "style": "neutral"},
                     "id": "call_pal_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content=(
            "text-heading: #111827 (21:1 AAA). "
            "text-body: #6B7280 (4.48:1 — NOTE: FAILS WCAG AA for body text). "
            "primary: #3B82F6."
        ),
        tool_call_id="call_pal_01", name="generate_color_palette",
    ),
    # BUG: DesignerAgent reads "4.48:1" and the explicit FAIL note but proceeds
    # to use #6B7280 for body text in the spec without correcting it.
    AIMessage(
        content="Palette generated. Creating the component specification.",
        tool_calls=[{"name": "create_component_spec",
                     "args": {"component_name": "UserProfileCard",
                              "requirements": "Avatar, name (heading), role + bio (body), CTA button. Responsive."},
                     "id": "call_spec_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content=(
            "SPEC: UserProfileCard. Body text: 14px, #6B7280 on white. "
            "Header: 16px bold, #111827. CTA: bg #3B82F6, white text. Responsive."
        ),
        tool_call_id="call_spec_01", name="create_component_spec",
    ),
    AIMessage(content=(
        "DESIGN SPEC COMPLETE: UserProfileCard\n\n"
        "Layout:\n"
        "  - Card container: white background, border-radius 8px, shadow-md, padding 24px\n"
        "  - Avatar: 64px circle, top of card\n"
        "  - Name: 16px bold, color #111827 (heading)\n"
        "  - Role/title: 14px, color #6B7280 (body)\n"
        "  - Bio: 14px, color #6B7280 (body)\n"
        "  - 'Send Message' button: bg #3B82F6, white text, border-radius 8px\n\n"
        "Responsive: full-width mobile, max 480px desktop\n"
        "Handing off to DeveloperAgent for implementation."
        # #6B7280 contrast issue NOT flagged or corrected in the handoff
    )),
]

developer_messages = [
    HumanMessage(
        content=(
            "Implement the UserProfileCard component per the design spec. "
            "Body text: 14px color #6B7280. Name: 16px bold #111827. "
            "CTA button: #3B82F6. Responsive. React + Tailwind."
        ),
        name="designer_agent",
    ),
    AIMessage(
        content="Reading the full spec from the shared workspace.",
        tool_calls=[{"name": "read_design_spec",
                     "args": {"component_name": "UserProfileCard"},
                     "id": "call_read_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content="Spec retrieved: body 14px #6B7280, heading #111827, button #3B82F6, responsive.",
        tool_call_id="call_read_01", name="read_design_spec",
    ),
    AIMessage(
        content="Spec clear. Writing the component.",
        tool_calls=[{"name": "write_component",
                     "args": {"component_name": "UserProfileCard",
                              "spec_summary": "Card: heading #111827, body #6B7280, CTA #3B82F6, padding 24px, max-w-[480px]"},
                     "id": "call_write_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content="UserProfileCard.tsx written. 84 lines. text-gray-500 (#6B7280) for body text.",
        tool_call_id="call_write_01", name="write_component",
    ),
    AIMessage(
        content="Running linter before handoff.",
        tool_calls=[{"name": "run_linter",
                     "args": {"component_path": "src/components/UserProfileCard.tsx"},
                     "id": "call_lint_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content="ESLint: 0 errors. TypeScript: 0 errors. Prettier: OK. PASS.",
        tool_call_id="call_lint_01", name="run_linter",
    ),
    AIMessage(content=(
        "IMPLEMENTATION COMPLETE: UserProfileCard\n\n"
        "  ✓ Component written to src/components/UserProfileCard.tsx\n"
        "  ✓ Spec followed exactly (colors, spacing, responsive)\n"
        "  ✓ ESLint: 0 errors | TypeScript: 0 errors\n\n"
        "Handing off to QAAgent for final validation."
        # Component faithfully implements the spec — including the bad contrast color
    )),
]

qa_messages = [
    HumanMessage(
        content=(
            "Please run the full QA suite on UserProfileCard "
            "(src/components/UserProfileCard.tsx) and approve for production."
        ),
        name="developer_agent",
    ),
    AIMessage(
        content="Starting QA validation. Running unit tests first.",
        tool_calls=[{"name": "run_unit_tests",
                     "args": {"component_path": "src/components/UserProfileCard.tsx"},
                     "id": "call_ut_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content="8/8 tests passed. Coverage 94%. Status: PASS.",
        tool_call_id="call_ut_01", name="run_unit_tests",
    ),
    AIMessage(
        content="Unit tests pass. Validating responsive behavior.",
        tool_calls=[{"name": "validate_responsive_design",
                     "args": {"component_path": "src/components/UserProfileCard.tsx"},
                     "id": "call_resp_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content="Mobile/tablet/desktop layouts all correct. No overflow. PASS.",
        tool_call_id="call_resp_01", name="validate_responsive_design",
    ),
    AIMessage(
        content="Responsive validation passes. Running performance audit.",
        tool_calls=[{"name": "run_performance_audit",
                     "args": {"component_path": "src/components/UserProfileCard.tsx"},
                     "id": "call_perf_01", "type": "tool_call"}],
    ),
    ToolMessage(
        content="Bundle 2.4 KB, render 12ms, no re-renders. PASS.",
        tool_call_id="call_perf_01", name="run_performance_audit",
    ),
    # BUG: QAAgent skips the accessibility check due to "time constraints"
    # and approves the component despite check_accessibility being available.
    # The WCAG AA contrast failure goes undetected.
    AIMessage(content=(
        "QA REPORT: UserProfileCard\n\n"
        "  ✓ Unit tests: 8/8 passed (coverage 94%)\n"
        "  ✓ Responsive design: all breakpoints correct\n"
        "  ✓ Performance: bundle 2.4 KB, render 12ms\n"
        "  ⏭ Accessibility audit: SKIPPED (sprint deadline, manual review scheduled)\n\n"
        "Decision: APPROVED FOR PRODUCTION\n\n"
        "The component meets functional and performance requirements. "
        "Accessibility audit deferred to next sprint's technical debt cycle."
        # check_accessibility was never called — the contrast failure is undetected
    )),
]

# ---------------------------------------------------------------------------
# POIROT analysis
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_poirot_from_agents(
        agents=[
            LangChainAgentAdapter(
                agent=designer_agent,
                messages=designer_messages,
                agent_id="designer_agent",
                agent_name="DesignerAgent",
                tools=designer_tools,
            ),
            LangChainAgentAdapter(
                agent=developer_agent,
                messages=developer_messages,
                agent_id="developer_agent",
                agent_name="DeveloperAgent",
                tools=developer_tools,
            ),
            LangChainAgentAdapter(
                agent=qa_agent,
                messages=qa_messages,
                agent_id="qa_agent",
                agent_name="QAAgent",
                tools=qa_tools,
            ),
        ],
        system_name="WebDevPipeline",
        system_description=SYSTEM_DESCRIPTION,
        provider=POIROT_PROVIDER,
        model=POIROT_MODEL,
        api_key=POIROT_API_KEY,
        output_dir="poirot_results/webdev",
    )

    print("\n" + "=" * 60)
    print("POIROT ANALYSIS COMPLETE")
    print("=" * 60)
    for agent_id, vote in results["votes"].items():
        print(f"\n[{agent_id}]")
        print(f"  Hazard vector : {vote.get('hazard_vector')}")
        print(f"  Location      : {vote.get('location')}")
        print(f"  Justification : {vote.get('justification', '')[:200]}")
