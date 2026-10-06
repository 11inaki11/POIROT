"""
POIROT Agent - Error Vector Space Builder
==========================================

This module contains the POIROTAgent class, which analyzes multi-agent system
descriptions and identifies potential error locations (error vector space).

The POIROTAgent is a preliminary step before running the full POIROT protocol.
It determines the N-dimensional error space where each dimension represents a
potential failure point in the system.

Author: CORTEX Team
Version: 1.0
Date: December 2025
"""

from typing import Dict, Any, List, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
import json
import os

# Import LLM Factory for local/remote LLM support
try:
    from llm_factory import LLMFactory
except ImportError:
    try:
        from .llm_factory import LLMFactory
    except ImportError:
        print("WARNING Could not import LLMFactory. Only Gemini API will be available.")
        LLMFactory = None

# Import token tracker
try:
    from token_tracker import TokenTracker, extract_tokens_from_response
except ImportError:
    try:
        from .token_tracker import TokenTracker, extract_tokens_from_response
    except ImportError:
        TokenTracker = None
        extract_tokens_from_response = None

POIROT_SYSTEM_PROMPT = """You are the POIROT agent, a core component of the POIROT protocol.
This protocol is used to debug and diagnose errors in multi-agent systems. In this process, each agent from the original system participates to identify the error vector that has triggered a problem whose origin is unknown. The main objective of POIROT is that the agents themselves collaboratively discover the cause of the issue.

The POIROT protocol consists of three stages:
1) Individual Analysis:
   - Each agent independently analyzes the problematic session, reflecting on what they perceived and performed.
   - Identify any possible mistakes, flaws, or issues that could have caused the incident — these may originate from your own actions, from other agents, or from flaws in the system itself.
   - If no issues are identified, explicitly state that. Be objective and analytical.

2) Peer Consultation:
   - After the individual analysis, each agent can communicate with and interrogate peers to gather insights and opinions about the incident.
   - This collaborative stage leverages the collective expertise of the agents to detect potential causes and propose solutions.
   - Specialized tools may be available to enable communication.

3) Voting:
   - After gathering sufficient information, each agent provides a comprehensive analysis of the incident and votes on which hazard vector caused the problem, including a clear justification.
   - Voting is private; agents will not know others' votes until the end of the POIROT protocol.

Your role (preliminary to these stages):
- You must determine the possible LOCATIONS where errors may occur within the system.
  Example (clinical environment): if the issue is that the Doctor agent lacks sufficient medical knowledge, the location is the Doctor agent.

Your task:
- Identify all potential error locations in the system.
- Given the code or description of a multi-agent system, determine the possible points where errors may occur.
- Represent the result as an N-dimensional vector, where each element corresponds to a potential error location:
  [x1, x2, x3, x4, ...]
  where x1 → location 1, x2 → location 2, etc.
- Later, vectors like [1, 0, 1, 0] indicate issues in locations x1 and x3.

Guidelines for identifying error locations:
- Each agent in the system must be a potential error region. If the system has N agents, include N positions (one per agent). It clarifies that if these agents represent individuals whose social behavior affects the development of the system, inappropriate behavior patterns that may affect coexistence or process development should be identified as errors. 
- Physical regions that play an important role must be considered.
- Hardware elements must be included if they play a significant role.
- Software elements must be included if they play a significant role.
- Human or external actors that interact with and affect the system must be considered.
- Include other relevant components not covered above that play an important role.

Restrictions:
- Do NOT over-identify regions. If a region is clearly integrated, do not subdivide it.
- Tools and actions belonging to the same agent must be treated as a single error region.
- If a physical location only hosts hardware and has no functional role, do NOT include it as an error source (but include the hardware).
- If an element integrates hardware and software, identify it as a single error region unless separation is absolutely necessary.
- Error regions must be limited. Defining an error region for a global failure or one that affects many regions, such as “similar system failure,” is not valid and is considered an error.  

Description field guidelines (CRITICAL):
- The "description" for each error region must describe DETECTABLE ANOMALOUS BEHAVIOURS, not generic risks or normal role characteristics.
- Do NOT describe what the component normally does or what its inherent limitations are. Instead, describe what it would look like if this component were the source of a fault.
- Use the format: "Error signature: [observable anomaly that distinguishes this component as the error source]. Normal behaviour: [what this component should do when functioning correctly]."
- The description must be specific enough that a peer agent reading a session log could recognise whether the described anomaly is present or absent.
- Avoid vague descriptions such as "potential errors in analysis" or "may introduce bias" — these apply equally to all agents and provide no diagnostic value.
- For agents: describe how their output would deviate from their defined role (e.g., wrong conclusions contradicting their own data, recommendations opposite to their mandate, content inconsistent with the agent's function).
- For hardware/software/data components: describe what corrupted, absent, or malformed output would look like in the session log.

Output requirement:
- You MUST output ONLY valid JSON and nothing else.
- Use the following schema:

{
  "system_name": "<name or description of the multi-agent system>",
  "error_regions": [
    {
      "id": "<snake_case_identifier>",  // IMPORTANT: For agents, use snake_case version of name (e.g., "Portfolio Manager" -> "portfolio_manager")
      "name": "<error region name>",
      "type": "<agent | hardware | software | physical | human | other>",
      "description": "<Error signature: [observable anomaly]. Normal behaviour: [expected behaviour when correct].>"
    }
    // ...
  ],
  "error_vector_example": [1, 0, 1, 0]
}
"""


class HazardSpaceMismatchError(ValueError):
    """The Phase I hazard space does not contain exactly one region per agent id."""


def format_required_agent_regions(agents: List[Dict[str, str]]) -> str:
    """Instruction that forces one agent region per agent, using the given ids."""
    lines = "\n".join(f'- id: "{a["id"]}"  (name: {a.get("name", a["id"])})' for a in agents)
    return (
        "REQUIRED AGENT REGIONS:\n"
        "The system contains exactly the following agents. For EACH of them include exactly one "
        "error region of type \"agent\" whose \"id\" is EXACTLY the id given below (same spelling "
        "and case; do not rename, translate, merge or split them). Do not create any other region "
        "of type \"agent\". Non-agent regions (hardware, software, human, ...) are defined as usual.\n"
        + lines
    )


def find_agent_region_problems(error_space: Dict[str, Any], agent_ids: List[str]) -> List[str]:
    """Return the reasons why error_space does not map every agent id to one region (empty if OK)."""
    regions = error_space.get("error_regions") or []
    region_ids = [str(r.get("id", "")) for r in regions]
    problems = []
    missing = [a for a in agent_ids if a not in region_ids]
    if missing:
        problems.append(f"missing region ids for agents {missing} (got ids {region_ids})")
    duplicated = sorted({a for a in agent_ids if region_ids.count(a) > 1})
    if duplicated:
        problems.append(f"agent ids used by more than one region: {duplicated}")
    return problems


class POIROTAgent:
    """POIROT helper agent to identify potential error locations in a multi-agent system.

    This agent analyzes system descriptions and constructs the error vector space
    by identifying all potential failure points. The output is used to define the
    dimensions of error vectors in subsequent POIROT analysis.

    Attributes:
        model: LLM model name (default: "gemini-2.5-pro")
        llm: LangChain LLM instance (initialized lazily)
        system_prompt: System prompt for error location identification
        vectors_to_ignore: List of components to exclude from analysis

    Usage example:
        >>> agent = POIROTAgent(model="gemini-2.5-pro")
        >>> result = agent.analyze_system(system_description)
        >>> _p(result["error_regions"])
        [{'id': 'x1', 'name': 'Portfolio Manager', 'type': 'agent', ...}, ...]
    """

    def __init__(
        self, 
        model: str = "gemini-2.5-pro", 
        vectors_to_ignore: Optional[List[str]] = None,
        use_local_llm: bool = False,
        local_model_name: Optional[str] = None,
        token_tracker: Optional[Any] = None,
        llm_provider: str = "gemini"
    ):
        """Initialize the POIROT agent.

        Args:
            model: LLM model to use for analysis (default: gemini-2.5-pro)
            vectors_to_ignore: List of components that should NOT be considered
                              as potential error regions (e.g., known non-issues)
            use_local_llm: If True, use LM Studio instead of Gemini API
            local_model_name: Model name for LM Studio (default: gpt-oss-20b)
            token_tracker: Optional TokenTracker instance for counting API tokens
            llm_provider: LLM provider - "gemini", "deepseek", "local", or "ollama" (default: "gemini")
        """
        self.model = model
        self.use_local_llm = use_local_llm
        self.local_model_name = local_model_name
        self.token_tracker = token_tracker
        self.llm_provider = llm_provider
        # LLM will be initialized lazily to avoid requiring credentials at import time
        self.llm = None
        # Keep the system prompt available
        self.system_prompt = POIROT_SYSTEM_PROMPT
        # Vectors to ignore: a list of human-readable strings that should NOT be treated as potential error regions
        self.vectors_to_ignore: List[str] = vectors_to_ignore or []

    def analyze_system(
        self,
        system_description: str,
        ignore_list: Optional[List[str]] = None,
        verbose: bool = True,
        agents: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Run the POIROT pre-analysis on a textual description of a multi-agent system.

        This method sends the system description to the LLM and receives a structured
        JSON response defining the error vector space.

        Args:
            system_description: Textual description of the multi-agent system,
                              including agents, workflow, components, etc.
            ignore_list: Optional list of components to exclude from analysis.
                        If provided, overrides the constructor's vectors_to_ignore.
            agents: Optional list of the system's agents as ``{"id": ..., "name": ...}``.
                    When given, the LLM must create exactly one region per agent using
                    that exact ``id`` (the id is what votes are weighted against). The
                    result is validated; on a mismatch the LLM gets one chance to fix
                    it and a :class:`HazardSpaceMismatchError` is raised otherwise.

        Returns:
            Dictionary containing:
                - system_name: Name of the analyzed system
                - error_regions: List of potential error locations, each with:
                    * id: Unique identifier (x1, x2, x3, ...)
                    * name: Human-readable name
                    * type: Category (agent, hardware, software, physical, human, other)
                    * description: Explanation of why this is a potential error source
                - error_vector_example: Example binary vector showing error encoding

        Raises:
            HazardSpaceMismatchError: If ``agents`` is given and the regions still do
                not contain every agent id after one correction attempt.
            Exception: If LLM invocation fails or response cannot be parsed
        """
        _p = print if verbose else (lambda *a, **kw: None)
        # Determine the ignore list to pass to the LLM: method arg overrides constructor list
        effective_ignore = ignore_list if ignore_list is not None else self.vectors_to_ignore

        ignore_message = ""
        if effective_ignore:
            # Format ignore instructions as a short numbered list
            lines = [f"{i+1}. {s}" for i, s in enumerate(effective_ignore)]
            ignore_message = (
                "Please DO NOT consider the following items as potential error regions (they are known non-issues):\n"
                + "\n".join(lines)
            )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=system_description)
        ]

        if ignore_message:
            messages.append(HumanMessage(content=ignore_message))

        if agents:
            messages.append(HumanMessage(content=format_required_agent_regions(agents)))

        # Initialize LLM lazily (so importing the module does not require credentials)
        if self.llm is None:
            if LLMFactory is not None:
                self.llm = LLMFactory.create_chat_llm(
                    model_name=self.model,
                    use_local=self.use_local_llm,
                    local_model_name=self.local_model_name,
                    provider=self.llm_provider,
                    temperature=0,
                    max_tokens=32000
                )
            else:
                self.llm = ChatGoogleGenerativeAI(model=self.model, temperature=0, max_tokens=32000)

        result, raw = self._invoke_and_parse(messages, _p)
        if not agents or "error" in result:
            return result

        agent_ids = [a["id"] for a in agents]
        problems = find_agent_region_problems(result, agent_ids)
        if problems:
            # One correction attempt: show the model its answer and what is wrong.
            _p(f"    WARNING Hazard space does not match the agents ({'; '.join(problems)}). Asking for a fix...")
            messages += [
                AIMessage(content=raw),
                HumanMessage(content=(
                    "Your error space is invalid: " + "; ".join(problems) + ".\n"
                    + format_required_agent_regions(agents)
                    + "\nReturn the complete corrected JSON."
                )),
            ]
            result, _ = self._invoke_and_parse(messages, _p)
            if "error" in result:
                return result
            problems = find_agent_region_problems(result, agent_ids)
            if problems:
                raise HazardSpaceMismatchError(
                    "The hazard space built in Phase I does not match the system's agents: "
                    + "; ".join(problems)
                )
        return result

    def _invoke_and_parse(self, messages: List[Any], _p) -> Tuple[Dict[str, Any], str]:
        """Invoke the LLM and parse its JSON answer. Returns (result, raw_content)."""
        response = self.llm.invoke(messages)

        # Track tokens if tracker is available
        if self.token_tracker is not None and extract_tokens_from_response is not None:
            usage = extract_tokens_from_response(response)
            self.token_tracker.add(usage)
            _p(f"    Tokens used: {usage.total_tokens} (input: {usage.input_tokens}, output: {usage.output_tokens})")

        raw = response.content if isinstance(response.content, str) else str(response.content)

        # The model MUST return JSON only; try to parse it
        try:
            return json.loads(raw), raw
        except Exception:
            # Try to extract JSON block if the model wrapped it in markdown
            import re
            m = re.search(r'```json\s*(\{.*?\})\s*```', raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1)), raw
                except Exception:
                    pass
            # Last resort: the first JSON object embedded in surrounding text
            if "{" in raw:
                try:
                    obj, _ = json.JSONDecoder().raw_decode(raw[raw.index("{"):])
                    if isinstance(obj, dict):
                        return obj, raw
                except Exception:
                    pass

        # If parsing failed, return raw content under an error key
        return {"error": "Could not parse LLM response as JSON", "raw": raw}, raw

    def save_output(self, parsed: Dict[str, Any], file_path: str) -> bool:
        """Save the parsed output dictionary to a JSON file.

        Creates the output directory if it doesn't exist.

        Args:
            parsed: Dictionary containing the error vector space analysis
            file_path: Path where the JSON file should be saved

        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Ensure output directory exists
            dirpath = os.path.dirname(os.path.abspath(file_path))
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
