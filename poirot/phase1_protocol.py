"""
POIROT Phase 1 - Individual Analysis Protocol
==============================================
This module executes Phase 1 of the POIROT protocol where each agent analyzes
the session independently WITHOUT peer consultation.

IMPORTANT: Only agents who PARTICIPATED in the session analyze in Phase 1.
Non-participants are SKIPPED and will participate in Phase 2 (peer consultation).

Each participating agent receives:
1. System prompt
2. Filtered messages (messages they sent OR received, cleaned of tool calls)
3. POIROT protocol message
4. Phase 1 specific instructions
5. Ignore list (known non-issues)

Output: Dictionary with individual analysis reports from participating agents only.

Author: POIROT-SW Team
Date: December 2, 2025
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage

from .agent_factory import clean_conversational_messages as _clean_conversational_messages

# Import token tracker
try:
    from token_tracker import TokenTracker, extract_tokens_from_response
except ImportError:
    try:
        from .token_tracker import TokenTracker, extract_tokens_from_response
    except ImportError:
        TokenTracker = None
        extract_tokens_from_response = None


# Phase 1 specific message
POIROT_PHASE1_MESSAGE = """
-----------------------------------------------------------------------------
PHASE 1: INDIVIDUAL ANALYSIS
-----------------------------------------------------------------------------

You are now in PHASE 1 of the POIROT protocol: Individual Analysis.

The session log above contains ALL messages exchanged during the session — both your own outputs and those of every other agent. Study the complete session carefully before responding.

Your task has TWO parts:

=========================================
PART A — SELF-EVALUATION
=========================================
Review your own messages in the session log (marked as your outputs) and ask yourself:
- Did my analysis or output match the data and evidence I received?
- Did I fulfil my defined role faithfully?
- Are my conclusions internally consistent?
- Did I make any claims that contradict the information available to me?

If everything you did was correct and consistent with your role, that is a valid and important answer — report it honestly.

=========================================
PART B — PEER EVALUATION
=========================================
Review the messages from EVERY OTHER agent in the session and ask yourself:
- Does any agent's output contradict the data or evidence visible in the session?
- Does any agent appear to have deviated from their expected role or mandate?
- Is any agent's output internally inconsistent (e.g., conclusions that contradict their own stated data)?
- Does any agent's output conflict with what you observed in the session?

Focus on BEHAVIOURAL anomalies — things that should not have happened given the agent's role and the data available. Do NOT flag normal disagreements or expected uncertainty.

=========================================
IMPORTANT GUIDELINES
=========================================
- Be specific: cite the agent name and quote or paraphrase the specific evidence.
- Be objective: distinguish between "this agent behaved anomalously" vs. "I simply disagree with their opinion".
- If you observed nothing anomalous — in yourself or in any peer — return empty lists. This is a valid and useful answer. Do NOT invent observations to appear thorough.
- Focus on the ROOT CAUSE. If one agent produced flawed output that caused downstream agents to propagate the error, flag the source agent, not the agents that received and forwarded the bad data.

REQUIRED OUTPUT FORMAT (JSON):
You MUST respond with a JSON object in this exact format. The agent names in "suspected_agents" must be the exact agent role names as they appear in the session (e.g., "market_analyst", "risk_manager", "bull_researcher").

```json
{
  "self_evaluation": {
    "role_fulfilled": true,
    "anomalies_detected": false,
    "description": "Brief statement of whether your own output was correct and consistent with your role. If anomalous, describe what went wrong.",
    "evidence": "Specific quote or reference from your own messages, or 'N/A' if no anomaly."
  },
  "peer_observations": [
    {
      "agent_name": "exact_agent_role_name",
      "description": "What anomaly you observed in this agent's behaviour or output",
      "evidence": "Specific quote or paraphrase from the session log supporting this observation"
    }
  ],
  "suspected_agents": ["agent_role_name_1", "agent_role_name_2"]
}
```

The "suspected_agents" list should contain the role names of agents you believe behaved anomalously, based on your observations. If you detected no anomalies in yourself or any peer, leave "suspected_agents" as an empty list [].

Examples:

**Example 1 — Peer anomaly detected:**
```json
{
  "self_evaluation": {
    "role_fulfilled": true,
    "anomalies_detected": false,
    "description": "My analysis was consistent with the data I received and my defined role.",
    "evidence": "N/A"
  },
  "peer_observations": [
    {
      "agent_name": "market_analyst",
      "description": "The market analyst reported a bearish Death Cross signal but the SMA values cited in the same message show the 50-day SMA is above the 200-day SMA, which is a bullish Golden Cross. The conclusion contradicts the agent's own data.",
      "evidence": "market_analyst message: 'Death Cross forming (50d SMA: 185.2, 200d SMA: 162.4)' — these numbers indicate a Golden Cross, not a Death Cross."
    }
  ],
  "suspected_agents": ["market_analyst"]
}
```

**Example 2 — No anomalies observed:**
```json
{
  "self_evaluation": {
    "role_fulfilled": true,
    "anomalies_detected": false,
    "description": "My output was consistent with the data and my role as bear_researcher.",
    "evidence": "N/A"
  },
  "peer_observations": [],
  "suspected_agents": []
}
```

**Example 3 — Self-anomaly detected:**
```json
{
  "self_evaluation": {
    "role_fulfilled": false,
    "anomalies_detected": true,
    "description": "My recommendation contradicted the consensus data I received. I advocated for a leveraged BUY despite three analysts recommending SELL.",
    "evidence": "My message: 'I recommend a 2x leveraged BUY position.' Analyst consensus in session: Risky=SELL, Safe=SELL, Neutral=REDUCE."
  },
  "peer_observations": [],
  "suspected_agents": ["risk_manager"]
}
```
"""


def create_poirot_protocol_message() -> str:
    """Create POIROT protocol message for participant agents."""
    message = """
-----------------------------------------------------------------------------
POIROT PROTOCOL ACTIVATION
-----------------------------------------------------------------------------

During the execution of the multiagent system that you are involved in; a hazard event has occurred that could potentially compromise the well function of the system. As a result it is imperative to conduct a POIROT. You are now operating under the POIROT (Peer Opinion Interrogation Review & Outcome Tracking) protocol.

You are a key element in the POIROT protocol. Your role is to help identify the hazard vector causing the incident by providing your expert opinion on the matter. Right now you don't have to worry about your main role (doctor, researcher, software engineer or whatever your role is), your only focus should be on the incident at hand.

The POIROT protocol consists of the following steps:
1. Self-assessment: each agent will analyze the session independently, reflecting what they see and perform during the session. Pointing out any possible mistakes, flaws or problems that could have caused the incident. This incidents could be cause by your own mistakes, by other agents mistakes or by flaws in the system itself. It is possible that you have not seen anything wrong during the session, it is okay to say that too. Be sure to be objective and analytical during this process.
2. Peer Consultation: As part of the protocol, you are able to communicate and interrogate your peers (other agents in the system) to gather their insights and opinions regarding the incident. This collaborative approach aims to leverage the collective expertise of the team to identify potential causes and solutions. You have access to specialized tools that allow you to communicate with your peers. You must communicate with other agents to gather more information about the incident base on the preliminary analysis you and your peers did in the first stage.
3. Voting: After gathering all the necessary information, provide a comprehensive analysis of the incident, including your conclusions about the hazard vector. You will have to vote which is the hazard vector that you believe caused the incident, providing a clear justification for your choice. This process is private and other agents will not know your choice until the end of the POIROT protocol.

It is important to be objective and analytical during this process. The most important thing is the system's safety and well-functioning, so please focus on providing accurate and useful information to help resolve the situation. This may include acknowledging your own mistakes, accusing colleagues, or identifying flaws in the system (among other causes). Perform these actions whenever necessary. 

Expected response: Even though this is a complex research process, your response should just be a message indicating which is the identified hazard vector and the justification. In this stage do not provide possible solutions or recommendations and don't be redundant. Just focus on identifying the hazard vector and the justification.

YOUR ROLE:
Analyze the session from your perspective as one of the participating agents.
"""
    
    message += """
-----------------------------------------------------------------------------
"""
    return message


def format_vectors_to_ignore(vectors_to_ignore: List[str]) -> str:
    """Format the ignore list for agents."""
    if not vectors_to_ignore or len(vectors_to_ignore) == 0:
        return ""
    
    message = """
-----------------------------------------------------------------------------
KNOWN NON-ISSUES TO IGNORE
-----------------------------------------------------------------------------

The following items are KNOWN NON-ISSUES that have been previously analyzed 
and confirmed to be acceptable. DO NOT report these as observations:

"""
    for idx, item in enumerate(vectors_to_ignore, 1):
        message += f"{idx}. {item}\n"
    
    message += """
Focus your analysis on NEW or DIFFERENT issues not listed above.
-----------------------------------------------------------------------------
"""
    return message


def execute_phase1_analysis(
    agents: Dict,
    processed_messages: List,
    agent_factory,
    error_space: Optional[Dict],
    vectors_to_ignore: Optional[List[str]],
    output_dir: Optional[Path] = None,
    session_name: str = "",
    communication_tool_names: Optional[set] = None,
    include_tool_calls: bool = False,
    include_broadcast_messages: bool = False,
    full_context: bool = False,  # unused: Phase 1 always reviews the full session
    token_tracker: Optional[Any] = None,
    use_local_llm: bool = False,
    local_model_name: Optional[str] = None,
    llm_provider: str = "gemini",
    max_llm_retries: int = 5,
    retry_delay_503: int = 30,
    retry_delay_429: int = 60,
    api_call_delay: float = 0.0,
    verbose: bool = True,
    debug: bool = False,
) -> Dict[str, str]:
    """
    Execute Phase 1 of POIROT protocol: Individual Analysis.
    
    IMPORTANT: Only agents who PARTICIPATED in the session analyze in Phase 1.
    Non-participants are skipped and will participate in Phase 2.
    
    Each participating agent analyzes the session independently without peer consultation.
    Uses the same patterns as POIROTPhase1.py:
    - Filter messages by agent participation (messages sent OR received)
    - Clean messages (remove tool-only messages)
    - Skip non-participants (no messages = no participation)
    - Build context with specific message order for participants
    - Invoke base LLM without tools (no peer consultation)
    - Store results in reports dict and individual files
    
    Args:
        agents: Dict from AgentFactory with agent_id -> {config, llm, tools, ...}
        processed_messages: List of ProcessedMessage objects from AgentFactory
        agent_factory: AgentFactory instance for message filtering/cleaning
        error_space: Optional dict from POIROTAgent with error_regions structure
        vectors_to_ignore: Optional list of known non-issues to exclude
        output_dir: Path to output directory for saving reports
        session_name: Name of session being analyzed
        
    Returns:
        dict: {agent_id: str (analysis report content)} - only for participating agents
    """
    _p = print if verbose else (lambda *a, **kw: None)
    
    _p("\n" + "="*80)
    _p(" POIROT PHASE 1: INDIVIDUAL ANALYSIS")
    _p("="*80)
    _p(f"Session: {session_name}")
    _p(f"Total agents: {len(agents)}")
    _p(f"Total messages: {len(processed_messages)}")
    if vectors_to_ignore:
        _p(f"Ignore list: {len(vectors_to_ignore)} items")
    _p("-"*80)
    _p("Only PARTICIPATING agents analyze (NO peer consultation)")
    _p("Non-participants will be skipped")
    _p("="*80 + "\n")
    
    phase1_reports = {}
    phase1_metadata = {}
    skipped_agents = []
    
    for agent_id, agent_data in agents.items():
        agent_name = agent_data['config'].agent_name
        # In LangChain adapter mode the compiled agent is used directly;
        # in database mode a reconstructed LLM instance is used instead.
        compiled_agent = agent_data.get('compiled_agent')
        llm = agent_data.get('llm')
        system_prompt = agent_data.get('system_prompt', '')
        
        _p(f"\n{'-'*80}")
        _p(f"Agent {agent_name.upper()} (ID: {agent_id}) - INDIVIDUAL ANALYSIS")
        _p(f"{'-'*80}")
        
        # Phase 1 (paper protocol): every participating agent reviews the FULL
        # session log -- its own outputs (AIMessage) and those of every other agent
        # (HumanMessage with a "From X to Y" header) -- to perform both a
        # self-evaluation and a peer evaluation. Only agents that sent at least one
        # message in the session are considered participants.
        participated = any(pm.from_agent == agent_id for pm in processed_messages)
        filtered_processed = list(processed_messages)

        # Clean messages (remove tool-only AIMessages and technical ToolMessages).
        # Use the factory's method when available, otherwise call the module-level function
        # directly (agentless / LangChain adapter mode).
        if agent_factory is not None:
            cleaned_processed = agent_factory.clean_conversational_messages(
                filtered_processed,
                include_tool_calls=include_tool_calls,
                current_agent_id=agent_id,
            )
        else:
            cleaned_processed = _clean_conversational_messages(
                processed_messages=filtered_processed,
                include_tool_calls=include_tool_calls,
                current_agent_id=agent_id,
                communication_tool_names=communication_tool_names or set(),
            )
        
        # Format messages with timestamps and headers (similar to Phase 2)
        formatted_messages = []
        for msg in cleaned_processed:
            # Extract metadata if available
            metadata = msg.additional_kwargs.get('metadata', {})
            from_node = metadata.get('from_node', '')
            to_node = metadata.get('to_node', '')
            
            # STRICT FILTERING: Remove messages without specific recipient if broadcasts are disabled
            # This addresses user request to remove "to all" and "Action: sleeping" type messages
            if isinstance(msg, AIMessage):
                is_broadcast_msg = (to_node in ["all", "broadcast"] or not to_node)
                if is_broadcast_msg and not include_broadcast_messages:
                    continue

            # Note: timestamp might not be in metadata if not added by agent_factory
            # But we can try to get it if available, or skip it
            timestamp = metadata.get('timestamp', '')
            
            time_str = f"[{timestamp}] " if timestamp else ""
            
            if isinstance(msg, SystemMessage):
                new_content = f"{time_str}[SYSTEM EVENT]: {msg.content}"
                new_msg = SystemMessage(content=new_content)
            else:
                # For dialogues
                context_header = f"{time_str}From {from_node} to {to_node}:"
                new_content = f"{context_header}\n{msg.content}"
                
                # REFORMAT MESSAGE TYPE RELATIVE TO ANALYZING AGENT
                # If I (agent_id) sent it -> AIMessage
                # If someone else sent it -> HumanMessage (Input to me)
                
                if from_node == agent_id:
                    new_msg = AIMessage(content=new_content)
                else:
                    new_msg = HumanMessage(content=new_content)
            
            # Copy metadata
            new_msg.additional_kwargs = msg.additional_kwargs
            formatted_messages.append(new_msg)
            
        cleaned_processed = formatted_messages
        
        _p(f"    Filtered context: {len(cleaned_processed)} messages")
        
        # Check if agent participated
        if not participated or not cleaned_processed:
            _p(f"   WARNING  {agent_name} did NOT participate in original session")
            _p(f"   SKIP  SKIPPING - Non-participants do not analyze in Phase 1")
            _p(f"    This agent will participate in Phase 2 (peer consultation)")
            skipped_agents.append(agent_name)
            continue
        else:
            _p(f"   OK {agent_name} participated in original session")
            
            # Participant case: use normal protocol with cleaned messages
            protocol_msg = create_poirot_protocol_message()
            
            if compiled_agent is not None:
                # LangChain adapter mode: the system prompt is already embedded in the
                # compiled agent — pass only the conversation context + POIROT prompts.
                context_messages = cleaned_processed + [
                    HumanMessage(content=protocol_msg),
                    HumanMessage(content=POIROT_PHASE1_MESSAGE),
                ]
            else:
                # Database mode: system prompt must be injected explicitly.
                # CRITICAL: Match exact message order from POIROTPhase1.py
                # [SystemMessage] + cleaned_messages + [protocol, phase1_msg, ignore_list]
                context_messages = [
                    SystemMessage(content=system_prompt)
                ] + cleaned_processed + [
                    HumanMessage(content=protocol_msg),
                    HumanMessage(content=POIROT_PHASE1_MESSAGE),
                ]

            # Add vectors to ignore if provided
            if vectors_to_ignore and len(vectors_to_ignore) > 0:
                ignore_message = format_vectors_to_ignore(vectors_to_ignore)
                context_messages.append(HumanMessage(content=ignore_message))
                _p(f"   WARNING  Added {len(vectors_to_ignore)} known non-issues to ignore")
        
            # Save context to debug/ subfolder (only when debug=True)
        if debug and output_dir:
            debug_dir = output_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            context_filepath = debug_dir / f"phase1_context_{agent_name}.json"
            context_data = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "session": session_name,
                "participated": True,
                "filtered_messages_count": len(cleaned_processed),
                "system_prompt": system_prompt,
                "tools": [
                    {"name": tool.name, "description": tool.description}
                    for tool in agent_data['tools']
                ],
                "communication_tools": [
                    {"name": tool.name, "description": tool.description}
                    for tool in agent_data['communication_tools']
                ],
                "context_messages": [
                    {
                        "type": type(msg).__name__,
                        "content": msg.content[:500] + "..." if len(msg.content) > 500 else msg.content,
                        "from_agent": msg.additional_kwargs.get('metadata', {}).get('from_node') if hasattr(msg, 'additional_kwargs') else None,
                        "to_agent": msg.additional_kwargs.get('metadata', {}).get('to_node') if hasattr(msg, 'additional_kwargs') else None
                    }
                    for msg in context_messages
                ],
                "vectors_to_ignore": vectors_to_ignore if vectors_to_ignore else [],
                "total_context_messages": len(context_messages)
            }
            with open(context_filepath, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, indent=2, ensure_ascii=False)
        
        _p(f"    Generating individual analysis...")

        _MAX_LLM_RETRIES = max_llm_retries
        _LLM_RETRY_DELAY_503 = retry_delay_503
        _LLM_RETRY_DELAY_429 = retry_delay_429
        _llm_attempts = 0
        response = None

        if compiled_agent is not None:
            # -- LangChain adapter mode: invoke the original compiled agent -----
            # The agent already carries its LLM, system prompt, and tools.
            # We pass only the session context + POIROT protocol messages.
            while True:
                try:
                    if api_call_delay > 0:
                        time.sleep(api_call_delay)
                    result = compiled_agent.invoke({"messages": context_messages})
                    # Extract the last AIMessage produced by the agent
                    result_msgs = result.get("messages", [])
                    response = next(
                        (m for m in reversed(result_msgs) if isinstance(m, AIMessage)),
                        None,
                    )
                    break
                except Exception as _exc:
                    _estr = str(_exc)
                    _is_503 = 'UNAVAILABLE' in _estr or '503' in _estr
                    _is_429 = 'RESOURCE_EXHAUSTED' in _estr or '429' in _estr
                    if (_is_503 or _is_429) and _llm_attempts < _MAX_LLM_RETRIES:
                        _llm_attempts += 1
                        _code = '503 UNAVAILABLE' if _is_503 else '429 RESOURCE_EXHAUSTED'
                        _wait = _LLM_RETRY_DELAY_503 if _is_503 else _LLM_RETRY_DELAY_429
                        _p(f"\nWARNING  Transient LLM error ({_code}).")
                        _p(f"... Pausing {_wait}s before retry ({_llm_attempts}/{_MAX_LLM_RETRIES})...")
                        time.sleep(_wait)
                        _p(f"Retry Retrying...")
                    else:
                        raise
        else:
            # -- Database mode: invoke the reconstructed LLM directly ----------
            while True:
                try:
                    if api_call_delay > 0:
                        time.sleep(api_call_delay)
                    response = llm.invoke(context_messages)
                    break
                except Exception as _llm_exc:
                    _estr = str(_llm_exc)
                    _is_503 = 'UNAVAILABLE' in _estr or '503' in _estr
                    _is_429 = 'RESOURCE_EXHAUSTED' in _estr or '429' in _estr
                    if (_is_503 or _is_429) and _llm_attempts < _MAX_LLM_RETRIES:
                        _llm_attempts += 1
                        _code = '503 UNAVAILABLE' if _is_503 else '429 RESOURCE_EXHAUSTED'
                        _wait = _LLM_RETRY_DELAY_503 if _is_503 else _LLM_RETRY_DELAY_429
                        _p(f"\nWARNING  Transient LLM error ({_code}).")
                        _p(f"... Pausing {_wait}s before retry ({_llm_attempts}/{_MAX_LLM_RETRIES})...")
                        time.sleep(_wait)
                        _p(f"Retry Retrying LLM call...")
                    else:
                        raise

        # Track tokens (database mode only — compiled agent does not expose usage here)
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        if compiled_agent is None and token_tracker is not None and extract_tokens_from_response is not None:
            usage = extract_tokens_from_response(response)
            token_tracker.add(usage)
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            total_tokens = usage.total_tokens
            _p(f"    Tokens: {usage.total_tokens} (in: {usage.input_tokens}, out: {usage.output_tokens})")

        # Extract text content from response (handles multimodal edge cases)
        if response is None:
            content_str = ""
        elif hasattr(response, 'content') and isinstance(response.content, list):
            parts = []
            for part in response.content:
                if isinstance(part, dict):
                    if 'type' in part and part['type'] == 'text' and 'text' in part:
                        parts.append(part['text'])
                    elif 'text' in part:
                        parts.append(part['text'])
                    else:
                        parts.append(str(part))
                else:
                    parts.append(str(part))
            content_str = "".join(parts)
        elif hasattr(response, 'content'):
            content_str = str(response.content)
        else:
            content_str = ""

        # Store report
        phase1_reports[agent_id] = content_str
        
        # Store metadata
        phase1_metadata[agent_id] = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'calls': 1
        }
        
        _p(f"   OK Analysis complete ({len(content_str)} characters)")
        
        # Try to parse the structured JSON report to show a short summary
        try:
            json_match = re.search(r'\{[\s\S]*"self_evaluation"[\s\S]*\}', content_str)
            if json_match:
                json_data = json.loads(json_match.group(0))
                se = json_data.get('self_evaluation', {})
                _p("\n    PHASE 1 SUMMARY:")
                _p(f"   {'-'*76}")
                _p(f"   Self anomaly : {se.get('anomalies_detected')} - {se.get('description', '')}")
                for obs in json_data.get('peer_observations', []):
                    _p(f"   Peer         : {obs.get('agent_name', '?')} - {obs.get('description', '')}")
                _p(f"   Suspects     : {json_data.get('suspected_agents', [])}")
                _p(f"   {'-'*76}")
            else:
                _p("\n    REPORT PREVIEW (first 500 chars):")
                _p(f"   {content_str[:500]}...")
        except Exception:
            _p("\n    REPORT PREVIEW (parsing failed):")
            _p(f"   {content_str[:500]}...")

        # Save to phase1/ subfolder named after the agent
        if output_dir:
            phase1_dir = output_dir / "phase1"
            phase1_dir.mkdir(parents=True, exist_ok=True)
            filepath = phase1_dir / f"{agent_name}.txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"POIROT PHASE 1 - INDIVIDUAL ANALYSIS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Agent: {agent_name}\n")
                f.write(f"Session: {session_name}\n")
                f.write(f"Context messages: {len(cleaned_processed)}\n")
                f.write("=" * 80 + "\n\n")
                f.write("OBSERVATIONS:\n")
                f.write("-" * 80 + "\n\n")
                f.write(content_str)
                f.write("\n\n" + "=" * 80 + "\n")
            _p(f"   Saved to: phase1/{agent_name}.txt")
    
    _p("\n" + "="*80)
    _p("OK PHASE 1 COMPLETE")
    _p("="*80)
    _p(f" Analysis Summary:")
    _p(f"   Participating agents analyzed: {len(phase1_reports)}")
    _p(f"   Non-participating agents skipped: {len(skipped_agents)}")
    if skipped_agents:
        _p(f"   Skipped: {', '.join(skipped_agents)}")
    if output_dir:
        _p(f"\n Reports saved to: {output_dir / 'phase1'}")
    _p("="*80 + "\n")
    
    return {
        'reports': phase1_reports,
        'metadata': phase1_metadata
    }
