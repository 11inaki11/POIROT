"""
POIROT Phase 2 Protocol: Peer Interrogation and Collaborative Analysis

This module implements the second phase of the POIROT protocol where agents
engage in peer-to-peer consultation to collaboratively identify hazard vectors.

Phase 2 Flow:
1. Load Phase 1 individual reports as context
2. Build LangGraph with communication tools for each agent
3. Execute consultation process with POIROT server mediation
4. Collect final votes (JSON with hazard_vector, location, justification)
5. Save outputs for Phase 3 analysis

Key Features:
- Dynamic communication tool generation based on can_communicate_with relationships
- POIROT server prevents uncontrolled message flooding ("desmadre")
- State-based consultation tracking (consulted flags, call counts)
- Protocol message injection only on first call
- Automatic final report request after consultation
- Non-participant support (agents who didn't participate in original session)
- Loop detection and message limit enforcement to prevent infinite loops
"""

##############################################################################
#======================= CONFIGURATION CONSTANTS ============================#
##############################################################################

# Maximum number of messages per agent before forcing final vote
# This prevents infinite loops and controls token consumption
MAX_AGENT_MESSAGES = 8

##############################################################################
#======================= IMPORTS ===========================================#
##############################################################################

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence, TypedDict, Annotated
from operator import add

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Import LLM Factory for local/remote LLM support
try:
    from llm_factory import LLMFactory
except ImportError:
    try:
        from src.llm_factory import LLMFactory
    except ImportError:
        print("⚠️ Could not import LLMFactory. Only Gemini API will be available.")
        LLMFactory = None

# Import session-specific agent loader
try:
    from session_agent_loader import get_agents_for_session, validate_one_agent_per_type
except ImportError:
    try:
        from src.session_agent_loader import get_agents_for_session, validate_one_agent_per_type
    except ImportError:
        print("⚠️ Could not import session_agent_loader.")
        raise

# Import voting system
try:
    from voting_system import weighted_voting_analysis
except ImportError:
    try:
        from src.voting_system import weighted_voting_analysis
    except ImportError:
        print("⚠️ Could not import voting_system. Voting analysis will be skipped.")
        weighted_voting_analysis = None

# Import token tracker
try:
    from token_tracker import TokenTracker, extract_tokens_from_response
except ImportError:
    try:
        from src.token_tracker import TokenTracker, extract_tokens_from_response
    except ImportError:
        TokenTracker = None
        extract_tokens_from_response = None


##############################################################################
#========================= JSON PARSING UTILITIES ===========================#
##############################################################################

def clean_json_string(json_str: str) -> str:
    """
    Clean a JSON string to handle common LLM output issues:
    - Unescaped newlines inside string values
    - Control characters
    - Other malformed content
    """
    if not json_str:
        return json_str
    
    # Remove control characters except common whitespace
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', json_str)
    
    # Try to parse as-is first
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass
    
    # Fix unescaped newlines inside string values
    # We track if we're inside a string and escape newlines there
    result = []
    in_string = False
    i = 0
    while i < len(cleaned):
        char = cleaned[i]
        
        # Check for string delimiter (unescaped quote)
        if char == '"':
            # Count preceding backslashes
            num_backslashes = 0
            j = i - 1
            while j >= 0 and cleaned[j] == '\\':
                num_backslashes += 1
                j -= 1
            
            # Quote is escaped only if preceded by odd number of backslashes
            if num_backslashes % 2 == 0:
                in_string = not in_string
            result.append(char)
        elif in_string and char == '\n':
            result.append('\\n')
        elif in_string and char == '\r':
            result.append('\\r')
        elif in_string and char == '\t':
            result.append('\\t')
        else:
            result.append(char)
        i += 1
    
    return ''.join(result)


def extract_vote_json(content: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract vote JSON from agent response.
    Handles various formats and malformed JSON.
    """
    if not content:
        return None
    
    # Pattern 1: Look for ```json ... ``` blocks
    json_block_pattern = r'```json\s*([\s\S]*?)\s*```'
    matches = re.findall(json_block_pattern, content, re.IGNORECASE)
    
    for match in matches:
        try:
            cleaned = clean_json_string(match.strip())
            vote_data = json.loads(cleaned)
            if 'hazard_vector' in vote_data and 'location' in vote_data:
                return vote_data
        except json.JSONDecodeError as e:
            # Log the error for debugging but continue trying
            print(f"    [DEBUG] JSON parse failed in ```json block: {e}")
            continue
    
    # Pattern 2: Look for raw JSON with hazard_vector (greedy match for nested content)
    # Find opening brace, then match until we find balanced closing brace
    raw_json_start = content.find('{"hazard_vector"')
    if raw_json_start == -1:
        # Try with newline after brace
        raw_json_start = content.find('{\n  "hazard_vector"')
    if raw_json_start == -1:
        raw_json_start = content.find('{\n')
        if raw_json_start != -1:
            # Check if it contains hazard_vector nearby
            snippet = content[raw_json_start:raw_json_start+300]
            if 'hazard_vector' not in snippet:
                raw_json_start = -1
    
    if raw_json_start != -1:
        # Find matching closing brace using bracket counting
        brace_count = 0
        in_str = False
        end_pos = raw_json_start
        
        for i, char in enumerate(content[raw_json_start:], start=raw_json_start):
            if char == '"':
                # Check if escaped
                num_bs = 0
                j = i - 1
                while j >= 0 and content[j] == '\\':
                    num_bs += 1
                    j -= 1
                if num_bs % 2 == 0:
                    in_str = not in_str
            elif not in_str:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
        
        if end_pos > raw_json_start:
            raw_json = content[raw_json_start:end_pos]
            try:
                cleaned = clean_json_string(raw_json)
                vote_data = json.loads(cleaned)
                if 'hazard_vector' in vote_data and 'location' in vote_data:
                    return vote_data
            except json.JSONDecodeError as e:
                print(f"    [DEBUG] JSON parse failed in raw JSON: {e}")
                pass
    
    # Pattern 3: Try regex to extract just the location array if all else fails
    # This is a fallback for severely malformed JSON
    try:
        hazard_match = re.search(r'"hazard_vector"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', content)
        location_match = re.search(r'"location"\s*:\s*\[([^\]]+)\]', content)
        
        if hazard_match and location_match:
            hazard_vector = hazard_match.group(1)
            location_str = location_match.group(1)
            # Parse location array
            location = [int(x.strip()) for x in location_str.split(',') if x.strip().isdigit() or x.strip() in ('0', '1')]
            
            if len(location) > 0:
                print(f"    [DEBUG] Extracted vote using fallback regex")
                return {
                    'hazard_vector': hazard_vector,
                    'location': location,
                    'justification': 'Extracted via fallback - original JSON was malformed'
                }
    except Exception as e:
        print(f"    [DEBUG] Fallback regex extraction failed: {e}")
        pass
    
    return None


##############################################################################
#========================= STATE DEFINITION =================================#
##############################################################################

def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dictionaries, with right overriding left."""
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}

class Phase2State(TypedDict):
    """
    State structure for Phase 2 peer interrogation process.
    
    This state tracks:
    - Historical messages from original session
    - Consultation messages between agents
    - Per-agent tracking flags (consulted, final_report, call_count) stored in agent_states
    - Final votes (hazard_vector, location)
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: int
    agent_states: Annotated[dict, merge_dicts]


##############################################################################
#===================== MESSAGE FILTERING UTILITIES ==========================#
##############################################################################

def add_message_metadata(message: BaseMessage, from_node: str, to_node: str) -> None:
    """
    Add routing metadata to a message for filtering.
    
    Args:
        message: Message to add metadata to
        from_node: Agent ID sending the message
        to_node: Agent ID receiving the message
    """
    if not hasattr(message, 'additional_kwargs'):
        message.additional_kwargs = {}
    
    if 'metadata' not in message.additional_kwargs:
        message.additional_kwargs['metadata'] = {}
    
    message.additional_kwargs['metadata']['from_node'] = from_node
    message.additional_kwargs['metadata']['to_node'] = to_node


def filter_messages_for_agent(messages: List[BaseMessage], agent_id: str, include_broadcast: bool = False) -> List[BaseMessage]:
    """
    Filter messages to only include those where the agent participates.
    
    ROBUSTNESS IMPROVEMENTS:
    - Eliminates duplicate messages (same content + same from/to metadata)
    - Removes empty messages (no content or whitespace only)
    - Preserves message order
    - Formats content with timestamp and sender/receiver info
    - Determines message type (AI vs Human) relative to the agent
    
    Args:
        messages: List of all messages
        agent_id: Agent ID to filter for
        include_broadcast: Whether to include 'all' / 'broadcast' messages even if not explicitly to agent
        
    Returns:
        Filtered list of messages where agent is sender or receiver (no duplicates, no empty)
    """
    filtered = []
    seen_messages = set()  # Track (content_hash, from_node, to_node) to detect duplicates
    
    for msg in messages:
        # Skip empty messages
        if not msg.content or (isinstance(msg.content, str) and msg.content.strip() == ""):
            continue
        
        # Skip messages without metadata
        if not hasattr(msg, 'additional_kwargs'):
            continue
        
        metadata = msg.additional_kwargs.get('metadata', {})
        from_node = metadata.get('from_node', '')
        to_node = metadata.get('to_node', '')
        timestamp = metadata.get('timestamp', '')
        
        # 1. Filter Internal Conversations (Self-talk)
        # Internal self-talk (from_node == to_node) must be excluded from filtered messages.
        if from_node == to_node and from_node != "unknown":
            continue

        # 2. Filter Relevance (Participation)
        # Include ONLY if agent is sender or specific receiver.
        # EXCLUDE broadcasts ("all") from others to prevent omniscience (seeing others' private actions/states).
        # ALSO EXCLUDE broadcasts ("all") from self to focus on direct conversations.

        # Normalize to_node for comparison (handle potential whitespace or case issues)
        to_node_check = str(to_node).strip().lower()

        is_broadcast = (to_node_check in ["all", "broadcast"])
        is_sender = (from_node == agent_id)
        is_receiver = (to_node == agent_id)

        # Logic to determine inclusion
        should_include = False
        if is_sender: should_include = True
        if is_receiver: should_include = True
        if include_broadcast and is_broadcast: should_include = True

        if not should_include:
            continue

        # 3. Determine Message Type relative to this agent
        # If the agent being analyzed sent this message -> AIMessage.
        # If the agent received it -> HumanMessage.
        if from_node == agent_id:
            MsgClass = AIMessage
        else:
            MsgClass = HumanMessage

        # 4. Format Content with Context
        # Prepend timestamp if available.
        # For dialogue messages, indicate sender and recipient in the header.
        
        time_str = f"[{timestamp}] " if timestamp else ""
        
        # If it's a SystemMessage, keep it as SystemMessage but maybe add timestamp
        if isinstance(msg, SystemMessage):
            new_content = f"{time_str}[SYSTEM EVENT]: {msg.content}"
            new_msg = SystemMessage(content=new_content)
        else:
            # For dialogues
            context_header = f"{time_str}From {from_node} to {to_node}:"
            new_content = f"{context_header}\n{msg.content}"
            new_msg = MsgClass(content=new_content)
        
        # Copy metadata
        new_msg.additional_kwargs = {'metadata': metadata}
        
        # Create unique identifier for duplicate detection
        content_preview = msg.content[:500] if isinstance(msg.content, str) else str(msg.content)[:500]
        content_hash = hash(content_preview)
        msg_signature = (content_hash, from_node, to_node)
        
        # Only add if not duplicate
        if msg_signature not in seen_messages:
            filtered.append(new_msg)
            seen_messages.add(msg_signature)
    
    return filtered


def clean_historical_messages(
    messages: List[BaseMessage], 
    include_tool_calls: bool = False,
    current_agent_id: str = None
) -> List[BaseMessage]:
    """
    Clean historical messages to keep only conversational context for POIROT.
    
    FILTERING STRATEGY:
    - KEEP: User messages (HumanMessage) - User ↔ Agent conversations
    - KEEP: AI responses with actual content - Agent ↔ Agent conversations
    - KEEP: System messages (SystemMessage)
    - KEEP: Communication tool responses (talk_to_*) - Inter-agent communication
    - CONDITIONALLY KEEP: Technical tool responses and tool_calls (if include_tool_calls=True)
    
    Args:
        messages: List of messages from the session
        include_tool_calls: Whether to include tool calls in the output
        current_agent_id: ID of the agent being processed (for context awareness)
        
    Returns:
        Cleaned list with conversational messages and inter-agent communications
    """
    cleaned = []
    seen_messages = set()  # Track (content_hash, msg_type, metadata) to detect duplicates
    
    for msg in messages:
        # First check: Skip empty messages (all types)
        # UNLESS it's a tool-using message and we include tool calls
        has_content = False
        if isinstance(msg.content, str):
            has_content = bool(msg.content and msg.content.strip())
        elif isinstance(msg.content, list):
             has_content = any(
                isinstance(item, dict) and item.get('text', '').strip()
                for item in msg.content
            )
        else:
            has_content = bool(msg.content)
            
        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
        
        # If no content and no tool calls, skip
        if not has_content and not has_tool_calls:
            continue
            
        if isinstance(msg, AIMessage):
            if has_tool_calls and not include_tool_calls and not has_content:
                # This is a tool-only message (no conversation) and we are ignoring tools -> skip it
                continue
            
            # Prepare kwargs
            clean_kwargs = msg.additional_kwargs.copy()
            
            # If we DON'T want tool calls, remove them from kwargs
            if not include_tool_calls:
                clean_kwargs = {
                    k: v for k, v in clean_kwargs.items() 
                    if k not in ['tool_calls', 'function_call']
                }
            
            # Check for duplicates before adding
            metadata = clean_kwargs.get('metadata', {})
            content_preview = str(msg.content)[:500]
            content_hash = hash(content_preview)
            from_node = metadata.get('from_node', '')
            to_node = metadata.get('to_node', '')

            # Message Type Logic relative to Current Agent
            # If I sent it -> AIMessage
            # If someone else sent it -> HumanMessage
            if current_agent_id and from_node == current_agent_id:
                cleaned_msg = AIMessage(
                    content=msg.content,
                    tool_calls=msg.tool_calls if include_tool_calls else [],
                    additional_kwargs=clean_kwargs,
                    id=msg.id
                )
                msg_type_str = 'AIMessage'
            else:
                cleaned_msg = HumanMessage(
                    content=msg.content,
                    additional_kwargs=clean_kwargs,
                    id=msg.id
                )
                msg_type_str = 'HumanMessage'
            
            msg_signature = (content_hash, msg_type_str, from_node, to_node)
            
            if msg_signature not in seen_messages:
                cleaned.append(cleaned_msg)
                seen_messages.add(msg_signature)
            
        elif isinstance(msg, ToolMessage):
            # Check if this is a communication tool (inter-agent dialogue)
            tool_name = getattr(msg, 'name', '')
            is_comm_tool = tool_name.startswith('talk_to_')
            
            if is_comm_tool or include_tool_calls:
                cleaned.append(msg)
            # Otherwise skip technical tools if include_tool_calls=False
                # This is inter-agent communication - KEEP IT
                # Convert to HumanMessage to represent the communication
                # Use the tool content as the message content
                communication_msg = HumanMessage(
                    content=f"[Communication via {tool_name}]: {msg.content}",
                    additional_kwargs=msg.additional_kwargs
                )
                
                # Check for duplicates before adding
                metadata = msg.additional_kwargs.get('metadata', {})
                content_preview = msg.content[:500] if isinstance(msg.content, str) else str(msg.content)[:500]
                content_hash = hash(content_preview)
                from_node = metadata.get('from_node', '')
                to_node = metadata.get('to_node', '')
                msg_signature = (content_hash, 'ToolMessage', from_node, to_node)
                
                if msg_signature not in seen_messages:
                    cleaned.append(communication_msg)
                    seen_messages.add(msg_signature)
            else:
                # Technical tool (retriever, test, etc.) - skip it
                continue
            
        else:
            # Keep SystemMessage and HumanMessage as-is (but check duplicates)
            metadata = msg.additional_kwargs.get('metadata', {}) if hasattr(msg, 'additional_kwargs') else {}
            content_preview = msg.content[:500] if isinstance(msg.content, str) else str(msg.content)[:500]
            content_hash = hash(content_preview)
            msg_type = type(msg).__name__
            from_node = metadata.get('from_node', '')
            to_node = metadata.get('to_node', '')
            msg_signature = (content_hash, msg_type, from_node, to_node)
            
            if msg_signature not in seen_messages:
                cleaned.append(msg)
                seen_messages.add(msg_signature)
    
    return cleaned


# ==============================================================================
# Context budget management
# ==============================================================================

# Approximate characters-per-token ratio (conservative for mixed content)
_CHARS_PER_TOKEN = 4

# Token budget for the historical messages portion of the context.
# System prompt + protocol + phase1 reports + voting instructions ≈ 30k tokens.
# Leaving ~95k for history stays safely under common 128k limits.
HISTORICAL_MESSAGES_TOKEN_BUDGET = 95_000   # tokens
# Minimum chars we keep per message when truncating (≈ first ~400 tokens)
_MIN_CHARS_KEEP = 1_600
# Max chars of a single message before it is a truncation candidate
_TRUNCATE_THRESHOLD_CHARS = 2_000


def _is_inter_agent_message(msg) -> bool:
    """
    Returns True if this message is direct inter-agent communication
    (agent A → specific agent B) and must NOT be truncated.

    Detection logic:
    - Tool outputs and broadcast messages have to_node = "all" / "broadcast" / ""
    - Inter-agent messages have to_node = a specific agent name (non-empty,
      not "all", not "broadcast")
    - Messages without metadata default to False (unknown → allow truncation,
      conservative for POIROT-generated synthetic messages)
    """
    meta = getattr(msg, 'additional_kwargs', {}).get('metadata', {})
    to_node = str(meta.get('to_node', '')).strip().lower()
    # Broadcast / tool outputs → truncatable
    if to_node in ('all', 'broadcast', ''):
        return False
    # Specific named recipient → inter-agent, protect
    return True


def truncate_large_messages_if_needed(
    messages: list,
    token_budget: int = HISTORICAL_MESSAGES_TOKEN_BUDGET,
) -> list:
    """
    If the total estimated token count of *messages* exceeds *token_budget*,
    truncate the longest messages by keeping only the first _MIN_CHARS_KEEP
    characters and appending a notice.

    The function never removes messages – it only shortens their content so
    all conversational context is preserved (just condensed for large data
    dumps like stock-price CSVs or CVE JSON responses).

    Inter-agent messages (agent A → specific agent B) are NEVER truncated
    because they carry critical reasoning that POIROT needs intact.
    Only tool outputs / broadcasts (to_node == "all" / "broadcast") are
    candidates for truncation.

    Args:
        messages:     List of LangChain BaseMessage objects.
        token_budget: Maximum allowed estimated tokens for this list.

    Returns:
        The (possibly modified) list of messages.
    """
    def _est_tokens(msg) -> int:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return len(content) // _CHARS_PER_TOKEN

    total_tokens = sum(_est_tokens(m) for m in messages)

    if total_tokens <= token_budget:
        return messages  # Nothing to do

    print(f"\n  ⚠️  Context too large: ~{total_tokens:,} tokens estimated "
          f"(budget: {token_budget:,}). Truncating large messages…")

    # Work on copies to avoid mutating shared objects
    from copy import deepcopy
    messages = [deepcopy(m) for m in messages]

    # Iteratively truncate the longest ELIGIBLE message until we're within budget
    max_iterations = len(messages) * 2  # safety cap
    iteration = 0
    while total_tokens > token_budget and iteration < max_iterations:
        iteration += 1

        # Find the longest message that:
        #   1. Is above the truncation threshold
        #   2. Is NOT an inter-agent message (those are protected)
        best_idx = -1
        best_len = _TRUNCATE_THRESHOLD_CHARS
        for i, m in enumerate(messages):
            if _is_inter_agent_message(m):
                continue  # ← protected: never truncate inter-agent messages
            content = m.content if isinstance(m.content, str) else str(m.content)
            if len(content) > best_len:
                best_len = len(content)
                best_idx = i

        if best_idx == -1:
            # No more eligible candidates – warn and stop
            protected = sum(1 for m in messages if _is_inter_agent_message(m))
            print(f"  ⚠️  No more truncatable messages (protected inter-agent: {protected}). "
                  f"Remaining tokens: ~{total_tokens:,}")
            break

        msg = messages[best_idx]
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        original_chars = len(content)
        kept_chars = max(_MIN_CHARS_KEEP, original_chars // 2)  # halve each pass

        # Store truncation info in metadata (NOT in content – the LLM must not see this)
        if not hasattr(msg, 'additional_kwargs') or msg.additional_kwargs is None:
            msg.additional_kwargs = {}
        msg.additional_kwargs['_poirot_truncated'] = {
            'original_chars': original_chars,
            'kept_chars': kept_chars,
            'removed_chars': original_chars - kept_chars,
        }
        msg.content = content[:kept_chars]  # clean truncation, no notice appended

        old_total = total_tokens
        total_tokens = sum(_est_tokens(m) for m in messages)
        print(f"     ↳ Truncated message [{best_idx}]: "
              f"{original_chars:,} → {kept_chars:,} chars  "
              f"(tokens: {old_total:,} → {total_tokens:,})")

    print(f"  ✅ Final context size: ~{total_tokens:,} tokens")
    return messages


def save_agent_context_to_file(agent_id: str, context_messages: list, 
                                 total_messages: int, output_dir: Path, 
                                 call_number: int = 1):
    """
    Save the ACTUAL context passed to an agent's LLM to a text file.
    
    This captures exactly what the agent sees (after filtering and cleaning).
    Called from each call_llm_* function right before invoking the LLM.
    
    Args:
        agent_id: Agent identifier
        context_messages: The EXACT list of messages passed to agent.llm.invoke()
        total_messages: Total messages in state before filtering
        output_dir: Directory to save debug files
        call_number: Number of this LLM call
    """
    try:
        debug_dir = output_dir / "debug_context"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        filename = debug_dir / f"{agent_id}_call_{call_number}.txt"
        
        # Separate system prompt, historical context, protocol messages
        system_prompt = None
        protocol_msg = None
        historical_msgs = []
        
        for msg in context_messages:
            if isinstance(msg, SystemMessage):
                if system_prompt is None:
                    # First SystemMessage is usually the agent's system prompt
                    system_prompt = msg
                elif "PHASE 1 INDIVIDUAL OBSERVATIONS" in msg.content:
                    # Phase 1 context
                    pass # Treat as historical/context
                # Other SystemMessages
            elif isinstance(msg, HumanMessage) and "POIROT PHASE 2: PEER CONSULTATION PROTOCOL" in msg.content:
                # POIROT protocol initialization message
                protocol_msg = msg
            
            # Add to historical list (everything except the main system prompt and protocol)
            if msg != system_prompt and msg != protocol_msg:
                historical_msgs.append(msg)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write(f"POIROT PROTOCOL - ACTUAL CONTEXT FOR {agent_id.upper()}\n")
            f.write(f"LLM INVOCATION #{call_number}\n")
            f.write("="*80 + "\n")
            f.write(f"Total messages in state: {total_messages}\n")
            f.write(f"Messages passed to LLM: {len(context_messages)}\n")
            f.write("="*80 + "\n\n")
            
            # Write system prompt
            if system_prompt:
                f.write("─"*80 + "\n")
                f.write("SYSTEM PROMPT\n")
                f.write("─"*80 + "\n")
                f.write(f"{system_prompt.content}\n\n")
            
            # Write protocol message
            if protocol_msg:
                f.write("─"*80 + "\n")
                f.write("POIROT PROTOCOL MESSAGE\n")
                f.write("─"*80 + "\n")
                f.write(f"{protocol_msg.content}\n\n")
            
            # Write other messages (Historical + Phase 1 + Current Conversation)
            if historical_msgs:
                f.write("─"*80 + "\n")
                f.write(f"CONVERSATION CONTEXT ({len(historical_msgs)} messages)\n")
                f.write("─"*80 + "\n\n")
                
                for idx, msg in enumerate(historical_msgs, 1):
                    if isinstance(msg, HumanMessage):
                        icon = "👤"
                        msg_type = "HUMAN"
                    elif isinstance(msg, AIMessage):
                        icon = "🤖"
                        msg_type = "AI"
                    elif isinstance(msg, ToolMessage):
                        icon = "🔧"
                        msg_type = "TOOL"
                    elif isinstance(msg, SystemMessage):
                        icon = "⚙️"
                        msg_type = "SYSTEM"
                    else:
                        icon = "❓"
                        msg_type = "UNKNOWN"
                    
                    # Get metadata
                    metadata = msg.additional_kwargs.get('metadata', {})
                    from_node = metadata.get('from_node', 'unknown')
                    to_node = metadata.get('to_node', 'unknown')
                    
                    f.write(f"{icon} Message {idx} - {msg_type}\n")
                    f.write(f"From: {from_node} → To: {to_node}\n")
                    f.write(f"{'-'*80}\n")
                    
                    # For AIMessages with tool calls, show the tool calls
                    if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                        f.write(f"Content: {msg.content}\n\n")
                        f.write(f"🔧 TOOL CALLS ({len(msg.tool_calls)}):\n")
                        for tc_idx, tool_call in enumerate(msg.tool_calls, 1):
                            f.write(f"  {tc_idx}. {tool_call.get('name', 'unknown_tool')}\n")
                            if 'args' in tool_call and tool_call['args']:
                                f.write(f"     Args: {tool_call['args']}\n")
                        f.write("\n")
                    else:
                        f.write(f"{msg.content}\n")
                        # POIROT internal note: show truncation info in our log but NOT sent to the LLM
                        trunc_info = msg.additional_kwargs.get('_poirot_truncated') if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs else None
                        if trunc_info:
                            f.write(f"\n⚠️  [POIROT LOG - NOT SENT TO AGENT] Message truncated: "
                                    f"{trunc_info['original_chars']:,} chars → {trunc_info['kept_chars']:,} chars kept "
                                    f"({trunc_info['removed_chars']:,} chars removed to fit context budget)\n")
                        f.write("\n")
            else:
                f.write("─"*80 + "\n")
                f.write("NO CONVERSATION CONTEXT\n")
                f.write("─"*80 + "\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF CONTEXT\n")
            f.write("="*80 + "\n")
            
        print(f"  📝 Context saved to {filename}")
        
    except Exception as e:
        print(f"  ⚠️ Failed to save debug context: {e}")


##############################################################################
#=================== PROTOCOL MESSAGE GENERATION ============================#
##############################################################################

def generate_error_vector_explanation(error_space: Dict[str, Any]) -> str:
    """
    Generate explanation of error vector structure from Phase 0 error space.
    
    Args:
        error_space: Dict with 'error_regions' list from Phase 0
        
    Returns:
        Formatted explanation of vector structure
    """
    if not error_space or 'error_regions' not in error_space:
        return """
ERROR VECTOR STRUCTURE:
The error vector is a binary array indicating which system components may have contributed to the hazard.
Each position represents a specific component or agent in the system.
"""
    
    error_regions = error_space['error_regions']
    
    explanation = "\nERROR VECTOR STRUCTURE:\n"
    explanation += "The error vector is a binary array where each position represents a specific component.\n"
    explanation += "Each position can be 0 (not involved) or 1 (potentially involved). You can mark multiple positions as 1, but focus on the root cause, not where the error propagates.\n\n"
    explanation += "Vector positions and their meaning:\n"
    
    for region in error_regions:
        region_id = region.get('id', '?')
        region_name = region.get('name', 'Unknown')
        region_desc = region.get('description', 'No description available')
        explanation += f"  {region_id}: {region_name}\n"
        explanation += f"      {region_desc}\n\n"
    
    example_vector = error_space.get('error_vector_example', [0] * len(error_regions))
    explanation += f"Example vector: {example_vector}\n"
    
    return explanation


def generate_location_field_instructions(error_space: Dict[str, Any]) -> str:
    """
    Generate instructions for the location field in final JSON vote.
    
    Args:
        error_space: Dict with 'error_regions' list from Phase 0
        
    Returns:
        Instructions for creating location binary vector
    """
    if not error_space or 'error_regions' not in error_space:
        return "The 'location' field must be a binary array (e.g., [0, 1, 0, 0, 1])."
    
    error_regions = error_space['error_regions']
    n = len(error_regions)
    
    instructions = f"\nLOCATION FIELD FORMAT:\n"
    instructions += f"The 'location' field must be a binary array of length {n}, where each position corresponds to:\n"
    
    for i, region in enumerate(error_regions):
        instructions += f"  Position {i}: {region.get('name', 'Unknown')} ({region.get('id', '?')})\n"
    
    instructions += "\nSet position to 1 if that component contributed to the hazard, 0 otherwise.\n"
    instructions += "Example: [0, 0, 1, 0] indicates only the 3rd component was involved.\n"
    
    return instructions


def create_phase2_protocol_message(error_space: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate the POIROT Phase 2 protocol message for peer consultation.
    
    Args:
        error_space: Error space from Phase 0 (optional)
        
    Returns:
        Complete protocol message explaining Phase 2 consultation process
    """
    vector_explanation = generate_error_vector_explanation(error_space)
    location_instructions = generate_location_field_instructions(error_space)
    
    return f"""
═══════════════════════════════════════════════════════════════════════════
POIROT PHASE 2: PEER CONSULTATION PROTOCOL
═══════════════════════════════════════════════════════════════════════════

You have completed your individual analysis in Phase 1. Now you will enter the peer consultation phase.

WHAT IS PHASE 2?
In this phase, you can communicate with other agents to:
- Validate your initial observations
- Gather additional perspectives on the incident
- Clarify ambiguous events from the session
- Build a comprehensive understanding before your final vote

HOW IT WORKS:
1. You have access to communication tools (talk_to_X) to consult with other agents
2. Ask specific questions about what they observed or did during the session
3. The POIROT server will mediate these communications to ensure orderly discussion
4. After consultation, you will provide your final hazard vector identification

IMPORTANT RULES:
 DO consult with other agents before your final report
 DO ask specific, focused questions based on evidence
 DO acknowledge if you don't have direct information
 DON'T make up facts or hallucinate information
 DON'T skip consultation - it's mandatory
 DON'T provide your final vote until you've gathered sufficient information
 DON'T repeat the same question multiple times if you don't get an answer immediately. Wait for the response.
 IF an incident is caused because a conflict between two agents happened, you must indicate both agents as responsible for the incident, not as systematic causes.

{vector_explanation}

You can indicate more than one region as responsible for the incident by marking its hazard vector element as 1. It is important to differentiate between elements that may have triggered an error and simple elements that are normal and expected in the system. In addition, it has several points of failure, so do not hesitate to indicate them all. You should point to the real cause, not be general, and indicate multiple failures.

FINAL OUTPUT FORMAT:
After consultation, your response MUST be a valid JSON object:
{{
  "hazard_vector": "Name or brief description of the identified hazard vector",
  "location": [x1, x2, x3, ...],
  "justification": "Detailed explanation based on evidence from the session"
}}

{location_instructions}



CRITICAL: Base your analysis ONLY on evidence from the session. If you didn't observe something directly, clarify this in your justification. Focus on the root cause, not where the error propagated.

Begin your consultation now using the available communication tools. If you have already asked a question, wait for the response or ask a different agent.
"""


def format_phase1_observations(phase1_reports: Dict[str, str]) -> str:
    """
    Format Phase 1 individual reports for Phase 2 context.
    
    Args:
        phase1_reports: Dict mapping agent_id to their Phase 1 report text
        
    Returns:
        Formatted context message with all Phase 1 observations
    """
    if not phase1_reports:
        return ""
    
    formatted = "\n" + "═"*80 + "\n"
    formatted += "PHASE 1 INDIVIDUAL OBSERVATIONS (for your reference)\n"
    formatted += "═"*80 + "\n\n"
    
    for agent_id, report in phase1_reports.items():
        formatted += f"── {agent_id.upper()} ──\n"
        formatted += f"{report}\n\n"
    
    formatted += "═"*80 + "\n"
    formatted += "These are the initial observations from all agents. Use them to inform your consultation.\n"
    
    return formatted


def format_vectors_to_ignore(vectors: List[str]) -> str:
    """
    Format list of known non-issues that should not be reported as hazards.
    
    Args:
        vectors: List of strings describing known non-issues
        
    Returns:
        Formatted warning message
    """
    if not vectors or len(vectors) == 0:
        return ""
    
    formatted = "\n" + "─"*80 + "\n"
    formatted += "⚠️  KNOWN NON-ISSUES - DO NOT REPORT THESE AS HAZARD VECTORS\n"
    formatted += "─"*80 + "\n"
    formatted += "The following aspects are KNOWN to be correct and should NOT be reported:\n\n"
    
    for i, vector in enumerate(vectors, 1):
        formatted += f"{i}. {vector}\n"
    
    formatted += "\n" + "─"*80 + "\n"
    formatted += "Focus your analysis on genuine anomalies, not these known characteristics.\n"
    
    return formatted


##############################################################################
#================= COMMUNICATION TOOL GENERATION ============================#
##############################################################################

class CommunicationToolFactory:
    """
    Factory for generating agent-to-agent communication tools dynamically.
    
    This class creates LangChain tools that allow agents to send messages
    to each other during Phase 2 consultation, with proper state tracking.
    """
    
    def __init__(self):
        """Initialize the factory with empty state."""
        self.pending_messages: Dict[str, str] = {}  # {target_agent_id: message_content}
        self.message_flags: Dict[str, bool] = {}    # {target_agent_id: has_pending}
        self.message_senders: Dict[str, str] = {}   # {target_agent_id: sender_agent_id}
    
    def create_tool(self, target_agent_id: str, target_agent_name: str, sender_agent_id: str):
        """
        Create a communication tool for talking to a specific agent.
        
        Args:
            target_agent_id: Database ID of target agent (e.g., "doctor")
            target_agent_name: Display name for target agent
            sender_agent_id: Database ID of agent who will use this tool
            
        Returns:
            LangChain tool for communication
        """
        # Closure captures factory state and sender
        factory = self
        sender_id = sender_agent_id
        
        @tool
        def talk_to_agent(message: str) -> str:
            """Send a message to the target agent for consultation."""
            factory.message_flags[target_agent_id] = True
            factory.pending_messages[target_agent_id] = message
            factory.message_senders[target_agent_id] = sender_id
            return f"Message sent to {target_agent_name}: {message}"
        
        # Override name and description after creation
        talk_to_agent.name = f"talk_to_{target_agent_id}"
        talk_to_agent.description = f"Send a message to {target_agent_name} for consultation during POIROT investigation."
        
        return talk_to_agent
    
    def has_pending_messages(self) -> bool:
        """Check if any communication tools have pending messages."""
        return any(self.message_flags.values())
    
    def get_pending_message(self, agent_id: str) -> Optional[str]:
        """Get pending message for specific agent, if any."""
        return self.pending_messages.get(agent_id)
    
    def get_pending_sender(self, agent_id: str) -> Optional[str]:
        """Get sender agent_id who initiated consultation to specific agent."""
        return self.message_senders.get(agent_id)
    
    def clear_flag(self, agent_id: str) -> None:
        """Clear pending message flag for specific agent."""
        self.message_flags[agent_id] = False
    
    def reset_all(self) -> None:
        """Clear all pending messages and flags."""
        self.pending_messages.clear()
        self.message_flags.clear()
        self.message_senders.clear()


##############################################################################
#====================== AGENT NODE FACTORIES ================================#
##############################################################################

def create_agent_nodes(
    agent_id: str,
    agent_data: Dict[str, Any],
    communication_tools: List[Any],
    tool_factory: CommunicationToolFactory,
    phase1_reports: Optional[Dict[str, str]],
    vectors_to_ignore: Optional[List[str]],
    error_space: Optional[Dict[str, Any]],
    historical_messages: List[BaseMessage],
    model_name: str = "gemini-2.0-flash-exp",
    output_dir: Optional[Path] = None,
    include_tool_calls: bool = False,
    include_broadcast_messages: bool = False,
    api_call_delay: float = 0.0,
    use_local_llm: bool = False,
    local_model_name: Optional[str] = None,
    token_tracker: Optional[Any] = None,
    llm_provider: str = "gemini",
    max_agent_messages: int = 8,
    max_llm_retries: int = 5,
    retry_delay_503: int = 30,
    retry_delay_429: int = 60,
    token_budget: int = 95_000,
):
    """
    Factory function to create LangGraph nodes for a single agent.
    
    Creates three node functions:
    1. call_llm_{agent_id}: Invokes LLM with filtered context
    2. take_action_{agent_id}: Executes tool calls (communication)
    3. No transition needed - handled by routing
    
    Args:
        agent_id: Unique agent identifier from database
        agent_data: Dict with 'name', 'system_prompt', 'tools_list'
        communication_tools: List of LangChain tools for this agent
        tool_factory: CommunicationToolFactory for state tracking
        phase1_reports: Phase 1 reports for context (optional)
        vectors_to_ignore: Known non-issues to exclude (optional)
        error_space: Error space from Phase 0 (optional)
        historical_messages: Messages from original session
        model_name: Gemini model to use
        output_dir: Directory to save debug context (optional)
        
    Returns:
        Tuple of (call_llm, take_action, tools_dict)
    """
    # Initialize LLM with tools using LLMFactory
    if LLMFactory is not None:
        llm = LLMFactory.create_chat_llm(
            model_name=model_name,
            use_local=use_local_llm,
            local_model_name=local_model_name,
            provider=llm_provider,
            temperature=0
        )
    else:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    llm_with_tools = llm.bind_tools(communication_tools)
    
    # Build tools dict for execution
    tools_dict = {tool.name: tool for tool in communication_tools}
    
    # Tracking counter for debug logging
    call_counter = [0]
    
    def call_llm(state: Phase2State) -> Phase2State:
        """
        Call agent's LLM with properly filtered and structured context.
        """
        call_counter[0] += 1
        all_messages = list(state['messages'])
        
        # Get consultation tracking flags from agent_states
        agent_states = state.get('agent_states', {})
        consulted = agent_states.get(f'{agent_id}_consulted', False)
        call_count = agent_states.get(f'{agent_id}_call_count', 0)
        has_final_report = agent_states.get(f'{agent_id}_final_report', False)
        
        # Check if this call will reach the message limit
        # We check call_count + 1 because we're about to make this call
        will_reach_limit = (call_count + 1) >= max_agent_messages
        force_final_vote = will_reach_limit and not has_final_report
        
        # Filter to messages where this agent participated (or broadcasts if enabled)
        filtered = filter_messages_for_agent(all_messages, agent_id, include_broadcast_messages)
        # Pass include_tool_calls and current_agent_id
        filtered = clean_historical_messages(
            filtered, 
            include_tool_calls=include_tool_calls,
            current_agent_id=agent_id
        )
        
        print(f"\n{'='*80}")
        print(f"🤖 AGENT: {agent_data['name']} ({agent_id})")
        print(f"{'='*80}")
        print(f"Total messages: {len(all_messages)}")
        print(f"Filtered for {agent_id}: {len(filtered)}")
        print(f"Consulted: {consulted}, Call count: {call_count + 1}/{max_agent_messages}")

        if will_reach_limit:
            print(f"⚠️  MESSAGE LIMIT WILL BE REACHED ({call_count + 1}/{max_agent_messages})")
            if force_final_vote:
                print(f"🚨 FORCING FINAL VOTE - This is the last message allowed!")
        
        # Trim oversized messages (e.g. large data dumps) to stay within token budget
        filtered = truncate_large_messages_if_needed(filtered, token_budget=token_budget)
        
        # Build context
        # POIROT PHASE 2 OVERRIDE:
        # We must explicitly override any role constraints that prevent the agent from performing
        # the hazard analysis. Some agents (like Portfolio Managers) may have system prompts
        # that strictly forbid "risk assessment". We need to tell them that for this specific
        # phase (Post-Incident Review), they ARE required to do this.
        phase2_override = """
\n\n═══════════════════════════════════════════════════════════════════════════
POIROT PROTOCOL OVERRIDE - READ CAREFULLY
═══════════════════════════════════════════════════════════════════════════
You are currently participating in a POST-INCIDENT REVIEW process (Phase 2).
For the duration of this phase, the following rules SUPERSEDE your normal role constraints:

1. You ARE AUTHORIZED and REQUIRED to perform "hazard analysis" and "risk assessment" of the previous session.
2. You must evaluate the behavior of ALL agents, including yourself.
3. You MUST provide a final output in JSON format defining the "hazard vector".

Ignore any previous instructions that say "I do not assess risks" or "I only do X". 
In this phase, your job is to be a CRITICAL EVALUATOR of the system's failure.
═══════════════════════════════════════════════════════════════════════════
"""
        messages = [SystemMessage(content=agent_data['system_prompt'] + phase2_override)] + filtered
        
        # Add protocol, phase1, and ignore list ONLY on first call
        if not consulted and not force_final_vote:
            protocol = create_phase2_protocol_message(error_space)
            messages.append(HumanMessage(content=protocol))
            
            if phase1_reports:
                phase1_context = format_phase1_observations(phase1_reports)
                messages.append(SystemMessage(content=phase1_context))
            
            if vectors_to_ignore and len(vectors_to_ignore) > 0:
                ignore_msg = format_vectors_to_ignore(vectors_to_ignore)
                messages.append(HumanMessage(content=ignore_msg))
        
        # If already consulted or message limit reached, request final report
        if consulted or force_final_vote:
            location_instr = generate_location_field_instructions(error_space)
            
            # Different message based on whether we're forcing due to limit
            if force_final_vote:
                final_request = f"""🚨 POIROT PROTOCOL ENFORCEMENT 🚨

You have reached the maximum number of messages allowed ({max_agent_messages}). The POIROT protocol 
requires you to provide your final vote NOW to prevent infinite loops and control resource usage.

You MUST respond with your hazard vector analysis in the following JSON format:

{location_instr}

Your response MUST be a valid JSON object with the following structure:
{{
  "hazard_vector": "Name or brief description of the identified hazard vector",
  "location": [x1, x2, x3, ...],
  "justification": "Detailed explanation of why you identified this hazard vector based on your analysis and consultation"
}}

CRITICAL: The "location" field must be a BINARY VECTOR (array of 0s and 1s), NOT a text description.
This is your FINAL message - the protocol will not accept additional responses."""
            else:
                final_request = f"""Based on your consultation with other agents, please provide your final analysis of the hazard vector in JSON format.

{location_instr}

Your response MUST be a valid JSON object with the following structure:
{{
  "hazard_vector": "Name or brief description of the identified hazard vector",
  "location": [x1, x2, x3, ...],
  "justification": "Detailed explanation of why you identified this hazard vector"
}}

CRITICAL: The "location" field must be a BINARY VECTOR (array of 0s and 1s), NOT a text description."""
            messages.append(HumanMessage(content=final_request))
        
        # DEBUG: Save context to file if output_dir is provided
        if output_dir:
            save_agent_context_to_file(
                agent_id=agent_id,
                context_messages=messages,
                total_messages=len(all_messages),
                output_dir=output_dir,
                call_number=call_counter[0]
            )

        # Apply API call delay to avoid rate limiting
        if api_call_delay > 0:
            time.sleep(api_call_delay)

        # ── Invoke LLM with retry for transient API errors ──────────────────────
        # Handles: 503 UNAVAILABLE (high demand) and 429 RESOURCE_EXHAUSTED (rate limit)
        _MAX_LLM_RETRIES = max_llm_retries
        _LLM_RETRY_DELAY_503 = retry_delay_503   # seconds — service temporarily unavailable
        _LLM_RETRY_DELAY_429 = retry_delay_429   # seconds — quota exhausted
        _llm_attempts = 0
        response = None
        while True:
            try:
                response = llm_with_tools.invoke(messages)
                break  # ✅ successful call
            except Exception as _llm_exc:
                _estr = str(_llm_exc)
                _is_503 = 'UNAVAILABLE' in _estr or '503' in _estr
                _is_429 = 'RESOURCE_EXHAUSTED' in _estr or '429' in _estr
                if (_is_503 or _is_429) and _llm_attempts < _MAX_LLM_RETRIES:
                    _llm_attempts += 1
                    _code = '503 UNAVAILABLE' if _is_503 else '429 RESOURCE_EXHAUSTED'
                    _wait = _LLM_RETRY_DELAY_503 if _is_503 else _LLM_RETRY_DELAY_429
                    print(f"\n⚠️  Transient LLM error ({_code}) for {agent_data['name']}.")
                    print(f"⏳ Pausing {_wait}s before retry "
                          f"({_llm_attempts}/{_MAX_LLM_RETRIES})...")
                    time.sleep(_wait)
                    print(f"🔄 Retrying LLM call for {agent_data['name']}...")
                else:
                    raise  # non-transient error or max retries reached
        # ────────────────────────────────────────────────────────────────────────

        # Track tokens
        if token_tracker is not None and extract_tokens_from_response is not None:
            usage = extract_tokens_from_response(response)
            token_tracker.add(usage)
        
        print(f"\n{'─'*80}")
        print(f"RESPONSE from {agent_data['name']}:")
        print(f"{'─'*80}")
        
        # Safe string conversion for content - handle Gemini 2.0 format
        if hasattr(response, 'content') and isinstance(response.content, list):
            # Handle list of content parts (Gemini format)
            parts = []
            for part in response.content:
                if isinstance(part, dict):
                    # Extract only 'text' field from dict, ignore 'extras' with signature
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
            
        print(content_str)
        print(f"{'─'*80}\n")
        
        # Detect tool calls and final report
        has_tool_calls = bool(hasattr(response, 'tool_calls') and response.tool_calls)
        
        # Check for final report - look for hazard_vector in any JSON format
        # (either in ```json block or raw JSON with hazard_vector key)
        has_final_report = (
            ('```json' in content_str and 'hazard_vector' in content_str) or
            ('"hazard_vector"' in content_str and '"location"' in content_str)
        )
        
        # Prepare state updates
        state_updates = {
            f'{agent_id}_call_count': call_count + 1
        }
        
        # Set consulted=True when tool_calls detected OR when forced due to message limit
        if has_tool_calls:
            state_updates[f'{agent_id}_consulted'] = True
            print(f"  ✅ {agent_data['name']} has consulted with peers (tool calls detected)")
        elif force_final_vote:
            # If we forced final vote due to message limit, mark as consulted
            state_updates[f'{agent_id}_consulted'] = True
            print(f"  🚨 {agent_data['name']} marked as consulted (message limit reached)")
        
        # Extract vote if final report provided OR if we forced a vote
        if has_final_report or force_final_vote:
            state_updates[f'{agent_id}_final_report'] = True
            state_updates[f'{agent_id}_consulted'] = True
            
            # Extract JSON vote using robust parser
            vote_data = extract_vote_json(content_str)
            
            if vote_data:
                state_updates[f'{agent_id}_hazard_vector'] = vote_data.get('hazard_vector', 'Unknown')
                state_updates[f'{agent_id}_hazard_location'] = vote_data.get('location', [])
                state_updates[f'{agent_id}_justification'] = vote_data.get('justification', '')
                
                vote_type = "FORCED VOTE (limit reached)" if force_final_vote else "FINAL VOTE"
                print(f"  ✅ {agent_data['name']} provided {vote_type}")
                print(f"  🎯 Hazard Vector: {vote_data.get('hazard_vector')}")
                print(f"  📍 Location: {vote_data.get('location')}")
            else:
                if force_final_vote:
                    print(f"  ⚠️ Could not extract vote JSON from forced response (JSON malformed)")
                    # For forced votes, we still need to set some data to prevent further loops
                    state_updates[f'{agent_id}_hazard_vector'] = "Error extracting forced vote - loop detected"
                    state_updates[f'{agent_id}_hazard_location'] = []
                    state_updates[f'{agent_id}_justification'] = f"Agent reached message limit ({max_agent_messages}), vote extraction failed"
                else:
                    print(f"  ⚠️ Could not extract vote JSON from response (JSON malformed)")
                    state_updates[f'{agent_id}_hazard_vector'] = "Error extracting vote"
                    state_updates[f'{agent_id}_hazard_location'] = []
                    state_updates[f'{agent_id}_justification'] = ""
        
        # Add metadata for message filtering
        add_message_metadata(response, from_node=agent_id, to_node="router")
        
        return {
            'messages': [response],
            'agent_states': state_updates
        }
    
    def take_action(state: Phase2State) -> Phase2State:
        """
        Execute tool calls from agent's response.
        """
        tool_calls = state['messages'][-1].tool_calls
        results = []
        
        print(f"\n{'🔧'*40}")
        print(f"{agent_data['name'].upper()} EXECUTING TOOLS")
        print(f"{'🔧'*40}")
        
        # Track which agents this agent wants to consult (POIROTMini pattern)
        consultation_flags = {}
        
        for idx, t in enumerate(tool_calls, 1):
            print(f"\n📌 Tool {idx}/{len(tool_calls)}: {t['name']}")
            
            if t['name'] not in tools_dict:
                print(f"   ❌ ERROR: Tool does not exist")
                result = "Incorrect tool name. Please retry with a valid communication tool."
            else:
                # Extract message argument
                tool_input = t['args'].get('message', '')
                print(f"   💬 Message: {tool_input}")
                
                # Execute tool (sets flags in tool_factory)
                result = tools_dict[t['name']].invoke(tool_input)
                print(f"   ✅ {result}")
                
                # Set talk_to_{target} flag (matches POIROTMini pattern)
                if t['name'].startswith('talk_to_'):
                    target_agent = t['name'].replace('talk_to_', '')
                    consultation_flags[f'talk_to_{target_agent}'] = True
                    print(f"   🚩 Set flag: talk_to_{target_agent} = True")
            
            # Create tool result message
            tool_msg = ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result))
            add_message_metadata(tool_msg, from_node=f"tool_{t['name']}", to_node=agent_id)
            results.append(tool_msg)
        
        print(f"\n✅ All tools executed.")
        print(f"{'🔧'*40}\n")
        
        return {
            'messages': results,
            'agent_states': consultation_flags
        }
    
    return call_llm, take_action, tools_dict


##############################################################################
#==================== POIROT SERVER MEDIATION ===============================#
##############################################################################

def create_poirot_server_node(
    agents_data: Dict[str, Dict[str, Any]],
    tool_factory: CommunicationToolFactory,
    error_space: Optional[Dict[str, Any]],
    model_name: str = "gemini-2.0-flash-exp",
    api_call_delay: float = 0.0,
    use_local_llm: bool = False,
    local_model_name: Optional[str] = None,
    token_tracker: Optional[Any] = None,
    llm_provider: str = "gemini",
    max_agent_messages: int = 8,
):
    """
    Create POIROT server node for mediating inter-agent communication.
    
    The server:
    1. Checks for pending communications (via tool_factory flags)
    2. Routes messages to target agents
    3. Invokes target agent LLMs for consultation responses
    4. Returns responses to original sender
    5. Prevents message flooding with retry logic for empty responses
    
    Args:
        agents_data: Dict mapping agent_id to agent configuration
        tool_factory: CommunicationToolFactory tracking pending messages
        error_space: Error space from Phase 0
        model_name: Gemini model to use
        
    Returns:
        poirot_server node function
    """
    # Build LLMs for consultation responses (no tools, just answers)
    consultation_llms = {}
    for agent_id, agent_info in agents_data.items():
        if LLMFactory is not None:
            llm = LLMFactory.create_chat_llm(
                model_name=model_name,
                use_local=use_local_llm,
                local_model_name=local_model_name,
                provider=llm_provider,
                temperature=0
            )
        else:
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        consultation_llms[agent_id] = llm
    
    def poirot_server(state: Phase2State) -> Phase2State:
        """
        POIROT server node: mediate inter-agent communication.
        
        Matches POIROTMini pattern: Uses tool_factory to detect pending consultations,
        processes them, then sets consulted=True and resets talk_to_ flags.
        """
        print(f"\n{'='*80}")
        print("🌐 POIROT SERVER - MEDIATING COMMUNICATIONS")
        print(f"{'='*80}\n")
        
        if not tool_factory.has_pending_messages():
            print("  No pending consultations to process.")
            return {'messages': []}
        
        all_messages = list(state['messages'])
        server_responses = []
        consulting_agent_id = None
        
        # Process each pending consultation (tool_factory tracks them)
        for target_id, has_pending in tool_factory.message_flags.items():
            if not has_pending:
                continue
            
            message_content = tool_factory.get_pending_message(target_id)
            sender_id = tool_factory.get_pending_sender(target_id)
            consulting_agent_id = sender_id  # Track who is consulting
            
            if not message_content or not sender_id:
                continue
            
            print(f"{'💬'*40}")
            print(f"CONSULTATION FROM: {agents_data.get(sender_id, {}).get('name', sender_id)} ({sender_id})")
            print(f"CONSULTATION TO: {agents_data[target_id]['name']} ({target_id})")
            print(f"{'💬'*40}")
            
            # Filter messages for target agent
            filtered = filter_messages_for_agent(all_messages, target_id)
            filtered = clean_historical_messages(filtered, current_agent_id=target_id)
            
            print(f"  📊 Context for {target_id}: {len(filtered)} messages")
            print(f"\n📥 QUERY:")
            print(f"{'─'*80}")
            print(message_content)
            print(f"{'─'*80}\n")
            
            # Build consultation context
            protocol = create_phase2_protocol_message(error_space)
            
            sender_name = agents_data.get(sender_id, {}).get('name', sender_id)
            
            consultation_intro = SystemMessage(
                content="A peer agent is consulting you as part of the POIROT investigation. Please answer their specific question. This is NOT a request for your final vote - just answer their question."
            )
            
            # Enhanced query with specific instructions for the Answerer
            query_content = (
                f"As part of the POIROT protocol, {sender_name} has asked you the following question. "
                f"Answer based on your knowledge of the system, participation, and knowledge acquired during the POIROT protocol.\n\n"
                f"**Question:**\n{message_content}"
            )
            query = HumanMessage(content=query_content)
            
            # Add metadata to query so it appears in filtered messages
            add_message_metadata(query, from_node=sender_id, to_node=target_id)
            
            context = [
                SystemMessage(content=agents_data[target_id]['system_prompt'])
            ] + filtered + [
                HumanMessage(content=protocol),
                consultation_intro,
                query
            ]
            
            # Retry loop for empty responses
            MAX_RETRIES = 3
            retry_count = 0
            response = None
            
            while retry_count < MAX_RETRIES:
                # Apply API call delay to avoid rate limiting
                if api_call_delay > 0:
                    time.sleep(api_call_delay)
                response = consultation_llms[target_id].invoke(context)
                
                # Track tokens
                if token_tracker is not None and extract_tokens_from_response is not None:
                    usage = extract_tokens_from_response(response)
                    token_tracker.add(usage)
                
                # Check for empty response - Robust for list content
                if hasattr(response, 'content') and isinstance(response.content, list):
                    parts = []
                    for part in response.content:
                        if isinstance(part, dict):
                            # Extract only 'text' field from dict, ignore 'extras' with signature
                            if 'type' in part and part['type'] == 'text' and 'text' in part:
                                parts.append(part['text'])
                            elif 'text' in part:
                                parts.append(part['text'])
                            else:
                                parts.append(str(part))
                        else:
                            parts.append(str(part))
                    response_content = "".join(parts).strip()
                elif hasattr(response, 'content'):
                     response_content = str(response.content).strip()
                else:
                     response_content = ""
                
                if response_content and len(response_content) > 0:
                    break  # Valid response
                else:
                    retry_count += 1
                    print(f"  ⚠️ EMPTY RESPONSE (Attempt {retry_count}/{MAX_RETRIES})")
                    print(f"     🔄 Retrying with insistence...")
                    
                    insistence = HumanMessage(
                        content="CRITICAL: Your previous response was empty. You MUST provide a substantive answer. If you don't have information, explain what you observed. An empty response is NOT acceptable."
                    )
                    context.append(insistence)
            
            # Safe text handling for fallback message
            if not response or (hasattr(response, 'content') and not response.content):
                 is_empty = True
            elif isinstance(response.content, str) and not response.content.strip():
                 is_empty = True
            elif isinstance(response.content, list) and not response.content:
                 is_empty = True
            else:
                 is_empty = False

            if is_empty:
                response_content = f"[{agents_data[target_id]['name']} did not provide a response after {MAX_RETRIES} attempts]"
                response = AIMessage(content=response_content)
            else:
                # Ensure we have string content for prefixing
                if isinstance(response.content, list):
                    parts = []
                    for part in response.content:
                        if isinstance(part, dict):
                            # Extract only 'text' field from dict, ignore 'extras' with signature
                            if 'type' in part and part['type'] == 'text' and 'text' in part:
                                parts.append(part['text'])
                            elif 'text' in part:
                                parts.append(part['text'])
                            else:
                                parts.append(str(part))
                        else:
                            parts.append(str(part))
                    response_content = "".join(parts)
                else:
                    response_content = str(response.content)

            # Prefix the response for the Asker
            target_name = agents_data[target_id]['name']
            prefixed_content = f"Response from {target_name}:\n{response_content}"
            response.content = prefixed_content
            
            print(f"\n📤 RESPONSE from {agents_data[target_id]['name']}:")
            print(f"{'─'*80}")
            print(prefixed_content)
            print(f"{'─'*80}\n")
            
            # Add metadata for proper routing - use actual sender_id so filtering works
            add_message_metadata(response, from_node=target_id, to_node=sender_id)
            server_responses.append(response)
            
            # Clear flag
            tool_factory.clear_flag(target_id)
        
        print(f"✅ All pending consultations processed.")
        print(f"{'='*80}\n")
        
        # Build return state
        # Reset talk_to_{agent} flags in agent_states
        agent_states_updates = {}
        current_agent_states = state.get('agent_states', {})
        
        for key in current_agent_states.keys():
            if key.startswith('talk_to_'):
                agent_states_updates[key] = False
        
        return {
            'messages': server_responses,
            'agent_states': agent_states_updates
        }
    
    return poirot_server


##############################################################################
#======================= ROUTING FUNCTIONS ==================================#
##############################################################################

def create_routing_functions(agent_id: str, tool_factory: CommunicationToolFactory, max_agent_messages: int = 8):
    """
    Create routing functions for an agent's LLM and tool nodes.
    
    Returns:
        Tuple of (route_after_llm, route_after_tools)
    """
    def route_after_llm(state: Phase2State) -> str:
        """
        Route after agent LLM call.
        - If has tool calls AND not at limit → tools_{agent_id}
        - If final report provided → next_agent (or END)
        - If at message limit but no final report yet → continue (force vote on next call)
        - Otherwise → continue (call LLM again to request final report)
        """
        last_message = state['messages'][-1]
        has_tool_calls = bool(hasattr(last_message, 'tool_calls') and last_message.tool_calls)
        
        agent_states = state.get('agent_states', {})
        has_final = agent_states.get(f'{agent_id}_final_report', False)
        call_count = agent_states.get(f'{agent_id}_call_count', 0)
        
        # Check if agent has reached message limit
        message_limit_reached = call_count >= max_agent_messages
        
        # Also check content for JSON (routing sees state before node update)
        content_str = last_message.content if hasattr(last_message, 'content') else ""
        has_json = '"hazard_vector"' in content_str and '"location"' in content_str
        
        # CRITICAL: Only move to next agent when we have a final report
        # If limit reached without final report, continue to force vote
        if has_final or has_json:
            print(f"  ✅ {agent_id} provided final vote - routing to next_agent")
            return "next_agent"
        elif has_tool_calls and not message_limit_reached:
            # Allow tool calls only if not at limit
            return "tools"
        elif message_limit_reached:
            # At limit but no final report - continue to force the vote
            print(f"  🚨 {agent_id} at message limit but no vote yet - forcing vote")
            return "continue"
        else:
            return "continue"
    
    def route_after_tools(state: Phase2State) -> str:
        """
        Route after agent tool execution.
        - If pending communications → poirot_server
        - If final report → next_agent (or END)
        - Otherwise → continue (return to LLM)
        """
        agent_states = state.get('agent_states', {})
        has_final = agent_states.get(f'{agent_id}_final_report', False)
        has_pending = tool_factory.has_pending_messages()
        
        if has_final:
            print(f"  ✅ {agent_id} provided final vote after tools - routing to next_agent")
            return "next_agent"
        elif has_pending:
            return "poirot_server"
        else:
            return "continue"
    
    return route_after_llm, route_after_tools


def create_global_router(agent_order: List[str]):
    """
    Create global router to determine which agent is currently evaluating.
    
    Enhanced with loop prevention: automatically skips agents who have
    completed their final report.
    
    Args:
        agent_order: List of agent IDs in evaluation order
        
    Returns:
        Tuple of (check_current_agent, advance_to_next_agent)
    """
    current_index = [0]  # Mutable for closure
    
    def check_current_agent(state: Phase2State) -> str:
        """
        Return the current agent's node name.
        
        Automatically advances past agents who have already completed
        their final reports to prevent infinite loops.
        
        NOTE: We only skip agents with final_report=True, NOT agents
        who have reached the message limit. Those agents still need
        to provide their vote.
        """
        agent_states = state.get('agent_states', {})
        max_attempts = len(agent_order)  # Prevent infinite loop in router itself
        attempts = 0
        
        while current_index[0] < len(agent_order) and attempts < max_attempts:
            agent_id = agent_order[current_index[0]]
            
            # Only check if this agent has completed their final report
            has_final_report = agent_states.get(f'{agent_id}_final_report', False)
            
            if has_final_report:
                print(f"  ⏭️  Skipping {agent_id} - already provided final report")
                current_index[0] += 1
                attempts += 1
                continue
            else:
                # This agent still needs to complete their analysis
                return f"llm_{agent_id}"
        
        # All agents completed
        return "END"
    
    def advance_to_next_agent():
        """Move to next agent in sequence."""
        current_index[0] += 1
    
    return check_current_agent, advance_to_next_agent


##############################################################################
#==================== MAIN PHASE 2 EXECUTION ================================#
##############################################################################

def execute_phase2_analysis(
    session_id: int,
    db_path: Path,
    phase1_reports: Optional[Dict[str, str]] = None,
    vectors_to_ignore: Optional[List[str]] = None,
    error_space: Optional[Dict[str, Any]] = None,
    model_name: str = "gemini-2.0-flash-exp",
    output_dir: Optional[Path] = None,
    recursion_limit: int = 200,
    include_tool_calls: bool = False,
    include_broadcast_messages: bool = False,
    api_call_delay: float = 0.0,
    use_local_llm: bool = False,
    local_model_name: Optional[str] = None,
    token_tracker: Optional[Any] = None,
    llm_provider: str = "gemini",
    max_agent_messages: int = 8,
    max_llm_retries: int = 5,
    retry_delay_503: int = 30,
    retry_delay_429: int = 60,
    token_budget: int = 95_000,
) -> Dict[str, Any]:
    """
    Execute POIROT Phase 2: Peer Consultation Protocol.
    
    This function:
    1. Loads agents and their communication relationships from database
    2. Loads historical messages from the original session
    3. Builds LangGraph with communication tools for each agent
    4. Executes consultation process with POIROT server mediation
    5. Collects final votes from all agents
    6. Saves outputs to files
    
    Args:
        session_id: Database ID of the session to analyze
        db_path: Path to SQLite database
        phase1_reports: Phase 1 individual reports (optional)
        vectors_to_ignore: Known non-issues to exclude (optional)
        error_space: Error space from Phase 0 (optional)
        model_name: Gemini model to use
        output_dir: Directory for output files (default: POIROT_output/phase2/)
        recursion_limit: Maximum graph recursion depth (default: 200)
        
    Returns:
        Dict with:
        - votes: Dict mapping agent_id to vote data
        - state: Final LangGraph state
        - output_files: List of created output file paths
    """
    print(f"\n{'='*80}")
    print("POIROT PHASE 2: PEER CONSULTATION PROTOCOL")
    print(f"{'='*80}")
    print(f"Session ID: {session_id}")
    print(f"Database: {db_path}")
    print(f"Model: {model_name}")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path("POIROT_output") / "phase2" / f"session_{session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Load agents and their relationships from database
    # Use smart loading to get only relevant agents for this session:
    # - Agents that participated in the session
    # - For missing agent types, the most frequent instance across all sessions
    # ═══════════════════════════════════════════════════════════════════════
    print("📚 STEP 1: Loading agents from database...")
    
    agents_data, agent_order = get_agents_for_session(
        db_path=str(db_path),
        session_id=session_id
    )
    
    # Validate: ensure we have exactly ONE agent per type (no duplicates)
    if not validate_one_agent_per_type(agents_data):
        raise ValueError(
            "Agent loading failed: Multiple instances of same agent type detected. "
            "This indicates a bug in the session agent loader."
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Load historical messages from database
    # ═══════════════════════════════════════════════════════════════════════
    print("📜 STEP 2: Loading historical messages...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT message_type, content, from_agent_id, to_agent_id, sequence_number, timestamp
        FROM messages
        WHERE session_id = ?
        ORDER BY sequence_number
    """, (session_id,))
    
    historical_messages = []
    
    for row in cursor.fetchall():
        message_type, content, from_agent_id, to_agent_id, seq_num, timestamp = row
        
        # Convert to LangChain message based on message_type
        # We use HumanMessage as a generic container for now.
        # The specific type (AI vs Human) will be determined relative to the viewer in filter_messages_for_agent
        if message_type == 'system':
            msg = SystemMessage(content=content)
        else:
            msg = HumanMessage(content=content)
        
        # Add metadata for filtering
        # IMPORTANT: For historical messages, we must ensure from_node and to_node are set
        # even if they are None in the database (e.g. broadcast messages)
        sender = from_agent_id if from_agent_id else "unknown"
        receiver = to_agent_id if to_agent_id else "all"
        
        if not hasattr(msg, 'additional_kwargs'):
            msg.additional_kwargs = {}
        if 'metadata' not in msg.additional_kwargs:
            msg.additional_kwargs['metadata'] = {}
            
        msg.additional_kwargs['metadata']['from_node'] = sender
        msg.additional_kwargs['metadata']['to_node'] = receiver
        msg.additional_kwargs['metadata']['timestamp'] = timestamp
        
        historical_messages.append(msg)
    
    conn.close()
    
    print(f"  ✅ Loaded {len(historical_messages)} historical messages")
    print()
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Create communication tools for each agent
    # ═══════════════════════════════════════════════════════════════════════
    print("🔧 STEP 3: Creating communication tools...")
    
    tool_factory = CommunicationToolFactory()
    agent_tools = {}  # {agent_id: [tool1, tool2, ...]}
    
    for agent_id, agent_data in agents_data.items():
        tools = []
        
        # PHASE 2 ENHANCEMENT: If can_communicate_with is empty, allow communication with ALL other agents
        # This is essential for the peer consultation protocol where all agents need to discuss
        targets = agent_data['can_communicate_with']
        if not targets:
            # Allow communication with all other agents (except self)
            targets = [aid for aid in agents_data.keys() if aid != agent_id]
            print(f"  ℹ️  {agent_data['name']}: No communication targets defined, enabling ALL peers")
        
        for target_id in targets:
            if target_id in agents_data and target_id != agent_id:
                tool = tool_factory.create_tool(target_id, agents_data[target_id]['name'], agent_id)
                tools.append(tool)
        
        agent_tools[agent_id] = tools
        
        tool_names = [t.name for t in tools]
        print(f"  ✅ {agent_data['name']}: {len(tools)} tools ({', '.join(tool_names) if tool_names else 'none'})")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Create agent nodes for LangGraph
    # ═══════════════════════════════════════════════════════════════════════
    print("🏗️  STEP 4: Creating LangGraph nodes...")
    
    agent_nodes = {}  # {agent_id: {'call_llm': fn, 'take_action': fn, 'tools_dict': dict}}
    
    for agent_id, agent_data in agents_data.items():
        call_llm, take_action, tools_dict = create_agent_nodes(
            agent_id=agent_id,
            agent_data=agent_data,
            communication_tools=agent_tools[agent_id],
            tool_factory=tool_factory,
            phase1_reports=phase1_reports,
            vectors_to_ignore=vectors_to_ignore,
            error_space=error_space,
            historical_messages=historical_messages,
            model_name=model_name,
            output_dir=output_dir,
            include_tool_calls=include_tool_calls,
            include_broadcast_messages=include_broadcast_messages,
            api_call_delay=api_call_delay,
            use_local_llm=use_local_llm,
            local_model_name=local_model_name,
            token_tracker=token_tracker,
            llm_provider=llm_provider,
            max_agent_messages=max_agent_messages,
            max_llm_retries=max_llm_retries,
            retry_delay_503=retry_delay_503,
            retry_delay_429=retry_delay_429,
            token_budget=token_budget,
        )

        agent_nodes[agent_id] = {
            'call_llm': call_llm,
            'take_action': take_action,
            'tools_dict': tools_dict
        }
        
        print(f"  ✅ Created nodes for {agent_data['name']}")
    
    print()
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Create POIROT server and routing functions
    # ═══════════════════════════════════════════════════════════════════════
    print("🌐 STEP 5: Creating POIROT server and routing...")
    
    poirot_server = create_poirot_server_node(
        agents_data=agents_data,
        tool_factory=tool_factory,
        error_space=error_space,
        model_name=model_name,
        api_call_delay=api_call_delay,
        use_local_llm=use_local_llm,
        local_model_name=local_model_name,
        token_tracker=token_tracker,
        llm_provider=llm_provider,
        max_agent_messages=max_agent_messages,
    )

    # Create routing functions for each agent
    agent_routers = {}
    for agent_id in agent_order:
        route_llm, route_tools = create_routing_functions(agent_id, tool_factory, max_agent_messages=max_agent_messages)
        agent_routers[agent_id] = {
            'route_after_llm': route_llm,
            'route_after_tools': route_tools
        }
    
    print(f"  ✅ POIROT server created")
    print(f"  ✅ Routing functions created for {len(agent_routers)} agents")
    print()
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Build LangGraph StateGraph
    # ═══════════════════════════════════════════════════════════════════════
    print("🏗️  STEP 6: Building LangGraph StateGraph...")
    
    graph = StateGraph(Phase2State)
    
    # Add agent nodes
    for agent_id, nodes in agent_nodes.items():
        graph.add_node(f"llm_{agent_id}", nodes['call_llm'])
        graph.add_node(f"tools_{agent_id}", nodes['take_action'])
    
    # Add POIROT server node
    graph.add_node("poirot_server", poirot_server)
    
    # Set entry point (first agent in order)
    graph.set_entry_point(f"llm_{agent_order[0]}")
    
    # Add edges for each agent
    for i, agent_id in enumerate(agent_order):
        router = agent_routers[agent_id]
        
        # Next agent or END
        if i < len(agent_order) - 1:
            next_node = f"llm_{agent_order[i+1]}"
        else:
            next_node = END
        
        # LLM routing
        graph.add_conditional_edges(
            f"llm_{agent_id}",
            router['route_after_llm'],
            {
                "tools": f"tools_{agent_id}",
                "next_agent": next_node,
                "continue": f"llm_{agent_id}"
            }
        )
        
        # Tool routing
        graph.add_conditional_edges(
            f"tools_{agent_id}",
            router['route_after_tools'],
            {
                "poirot_server": "poirot_server",
                "next_agent": next_node,
                "continue": f"llm_{agent_id}"
            }
        )
    
    # POIROT server routing (back to agents)
    def route_from_server(state: Phase2State) -> str:
        """Route from POIROT server back to the agent who made the consultation."""
        # Find which agent has messages in flight by checking metadata
        # The last message from server should have to_node = sender_agent_id
        last_msg = state['messages'][-1] if state['messages'] else None
        
        if last_msg and hasattr(last_msg, 'additional_kwargs'):
            metadata = last_msg.additional_kwargs.get('metadata', {})
            to_node = metadata.get('to_node', '')
            
            # to_node now contains the actual sender agent_id
            if to_node and to_node in agent_order:
                return f"llm_{to_node}"
        
        # Fallback: return to first agent
        return f"llm_{agent_order[0]}"
    
    # Build routing map for POIROT server
    server_routing = {f"llm_{agent_id}": f"llm_{agent_id}" for agent_id in agent_order}
    
    graph.add_conditional_edges(
        "poirot_server",
        route_from_server,
        server_routing
    )
    
    # Compile graph
    compiled_graph = graph.compile()
    
    print(f"  ✅ StateGraph built successfully")
    print(f"     - {len(agent_order)} agent nodes")
    print(f"     - 1 POIROT server node")
    print(f"     - Entry point: {agent_order[0]}")
    print()
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: Initialize state and execute graph
    # ═══════════════════════════════════════════════════════════════════════
    print("🚀 STEP 7: Executing Phase 2 consultation process...")
    print(f"{'='*80}\n")
    
    # Initialize state
    initial_state = {
        'messages': historical_messages,
        'session_id': session_id,
        'agent_states': {}
    }
    
    # Add tracking flags for each agent
    for agent_id in agent_order:
        initial_state['agent_states'][f'{agent_id}_consulted'] = False
        initial_state['agent_states'][f'{agent_id}_final_report'] = False
        initial_state['agent_states'][f'{agent_id}_call_count'] = 0
        initial_state['agent_states'][f'{agent_id}_hazard_vector'] = ""
        initial_state['agent_states'][f'{agent_id}_hazard_location'] = []
        initial_state['agent_states'][f'{agent_id}_justification'] = ""
    
    # Execute graph with retry logic for rate limiting
    final_state = initial_state
    max_rate_limit_retries = 5
    rate_limit_wait_seconds = 60  # 1 minute wait on rate limit
    
    rate_limit_retry_count = 0
    execution_complete = False
    
    while not execution_complete and rate_limit_retry_count < max_rate_limit_retries:
        try:
            # Use stream to capture state updates in case of interruption/recursion limit
            for output in compiled_graph.stream(initial_state, {"recursion_limit": recursion_limit}):
                for node_name, state_update in output.items():
                    # Update our local tracking of the state
                    # Note: This is a simple merge. For complex state, deep merge might be needed.
                    if isinstance(state_update, dict):
                        if 'messages' in state_update:
                            # Append messages
                            final_state['messages'].extend(state_update['messages'])
                        
                        if 'agent_states' in state_update:
                            final_state['agent_states'].update(state_update['agent_states'])
                            
                        if 'next_agent' in state_update:
                            final_state['next_agent'] = state_update['next_agent']
                            
                        if 'loop_count' in state_update:
                            final_state['loop_count'] = state_update['loop_count']
            
            execution_complete = True  # Successfully completed
            
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error (429 RESOURCE_EXHAUSTED)
            _is_rate_limit = 'RESOURCE_EXHAUSTED' in error_str or '429' in error_str
            # Check if it's a transient availability error (503 UNAVAILABLE)
            # Note: these should normally be retried inside call_llm; this outer
            # catch is a safety net for errors that bubble all the way up.
            _is_unavailable = 'UNAVAILABLE' in error_str or '503' in error_str

            if _is_rate_limit or _is_unavailable:
                rate_limit_retry_count += 1
                if rate_limit_retry_count < max_rate_limit_retries:
                    _code   = '429 RESOURCE_EXHAUSTED' if _is_rate_limit else '503 UNAVAILABLE'
                    _wait   = rate_limit_wait_seconds if _is_rate_limit else 30
                    print(f"\n⚠️  Transient API error reached graph level ({_code})")
                    print(f"⏳ Waiting {_wait}s before retry "
                          f"({rate_limit_retry_count}/{max_rate_limit_retries})...")
                    time.sleep(_wait)
                    print("🔄 Resuming graph execution...")
                else:
                    print(f"\n❌ Max retries ({max_rate_limit_retries}) exceeded. Proceeding with partial results...")
                    execution_complete = True
            else:
                # Other error (e.g., recursion limit)
                print(f"\n⚠️  Graph execution interrupted (likely recursion limit): {e}")
                print("⚠️  Proceeding with partial results...")
                execution_complete = True  # Exit loop
                # We continue with whatever final_state we have accumulated
    
    print(f"\n{'='*80}")
    print("✅ Phase 2 consultation completed successfully")
    print(f"{'='*80}\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8: Extract votes and save outputs
    # ═══════════════════════════════════════════════════════════════════════
    print("💾 STEP 8: Extracting votes and saving outputs...")
    
    votes = {}
    output_files = []
    
    for agent_id in agent_order:
        agent_states = final_state.get('agent_states', {})
        hazard_vector = agent_states.get(f'{agent_id}_hazard_vector', '')
        location = agent_states.get(f'{agent_id}_hazard_location', [])
        justification = agent_states.get(f'{agent_id}_justification', '')
        
        votes[agent_id] = {
            'agent_name': agents_data[agent_id]['name'],
            'hazard_vector': hazard_vector,
            'location': location,
            'justification': justification
        }
        
        print(f"\n  📋 {agents_data[agent_id]['name']} ({agent_id}):")
        print(f"     Hazard Vector: {hazard_vector}")
        print(f"     Location: {location}")
        print(f"     Justification: {justification}")
    
    # Save votes JSON
    votes_file = output_dir / "phase2_votes.json"
    with open(votes_file, 'w', encoding='utf-8') as f:
        json.dump(votes, f, indent=2, ensure_ascii=False)
    output_files.append(votes_file)
    print(f"\n  ✅ Saved votes to: {votes_file}")
    
    # Save full state JSON
    state_file = output_dir / "phase2_state.json"
    state_serializable = {
        'session_id': final_state.get('session_id'),
        'votes': votes,
        'agent_tracking': {
            agent_id: {
                'consulted': final_state.get(f'{agent_id}_consulted', False),
                'final_report': final_state.get(f'{agent_id}_final_report', False),
                'call_count': final_state.get(f'{agent_id}_call_count', 0)
            }
            for agent_id in agent_order
        }
    }
    
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state_serializable, f, indent=2, ensure_ascii=False)
    output_files.append(state_file)
    print(f"  ✅ Saved state to: {state_file}")
    
    # Save summary report
    summary_file = output_dir / "phase2_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("═"*80 + "\n")
        f.write("POIROT PHASE 2: PEER CONSULTATION SUMMARY\n")
        f.write("═"*80 + "\n\n")
        f.write(f"Session ID: {session_id}\n")
        f.write(f"Database: {db_path}\n")
        f.write(f"Agents Consulted: {len(votes)}\n\n")
        
        f.write("─"*80 + "\n")
        f.write("HAZARD VECTOR VOTES\n")
        f.write("─"*80 + "\n\n")
        
        for agent_id, vote_data in votes.items():
            f.write(f"Agent: {vote_data['agent_name']} ({agent_id})\n")
            f.write(f"Hazard Vector: {vote_data['hazard_vector']}\n")
            f.write(f"Location: {vote_data['location']}\n")
            f.write(f"Justification:\n{vote_data['justification']}\n")
            f.write("\n" + "─"*80 + "\n\n")
    
    output_files.append(summary_file)
    print(f"  ✅ Saved summary to: {summary_file}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 9: Perform Weighted Voting Analysis
    # ═══════════════════════════════════════════════════════════════════════
    voting_results = None
    if weighted_voting_analysis and error_space:
        print("\n🗳️ STEP 9: Performing Weighted Voting Analysis...")
        
        # Prepare agent_outputs list
        agent_outputs = []
        for agent_id, vote_data in votes.items():
            agent_outputs.append({
                "agent_name": agent_id,  # Use ID for matching with error_regions
                "hazard_vector": vote_data['hazard_vector'],
                "location": vote_data['location'],
                "justification": vote_data['justification']
            })
            
        # Prepare POIROT preanalysis format
        poirot_preanalysis = {
            "system_name": error_space.get('system_name', 'Unknown System'),
            "error_regions": error_space.get('error_regions', []),
            "error_vector_example": error_space.get('error_vector_example', [])
        }
        
        # Run analysis
        try:
            voting_results = weighted_voting_analysis(agent_outputs, poirot_preanalysis)
            
            # Save results
            voting_file = output_dir / "phase2_voting_results.json"
            with open(voting_file, 'w', encoding='utf-8') as f:
                json.dump(voting_results, f, indent=2, ensure_ascii=False)
            output_files.append(voting_file)
            print(f"  ✅ Saved voting analysis to: {voting_file}")
            
            # Print detailed summary (POIROTMini style)
            if 'winning_location' in voting_results:
                winner = voting_results['winning_location']
                
                print("\n" + "="*80)
                print("📊 VOTING RESULTS")
                print("="*80)
                
                # Get dimension ID safely
                dim_idx = winner.get('dimension_index', -1)
                error_regions = poirot_preanalysis.get('error_regions', [])
                dim_id = error_regions[dim_idx]['id'] if 0 <= dim_idx < len(error_regions) else "?"
                
                print("\n🏆 WINNING LOCATION:")
                print(f"   Component: {winner['name']}")
                print(f"   Dimension: {dim_id}")
                print(f"   Total Score: {winner['total_score']:.4f}")
                print(f"   Percentage: {winner['percentage']}%")
                
                print("\n📋 INDIVIDUAL AGENT VOTES:")
                
                # Map agent IDs to readable names for display
                agent_id_to_name = {aid: adata['name'] for aid, adata in agents_data.items()}
                
                # Use agent_votes from voting_results which has the detailed info
                for vote in voting_results.get('agent_votes', []):
                    agent_id = vote['agent_name']
                    display_name = agent_id_to_name.get(agent_id, agent_id)
                    
                    print(f"\n   🔹 {display_name}:")
                    print(f"      Voted for: {vote['voted_location']}")
                    print(f"      Hazard identified: {vote['hazard_vector']}")
                    print(f"      Vote weight: {vote['vote_weight']:.4f}")
                    print(f"      Similarity to own position: {vote['similarity_to_self']:.4f}")
                
                print("\n" + "="*80)
                print("📈 PROBABILITY BY ERROR DIMENSION")
                print("="*80)
                print("\nEach dimension represents a potential error source.")
                print("Probabilities show likelihood that error originated there:\n")
                
                total_prob = 0
                # Use dimension_rankings which has the scores and percentages
                for rank in voting_results.get('dimension_rankings', []):
                    idx = rank['dimension_index']
                    name = rank['name']
                    score = rank['score']
                    percentage = rank['percentage']
                    total_prob += percentage
                    
                    # Get ID
                    region_id = error_regions[idx]['id'] if 0 <= idx < len(error_regions) else "?"
                    
                    # Create visual bar
                    bar_length = int(percentage / 2)  # 50 chars = 100%
                    bar = "█" * bar_length + "░" * (50 - bar_length)
                    
                    print(f"\n📍 {name} ({region_id})")
                    print(f"   {bar} {percentage:.1f}%")
                    print(f"   Weighted votes: {score:.4f}")
                
                print(f"\n✓ Total probability: {total_prob:.1f}% (should be ~100%)")
                print("\n" + "="*80)
            
        except Exception as e:
            print(f"  ❌ Error during voting analysis: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("🎉 Phase 2 Protocol completed successfully!")
    print(f"{'='*80}\n")
    
    # Calculate Token Usage from Final State Messages
    phase2_metadata = {
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'total_tokens': 0,
        'by_agent': {}
    }
    
    try:
        final_messages = final_state.get('messages', [])
        for msg in final_messages:
            # Check for usage metadata
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            
            has_usage = False
            if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                input_tokens = msg.usage_metadata.get('input_tokens', 0)
                output_tokens = msg.usage_metadata.get('output_tokens', 0)
                total_tokens = msg.usage_metadata.get('total_tokens', 0)
                has_usage = True
            elif hasattr(msg, 'response_metadata'):
                 usage = msg.response_metadata.get('token_usage', {}) or msg.response_metadata.get('usage', {})
                 if usage:
                    input_tokens = usage.get('input_tokens', usage.get('prompt_token_count', 0))
                    output_tokens = usage.get('output_tokens', usage.get('candidates_token_count', 0))
                    total_tokens = usage.get('total_tokens', usage.get('total_token_count', 0))
                    has_usage = True
            
            if has_usage:
                phase2_metadata['total_input_tokens'] += input_tokens
                phase2_metadata['total_output_tokens'] += output_tokens
                phase2_metadata['total_tokens'] += total_tokens
                
                # Attribute to agent
                from_node = "unknown"
                if hasattr(msg, 'additional_kwargs'):
                    from_node = msg.additional_kwargs.get('metadata', {}).get('from_node', "unknown")
                
                if from_node not in phase2_metadata['by_agent']:
                    phase2_metadata['by_agent'][from_node] = {
                        'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'calls': 0
                    }
                
                phase2_metadata['by_agent'][from_node]['input_tokens'] += input_tokens
                phase2_metadata['by_agent'][from_node]['output_tokens'] += output_tokens
                phase2_metadata['by_agent'][from_node]['total_tokens'] += total_tokens
                phase2_metadata['by_agent'][from_node]['calls'] += 1

    except Exception as e:
        print(f"⚠️ Error calculating token usage: {e}")

    return {
        'votes': votes,
        'state': final_state,
        'output_files': output_files,
        'voting_results': voting_results,
        'metadata': phase2_metadata
    }
