"""
POIROT-SW Agent Factory
=======================

This module is responsible for:
1. Loading agent configurations from the POIROT database
2. Creating LangChain agent instances with proper LLM configuration
3. Processing and enriching messages with routing metadata
4. Managing agent communication tools

The AgentFactory bridges the gap between database schema and executable LangChain agents,
enabling POIROT to work with ANY multi-agent system.

Author: POIROT-SW Team
Date: December 2025
"""

import sqlite3
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool, StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI

from .llm_factory import LLMFactory


##############################################################################
#======================== MODULE-LEVEL UTILITIES ==========================#
##############################################################################

def clean_conversational_messages(
    processed_messages: List["ProcessedMessage"],
    include_tool_calls: bool = False,
    current_agent_id: Optional[str] = None,
    communication_tool_names: Optional[set] = None,
) -> List:
    """
    Filter a list of ProcessedMessage objects down to conversational context only.

    Args:
        processed_messages: List of ProcessedMessage objects to filter.
        include_tool_calls: If True, keep non-communication tool calls/results
            that belong to current_agent_id.
        current_agent_id: ID of the agent being analyzed (used for tool filtering).
        communication_tool_names: Set of tool names considered inter-agent
            communication (e.g. talk_to_*). Pass an empty set when no such
            tools exist (e.g. raw LangChain agents).

    Returns:
        Filtered list of LangChain BaseMessage objects.
    """
    if communication_tool_names is None:
        communication_tool_names = set()

    cleaned = []
    seen_messages = set()

    for pm in processed_messages:
        msg = pm.message

        if pm.msg_type in ["human", "system"]:
            content_hash = hash(msg.content) if msg.content else 0
            msg_key = (content_hash, pm.msg_type, pm.from_agent, pm.to_agent)
            if msg_key not in seen_messages:
                seen_messages.add(msg_key)
                cleaned.append(msg)
            continue

        is_my_tool_usage = (current_agent_id and pm.from_agent == current_agent_id)

        if pm.msg_type == "ai":
            has_content = msg.content and msg.content.strip()
            is_comm_tool = pm.is_tool_call and pm.tool_name in communication_tool_names
            keep_as_tool_call = pm.is_tool_call and include_tool_calls and is_my_tool_usage
            if has_content or is_comm_tool or keep_as_tool_call:
                content_hash = hash(msg.content) if msg.content else 0
                msg_key = (content_hash, pm.msg_type, pm.from_agent)
                if msg_key not in seen_messages:
                    seen_messages.add(msg_key)
                    cleaned.append(msg)
            continue

        if pm.msg_type == "tool":
            is_comm_tool = pm.tool_name in communication_tool_names
            is_tool_result_for_me = (current_agent_id and pm.to_agent == current_agent_id)
            keep_as_tool_result = include_tool_calls and is_tool_result_for_me
            if is_comm_tool or keep_as_tool_result:
                content_hash = hash(msg.content) if msg.content else 0
                msg_key = (content_hash, pm.msg_type, pm.tool_name)
                if msg_key not in seen_messages:
                    seen_messages.add(msg_key)
                    cleaned.append(msg)

    return cleaned


##############################################################################
#======================== DATA STRUCTURES ================================#
##############################################################################

@dataclass
class AgentConfig:
    """
    Configuration for a single agent loaded from database.
    
    Attributes:
        agent_id: Unique identifier for the agent
        agent_name: Human-readable name
        agent_type: Category (e.g., "decision_maker", "analyst")
        system_name: Name of the multi-agent system
        system_prompt: Complete system prompt defining agent's role
        llm_model: LLM model identifier (e.g., "gemini-2.5-pro")
        temperature: LLM temperature setting
        max_tokens: Maximum tokens for LLM responses
        tools: List of tool definitions for this agent
        can_communicate_with: List of agent_ids this agent can talk to
    """
    agent_id: str
    agent_name: str
    agent_type: str
    system_name: str
    system_prompt: str
    llm_model: str
    temperature: float
    max_tokens: int
    tools: List[Dict[str, Any]]
    can_communicate_with: List[str]


@dataclass
class ProcessedMessage:
    """
    Enriched message with routing metadata for POIROT analysis.
    
    Attributes:
        message: Original LangChain message object
        from_agent: Source agent identifier
        to_agent: Destination agent identifier (None for user messages)
        msg_type: Message type ("human", "ai", "tool", "system")
        is_tool_call: True if this is an AI message with tool_calls
        tool_name: Name of tool if is_tool_call or ToolMessage
        is_communication: True if this is inter-agent communication
    """
    message: BaseMessage
    from_agent: Optional[str]
    to_agent: Optional[str]
    msg_type: str
    is_tool_call: bool
    tool_name: Optional[str]
    is_communication: bool


##############################################################################
#======================== AGENT FACTORY CLASS =============================#
##############################################################################

class AgentFactory:
    """
    Factory for creating LangChain agents from POIROT database configuration.
    
    This class handles:
    - Loading agent configurations from SQLite database
    - Creating LangChain LLM instances with proper settings
    - Generating communication tools for inter-agent dialogue
    - Processing messages with metadata for routing
    - Filtering messages relevant to specific agents
    
    Usage:
        factory = AgentFactory(database_path="system.db")
        agents = factory.create_all_agents()
        
        # Use agents
        doctor_agent = agents["doctor"]
        response = doctor_agent["llm"].invoke(messages)
        
        # Process messages
        processed = factory.process_messages(raw_messages)
    """
    
    def __init__(
        self,
        database_path: str,
        api_key: Optional[str] = None,
        use_local_llm: bool = False,
        local_model_name: Optional[str] = None,
        llm_provider: str = "gemini",
        ollama_num_ctx: int = 131072,
        model_override: Optional[str] = None,
    ):
        """
        Initialize the Agent Factory.

        Args:
            database_path: Path to SQLite database with agent configurations
            api_key: Google API key for Gemini models (or set GOOGLE_API_KEY env var)
            use_local_llm: If True, use LM Studio instead of Gemini API
            local_model_name: Model name for LM Studio (default: gpt-oss-20b)
            llm_provider: LLM provider - "gemini", "deepseek", "local", or "ollama" (default: "gemini")
            ollama_num_ctx: Context window size passed to Ollama (default: 131072 = 128K).
                            Only relevant when llm_provider="ollama".
            model_override: When set, all agents use this model instead of the DB value.
                            Useful to force a specific model from the CLI without editing the DB.

        Raises:
            FileNotFoundError: If database file doesn't exist
            ValueError: If database schema is invalid or API key missing (when not using local)
        """
        self.database_path = Path(database_path)
        self.use_local_llm = use_local_llm
        self.local_model_name = local_model_name
        self.llm_provider = llm_provider
        self.ollama_num_ctx = ollama_num_ctx
        self.model_override = model_override
        
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {database_path}")
        
        # Set API key into the correct environment variable for the provider
        if api_key:
            if llm_provider == "deepseek":
                os.environ["DEEPSEEK_API_KEY"] = api_key
            else:
                os.environ["GOOGLE_API_KEY"] = api_key
        
        # Verify API key is available (only required if not using local LLM, DeepSeek, or Ollama)
        if not use_local_llm and llm_provider == "gemini" and "GOOGLE_API_KEY" not in os.environ:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter. Alternatively, use use_local_llm=True for local LLM, llm_provider='deepseek', or llm_provider='ollama'."
            )
        
        if llm_provider == "deepseek" and "DEEPSEEK_API_KEY" not in os.environ:
            raise ValueError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable in your .env file."
            )
        
        # Load agent configurations from database
        self.agent_configs = self._load_agent_configs()
        
        # Communication tool names (used to identify inter-agent messages)
        self.communication_tool_names = set()
        
        print(f"✅ AgentFactory initialized: {len(self.agent_configs)} agents loaded")
    
    def _load_agent_configs(self) -> Dict[str, AgentConfig]:
        """
        Load all agent configurations from database.
        
        Returns:
            Dict mapping agent_id to AgentConfig objects
        
        Raises:
            sqlite3.Error: If database query fails
        """
        configs = {}
        
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Query agents table
            cursor.execute("""
                SELECT
                    agent_id, agent_name, agent_type, system_name,
                    system_prompt, llm_model, temperature, max_tokens,
                    has_tools, can_communicate_with
                FROM agents
            """)
            
            agents = cursor.fetchall()
            
            for agent_row in agents:
                agent_id = agent_row["agent_id"]
                
                # Load tools for this agent
                tools = []
                if agent_row["has_tools"]:
                    cursor.execute("""
                        SELECT tool_name, tool_description, tool_schema, tool_code, is_enabled
                        FROM agent_tools
                        WHERE agent_id = ? AND is_enabled = 1
                    """, (agent_id,))
                    
                    tool_rows = cursor.fetchall()
                    for tool_row in tool_rows:
                        tools.append({
                            "name": tool_row["tool_name"],
                            "description": tool_row["tool_description"],
                            "schema": json.loads(tool_row["tool_schema"]) if tool_row["tool_schema"] else {},
                            "code": tool_row["tool_code"]
                        })
                
                # Parse can_communicate_with JSON
                can_communicate_with = []
                if agent_row["can_communicate_with"]:
                    can_communicate_with = json.loads(agent_row["can_communicate_with"])
                
                # Create AgentConfig
                config = AgentConfig(
                    agent_id=agent_id,
                    agent_name=agent_row["agent_name"],
                    agent_type=agent_row["agent_type"] or "agent",
                    system_name=agent_row["system_name"],
                    system_prompt=agent_row["system_prompt"] or "",
                    llm_model=self.model_override or agent_row["llm_model"] or "gemini-2.5-pro",
                    temperature=agent_row["temperature"] if agent_row["temperature"] is not None else 0,
                    max_tokens=agent_row["max_tokens"] or 8000,
                    tools=tools,
                    can_communicate_with=can_communicate_with
                )
                
                configs[agent_id] = config
            
            print(f"📊 Loaded {len(configs)} agent configurations from database")
            
        finally:
            conn.close()
        
        return configs
    
    def create_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        Create LangChain agent instances for all agents in database.
        
        Returns:
            Dict mapping agent_id to agent data:
            {
                "agent_id": {
                    "config": AgentConfig object,
                    "llm": ChatGoogleGenerativeAI instance (with tools bound),
                    "tools": List of LangChain tool objects,
                    "tools_dict": Dict mapping tool_name to tool object,
                    "system_prompt": System prompt string,
                    "communication_tools": List of inter-agent communication tools
                }
            }
        
        Example:
            agents = factory.create_all_agents()
            doctor = agents["doctor"]
            response = doctor["llm"].invoke([
                SystemMessage(content=doctor["system_prompt"]),
                HumanMessage(content="Analyze this patient...")
            ])
        """
        agents = {}
        
        for agent_id, config in self.agent_configs.items():
            print(f"\n🤖 Creating agent: {config.agent_name} ({agent_id})")
            
            # Create base LLM instance using LLMFactory
            llm = LLMFactory.create_chat_llm(
                model_name=config.llm_model,
                use_local=self.use_local_llm,
                local_model_name=self.local_model_name,
                provider=self.llm_provider,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                num_ctx=self.ollama_num_ctx,
            )
            
            # Log which LLM backend is being used
            llm_info = LLMFactory.get_model_info(
                self.use_local_llm, config.llm_model, self.local_model_name, self.llm_provider
            )
            print(f"   📡 Using LLM: {llm_info}")
            
            # Create communication tools for this agent
            communication_tools = self._create_communication_tools(config)
            
            # Create domain-specific tools (placeholders for now)
            domain_tools = self._create_domain_tools(config)
            
            # Combine all tools
            all_tools = communication_tools + domain_tools
            
            # Create tools dictionary for execution
            tools_dict = {t.name: t for t in all_tools}
            
            # Bind tools to LLM
            if all_tools:
                llm = llm.bind_tools(all_tools)
                print(f"   ✅ Bound {len(all_tools)} tools to LLM")
            
            # Store agent data
            agents[agent_id] = {
                "config": config,
                "llm": llm,
                "tools": all_tools,
                "tools_dict": tools_dict,
                "system_prompt": config.system_prompt,
                "communication_tools": communication_tools
            }
            
            print(f"   ✅ Agent created successfully")
        
        return agents
    
    def _create_communication_tools(self, config: AgentConfig) -> List:
        """
        Create inter-agent communication tools for this agent.
        
        For each agent in can_communicate_with, creates a talk_to_X tool.
        
        Args:
            config: Agent configuration
        
        Returns:
            List of LangChain tool objects for communication
        """
        tools = []
        
        for target_agent_id in config.can_communicate_with:
            # Get target agent name
            target_config = self.agent_configs.get(target_agent_id)
            if not target_config:
                print(f"   ⚠️  Warning: Agent {target_agent_id} not found in configs")
                continue
            
            target_name = target_config.agent_name
            
            # Create communication tool
            tool_name = f"talk_to_{target_agent_id}"
            
            # Track this as a communication tool
            self.communication_tool_names.add(tool_name)
            
            # Create tool function dynamically - SAME FORMAT AS graphPOIROTMini
            def make_communication_tool(target_id: str, target_nm: str, tool_nm: str):
                def talk_to_agent(message: str) -> str:
                    """Send a message to another agent for consultation."""
                    return f"Message sent to {target_nm}: {message}"
                
                # Create StructuredTool with proper name and description
                comm_tool = StructuredTool.from_function(
                    func=talk_to_agent,
                    name=tool_nm,
                    description=f"Send a message to {target_nm} for consultation."
                )
                
                return comm_tool
            
            comm_tool = make_communication_tool(target_agent_id, target_name, tool_name)
            tools.append(comm_tool)
        
        if tools:
            print(f"   📞 Created {len(tools)} communication tools")
        
        return tools
    
    def _create_domain_tools(self, config: AgentConfig) -> List:
        """
        Create domain-specific tools from database tool definitions.
        
        If 'tool_code' is present in the definition, it dynamically executes
        the code to create the actual tool function. Otherwise, creates a placeholder.
        
        Args:
            config: Agent configuration with tools list
        
        Returns:
            List of LangChain tool objects
        """
        tools = []
        
        for tool_def in config.tools:
            tool_name = tool_def["name"]
            tool_description = tool_def.get("description", "No description")
            tool_code = tool_def.get("code")
            
            # Skip communication tools (already created)
            if tool_name.startswith("talk_to_"):
                continue
            
            tool_created = False
            
            # Try to create from code if available
            if tool_code:
                try:
                    # Create a local scope to execute the code
                    local_scope = {}
                    # We use globals() to allow imports in the tool code to work if they are standard libraries
                    # But for safety, maybe we should restrict? 
                    # For now, we assume the database is trusted.
                    exec(tool_code, globals(), local_scope)
                    
                    # Look for the function in the local scope
                    # We expect the function name to match the tool name
                    if tool_name in local_scope and callable(local_scope[tool_name]):
                        func = local_scope[tool_name]
                        
                        # Create StructuredTool from the function
                        # LangChain will parse the docstring and type hints automatically
                        tool_obj = StructuredTool.from_function(
                            func=func,
                            name=tool_name,
                            description=tool_description or func.__doc__
                        )
                        tools.append(tool_obj)
                        tool_created = True
                        print(f"   🛠️  Created tool '{tool_name}' from source code")
                    else:
                        # If exact name match fails, look for any callable
                        # This handles cases where the function name in code might differ slightly
                        # but usually it should match.
                        callables = [v for k, v in local_scope.items() if callable(v) and not k.startswith("__")]
                        if len(callables) == 1:
                            func = callables[0]
                            tool_obj = StructuredTool.from_function(
                                func=func,
                                name=tool_name,
                                description=tool_description or func.__doc__
                            )
                            tools.append(tool_obj)
                            tool_created = True
                            print(f"   🛠️  Created tool '{tool_name}' from source code (inferred function)")
                        else:
                            print(f"   ⚠️  Warning: Tool code for '{tool_name}' did not define a matching function. Falling back to placeholder.")
                            
                except Exception as e:
                    print(f"   ❌ Error creating tool '{tool_name}' from code: {e}. Falling back to placeholder.")

            # Fallback to placeholder if creation failed or no code
            if not tool_created:
                # Create placeholder tool - MOCK for POIROT analysis
                def make_placeholder_tool(t_name: str, t_desc: str):
                    def placeholder_tool(*args, **kwargs) -> str:
                        """This tool is not available in the POIROT analysis environment."""
                        return "This tool is not available during the POIROT procedure. Please limit yourself to using the information you already have available."
                    
                    # Create StructuredTool with proper name and description
                    p_tool = StructuredTool.from_function(
                        func=placeholder_tool,
                        name=t_name,
                        description=t_desc if t_desc else "Execute a domain-specific tool."
                    )
                    
                    return p_tool
                
                p_tool = make_placeholder_tool(tool_name, tool_description)
                tools.append(p_tool)
        
        if tools:
            print(f"   Total domain tools: {len(tools)}")
        
        return tools
    
    def process_messages(self, messages: List[BaseMessage], 
                         session_id: Optional[str] = None) -> List[ProcessedMessage]:
        """
        Process raw messages and enrich them with routing metadata.
        
        This function:
        1. Loads messages from database if session_id provided, or uses messages list
        2. Analyzes each message to determine from_agent and to_agent
        3. Identifies tool calls and inter-agent communications
        4. Returns ProcessedMessage objects with full metadata
        
        Args:
            messages: List of LangChain messages (if session_id is None)
            session_id: Optional session_id to load messages from database
        
        Returns:
            List of ProcessedMessage objects with routing metadata
        
        Example:
            processed = factory.process_messages(session_id="trading_session_001")
            for msg in processed:
                if msg.is_communication:
                    print(f"{msg.from_agent} -> {msg.to_agent}: {msg.message.content}")
        """
        if session_id:
            messages = self._load_messages_from_db(session_id)
        
        processed = []
        
        for msg in messages:
            # Extract metadata from message if available
            from_agent = self._extract_from_agent(msg)
            to_agent = self._extract_to_agent(msg)
            
            # Determine message type
            msg_type = self._get_message_type(msg)
            
            # Check if this is a tool call
            is_tool_call = isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls
            
            # Extract tool name if applicable
            tool_name = None
            if is_tool_call:
                tool_name = msg.tool_calls[0]["name"] if msg.tool_calls else None
            elif isinstance(msg, ToolMessage):
                tool_name = msg.name if hasattr(msg, 'name') else None
            
            # Check if this is inter-agent communication
            is_communication = tool_name in self.communication_tool_names if tool_name else False
            
            # Create ProcessedMessage
            processed_msg = ProcessedMessage(
                message=msg,
                from_agent=from_agent,
                to_agent=to_agent,
                msg_type=msg_type,
                is_tool_call=is_tool_call,
                tool_name=tool_name,
                is_communication=is_communication
            )
            
            processed.append(processed_msg)
        
        return processed
    
    def _load_messages_from_db(self, session_id: str) -> List[BaseMessage]:
        """
        Load messages from database for a specific session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of LangChain message objects
        """
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        messages = []
        
        try:
            cursor.execute("""
                SELECT
                    message_id, session_id, from_agent_id, to_agent_id,
                    content, message_type, tool_calls, tool_results,
                    timestamp, sequence_number
                FROM messages
                WHERE session_id = ?
                ORDER BY sequence_number ASC
            """, (session_id,))

            rows = cursor.fetchall()

            for row in rows:
                msg_type = row["message_type"]
                content = row["content"]
                from_agent = row["from_agent_id"]
                to_agent = row["to_agent_id"]
                
                # Create appropriate message type
                if msg_type == "human":
                    msg = HumanMessage(content=content)
                elif msg_type == "ai":
                    msg = AIMessage(content=content)
                    # Add tool_calls if present
                    if row["tool_calls"]:
                        msg.tool_calls = json.loads(row["tool_calls"])
                elif msg_type == "tool":
                    msg = ToolMessage(content=content, tool_call_id=row["message_id"])
                elif msg_type == "system":
                    msg = SystemMessage(content=content)
                else:
                    # Unknown type, default to HumanMessage
                    msg = HumanMessage(content=content)
                
                # Add metadata
                if not hasattr(msg, 'additional_kwargs'):
                    msg.additional_kwargs = {}
                if 'metadata' not in msg.additional_kwargs:
                    msg.additional_kwargs['metadata'] = {}
                
                msg.additional_kwargs['metadata']['from_node'] = from_agent
                msg.additional_kwargs['metadata']['to_node'] = to_agent
                
                messages.append(msg)
            
            print(f"📥 Loaded {len(messages)} messages for session {session_id}")
            
        finally:
            conn.close()
        
        return messages
    
    def _extract_from_agent(self, msg: BaseMessage) -> Optional[str]:
        """Extract from_agent metadata from message."""
        if hasattr(msg, 'additional_kwargs') and 'metadata' in msg.additional_kwargs:
            return msg.additional_kwargs['metadata'].get('from_node')
        return None
    
    def _extract_to_agent(self, msg: BaseMessage) -> Optional[str]:
        """Extract to_agent metadata from message."""
        if hasattr(msg, 'additional_kwargs') and 'metadata' in msg.additional_kwargs:
            return msg.additional_kwargs['metadata'].get('to_node')
        return None
    
    def _get_message_type(self, msg: BaseMessage) -> str:
        """Determine message type string."""
        if isinstance(msg, HumanMessage):
            return "human"
        elif isinstance(msg, AIMessage):
            return "ai"
        elif isinstance(msg, ToolMessage):
            return "tool"
        elif isinstance(msg, SystemMessage):
            return "system"
        else:
            return "unknown"
    
    def filter_messages_for_agent(self, processed_messages: List[ProcessedMessage], 
                                  agent_id: str) -> List[BaseMessage]:
        """
        Filter messages to only include those relevant to a specific agent.
        
        Returns messages where agent is sender OR receiver.
        Removes duplicates and empty messages.
        
        Args:
            processed_messages: List of ProcessedMessage objects
            agent_id: Agent identifier to filter for
        
        Returns:
            List of LangChain message objects relevant to this agent
        
        Example:
            doctor_messages = factory.filter_messages_for_agent(processed, "doctor")
        """
        filtered = []
        seen_messages = set()
        
        for pm in processed_messages:
            # Check if agent is involved
            if pm.from_agent == agent_id or pm.to_agent == agent_id:
                # Check for duplicates
                content_hash = hash(pm.message.content) if pm.message.content else 0
                msg_key = (content_hash, pm.from_agent, pm.to_agent)
                
                if msg_key in seen_messages:
                    continue
                
                # Skip empty messages
                if not pm.message.content or pm.message.content.strip() == "":
                    continue
                
                seen_messages.add(msg_key)
                filtered.append(pm.message)
        
        return filtered
    
    def clean_conversational_messages(
        self,
        processed_messages: List[ProcessedMessage],
        include_tool_calls: bool = False,
        current_agent_id: str = None
    ) -> List[BaseMessage]:
        """
        Clean messages to keep only conversational context.

        Delegates to the module-level clean_conversational_messages() function,
        passing this factory's communication_tool_names set.
        """
        return clean_conversational_messages(
            processed_messages=processed_messages,
            include_tool_calls=include_tool_calls,
            current_agent_id=current_agent_id,
            communication_tool_names=self.communication_tool_names,
        )
    
    def add_message_metadata(self, message: BaseMessage, from_node: str, to_node: str) -> BaseMessage:
        """
        Add routing metadata to a message (same as graphPOIROTMini function).
        
        Args:
            message: Message to add metadata to
            from_node: Source node name
            to_node: Destination node name
        
        Returns:
            Message with metadata added
        """
        if not hasattr(message, 'additional_kwargs'):
            message.additional_kwargs = {}
        
        if 'metadata' not in message.additional_kwargs:
            message.additional_kwargs['metadata'] = {}
        
        message.additional_kwargs['metadata']['from_node'] = from_node
        message.additional_kwargs['metadata']['to_node'] = to_node
        
        return message
    
    def get_agent_summary(self, agent_id: str) -> str:
        """
        Get a human-readable summary of an agent's configuration.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Formatted string with agent details
        """
        if agent_id not in self.agent_configs:
            return f"❌ Agent '{agent_id}' not found"
        
        config = self.agent_configs[agent_id]
        
        summary = f"""
Agent: {config.agent_name} ({config.agent_id})
Type: {config.agent_type}
System: {config.system_name}
Model: {config.llm_model}
Temperature: {config.temperature}
Max Tokens: {config.max_tokens}
Tools: {len(config.tools)}
Can Communicate With: {', '.join(config.can_communicate_with) if config.can_communicate_with else 'None'}
"""
        return summary.strip()
    
    def get_all_agents_summary(self) -> str:
        """
        Get summary of all agents in the system.
        
        Returns:
            Formatted string with all agent summaries
        """
        summary = f"POIROT Agent Factory - {len(self.agent_configs)} Agents\n"
        summary += "=" * 80 + "\n\n"
        
        for agent_id in self.agent_configs:
            summary += self.get_agent_summary(agent_id) + "\n\n"
        
        return summary


##############################################################################
#======================== USAGE EXAMPLE ===================================#
##############################################################################

if __name__ == "__main__":
    """
    Example usage of AgentFactory
    """
    print("="*80)
    print("POIROT-SW AGENT FACTORY - EXAMPLE USAGE")
    print("="*80)
    
    # Initialize factory with example database
    factory = AgentFactory(
        database_path="database/example_trading_system.db"
    )
    
    # Print all agents summary
    print("\n" + factory.get_all_agents_summary())
    
    # Create all agents
    agents = factory.create_all_agents()
    
    # Example: Use an agent
    if "portfolio_manager" in agents:
        print("\n" + "="*80)
        print("EXAMPLE: Using Portfolio Manager Agent")
        print("="*80)
        
        pm = agents["portfolio_manager"]
        
        # Create test message
        test_messages = [
            SystemMessage(content=pm["system_prompt"]),
            HumanMessage(content="What is the current portfolio risk level?")
        ]
        
        print("\n📤 Invoking LLM...")
        # Note: This would actually call the LLM in a real scenario
        # response = pm["llm"].invoke(test_messages)
        print("✅ Agent ready for invocation")
    
    # Example: Process messages
    print("\n" + "="*80)
    print("EXAMPLE: Processing Messages")
    print("="*80)
    
    # Create sample messages
    sample_messages = [
        HumanMessage(content="Analyze market conditions"),
        AIMessage(content="I'll analyze the data...", tool_calls=[
            {"name": "get_stock_price", "args": {"ticker": "AAPL"}}
        ]),
        ToolMessage(content="Price: $150.25", tool_call_id="call_1"),
        AIMessage(content="Based on the price...")
    ]
    
    # Add metadata
    sample_messages[0] = factory.add_message_metadata(sample_messages[0], "user", "portfolio_manager")
    sample_messages[1] = factory.add_message_metadata(sample_messages[1], "portfolio_manager", "user")
    sample_messages[2] = factory.add_message_metadata(sample_messages[2], "tool", "portfolio_manager")
    sample_messages[3] = factory.add_message_metadata(sample_messages[3], "portfolio_manager", "user")
    
    processed = factory.process_messages(sample_messages)
    
    print(f"\n✅ Processed {len(processed)} messages")
    for i, pm in enumerate(processed):
        print(f"   {i+1}. {pm.msg_type}: {pm.from_agent} -> {pm.to_agent}")
        if pm.is_tool_call:
            print(f"      Tool: {pm.tool_name}")
    
    print("\n" + "="*80)
    print("AGENT FACTORY EXAMPLE COMPLETE")
    print("="*80)
