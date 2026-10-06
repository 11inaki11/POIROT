"""
POIROT-SW Main Pipeline
=======================

This is the main execution pipeline for POIROT-SW. It orchestrates all phases
of the POIROT protocol for analyzing multi-agent systems.

Usage:
    from poirot_pipeline import POIROTPipeline
    
    pipeline = POIROTPipeline(
        system_name="MyTradingBot",
        system_description="...",
        database_path="my_system.db"
    )
    
    results = pipeline.run_full_analysis()

Author: CORTEX Team
Version: 1.0
Date: December 2025
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from .poirot_agent import POIROTAgent
    from .agent_factory import AgentFactory
    from .phase1_protocol import execute_phase1_analysis
    from .phase2_protocol import execute_phase2_analysis
    from .token_tracker import TokenTracker
except ImportError:
    from poirot_agent import POIROTAgent
    from agent_factory import AgentFactory
    from phase1_protocol import execute_phase1_analysis
    from phase2_protocol import execute_phase2_analysis
    from token_tracker import TokenTracker


class POIROTPipeline:
    """Main pipeline for POIROT-SW multi-agent system analysis.
    
    This class orchestrates all phases of the POIROT protocol:
    - Phase 0: Error vector space construction (POIROT Agent)
    - Phase 1: Agent factory (load agents from database)
    - Phase 2: Communication system setup
    - Phase 3: Graph builder (construct LangGraph)
    - Phase 4: Execution and logging
    - Phase 5: Analysis engine (POIROT Phase 1 & 2)
    - Phase 6: Visualization
    
    Attributes:
        system_name: Name of the multi-agent system being analyzed
        system_description: Textual description of the system architecture
        database_path: Path to SQLite database with system data
        output_dir: Directory for results and intermediate files
        config: Pipeline configuration options
    """
    
    def __init__(
        self,
        system_name: str,
        system_description: str,
        database_path: str,
        output_dir: str = "poirot_results",
        ignore_list: Optional[List[str]] = None,
        llm_model: str = "gemini-2.5-pro",
        session_id: Optional[str] = None,
        include_tool_calls: bool = False,
        include_broadcast_messages: bool = False,
        full_context: bool = False,
        api_call_delay: float = 0.0,
        use_local_llm: bool = False,
        local_model_name: Optional[str] = None,
        llm_provider: str = "gemini",  # "gemini", "deepseek", "local", or "ollama"
        ollama_num_ctx: int = 131072,  # Ollama context window in tokens (default 128K)
        token_budget: int = 95_000,
        max_agent_messages: int = 8,
        max_llm_retries: int = 5,
        retry_delay_503: int = 30,
        retry_delay_429: int = 60,
        verbose: bool = True,
    ):
        """Initialize the POIROT pipeline.

        Args:
            system_name: Name of the multi-agent system (e.g., "StockTradingBot")
            system_description: Detailed description of system architecture
            database_path: Path to SQLite database (schema-compliant)
            output_dir: Directory for output files (default: "poirot_results")
            ignore_list: Components to exclude from error vector space
            llm_model: LLM model for POIROT Agent (default: "gemini-2.5-pro")
            session_id: Specific session to analyze (default: most recent session)
            include_tool_calls: Whether to include tool calls in agent context (default: False)
            use_local_llm: If True, use LM Studio instead of Gemini API
            local_model_name: Model name for LM Studio (default: gpt-oss-20b)
            llm_provider: LLM provider - "gemini", "deepseek", "local", or "ollama" (default: "gemini")
            ollama_num_ctx: Context window size for Ollama models (default: 131072 = 128K).
                            Ignored for non-Ollama providers.
        """
        self.system_name = system_name
        self.system_description = system_description
        self.database_path = database_path
        self.output_dir = output_dir
        self.ignore_list = ignore_list or []
        self.llm_model = llm_model
        self.session_id = session_id
        self.include_tool_calls = include_tool_calls
        self.include_broadcast_messages = include_broadcast_messages
        self.full_context = full_context
        self.api_call_delay = api_call_delay
        self.use_local_llm = use_local_llm
        self.local_model_name = local_model_name
        self.llm_provider = llm_provider
        self.ollama_num_ctx = ollama_num_ctx
        self.token_budget = token_budget
        self.max_agent_messages = max_agent_messages
        self.max_llm_retries = max_llm_retries
        self.retry_delay_503 = retry_delay_503
        self.retry_delay_429 = retry_delay_429
        self.verbose = verbose
        self._p = print if verbose else (lambda *a, **kw: None)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components (will be set during execution)
        self.poirot_agent = None
        self.error_vector_space = None
        self.agent_factory = None
        self.agents = None
        self.processed_messages = None  # Messages with metadata from agent factory
        self.graph = None
        self.analysis_results = None
        
        # Token tracking
        self.token_tracker = TokenTracker()
        
        # Execution metadata
        self.execution_metadata = {
            "pipeline_started": None,
            "pipeline_completed": None,
            "phases_completed": []
        }
    
    # ========================================================================
    # PHASE 0: ERROR VECTOR SPACE CONSTRUCTION
    # ========================================================================
    
    def _get_error_space_path(self) -> str:
        """Get path for error vector space JSON file."""
        # Sanitize system name for filename
        safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' 
                           for c in self.system_name)
        return os.path.join(self.output_dir, f"{safe_name}_error_space.json")
    
    def _load_existing_error_space(self) -> Optional[Dict[str, Any]]:
        """Load existing error vector space if it exists.
        
        Returns:
            Dictionary with error vector space, or None if doesn't exist
        """
        path = self._get_error_space_path()
        
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if "error_regions" in data and "system_name" in data:
                return data
            else:
                self._p(f"WARNING  Warning: Existing error space file has invalid structure")
                return None
                
        except Exception as e:
            self._p(f"WARNING  Warning: Could not load existing error space: {e}")
            return None
    
    def run_phase0_error_space_construction(
        self, 
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """Phase 0: Construct error vector space using POIROT Agent.
        
        This phase identifies all potential error locations in the system
        and creates the N-dimensional error vector space.
        
        Args:
            force_rerun: If True, re-run even if cached result exists
        
        Returns:
            Dictionary containing error vector space definition
        """
        self._p("=" * 80)
        self._p("PHASE 0: ERROR VECTOR SPACE CONSTRUCTION")
        self._p("=" * 80)
        
        # Check if error space already exists
        if not force_rerun:
            existing = self._load_existing_error_space()
            if existing:
                self._p(f"\nOK Using existing error vector space from cache")
                self._p(f"   Path: {self._get_error_space_path()}")
                self._p(f"   Dimensions: {len(existing['error_regions'])}")
                self._p(f"\n Use force_rerun=True to regenerate")
                
                self.error_vector_space = existing
                self.execution_metadata["phases_completed"].append("phase0_cached")
                return existing
        
        # Initialize POIROT Agent
        self._p(f"\nAgent Initializing POIROT Agent ({self.llm_model})...")
        if self.use_local_llm:
            self._p(f"    Using Local LLM: {self.local_model_name or 'gpt-oss-20b'}")
        elif self.llm_provider == "deepseek":
            self._p(f"   - Using DeepSeek API: {self.llm_model}")
        self.poirot_agent = POIROTAgent(
            model=self.llm_model,
            vectors_to_ignore=self.ignore_list,
            use_local_llm=self.use_local_llm,
            local_model_name=self.local_model_name,
            token_tracker=self.token_tracker,
            llm_provider=self.llm_provider
        )
        
        # Run analysis
        self._p(f" Analyzing system: {self.system_name}")
        self._p(f"   Description length: {len(self.system_description)} characters")
        
        if self.ignore_list:
            self._p(f"   Ignoring {len(self.ignore_list)} components")
        
        self._p("\n... Running POIROT Agent analysis...")
        
        # The agents that will vote in Phase 2 must each own the region whose id is
        # their agent_id, so the hazard space is built against that list.
        session_agents = self._get_voting_agents()
        self._p(f"   Agents (required region ids): {[a['id'] for a in session_agents]}")

        result = self.poirot_agent.analyze_system(
            self.system_description,
            ignore_list=self.ignore_list,
            verbose=self.verbose,
            agents=session_agents,
        )
        
        # Check for errors
        if "error" in result:
            self._p(f"\nERROR ERROR: POIROT Agent analysis failed")
            self._p(f"   {result['error']}")
            if "raw" in result:
                self._p(f"\n   Raw response (first 200 chars):")
                self._p(f"   {result['raw'][:200]}...")
            raise RuntimeError("Phase 0 failed: POIROT Agent could not analyze system")
        
        # Validate result
        if "error_regions" not in result or "system_name" not in result:
            self._p(f"\nERROR ERROR: Invalid response structure from POIROT Agent")
            raise RuntimeError("Phase 0 failed: Invalid response structure")
        
        # Save result
        output_path = self._get_error_space_path()
        self._p(f"\nSaved Saving error vector space to: {output_path}")
        
        saved = self.poirot_agent.save_output(result, output_path)
        if not saved:
            self._p(f"WARNING  Warning: Could not save error vector space to file")
        
        # Store in pipeline
        self.error_vector_space = result
        
        # Print summary
        self._p(f"\nOK Phase 0 complete!")
        self._p(f"\n   System: {result['system_name']}")
        self._p(f"   Error dimensions: {len(result['error_regions'])}")
        self._p(f"\n   Error regions identified:")
        
        for i, region in enumerate(result['error_regions'], 1):
            self._p(f"      {i}. {region['id']}: {region['name']} ({region['type']})")
        
        # Update metadata
        self.execution_metadata["phases_completed"].append("phase0_completed")
        
        return result
    
    # ========================================================================
    # PHASE 1: AGENT FACTORY
    # ========================================================================
    
    def run_phase1_agent_factory(self) -> Dict[str, Any]:
        """Phase 1: Load agents from database and create LangChain instances.
        
        This phase:
        - Initializes AgentFactory with database
        - Loads agent configurations from agents and agent_tools tables
        - Creates LangChain agent instances with LLM and tools
        - Sets up inter-agent communication tools
        
        Returns:
            Dictionary mapping agent_id to agent data:
            {
                "agent_id": {
                    "config": AgentConfig,
                    "llm": ChatGoogleGenerativeAI (with tools bound),
                    "tools": List of tool objects,
                    "tools_dict": Dict mapping tool_name to tool,
                    "system_prompt": str,
                    "communication_tools": List of comm tools
                }
            }
        """
        self._p("\n" + "=" * 80)
        self._p("PHASE 1: AGENT FACTORY")
        self._p("=" * 80)
        
        # Verify database exists
        if not os.path.exists(self.database_path):
            raise FileNotFoundError(
                f"Database not found: {self.database_path}\n"
                "Please ensure the database exists before running Phase 1."
            )
        
        # Initialize AgentFactory
        self._p(f"\n Initializing Agent Factory...")
        self._p(f"   Database: {self.database_path}")
        
        try:
            self.agent_factory = AgentFactory(
                database_path=self.database_path,
                use_local_llm=self.use_local_llm,
                local_model_name=self.local_model_name,
                llm_provider=self.llm_provider,
                ollama_num_ctx=self.ollama_num_ctx,
                model_override=self.llm_model,
            )
        except Exception as e:
            self._p(f"\nERROR ERROR: Could not initialize Agent Factory")
            self._p(f"   {e}")
            raise RuntimeError(f"Phase 1 failed: {e}")
        
        # Create all agents
        self._p(f"\nAgent Creating LangChain agent instances...")
        
        try:
            self.agents = self.agent_factory.create_all_agents()
        except Exception as e:
            self._p(f"\nERROR ERROR: Could not create agents")
            self._p(f"   {e}")
            raise RuntimeError(f"Phase 1 failed: {e}")
        
        # Print summary
        self._p(f"\nOK Phase 1 complete!")
        self._p(f"\n   Agents created: {len(self.agents)}")
        
        for agent_id, agent_data in self.agents.items():
            config = agent_data["config"]
            num_tools = len(agent_data["tools"])
            num_comm = len(agent_data["communication_tools"])
            
            self._p(f"\n   {config.agent_name} ({agent_id}):")
            self._p(f"      Type: {config.agent_type}")
            self._p(f"      Model: {config.llm_model}")
            self._p(f"      Temperature: {config.temperature}")
            self._p(f"      Total Tools: {num_tools} ({num_comm} communication + {num_tools - num_comm} domain)")
            
            if config.can_communicate_with:
                self._p(f"      Can communicate with: {', '.join(config.can_communicate_with)}")
        
        # Update metadata
        self.execution_metadata["phases_completed"].append("phase1_completed")
        
        # Load and process messages from database (needed for Phase 1 protocol)
        self._p(f"\n Loading session messages from database...")
        if self.session_id:
            self._p(f"    Target session: {self.session_id}")
        else:
            self._p(f"    Target session: Most recent")
        raw_messages = self._load_messages_from_database()
        if raw_messages:
            self.processed_messages = self.agent_factory.process_messages(raw_messages)
            self._p(f"   OK Processed {len(self.processed_messages)} messages")
        else:
            self._p(f"   WARNING  No messages found in database")
            self.processed_messages = []
        
        # Save agent configurations to JSON for inspection
        self._p(f"\nSaved Saving agent configurations to JSON...")
        self._save_agent_configurations()
        
        # Return agents dictionary for external use
        return {
            "num_agents": len(self.agents),
            "agent_ids": list(self.agents.keys()),
            "num_messages": len(self.processed_messages),
            "agents": self.agents  # Full agent data
        }
    
    # ========================================================================
    # PHASE 1B: POIROT PHASE 1 PROTOCOL (INDIVIDUAL ANALYSIS)
    # ========================================================================
    
    def run_phase1_protocol(self) -> Dict[str, Any]:
        """Phase 1B: Run POIROT Phase 1 protocol (individual agent analysis).
        
        Each agent analyzes the session independently WITHOUT peer consultation.
        Requires Phase 1 (agent factory) to be completed first.
        
        Returns:
            Dictionary with phase1_reports {agent_id: analysis_content}
        """
        self._p("\n" + "=" * 80)
        self._p("PHASE 1B: POIROT PHASE 1 PROTOCOL")
        self._p("=" * 80)
        
        # Validate prerequisites
        if self.agents is None:
            raise RuntimeError("Phase 1 (agent factory) must be run before Phase 1 protocol")
        
        if self.agent_factory is None:
            raise RuntimeError("Agent factory not initialized")
        
        if self.processed_messages is None:
            raise RuntimeError("No processed messages available (run Phase 1 factory first)")
        
        # Get session name from database
        session_name = self._get_session_name_from_database()
        
        # Import session agent loader
        try:
            from .session_agent_loader import get_agents_for_session, validate_one_agent_per_type
        except ImportError:
            try:
                from session_agent_loader import get_agents_for_session, validate_one_agent_per_type
            except ImportError:
                self._p("WARNING Could not import session_agent_loader, using fallback logic")
                get_agents_for_session = None
        
        # --- SMART AGENT SELECTION LOGIC ---
        # Instead of just filtering participants, we now use intelligent selection:
        # 1. Agents that participated in the session
        # 2. For missing agent types, the most frequent instance (as judges)
        
        if get_agents_for_session is not None and self.session_id:
            self._p(f"\n Using smart agent selection for session: {self.session_id[:8]}...")
            
            # Get the correct agent IDs for this session (participants + judges)
            session_agents_data, _ = get_agents_for_session(
                db_path=str(Path(self.database_path)),
                session_id=self.session_id
            )
            
            # Validate no duplicates
            if not validate_one_agent_per_type(session_agents_data):
                raise ValueError(
                    "Agent selection failed: Multiple instances of same agent type. "
                    "This should not happen with get_agents_for_session."
                )
                
            # Select these agents from self.agents
            selected_agent_ids = set(session_agents_data.keys())
            active_agents = {
                aid: agent_data
                for aid, agent_data in self.agents.items()
                if aid in selected_agent_ids
            }
            
            self._p(f"\n Smart Agent Selection Results:")
            self._p(f"   Total Agents in DB: {len(self.agents)}")
            self._p(f"   Selected for Analysis: {len(active_agents)}")
            
        else:
            # Fallback: Use old logic (only participants, no judges)
            self._p(f"\n Agent Participation Analysis (Fallback Mode):")
            
            participating_agent_ids = set()
            for pm in self.processed_messages:
                # Add sender
                if pm.from_agent and pm.from_agent.strip():
                    participating_agent_ids.add(pm.from_agent)
                
                # Add receiver
                if pm.to_agent and pm.to_agent.strip() and pm.to_agent not in ['all', 'None', 'unknown']:
                    participating_agent_ids.add(pm.to_agent)
                    
            # Filter the agents dictionary
            active_agents = {
                aid: agent_data 
                for aid, agent_data in self.agents.items() 
                if aid in participating_agent_ids
            }
            
            self._p(f"   Total Agents in DB: {len(self.agents)}")
            self._p(f"   Selected for Analysis: {len(active_agents)}")
        
        if not active_agents:
            self._p("   WARNING  No active agents found in this session analysis. Skipping Phase 1 Protocol.")
            return {}
            
        # Execute Phase 1 protocol using ONLY active agents
        phase1_result = execute_phase1_analysis(
            agents=active_agents,
            processed_messages=self.processed_messages,
            agent_factory=self.agent_factory,
            error_space=self.error_vector_space,
            vectors_to_ignore=self.ignore_list,  # Use pipeline's ignore list
            output_dir=Path(self.output_dir),
            session_name=session_name,
            include_tool_calls=self.include_tool_calls,
            include_broadcast_messages=self.include_broadcast_messages,
            full_context=self.full_context,
            token_tracker=self.token_tracker,
            use_local_llm=self.use_local_llm,
            local_model_name=self.local_model_name,
            llm_provider=self.llm_provider,
            max_llm_retries=self.max_llm_retries,
            retry_delay_503=self.retry_delay_503,
            retry_delay_429=self.retry_delay_429,
            api_call_delay=self.api_call_delay,
        )

        # Store results
        if isinstance(phase1_result, dict) and 'reports' in phase1_result:
            self.analysis_results = phase1_result['reports']
            if 'metadata' in phase1_result:
                self.execution_metadata['phase1_metrics'] = phase1_result['metadata']
        else:
            self.analysis_results = phase1_result

        # Mark completion
        self.execution_metadata["phases_completed"].append("phase1_protocol_completed")
        
        self._p(f"\nOK Phase 1 Protocol completed: {len(self.analysis_results)} agent reports generated")
        
        return {
            "num_reports": len(self.analysis_results),
            "agent_ids": list(self.analysis_results.keys()),
            "total_content_length": sum(len(content) for content in self.analysis_results.values())
        }
    
    # ========================================================================
    # PHASE 2: PEER CONSULTATION PROTOCOL
    # ========================================================================
    
    def run_phase2_protocol(self) -> Dict[str, Any]:
        """Phase 2: Run POIROT Phase 2 protocol (peer consultation).
        
        Agents engage in peer-to-peer consultation to collaboratively identify
        hazard vectors. Requires Phase 1B to be completed first to provide
        individual analysis context.
        
        Returns:
            Dictionary with:
            - votes: Dict mapping agent_id to vote data
            - num_agents: Number of agents who voted
            - output_files: List of created output file paths
        """
        self._p("\n" + "=" * 80)
        self._p("PHASE 2: POIROT PHASE 2 PROTOCOL (PEER CONSULTATION)")
        self._p("=" * 80)
        
        # Validate prerequisites
        if self.analysis_results is None:
            raise RuntimeError("Phase 1B (individual protocol) must be run before Phase 2")
        
        if self.error_vector_space is None:
            raise RuntimeError("Phase 0 (error space) must be run before Phase 2")
        
        # Get session ID
        session_id = self._get_session_id()
        
        # Execute Phase 2 protocol
        result = execute_phase2_analysis(
            session_id=session_id,
            db_path=Path(self.database_path),
            phase1_reports=self.analysis_results,  # Use Phase 1B individual reports
            vectors_to_ignore=self.ignore_list,
            error_space=self.error_vector_space,
            model_name=self.llm_model,
            output_dir=Path(self.output_dir) / "phase2",
            recursion_limit=300,  # Increased limit for full agent consultation
            include_tool_calls=self.include_tool_calls,
            include_broadcast_messages=self.include_broadcast_messages,
            full_context=self.full_context,
            api_call_delay=self.api_call_delay,
            use_local_llm=self.use_local_llm,
            local_model_name=self.local_model_name,
            token_tracker=self.token_tracker,
            llm_provider=self.llm_provider,
            max_agent_messages=self.max_agent_messages,
            max_llm_retries=self.max_llm_retries,
            retry_delay_503=self.retry_delay_503,
            retry_delay_429=self.retry_delay_429,
            token_budget=self.token_budget,
        )

        # Store results
        self.phase2_votes = result.get('votes', {})
        
        # Capture metrics
        if 'metadata' in result:
            self.execution_metadata['phase2_metrics'] = result['metadata']
        
        # Mark completion
        self.execution_metadata["phases_completed"].append("phase2_protocol_completed")
        
        self._p(f"\nOK Phase 2 Protocol completed: {len(self.phase2_votes)} agent votes collected")
        
        return {
            "votes": self.phase2_votes,
            "num_agents": len(self.phase2_votes),
            "output_files": result.get('output_files', []),
            "voting_results": result.get('voting_results')
        }
    
    # ========================================================================
    # PHASE 3: GRAPH BUILDER (TODO)
    # ========================================================================
    
    def run_phase3_graph_builder(self) -> Any:
        """Phase 3: Build LangGraph StateGraph dynamically.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        self._p("\n" + "=" * 80)
        self._p("PHASE 3: GRAPH BUILDER")
        self._p("=" * 80)
        self._p("\nWARNING  Phase 3 not yet implemented")
        self._p("   Coming soon: Dynamic StateGraph construction")
        
        # TODO: Implement graph builder
        
        raise NotImplementedError("Phase 3: Graph Builder not yet implemented")
    
    # ========================================================================
    # PHASE 4: EXECUTION & LOGGING (TODO)
    # ========================================================================
    
    def run_phase4_execution(self, input_query: str) -> Dict[str, Any]:
        """Phase 4: Execute multi-agent workflow and log data.
        
        Args:
            input_query: Initial problem/query for the system
        
        Returns:
            Execution results and session metadata
        """
        self._p("\n" + "=" * 80)
        self._p("PHASE 4: EXECUTION & LOGGING")
        self._p("=" * 80)
        self._p("\nWARNING  Phase 4 not yet implemented")
        self._p("   Coming soon: Multi-agent workflow execution")
        
        # TODO: Implement execution engine
        
        raise NotImplementedError("Phase 4: Execution Engine not yet implemented")
    
    # ========================================================================
    # PHASE 5: ANALYSIS ENGINE (TODO)
    # ========================================================================
    
    def run_phase5_analysis(self, session_id: str) -> Dict[str, Any]:
        """Phase 5: Run POIROT Phase 1 & 2 analysis.
        
        Args:
            session_id: ID of session to analyze
        
        Returns:
            Analysis results including error vectors and trust scores
        """
        self._p("\n" + "=" * 80)
        self._p("PHASE 5: POIROT ANALYSIS ENGINE")
        self._p("=" * 80)
        self._p("\nWARNING  Phase 5 not yet implemented")
        self._p("   Coming soon: POIROT Phase 1 & 2 analysis")
        
        # TODO: Implement analysis engine
        
        raise NotImplementedError("Phase 5: Analysis Engine not yet implemented")
    
    # ========================================================================
    # PHASE 6: VISUALIZATION (TODO)
    # ========================================================================
    
    def run_phase6_visualization(self) -> None:
        """Phase 6: Generate visualizations and reports."""
        self._p("\n" + "=" * 80)
        self._p("PHASE 6: VISUALIZATION")
        self._p("=" * 80)
        self._p("\nWARNING  Phase 6 not yet implemented")
        self._p("   Coming soon: Result visualization")
        
        # TODO: Implement visualizer
        
        raise NotImplementedError("Phase 6: Visualizer not yet implemented")
    
    # ========================================================================
    # FULL PIPELINE EXECUTION
    # ========================================================================
    
    def run_full_analysis(
        self,
        input_query: Optional[str] = None,
        session_id: Optional[str] = None,
        phases: Optional[List[str]] = None,
        specific_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete POIROT analysis pipeline.
        
        Args:
            input_query: Initial query for Phase 4 (execution)
            session_id: Session ID for Phase 5 (analysis) - if None, uses Phase 4 result
            phases: List of phases to run (default: all available)
            specific_session_id: ID of the specific session to analyze (overrides logic to analyze all/recent)
        
        Returns:
            Dictionary containing results from all executed phases
        """
        # Set session filter if provided
        if specific_session_id:
            self.session_id = specific_session_id
            self._p(f"i  Filtering analysis to specific session: {specific_session_id}")
            
        self.execution_metadata["pipeline_started"] = datetime.now().isoformat()
        
        self._p("\n" + "=" * 80)
        self._p("POIROT-SW FULL ANALYSIS PIPELINE")
        self._p("=" * 80)
        self._p(f"\nSystem: {self.system_name}")
        self._p(f"Database: {self.database_path}")
        self._p(f"Output: {self.output_dir}")
        self._p(f"Timestamp: {self.execution_metadata['pipeline_started']}")
        if self.session_id:
             self._p(f"Target Session: {self.session_id}")
        
        results = {
            "system_name": self.system_name,
            "metadata": self.execution_metadata
        }
        
        try:
            # Phase 0: Error vector space construction
            if phases is None or "phase0" in phases:
                results["phase0_error_space"] = self.run_phase0_error_space_construction()
            
            # Phase 1: Agent factory
            if phases is None or "phase1" in phases:
                try:
                    results["phase1_agents"] = self.run_phase1_agent_factory()
                except NotImplementedError:
                    self._p("\nSKIP  Skipping Phase 1 (not implemented)")
            
            # Phase 1B: POIROT Phase 1 Protocol (individual analysis)
            if phases is None or "phase1_protocol" in phases:
                try:
                    results["phase1_protocol"] = self.run_phase1_protocol()
                except NotImplementedError:
                    self._p("\nSKIP  Skipping Phase 1 Protocol (not implemented)")
            
            # Phase 2: POIROT Phase 2 Protocol (peer consultation)
            if phases is None or "phase2_protocol" in phases:
                try:
                    results["phase2_protocol"] = self.run_phase2_protocol()
                except NotImplementedError:
                    self._p("\nSKIP  Skipping Phase 2 Protocol (not implemented)")

            # Mark completion
            self.execution_metadata["pipeline_completed"] = datetime.now().isoformat()
            
            # Add token usage to results
            token_summary = self.token_tracker.get_summary()
            results["token_usage"] = token_summary
            
            # Print summary
            self._p("\n" + "=" * 80)
            self._p("PIPELINE EXECUTION SUMMARY")
            self._p("=" * 80)
            self._p(f"\nPhases completed: {len(self.execution_metadata['phases_completed'])}")
            for phase in self.execution_metadata['phases_completed']:
                self._p(f"   OK {phase}")
            
            # Print token usage
            self._p(f"\n Token Usage (Gemini API):")
            self._p(f"   Total tokens: {token_summary['total_tokens']:,}")
            self._p(f"   Input tokens: {token_summary['input_tokens']:,}")
            self._p(f"   Output tokens: {token_summary['output_tokens']:,}")
            self._p(f"   API calls: {token_summary['call_count']}")
            
            self._p(f"\nResults saved to: {self.output_dir}")
            self._p(f"Completed at: {self.execution_metadata['pipeline_completed']}")
            
            return results
            
        except Exception as e:
            self._p(f"\nERROR PIPELINE ERROR: {e}")
            self.execution_metadata["pipeline_error"] = str(e)
            raise
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _save_agent_configurations(self) -> bool:
        """Save complete agent configurations to JSON for inspection.
        
        This saves all agent data including:
        - System prompt
        - Tools (with descriptions)
        - LLM configuration
        - Communication capabilities
        
        Returns:
            True if successful, False otherwise
        """
        if not self.agents:
            self._p("   WARNING  No agents to save")
            return False
        
        output_path = os.path.join(self.output_dir, "agent_configurations.json")
        
        try:
            agent_configs = {}
            
            for agent_id, agent_data in self.agents.items():
                config = agent_data["config"]
                
                # Extract tool information
                tools_info = []
                for tool in agent_data["tools"]:
                    tools_info.append({
                        "name": tool.name,
                        "description": tool.description
                    })
                
                # Extract communication tools
                comm_tools_info = []
                for tool in agent_data["communication_tools"]:
                    comm_tools_info.append({
                        "name": tool.name,
                        "description": tool.description
                    })
                
                # Build complete configuration
                agent_configs[agent_id] = {
                    "agent_name": config.agent_name,
                    "agent_type": config.agent_type,
                    "system_name": config.system_name,
                    "system_prompt": config.system_prompt,
                    "llm_configuration": {
                        "model": config.llm_model,
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens
                    },
                    "tools": {
                        "total": len(tools_info),
                        "communication": len(comm_tools_info),
                        "domain": len(tools_info) - len(comm_tools_info),
                        "all_tools": tools_info,
                        "communication_tools": comm_tools_info
                    },
                    "communication": {
                        "can_communicate_with": config.can_communicate_with
                    }
                }
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(agent_configs, f, indent=2, ensure_ascii=False)
            
            self._p(f"   OK Agent configurations saved to: {output_path}")
            self._p(f"    Contains: system prompts, tools, LLM configs for {len(agent_configs)} agents")
            
            return True
            
        except Exception as e:
            self._p(f"   WARNING  Could not save agent configurations: {e}")
            return False
    
    def get_error_space_summary(self) -> str:
        """Get human-readable summary of error vector space.
        
        Returns:
            Formatted string with error space information
        """
        if not self.error_vector_space:
            return "Error vector space not yet constructed. Run phase0 first."
        
        lines = [
            f"System: {self.error_vector_space['system_name']}",
            f"Dimensions: {len(self.error_vector_space['error_regions'])}",
            "\nError Regions:"
        ]
        
        for region in self.error_vector_space['error_regions']:
            lines.append(f"  {region['id']}: {region['name']} ({region['type']})")
            lines.append(f"     {region['description']}")
        
        return "\n".join(lines)
    
    def save_pipeline_state(self, filepath: Optional[str] = None) -> bool:
        """Save current pipeline state to JSON file.
        
        Args:
            filepath: Path to save state (default: {output_dir}/pipeline_state.json)
        
        Returns:
            True if successful, False otherwise
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, "pipeline_state.json")
        
        state = {
            "system_name": self.system_name,
            "database_path": self.database_path,
            "output_dir": self.output_dir,
            "execution_metadata": self.execution_metadata,
            "error_vector_space": self.error_vector_space,
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    def _load_messages_from_database(self) -> List:
        """Load messages from database for Phase 1 protocol.
        
        Returns:
            List of BaseMessage objects from the most recent session
        """
        import sqlite3
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
        
        try:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get target session (specific or most recent)
            if self.session_id:
                cursor.execute("""
                    SELECT session_id, system_name 
                    FROM sessions 
                    WHERE session_id = ?
                """, (self.session_id,))
            else:
                cursor.execute("""
                    SELECT session_id, system_name 
                    FROM sessions 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
            session_row = cursor.fetchone()
            
            if not session_row:
                return []
            
            session_id = session_row['session_id']
            
            # Get all messages for this session
            cursor.execute("""
                SELECT * FROM messages 
                WHERE session_id = ? 
                ORDER BY sequence_number ASC
            """, (session_id,))
            
            messages = []
            for row in cursor.fetchall():
                msg_type = row['message_type']
                content = row['content']
                
                # Convert to LangChain message types
                if msg_type == 'human':
                    msg = HumanMessage(content=content)
                elif msg_type == 'ai':
                    # Check if it's a tool call
                    if row['is_tool_call']:
                        tool_calls = [{
                            'name': row['tool_name'],
                            'args': json.loads(row['tool_input']) if row['tool_input'] else {},
                            'id': row['message_id']
                        }]
                        msg = AIMessage(content=content, tool_calls=tool_calls)
                    else:
                        msg = AIMessage(content=content)
                elif msg_type == 'tool':
                    msg = ToolMessage(
                        content=row['tool_output'] or content,
                        tool_call_id=row['message_id']
                    )
                elif msg_type == 'system':
                    msg = SystemMessage(content=content)
                else:
                    # Default to HumanMessage
                    msg = HumanMessage(content=content)
                
                # Add metadata from database (match AgentFactory format)
                msg.additional_kwargs = {
                    'metadata': {
                        'from_node': row['from_agent_id'],
                        'to_node': row['to_agent_id']
                    },
                    'timestamp': row['timestamp'],
                    'sequence_number': row['sequence_number']
                }
                
                messages.append(msg)
            
            conn.close()
            return messages
            
        except Exception as e:
            self._p(f"   WARNING  Error loading messages: {e}")
            return []
    
    def _get_voting_agents(self) -> List[Dict[str, str]]:
        """Agents that vote in Phase 2 for the analyzed session, as {"id", "name"}.

        Uses the same selection as Phase 2 (get_agents_for_session on the target
        session) so that Phase 0 can require one region per voting agent.
        """
        try:
            from .session_agent_loader import get_agents_for_session
        except ImportError:
            from session_agent_loader import get_agents_for_session

        agents_data, agent_order = get_agents_for_session(
            db_path=str(Path(self.database_path)),
            session_id=self._get_session_id(),
        )
        return [{"id": aid, "name": agents_data[aid].get("name", aid)} for aid in agent_order]

    def _get_session_id(self) -> str:
        """Get session ID from database.
        
        Returns:
            Session ID (either user-specified or most recent)
        """
        import sqlite3
        
        if self.session_id:
            return str(self.session_id)
        
        # Get most recent session
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute("SELECT session_id FROM sessions ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return row[0]
        except Exception as e:
            raise RuntimeError(f"Could not determine session ID: {e}")
        
        raise RuntimeError("No sessions found in database")
    
    def _get_session_name_from_database(self) -> str:
        """Get session name from database.
        
        Returns:
            Session name from most recent session, or system name as fallback
        """
        import sqlite3
        
        try:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if self.session_id:
                cursor.execute("""
                    SELECT system_name, session_number, session_notes
                    FROM sessions 
                    WHERE session_id = ?
                """, (self.session_id,))
            else:
                cursor.execute("""
                    SELECT system_name, session_number, session_notes
                    FROM sessions 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Create descriptive session name
                session_name = f"{row['system_name']} Session {row['session_number']}"
                if row['session_notes']:
                    session_name += f" - {row['session_notes']}"
                return session_name
            
        except Exception:
            pass
        
        return self.system_name
    
    def list_available_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions available in the database.
        
        Returns:
            List of session dictionaries with metadata
        """
        import sqlite3
        
        try:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    session_id,
                    system_name,
                    session_number,
                    created_at,
                    session_notes,
                    has_ground_truth,
                    execution_time_seconds
                FROM sessions 
                ORDER BY created_at DESC
            """)
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    'session_id': row['session_id'],
                    'system_name': row['system_name'],
                    'session_number': row['session_number'],
                    'created_at': row['created_at'],
                    'session_notes': row['session_notes'],
                    'has_ground_truth': bool(row['has_ground_truth']),
                    'execution_time_seconds': row['execution_time_seconds']
                })
            
            conn.close()
            return sessions
            
        except Exception as e:
            self._p(f"WARNING  Error listing sessions: {e}")
            return []
    
    def print_available_sessions(self) -> None:
        """Print a formatted list of available sessions."""
        sessions = self.list_available_sessions()
        
        if not sessions:
            self._p("\nWARNING  No sessions found in database")
            return
        
        self._p("\n" + "="*80)
        self._p("AVAILABLE SESSIONS")
        self._p("="*80)
        self._p(f"\nFound {len(sessions)} session(s) in database:\n")
        
        for i, session in enumerate(sessions, 1):
            self._p(f"{i}. {session['session_id']}")
            self._p(f"   System: {session['system_name']} (Session #{session['session_number']})")
            self._p(f"   Created: {session['created_at']}")
            if session['session_notes']:
                self._p(f"   Notes: {session['session_notes']}")
            if session['has_ground_truth']:
                self._p(f"   OK Has ground truth")
            if session['execution_time_seconds']:
                self._p(f"   Duration: {session['execution_time_seconds']:.2f}s")
            self._p()
        
        self._p("="*80 + "\n")
