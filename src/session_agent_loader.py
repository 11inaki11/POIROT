"""
Session-Specific Agent Loader for POIROT
=========================================

This module handles the correct loading of agents for a specific session:
1. Loads agents that actually participated in the session
2. For missing agent types, finds the most frequent instance across all sessions
3. Ensures exactly ONE instance per agent type (name-based, not ID-based)

Author: POIROT-SW Team
Date: February 2026
"""

import sqlite3
from typing import Dict, List, Tuple, Set
from collections import Counter
import json


def get_agents_for_session(
    db_path: str,
    session_id: str
) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Load the correct agent instances for a specific session.
    
    Logic:
    1. Find agents that participated in this specific session (from messages table)
    2. Get all unique agent types (names) from the system architecture
    3. For any agent type NOT in the session, find the most frequent instance across all sessions
    4. Return exactly ONE agent instance per agent type
    
    Args:
        db_path: Path to SQLite database
        session_id: Session identifier to analyze
    
    Returns:
        Tuple of:
        - agents_data: Dict mapping agent_id to agent data {name, system_prompt, can_communicate_with}
        - agent_order: List of agent_ids in deterministic order
    
    Example:
        In session 1, we have: [query_parser_v1, cve_search_v1, summarizer_v1]
        risk_analysis didn't participate in session 1
        
        Looking at all sessions:
        - risk_analysis_v1 appears in 5 sessions
        - risk_analysis_v2 appears in 2 sessions
        
        Result: We add risk_analysis_v1 as judge for this session
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get system_name for this session
        cursor.execute("SELECT system_name FROM sessions WHERE session_id = ?", (session_id,))
        session_row = cursor.fetchone()
        if not session_row:
            raise ValueError(f"Session {session_id} not found in database")
        system_name = session_row[0]
        
        print(f"\n🔍 SMART AGENT LOADING FOR SESSION: {session_id[:8]}...")
        print(f"   System: {system_name}")
        print(f"{'─'*80}")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Find agents that participated in THIS session
        # ═══════════════════════════════════════════════════════════════════
        print("\n📊 STEP 1: Analyzing agents in this session...")
        
        cursor.execute("""
            SELECT DISTINCT from_agent_id 
            FROM messages 
            WHERE session_id = ? 
            AND from_agent_id != 'system'
            ORDER BY from_agent_id
        """, (session_id,))
        
        participating_agent_ids = {row[0] for row in cursor.fetchall()}
        
        print(f"   Found {len(participating_agent_ids)} participating agents:")
        
        # Get agent details for participating agents
        agents_in_session = {}
        cursor.execute("""
            SELECT agent_id, agent_name, system_prompt, can_communicate_with
            FROM agents
            WHERE agent_id IN ({})
        """.format(','.join('?' * len(participating_agent_ids))), tuple(participating_agent_ids))
        
        for row in cursor.fetchall():
            agent_id, agent_name, system_prompt, can_communicate_json = row
            can_communicate = json.loads(can_communicate_json) if can_communicate_json else []
            
            agents_in_session[agent_id] = {
                'name': agent_name,
                'system_prompt': system_prompt,
                'can_communicate_with': can_communicate
            }
            print(f"      ✓ {agent_name} ({agent_id})")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: Get all unique agent types (names) in the system
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n📚 STEP 2: Analyzing complete system architecture...")
        
        cursor.execute("""
            SELECT DISTINCT agent_name
            FROM agents
            WHERE system_name = ?
            ORDER BY agent_name
        """, (system_name,))
        
        all_agent_types = {row[0] for row in cursor.fetchall()}
        
        print(f"   Total agent types in system: {len(all_agent_types)}")
        for agent_type in sorted(all_agent_types):
            print(f"      - {agent_type}")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: Find missing agent types (by name, not ID!)
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n🔎 STEP 3: Identifying missing agent types...")
        
        participating_agent_names = {data['name'] for data in agents_in_session.values()}
        missing_agent_types = all_agent_types - participating_agent_names
        
        if missing_agent_types:
            print(f"   Missing agent types: {len(missing_agent_types)}")
            for agent_type in sorted(missing_agent_types):
                print(f"      ⚠️ {agent_type}")
        else:
            print(f"   ✅ All agent types present in this session!")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: For missing types, find most frequent instance
        # ═══════════════════════════════════════════════════════════════════
        if missing_agent_types:
            print(f"\n🎯 STEP 4: Finding most frequent instances for missing types...")
            
            for agent_type in sorted(missing_agent_types):
                # Get all instances of this agent type
                cursor.execute("""
                    SELECT agent_id
                    FROM agents
                    WHERE agent_name = ? AND system_name = ?
                """, (agent_type, system_name))
                
                agent_instances = [row[0] for row in cursor.fetchall()]
                
                if not agent_instances:
                    print(f"      ⚠️ {agent_type}: No instances found in database!")
                    continue
                
                # Count how many sessions each instance participated in
                instance_counts = Counter()
                
                for instance_id in agent_instances:
                    cursor.execute("""
                        SELECT COUNT(DISTINCT session_id)
                        FROM messages
                        WHERE from_agent_id = ?
                    """, (instance_id,))
                    
                    count = cursor.fetchone()[0]
                    instance_counts[instance_id] = count
                
                # Select the most frequent instance
                most_frequent_id, frequency = instance_counts.most_common(1)[0]
                
                print(f"      🏆 {agent_type}:")
                print(f"         Available instances: {len(agent_instances)}")
                for instance_id in agent_instances:
                    marker = "👑" if instance_id == most_frequent_id else "  "
                    print(f"         {marker} {instance_id}: {instance_counts[instance_id]} sessions")
                
                # Load this agent's data
                cursor.execute("""
                    SELECT agent_id, agent_name, system_prompt, can_communicate_with
                    FROM agents
                    WHERE agent_id = ?
                """, (most_frequent_id,))
                
                row = cursor.fetchone()
                if row:
                    agent_id, agent_name, system_prompt, can_communicate_json = row
                    can_communicate = json.loads(can_communicate_json) if can_communicate_json else []
                    
                    agents_in_session[agent_id] = {
                        'name': agent_name,
                        'system_prompt': system_prompt,
                        'can_communicate_with': can_communicate
                    }
                    print(f"         ✅ Added {most_frequent_id} as judge for POIROT analysis")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: Create final agent list
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n✅ FINAL AGENT ROSTER:")
        agent_order = sorted(agents_in_session.keys())
        
        for agent_id in agent_order:
            data = agents_in_session[agent_id]
            status = "📝 Participated" if agent_id in participating_agent_ids else "⚖️ Judge (most frequent)"
            comm_list = ", ".join(data['can_communicate_with']) if data['can_communicate_with'] else "none"
            print(f"   {status}: {data['name']} ({agent_id})")
            print(f"      Can communicate with: {comm_list}")
        
        print(f"\n{'─'*80}")
        print(f"✅ Loaded {len(agents_in_session)} agents total")
        print(f"   - {len(participating_agent_ids)} participated in session")
        print(f"   - {len(agents_in_session) - len(participating_agent_ids)} added as judges")
        print(f"{'─'*80}\n")
        
        return agents_in_session, agent_order
    
    finally:
        conn.close()


def validate_one_agent_per_type(agents_data: Dict[str, Dict]) -> bool:
    """
    Validate that we have exactly one agent instance per agent type (name).
    
    Args:
        agents_data: Dict mapping agent_id to agent data
    
    Returns:
        True if validation passes, False otherwise
    """
    agent_names = [data['name'] for data in agents_data.values()]
    unique_names = set(agent_names)
    
    if len(agent_names) != len(unique_names):
        print(f"\n❌ VALIDATION FAILED: Duplicate agent types detected!")
        print(f"   Total agents: {len(agent_names)}")
        print(f"   Unique types: {len(unique_names)}")
        
        # Find duplicates
        name_counts = Counter(agent_names)
        for name, count in name_counts.items():
            if count > 1:
                print(f"   ⚠️ {name}: {count} instances")
                # Show which IDs
                duplicate_ids = [aid for aid, data in agents_data.items() if data['name'] == name]
                for dup_id in duplicate_ids:
                    print(f"      - {dup_id}")
        
        return False
    
    print(f"✅ Validation passed: {len(unique_names)} unique agent types, no duplicates")
    return True
