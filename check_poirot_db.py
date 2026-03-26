"""
Check POIROT Database Schema
=============================

Verifies that a database has the correct structure for POIROT-SW validation.

Usage:
    python check_poirot_db.py path/to/database.db
    python check_poirot_db.py CVE_Analyst/poirot_template.db
"""

import sqlite3
import sys
import os
from pathlib import Path

# Required tables and their expected columns
REQUIRED_SCHEMA = {
    "sessions": {
        "columns": ["id", "ticker", "description", "created_at", "system_name"],
        "required": True
    },
    "conversation_messages": {
        "columns": ["id", "session_id", "agent_id", "agent_name", "content", "timestamp"],
        "required": True
    },
    "agent_outputs": {
        "columns": ["id", "session_id", "agent_id", "agent_name", "output_data", "timestamp"],
        "required": True
    },
    "final_decisions": {
        "columns": ["id", "session_id", "decision", "rationale", "timestamp"],
        "required": True
    },
    "graph_executions": {
        "columns": ["id", "session_id", "node_name", "execution_order", "timestamp", "status"],
        "required": True
    },
    "agent_configurations": {
        "columns": ["id", "agent_id", "agent_name", "system_prompt", "tools", "model_info"],
        "required": True
    },
    "injected_errors": {
        "columns": ["id", "session_id", "error_type", "affected_agent_id", "ground_truth_vector", "description"],
        "required": True  # Critical for validation
    }
}


def check_database(db_path: str) -> bool:
    """
    Check if database has required POIROT structure.
    
    Returns:
        True if database is valid, False otherwise
    """
    
    print("\n" + "="*80)
    print(f"🔍 CHECKING DATABASE: {db_path}")
    print("="*80)
    
    # Check file exists
    if not os.path.exists(db_path):
        print(f"\n❌ ERROR: Database file not found")
        return False
    
    print(f"\n✅ File exists ({os.path.getsize(db_path)} bytes)")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}
        
        print(f"\n📊 Found {len(existing_tables)} tables:")
        for table in sorted(existing_tables):
            print(f"   - {table}")
        
        # Check required tables
        print(f"\n🔎 Validating schema...")
        
        all_valid = True
        for table_name, schema in REQUIRED_SCHEMA.items():
            print(f"\n   Checking table: {table_name}")
            
            if table_name not in existing_tables:
                if schema["required"]:
                    print(f"      ❌ MISSING (REQUIRED)")
                    all_valid = False
                else:
                    print(f"      ⚠️  MISSING (optional)")
                continue
            
            # Check columns
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_cols = {row[1] for row in cursor.fetchall()}
            
            missing_cols = set(schema["columns"]) - existing_cols
            extra_cols = existing_cols - set(schema["columns"])
            
            if missing_cols:
                print(f"      ❌ Missing columns: {', '.join(missing_cols)}")
                all_valid = False
            else:
                print(f"      ✅ All required columns present")
            
            if extra_cols:
                print(f"      ℹ️  Extra columns: {', '.join(extra_cols)}")
        
        # Check data presence
        print(f"\n📈 Checking data...")
        
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        print(f"   Sessions: {session_count}")
        
        if session_count == 0:
            print(f"      ⚠️  No sessions found (database is empty)")
        
        cursor.execute("SELECT COUNT(*) FROM injected_errors")
        error_count = cursor.fetchone()[0]
        print(f"   Injected errors: {error_count}")
        
        if error_count == 0:
            print(f"      ⚠️  No ground truth errors (cannot validate)")
        
        # Check for sessions with ground truth
        cursor.execute("""
            SELECT COUNT(DISTINCT s.id)
            FROM sessions s
            JOIN injected_errors e ON s.id = e.session_id
        """)
        gt_session_count = cursor.fetchone()[0]
        print(f"   Sessions with ground truth: {gt_session_count}")
        
        if gt_session_count > 0:
            # Show some examples
            cursor.execute("""
                SELECT s.id, s.ticker, e.error_type, e.ground_truth_vector
                FROM sessions s
                JOIN injected_errors e ON s.id = e.session_id
                LIMIT 5
            """)
            
            print(f"\n📋 Sample sessions:")
            for row in cursor.fetchall():
                session_id, ticker, error_type, gt_vector = row
                print(f"      Session {session_id}: {ticker} - {error_type}")
                print(f"         GT Vector: {gt_vector}")
        
        conn.close()
        
        # Final verdict
        print("\n" + "="*80)
        if all_valid and session_count > 0 and gt_session_count > 0:
            print("✅ DATABASE IS VALID FOR POIROT-SW")
            print(f"   Ready to run validation with {gt_session_count} sessions")
            return True
        elif all_valid and session_count > 0:
            print("⚠️  DATABASE STRUCTURE IS VALID BUT NO GROUND TRUTH")
            print("   Cannot run validation without injected_errors")
            return False
        else:
            print("❌ DATABASE HAS STRUCTURAL ISSUES")
            print("   Fix schema errors before running validation")
            return False
    
    except sqlite3.Error as e:
        print(f"\n❌ SQLite Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_poirot_db.py <path_to_database.db>")
        print("\nExample:")
        print("  python check_poirot_db.py CVE_Analyst/poirot_template.db")
        print("  python check_poirot_db.py TradingAgents/poirot_template.db")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    # Make path absolute
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.getcwd(), db_path)
    
    is_valid = check_database(db_path)
    
    if is_valid:
        print("\n✨ Next steps:")
        print(f"   1. python validate_system_CVE.py  # Run single validation")
        print(f"   2. python run_validation_batch.py --runs 10  # Run batch")
        print(f"   3. python generate_charts.py  # Analyze results")
        sys.exit(0)
    else:
        print("\n❌ Database validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
