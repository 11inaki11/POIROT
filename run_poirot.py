"""
POIROT LAUNCHER
===============

Use this script to analyze any multi-agent system database with POIROT.

CONFIGURATION:
1. Update 'DB_PATH' to point to your SQLite database.
2. Update 'SYSTEM_NAME' and 'SYSTEM_DESCRIPTION'.
3. (Optional) Add items to 'IGNORE_LIST'.

The script will automatically:
- Detect available sessions in the database.
- Ask you which one to analyze.
- Create a dedicated results directory.
- Generate the Error Space (if new) and run the full diagnosis.
"""

import sys
import os
import sqlite3
import argparse
from typing import List, Tuple
from dotenv import load_dotenv

# Add src to path for running directly (before pip install)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import poirot

# Load API Keys from .env
load_dotenv()

# ==============================================================================
# CONFIGURATION SECTION - EDIT THIS
# ==============================================================================

# 1. Path to your .db file
DB_PATH = r"path/to/your_system.db"

# 2. System Identity
SYSTEM_NAME = "MyAgentSystem"
SYSTEM_DESCRIPTION = """
Brief description of your multi-agent system architecture.
List the agents, their roles, and how they communicate.
"""

# 3. Ignore List (known non-issues to filter out of the analysis)
IGNORE_LIST = [
    # "Example: memory limitations applied for testing purposes",
]

# 4. Context Configuration
INCLUDE_TOOL_CALLS = True
INCLUDE_BROADCAST_MESSAGES = True

# ==============================================================================
# END OF CONFIGURATION
# ==============================================================================


def get_available_sessions(db_path: str) -> List[Tuple]:
    """Retrieve session list from the database."""
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at: {db_path}")
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT session_id, created_at, updated_at FROM sessions ORDER BY created_at DESC"
            )
            sessions = [(row[0], row[1], row[2], "N/A") for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            cursor.execute("SELECT session_id FROM sessions")
            sessions = [(row[0], "Unknown", "Unknown", "N/A") for row in cursor.fetchall()]
        conn.close()
        return sessions
    except Exception as e:
        print(f"Error reading database: {e}")
        return []


def select_session(sessions: List[Tuple]) -> str:
    """Interactive terminal menu to select a session."""
    print(f"\nFOUND {len(sessions)} SESSIONS IN DATABASE:")
    print("=" * 90)
    print(f"{'#':<4} | {'SESSION ID':<40} | {'DATE':<25} | {'STATUS':<10}")
    print("-" * 90)

    for i, (sid, start, end, status) in enumerate(sessions):
        display_sid = (sid[:37] + "...") if len(sid) > 40 else sid
        print(f"{i+1:<4} | {display_sid:<40} | {start:<25} | {status:<10}")
    print("=" * 90)

    if len(sessions) == 1:
        print(f"\nOnly one session found. Auto-selecting: {sessions[0][0]}")
        return sessions[0][0]

    while True:
        try:
            choice = input(f"\nSelect session number (1-{len(sessions)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx][0]
            print("Invalid number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def main():
    parser = argparse.ArgumentParser(
        description="POIROT - Automated Multi-Agent Forensics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=["gemini", "deepseek", "local", "ollama"],
        default="gemini",
        help="LLM provider: gemini (default), deepseek, local (LM Studio), or ollama",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override default model for the selected provider",
    )
    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="Shorthand for --provider local",
    )
    args = parser.parse_args()

    provider = "local" if args.local else args.provider
    api_key = os.getenv("GOOGLE_API_KEY" if provider == "gemini" else "DEEPSEEK_API_KEY")

    print("\nPOIROT - Automated Multi-Agent Forensics\n")

    if provider == "local":
        print("Using LOCAL LLM via LM Studio (http://localhost:1234)\n")
    elif provider == "ollama":
        print("Using Ollama LOCAL LLM (http://localhost:11434)\n")
    elif provider == "deepseek":
        print("Using DeepSeek API\n")

    # 1. Validate database and select session
    sessions = get_available_sessions(DB_PATH)
    if not sessions:
        print("No sessions found to analyze. Exiting.")
        return

    selected_session_id = select_session(sessions)
    print(f"\nTarget Session Selected: {selected_session_id}")

    # 2. Output directory alongside the database
    db_parent_dir = os.path.dirname(os.path.abspath(DB_PATH))
    output_path = os.path.join(db_parent_dir, f"POIROT_Analysis_{SYSTEM_NAME}")
    print(f"Output Directory: {output_path}\n")

    # 3. Run full analysis via the public API
    try:
        results = poirot.run_poirot(
            database_path=DB_PATH,
            system_name=SYSTEM_NAME,
            system_description=SYSTEM_DESCRIPTION,
            provider=provider,
            api_key=api_key,
            session_id=selected_session_id,
            output_dir=output_path,
            model=args.model,
            ignore_list=IGNORE_LIST,
            include_tool_calls=INCLUDE_TOOL_CALLS,
            include_broadcast_messages=INCLUDE_BROADCAST_MESSAGES,
        )

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Full report saved in: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
