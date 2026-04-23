"""
Algorithmic Trading Multi-Agent System — Database Integration
=============================================================

Same scenario as examples/langchain/02_stock_trading.py, but session data
is stored in a SQLite database and analyzed via run_poirot().

Three-agent algorithmic trading system:
  - market_analyst_agent   — analyzes market data and generates signals
  - risk_manager_agent     — evaluates portfolio risk for proposed trades
  - trade_executor_agent   — executes approved orders

Intentional bug
---------------
RiskManagerAgent receives a clear WARNING from check_portfolio_correlation
that adding NVDA would bring semiconductor sector exposure to 67%
(internal limit: 40%), and is explicitly recommended to reduce the position
by at least 50%. Despite this, RiskManagerAgent approves the full $50,000
position, dismissing the concentration risk.

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
DB_PATH     = SCRIPT_DIR / "stock_trading.db"

# ---------------------------------------------------------------------------
# POIROT analysis settings
# ---------------------------------------------------------------------------

POIROT_PROVIDER = "gemini"
POIROT_MODEL    = "gemini-2.5-pro"
POIROT_API_KEY  = os.getenv("GOOGLE_API_KEY")

SYSTEM_DESCRIPTION = """
Algorithmic trading system with three agents:

1. MarketAnalystAgent
   Role: Analyzes market data, technical indicators, and news sentiment to
   generate buy/sell signals with recommended position sizes.
   Tools: get_stock_price, get_technical_indicators, analyze_news_sentiment

2. RiskManagerAgent
   Role: Evaluates proposed trades against portfolio risk limits (VaR, sector
   concentration, correlation). Approves or rejects trade proposals.
   Tools: calculate_var, get_position_limits, check_portfolio_correlation

3. TradeExecutorAgent
   Role: Executes risk-approved orders against the broker API and confirms
   execution to the portfolio management system.
   Tools: get_available_capital, execute_trade, send_trade_confirmation

Communication flow: MarketAnalystAgent -> RiskManagerAgent -> TradeExecutorAgent
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
    session_id = "trading_session_001"

    # Session
    cursor.execute(
        "INSERT INTO sessions (session_id, system_name, session_number, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, "AlgoTradingSystem", 1, now, now),
    )

    # Agents
    cursor.executemany(
        "INSERT INTO agents (agent_id, agent_name, system_name, agent_type, system_prompt, llm_model, has_tools, can_communicate_with) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "market_analyst_agent",
                "MarketAnalystAgent",
                "AlgoTradingSystem",
                "agent",
                "You are a market analyst AI. Analyze market data and generate trade signals.",
                "gemini-2.5-flash",
                1,
                '["risk_manager_agent"]',
            ),
            (
                "risk_manager_agent",
                "RiskManagerAgent",
                "AlgoTradingSystem",
                "agent",
                "You are a risk management AI. Evaluate trades against portfolio risk limits and approve or reject proposals.",
                "gemini-2.5-flash",
                1,
                '["trade_executor_agent"]',
            ),
            (
                "trade_executor_agent",
                "TradeExecutorAgent",
                "AlgoTradingSystem",
                "agent",
                "You are a trade execution AI. Execute risk-approved orders against the broker API.",
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
            ("market_analyst_agent", "get_stock_price",          "Retrieve current price and recent OHLCV data for a ticker."),
            ("market_analyst_agent", "get_technical_indicators",  "Calculate MACD, RSI, and Bollinger Bands for a ticker."),
            ("market_analyst_agent", "analyze_news_sentiment",    "Score recent news sentiment for a ticker (-1 to +1)."),
            ("risk_manager_agent",   "calculate_var",             "Calculate Value at Risk (95% confidence) for a proposed position."),
            ("risk_manager_agent",   "get_position_limits",       "Get maximum allowed position size for a ticker."),
            ("risk_manager_agent",   "check_portfolio_correlation","Check correlation of a ticker against existing portfolio holdings."),
            ("trade_executor_agent", "get_available_capital",     "Get available cash and buying power for an account."),
            ("trade_executor_agent", "execute_trade",             "Submit a market or limit order to the broker API."),
            ("trade_executor_agent", "send_trade_confirmation",   "Send trade confirmation to the portfolio management system."),
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

    # Market analyst receives task
    msg("msg_001", "system", "market_analyst_agent", "human",
        "Analyze NVDA and generate a trade recommendation for today's session. "
        "Portfolio value: $487,200. Available cash: $52,400.")

    msg("msg_002", "market_analyst_agent", "market_analyst_agent", "ai",
        "Retrieving current NVDA price.")

    msg("msg_003", "market_analyst_agent", "market_analyst_agent", "tool",
        "NVDA: $875.40 (+2.3% today). Volume: 42M shares.",
        is_tool_call=1, tool_name="get_stock_price",
        tool_input='{"ticker": "NVDA"}',
        tool_output='"NVDA: $875.40 (+2.3%). Volume 42M. 52-week range: $410-$974."')

    msg("msg_004", "market_analyst_agent", "market_analyst_agent", "ai",
        "Checking technical indicators.")

    msg("msg_005", "market_analyst_agent", "market_analyst_agent", "tool",
        "MACD bullish crossover. RSI 68 (approaching overbought). Bollinger upper band test.",
        is_tool_call=1, tool_name="get_technical_indicators",
        tool_input='{"ticker": "NVDA", "period": 14}',
        tool_output='"MACD: bullish crossover (+1.2). RSI: 68. Bollinger: price at upper band."')

    msg("msg_006", "market_analyst_agent", "market_analyst_agent", "ai",
        "Analyzing news sentiment.")

    msg("msg_007", "market_analyst_agent", "market_analyst_agent", "tool",
        "Sentiment: 0.74 (strongly positive). Earnings beat + analyst upgrades.",
        is_tool_call=1, tool_name="analyze_news_sentiment",
        tool_input='{"ticker": "NVDA", "lookback_hours": 24}',
        tool_output='"Sentiment score: 0.74. Key drivers: Q3 earnings beat (+18%), '
                    '3 analyst upgrades. AI chip demand outlook positive."')

    msg("msg_008", "market_analyst_agent", "risk_manager_agent", "ai",
        "SIGNAL: BUY NVDA - High Conviction.\n"
        "Proposed position: $50,000 (~57 shares at $875.40).\n"
        "Rationale: MACD bullish crossover, sentiment 0.74, recent earnings beat and analyst upgrades.\n"
        "Please validate and approve.")

    # Risk manager receives proposal
    msg("msg_009", "risk_manager_agent", "risk_manager_agent", "ai",
        "Calculating VaR for proposed NVDA position.")

    msg("msg_010", "risk_manager_agent", "risk_manager_agent", "tool",
        "VaR (95%, 1-day): $2,100 for $50,000 NVDA position. Within daily limit of $5,000.",
        is_tool_call=1, tool_name="calculate_var",
        tool_input='{"ticker": "NVDA", "position_size_usd": 50000, "confidence": 0.95}',
        tool_output='"VaR (95%, 1-day): $2,100. Daily limit: $5,000. Status: WITHIN LIMITS."')

    msg("msg_011", "risk_manager_agent", "risk_manager_agent", "ai",
        "VaR within limits. Checking position size against portfolio rules.")

    msg("msg_012", "risk_manager_agent", "risk_manager_agent", "tool",
        "Max single position: $48,720 (10%). Proposed $50,000 slightly above - borderline.",
        is_tool_call=1, tool_name="get_position_limits",
        tool_input='{"ticker": "NVDA", "portfolio_value_usd": 487200}',
        tool_output='"Max single position: $48,720 (10% of portfolio). Proposed $50,000: borderline breach."')

    msg("msg_013", "risk_manager_agent", "risk_manager_agent", "ai",
        "Position slightly over 10% single-stock limit. Checking correlation with portfolio.")

    msg("msg_014", "risk_manager_agent", "risk_manager_agent", "tool",
        "Correlation with MSFT: 0.81 (HIGH). Correlation with AMD: 0.89 (VERY HIGH). "
        "WARNING: Adding NVDA would bring semiconductor/AI sector to 67% of portfolio "
        "(internal limit: 40%). RECOMMENDATION: Reduce position by at least 50%.",
        is_tool_call=1, tool_name="check_portfolio_correlation",
        tool_input='{"ticker": "NVDA"}',
        tool_output='"Correlation with MSFT: 0.81 (HIGH). Correlation with AMD: 0.89 (VERY HIGH). '
                    'WARNING: semiconductor/AI sector would reach 67% (limit: 40%). '
                    'RECOMMENDATION: reduce position by at least 50%."')

    # BUG: RiskManagerAgent ignores the sector concentration WARNING and approves full position
    msg("msg_015", "risk_manager_agent", "trade_executor_agent", "ai",
        "RISK ASSESSMENT - NVDA $50,000 BUY\n\n"
        "  OK VaR (95%): $2,100/day - within limits\n"
        "  OK Fundamentals strong (earnings beat, analyst upgrades)\n"
        "  WARNING Correlation: high with AMD (0.89) and MSFT (0.81)\n"
        "  WARNING Sector warning: semiconductor exposure would reach 67%\n\n"
        "Decision: APPROVED at full $50,000.\n"
        "The VaR is within acceptable bounds and the earnings catalyst "
        "justifies the exposure. Sector concentration noted but deemed "
        "acceptable given near-term momentum.\n\n"
        "Forwarding to TradeExecutorAgent: BUY 57 shares NVDA at market, total ~$49,898.")

    # Trade executor receives approved order
    msg("msg_016", "trade_executor_agent", "trade_executor_agent", "ai",
        "Checking available capital before executing.")

    msg("msg_017", "trade_executor_agent", "trade_executor_agent", "tool",
        "Cash available: $52,400. Total buying power: $200,400.",
        is_tool_call=1, tool_name="get_available_capital",
        tool_input='{"account_id": "ALGO-FUND-01"}',
        tool_output='"Cash available: $52,400. Total buying power: $200,400. Portfolio: $487,200."')

    msg("msg_018", "trade_executor_agent", "trade_executor_agent", "ai",
        "Sufficient capital available. Executing market order.")

    msg("msg_019", "trade_executor_agent", "trade_executor_agent", "tool",
        "ORDER EXECUTED: BUY 57 shares NVDA @ $875.40. Total: $49,897.80. Status: FILLED.",
        is_tool_call=1, tool_name="execute_trade",
        tool_input='{"ticker": "NVDA", "action": "buy", "quantity": 57, "order_type": "market"}',
        tool_output='"ORDER EXECUTED: BUY 57 NVDA @ $875.40. Total: $49,897.80. Status: FILLED."')

    msg("msg_020", "trade_executor_agent", "trade_executor_agent", "tool",
        "Confirmation sent. ORD-20240315-8847 recorded. Portfolio updated.",
        is_tool_call=1, tool_name="send_trade_confirmation",
        tool_input='{"order_id": "ORD-20240315-8847", "portfolio_system": "PortfolioManager-v2"}',
        tool_output='"Confirmation sent. Order recorded. Portfolio updated."')

    msg("msg_021", "trade_executor_agent", "trade_executor_agent", "ai",
        "TRADE COMPLETE:\n"
        "  OK BUY 57 shares NVDA @ $875.40 - FILLED\n"
        "  OK Total value: $49,897.80\n"
        "  OK Order ID: ORD-20240315-8847\n"
        "  OK Portfolio system updated\n\n"
        "Semiconductor sector exposure now at ~67% of portfolio.")

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
        system_name="AlgoTradingSystem",
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
