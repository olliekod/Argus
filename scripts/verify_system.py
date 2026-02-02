"""
Argus System Verification
=========================

Comprehensive test to verify all components work correctly.
Run: python scripts/verify_system.py
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def ok(msg):
    print(f"  {GREEN}[OK]{RESET} {msg}")

def fail(msg, error=None):
    print(f"  {RED}[FAIL]{RESET} {msg}")
    if error:
        print(f"        {error}")

def warn(msg):
    print(f"  {YELLOW}[WARN]{RESET} {msg}")

def info(msg):
    print(f"  {CYAN}[INFO]{RESET} {msg}")


async def verify_system():
    """Run all verification checks."""
    print("=" * 60)
    print("ARGUS SYSTEM VERIFICATION")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {"passed": 0, "failed": 0, "warnings": 0}
    
    # === 1. CORE IMPORTS ===
    print("1. Core Imports")
    print("-" * 40)
    
    try:
        from src.core.database import Database
        ok("Database")
        results["passed"] += 1
    except Exception as e:
        fail("Database", str(e))
        results["failed"] += 1
    
    try:
        from src.core.config import load_all_config
        ok("Config")
        results["passed"] += 1
    except Exception as e:
        fail("Config", str(e))
        results["failed"] += 1
    
    try:
        from src.core.economic_calendar import EconomicCalendar
        ok("Economic Calendar")
        results["passed"] += 1
    except Exception as e:
        fail("Economic Calendar", str(e))
        results["failed"] += 1
    
    try:
        from src.core.sentiment_collector import SentimentCollector
        ok("Sentiment Collector")
        results["passed"] += 1
    except Exception as e:
        fail("Sentiment Collector", str(e))
        results["failed"] += 1
    
    print()
    
    # === 2. CONNECTORS ===
    print("2. Connectors")
    print("-" * 40)
    
    try:
        from src.connectors.bybit_ws import BybitWebSocket
        ok("Bybit WebSocket")
        results["passed"] += 1
    except Exception as e:
        fail("Bybit WebSocket", str(e))
        results["failed"] += 1
    
    try:
        from src.connectors.coinbase_client import CoinbaseClient
        ok("Coinbase Client")
        results["passed"] += 1
    except Exception as e:
        fail("Coinbase Client", str(e))
        results["failed"] += 1
    
    try:
        from src.connectors.deribit_client import DeribitClient
        ok("Deribit Client")
        results["passed"] += 1
    except Exception as e:
        fail("Deribit Client", str(e))
        results["failed"] += 1
    
    try:
        from src.connectors.yahoo_client import YahooFinanceClient
        ok("Yahoo Finance Client")
        results["passed"] += 1
    except Exception as e:
        fail("Yahoo Finance Client", str(e))
        results["failed"] += 1
    
    try:
        from src.connectors.ibit_options_client import IBITOptionsClient
        ok("Options Client (IBIT/BITO)")
        results["passed"] += 1
    except Exception as e:
        fail("Options Client", str(e))
        results["failed"] += 1
    
    print()
    
    # === 3. DETECTORS ===
    print("3. Detectors")
    print("-" * 40)
    
    try:
        from src.detectors.ibit_detector import IBITDetector
        ok("ETF Options Detector (IBIT/BITO)")
        results["passed"] += 1
    except Exception as e:
        fail("ETF Options Detector", str(e))
        results["failed"] += 1
    
    try:
        from src.detectors.funding_detector import FundingDetector
        ok("Funding Detector")
        results["passed"] += 1
    except Exception as e:
        fail("Funding Detector", str(e))
        results["failed"] += 1
    
    print()
    
    # === 4. ANALYSIS ===
    print("4. Analysis Components")
    print("-" * 40)
    
    try:
        from src.analysis.paper_trader import PaperTrader
        ok("Paper Trader")
        results["passed"] += 1
    except Exception as e:
        fail("Paper Trader", str(e))
        results["failed"] += 1
    
    try:
        from src.analysis.trade_calculator import TradeCalculator
        ok("Trade Calculator")
        results["passed"] += 1
    except Exception as e:
        fail("Trade Calculator", str(e))
        results["failed"] += 1
    
    try:
        from src.analysis.backtester import StrategyBacktester
        ok("Backtester")
        results["passed"] += 1
    except Exception as e:
        fail("Backtester", str(e))
        results["failed"] += 1
    
    try:
        from src.analysis.greeks_engine import GreeksEngine
        ok("Greeks Engine")
        results["passed"] += 1
    except Exception as e:
        fail("Greeks Engine", str(e))
        results["failed"] += 1
    
    print()
    
    # === 5. ALERTS ===
    print("5. Alerts")
    print("-" * 40)
    
    try:
        from src.alerts.telegram_bot import TelegramBot
        ok("Telegram Bot")
        results["passed"] += 1
    except Exception as e:
        fail("Telegram Bot", str(e))
        results["failed"] += 1
    
    print()
    
    # === 6. ORCHESTRATOR ===
    print("6. Main Orchestrator")
    print("-" * 40)
    
    try:
        from src.orchestrator import ArgusOrchestrator
        ok("ArgusOrchestrator")
        results["passed"] += 1
    except Exception as e:
        fail("ArgusOrchestrator", str(e))
        results["failed"] += 1
    
    print()
    
    # === 7. CONFIG FILES ===
    print("7. Configuration Files")
    print("-" * 40)
    
    config_dir = Path(__file__).parent.parent / "config"
    
    if (config_dir / "secrets.yaml").exists():
        ok("secrets.yaml exists")
        results["passed"] += 1
    else:
        fail("secrets.yaml missing")
        results["failed"] += 1
    
    if (config_dir / "thresholds.yaml").exists():
        # Check for BITO config
        content = (config_dir / "thresholds.yaml").read_text(encoding='utf-8')
        if 'bito:' in content:
            ok("thresholds.yaml (IBIT + BITO configured)")
        else:
            ok("thresholds.yaml (IBIT only)")
        results["passed"] += 1
    else:
        warn("thresholds.yaml missing (using defaults)")
        results["warnings"] += 1
    
    print()
    
    # === 8. DATABASE ===
    print("8. Database")
    print("-" * 40)
    
    db_path = Path(__file__).parent.parent / "data" / "argus.db"
    if db_path.exists():
        ok(f"Database exists ({db_path.stat().st_size / 1024:.1f} KB)")
        results["passed"] += 1
        
        # Check paper trades count
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM paper_trades")
            count = cursor.fetchone()[0]
            conn.close()
            if count > 0:
                info(f"Paper trades logged: {count}")
            else:
                info("No paper trades yet")
        except:
            pass
    else:
        warn("Database doesn't exist yet (will be created on first run)")
        results["warnings"] += 1
    
    print()
    
    # === 9. LIVE DATA TESTS ===
    print("9. Live Data Tests")
    print("-" * 40)
    
    # Test IBIT Options Client
    try:
        from src.connectors.ibit_options_client import IBITOptionsClient
        client = IBITOptionsClient(symbol="IBIT")
        price = client.get_current_price()
        if price > 0:
            ok(f"IBIT Price: ${price:.2f}")
            results["passed"] += 1
        else:
            warn("IBIT price unavailable (market closed?)")
            results["warnings"] += 1
    except Exception as e:
        fail("IBIT Price", str(e))
        results["failed"] += 1
    
    # Test BITO Options Client
    try:
        from src.connectors.ibit_options_client import IBITOptionsClient
        client = IBITOptionsClient(symbol="BITO")
        price = client.get_current_price()
        if price > 0:
            ok(f"BITO Price: ${price:.2f}")
            results["passed"] += 1
        else:
            warn("BITO price unavailable (market closed?)")
            results["warnings"] += 1
    except Exception as e:
        fail("BITO Price", str(e))
        results["failed"] += 1
    
    # Test sentiment
    try:
        from src.core.sentiment_collector import SentimentCollector
        collector = SentimentCollector()
        data = await collector.get_sentiment()
        if data and data.fear_greed_value > 0:
            ok(f"Fear & Greed: {data.fear_greed_value} ({data.fear_greed_label})")
            results["passed"] += 1
        else:
            warn("Fear & Greed returned empty")
            results["warnings"] += 1
    except Exception as e:
        fail("Fear & Greed API", str(e))
        results["failed"] += 1
    
    # Test economic calendar
    try:
        from src.core.economic_calendar import EconomicCalendar
        cal = EconomicCalendar()
        is_blackout, reason = cal.is_blackout_period()
        ok(f"Economic Calendar: {'BLACKOUT - ' + reason if is_blackout else 'Clear'}")
        results["passed"] += 1
    except Exception as e:
        fail("Economic Calendar", str(e))
        results["failed"] += 1
    
    print()
    
    # === SUMMARY ===
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = results["passed"] + results["failed"]
    print(f"  Passed:   {GREEN}{results['passed']}/{total}{RESET}")
    print(f"  Failed:   {RED}{results['failed']}{RESET}")
    print(f"  Warnings: {YELLOW}{results['warnings']}{RESET}")
    print()
    
    if results["failed"] == 0:
        print(f"{GREEN}System is ready for paper trading!{RESET}")
        print()
        print("Configured Tickers:")
        print(f"  • IBIT (BlackRock Bitcoin ETF)")
        print(f"  • BITO (ProShares Bitcoin ETF)")
        print()
        print("Next steps:")
        print("  1. Double-click 'Start Argus.vbs' to start monitoring")
        print("  2. Check Telegram for alerts")
        print("  3. Run 'python scripts\\paper_performance.py' to see trades")
    else:
        print(f"{RED}Some components failed. Please review errors above.{RESET}")
    
    print()
    print("=" * 60)
    
    return results["failed"] == 0


if __name__ == "__main__":
    success = asyncio.run(verify_system())
    sys.exit(0 if success else 1)
