"""
Argus Full End-to-End Test Suite
=================================

Comprehensive test covering the entire application:
- All connectors
- All detectors
- All core modules
- Orchestrator integration
- Alerts system

Usage: python test_argus_e2e.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, date, timedelta, timezone

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = []
        self.failed = []
    
    def add(self, name: str, success: bool, error: str = None):
        if success:
            self.passed.append(name)
        else:
            self.failed.append((name, error))
    
    def summary(self):
        total = len(self.passed) + len(self.failed)
        return len(self.passed), total


results = TestResults()


def test_import(module_path: str, name: str):
    """Test that a module can be imported."""
    try:
        __import__(module_path)
        results.add(name, True)
        return True
    except Exception as e:
        results.add(name, False, str(e))
        return False


# Prevent pytest from collecting this helper as a test.
test_import.__test__ = False


async def run_all_tests():
    """Run comprehensive end-to-end tests."""
    print("\n" + "=" * 60)
    print("🔍 ARGUS FULL E2E TEST SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # =========================================================================
    # SECTION 1: Core Module Imports
    # =========================================================================
    print("📦 SECTION 1: Core Modules")
    print("-" * 40)
    
    core_modules = [
        ("src.core.config", "Config"),
        ("src.core.database", "Database"),
        ("src.core.logger", "Logger"),
        ("src.core.utils", "Utils"),
        ("src.core.conditions_monitor", "Conditions Monitor"),
        ("src.core.gap_risk_tracker", "Gap Risk Tracker"),
        ("src.core.reddit_monitor", "Reddit Monitor"),
        ("src.core.warmth_monitor", "Warmth Monitor"),
        ("src.core.economic_calendar", "Economic Calendar"),
        ("src.core.sentiment_collector", "Sentiment Collector"),
    ]
    
    for module, name in core_modules:
        success = test_import(module, name)
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    # =========================================================================
    # SECTION 2: Connector Imports
    # =========================================================================
    print("\n🔌 SECTION 2: Connectors")
    print("-" * 40)
    
    connectors = [
        ("src.connectors.bybit_ws", "Bybit WebSocket"),
        ("src.connectors.coinbase_client", "Coinbase Client"),
        ("src.connectors.okx_client", "OKX Client"),
        ("src.connectors.deribit_client", "Deribit Client"),
        ("src.connectors.coinglass_client", "Coinglass Client"),
        ("src.connectors.yahoo_client", "Yahoo Finance Client"),
        ("src.connectors.binance_ws", "Binance WebSocket"),
        ("src.connectors.ibit_options_client", "IBIT Options Client"),
    ]
    
    for module, name in connectors:
        success = test_import(module, name)
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    # =========================================================================
    # SECTION 3: Detector Imports
    # =========================================================================
    print("\n🎯 SECTION 3: Detectors")
    print("-" * 40)
    
    detectors = [
        ("src.detectors.base_detector", "Base Detector"),
        ("src.detectors.funding_detector", "Funding Detector"),
        ("src.detectors.basis_detector", "Basis Detector"),
        ("src.detectors.cross_exchange_detector", "Cross Exchange Detector"),
        ("src.detectors.liquidation_detector", "Liquidation Detector"),
        ("src.detectors.options_iv_detector", "Options IV Detector"),
        ("src.detectors.volatility_detector", "Volatility Detector"),
        ("src.detectors.ibit_detector", "IBIT Detector"),
    ]
    
    for module, name in detectors:
        success = test_import(module, name)
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    # =========================================================================
    # SECTION 4: Alerts & Analysis
    # =========================================================================
    print("\n📢 SECTION 4: Alerts & Analysis")
    print("-" * 40)
    
    alerts = [
        ("src.alerts.telegram_bot", "Telegram Bot"),
        ("src.analysis.daily_review", "Daily Review"),
    ]
    
    for module, name in alerts:
        success = test_import(module, name)
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    # =========================================================================
    # SECTION 5: Orchestrator & Trading
    # =========================================================================
    print("\n🎛️ SECTION 5: Orchestrator & Trading")
    print("-" * 40)
    
    trading_modules = [
        ("src.orchestrator", "Orchestrator"),
        ("src.analysis.gpu_engine", "GPU Engine"),
        ("src.trading.paper_trader_farm", "Paper Trader Farm"),
        ("src.trading.trader_config_generator", "Trader Generator"),
    ]
    
    for module, name in trading_modules:
        success = test_import(module, name)
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    # =========================================================================
    # SECTION 6: Functional Tests (GPU & Farm)
    # =========================================================================
    print("\n⚙️ SECTION 6: Functional Tests (GPU & Farm)")
    print("-" * 40)
    
    # Test GPU Engine
    try:
        from src.analysis.gpu_engine import get_gpu_engine
        gpu = get_gpu_engine()
        # Test basic PoP
        pop = gpu.monte_carlo_pop(S=100.0, short_strike=90.0, long_strike=85.0, credit=1.0, T=0.1, sigma=0.5, simulations=10000)
        assert 0 <= pop <= 100
        results.add("GPU: Monte Carlo PoP", True)
        print(f"  ✅ GPU: Monte Carlo PoP (10K sims, Result: {pop:.1f}%)")
        
        # Test Heston PoP & Touch Risk
        h_res = gpu.monte_carlo_pop_heston(S=42.0, short_strike=38.0, long_strike=34.0, credit=0.80, T=0.1, v0=0.65**2, simulations=10000)
        assert 'pop' in h_res and 'prob_of_touch_stop' in h_res
        results.add("GPU: Heston Model & Touch Risk", True)
        print(f"  ✅ GPU: Heston Model & Touch Risk (PoP: {h_res['pop']:.1f}%, Touch: {h_res['prob_of_touch_stop']:.1f}%)")
        
        # Test Batch Greeks
        batch = gpu.batch_greeks(100.0, [95, 100, 105], 0.1, [0.4, 0.4, 0.4])
        assert len(batch['delta']) == 3
        results.add("GPU: Batch Greeks (Vectorized)", True)
        print("  ✅ GPU: Batch Greeks (Vectorized)")
    except Exception as e:
        results.add("GPU Infrastructure", False, str(e))
        print(f"  ❌ GPU Infrastructure: {e}")
    
    # Test Trader Farm Scaling & Cap
    # Test Trader Farm Scaling & Cap
    try:
        from src.trading.trader_config_generator import generate_all_configs
        configs = generate_all_configs(full_coverage=True)
        # Verify 400K cap
        assert len(configs) <= 400000, f"Farm over cap! Found {len(configs)} traders"
        results.add("Farm: 400K Population Cap", True)
        print(f"  ✅ Farm: 400K Population Cap ({len(configs):,} traders)")
        
        # Verify Budget Tiers (2%, 5%, 9%, 14%)
        budgets = set(c.position_size_pct for c in configs)
        expected_budgets = {2, 5, 9, 14}
        assert expected_budgets.issubset(budgets), f"Missing budget tiers. Found: {budgets}"
        results.add("Farm: Budget Tiers (2, 5, 9, 14%)", True)
        print(f"  ✅ Farm: Budget Tiers (2, 5, 9, 14%) verified")
        
        # Test Portfolio Nuke (Stress Test)
        from src.analysis.gpu_engine import get_gpu_engine
        import torch
        gpu = get_gpu_engine()
        # Create small mock tensor for 100 traders
        mock_params = torch.rand((100, 8), device=gpu._device)
        nuke_res = gpu.portfolio_stress_test(mock_params, S=42, T=0.1, v0=0.65**2, num_paths=100)
        assert 'avg_stopped_pct' in nuke_res
        results.add("Farm: Nuke Stress Test", True)
        print(f"  ✅ Farm: Nuke Stress Test (Stopped: {nuke_res['avg_stopped_pct']:.1f}%)")
        
        # Test session filter logic
        c = configs[0]
        assert hasattr(c, 'session_filter')
        results.add("Farm: Configuration Schema", True)
        print("  ✅ Farm: Configuration Schema")
    except Exception as e:
        results.add("Farm Scaling Tests", False, str(e))
        print(f"  ❌ Farm Scaling Tests: {e}")

    # Test IBIT Detector + Farm Signal Loop
    try:
        from src.detectors.ibit_detector import IBITDetector
        from src.trading.paper_trader_farm import PaperTraderFarm
        
        farm = PaperTraderFarm(db=None, full_coverage=False, total_traders=1000)
        await farm.initialize()
        
        detector = IBITDetector({}, None)
        detector.set_paper_trader_farm(farm)
        
        # Simulate a signal and evaluate
        # Note: We need a mock recommendation
        from dataclasses import dataclass
        @dataclass
        class MockRec:
            dte: int = 14
            probability_of_profit: float = 75
            short_strike: float = 40
            long_strike: float = 35
            net_credit: float = 0.50
            expiration: str = "2026-02-14"
            
        # Prepare a "market hours" signal (10:30 AM ET)
        market_time = datetime.now(timezone.utc).replace(hour=10, minute=30, second=0, microsecond=0)
        
        farm_trades = await farm.evaluate_signal(
            symbol="IBIT",
            signal_data={
                'iv': 65,
                'warmth': 8,
                'dte': 45,
                'pop': 75,
                'direction': 'bullish',
                'credit': 0.50,
                'timestamp': market_time.isoformat()
            }
        )
        assert len(farm_trades) > 0, f"No traders entered from a population of {len(farm.trader_configs)}. Check thresholds vs signal."
        assert len(farm.active_traders) == len(farm_trades), "Lazy loading failed: active_traders count mismatch"
        results.add("Signal Loop: Detector -> Farm", True)
        print(f"  ✅ Signal Loop: Detector -> Farm ({len(farm_trades)} traders entered, lazy loading OK)")
    except Exception as e:
        results.add("Signal Loop Integration", False, str(e))
        print(f"  ❌ Signal Loop Integration: {e}")

    # Test Database Persistence
    try:
        from src.core.database import Database
        db = Database(":memory:")
        await db.connect()
        await db.execute("CREATE TABLE test (val TEXT)")
        await db.execute("INSERT INTO test VALUES (?)", ("PASS",))
        res = await db.fetch_one("SELECT val FROM test")
        assert res[0] == "PASS"
        await db.close()
        results.add("Database Persistence", True)
        print("  ✅ Database Persistence")
    except Exception as e:
        results.add("Database Persistence", False, str(e))
        print(f"  ❌ Database Persistence: {e}")

    # =========================================================================
    # SECTION 7: System-Wide Checks
    # =========================================================================
    print("\n🌍 SECTION 7: System-Wide Checks")
    print("-" * 40)
    
    # Review + GPU integration
    try:
        from src.analysis.daily_review import DailyReview, DailyReviewData
        review = DailyReview(starting_balance=5000)
        data = DailyReviewData(
            date=date.today(),
            trades_today=5,
            pnl_today=120.0,
            pnl_today_pct=2.4,
            open_positions=[],
            trades_mtd=20,
            pnl_mtd=500.0,
            pnl_mtd_pct=10.0,
            win_rate_mtd=85.0,
            account_value=5500.0,
            starting_balance=5000.0,
            conditions_score=8,
            conditions_label="warming",
            btc_iv=65.0,
            gap_risk_level="low",
            gpu_stats="RTX 4080: 1.2B sims/sec"  # The new field
        )
        report = review.format_review(data)
        assert "RTX 4080" in report
        results.add("Review: GPU Stats Integration", True)
        print("  ✅ Review: GPU Stats Integration")
    except Exception as e:
        results.add("Daily Review System", False, str(e))
        print(f"  ❌ Daily Review System: {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    passed, total = results.summary()
    
    print("\n" + "=" * 60)
    print("📊 MASTER VERIFICATION SUMMARY")
    print("=" * 60)
    
    if results.failed:
        print("\n❌ FAILURES:")
        for name, error in results.failed:
            print(f"  • {name}: {error[:80]}...")
    
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL SYSTEMS GREEN! Argus is ready for market open.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} components offline. Do not deploy yet.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
