"""
Paper Trading Test Suite
=========================

Comprehensive tests for paper trading infrastructure:
- TraderConfig generation and validation
- Entry condition evaluation (6 conditions)
- Exit strategy logic (6 strategies)
- Paper trader farm operations
- Leaderboard and reporting

Run with: python test_paper_trader.py
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, '.')


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = []
        self.failed = []
    
    def add(self, name: str, passed: bool, error: str = None):
        if passed:
            self.passed.append(name)
        else:
            self.failed.append((name, error or "Unknown error"))
    
    def summary(self):
        return len(self.passed), len(self.passed) + len(self.failed)


async def run_paper_trader_tests():
    """Run all paper trading tests."""
    print("=" * 60)
    print("PAPER TRADING TEST SUITE")
    print("=" * 60)
    
    results = TestResults()
    
    # =========================================================================
    # SECTION 1: Imports
    # =========================================================================
    print("\n📦 SECTION 1: Import Tests")
    print("-" * 40)
    
    try:
        from src.trading.paper_trader import (
            PaperTrader, TraderConfig, PaperTrade, StrategyType
        )
        results.add("Paper Trader Imports", True)
        print("  ✅ paper_trader imports")
    except Exception as e:
        results.add("Paper Trader Imports", False, str(e))
        print(f"  ❌ paper_trader imports: {e}")
        return 1
    
    try:
        from src.trading.paper_trader_farm import PaperTraderFarm
        results.add("Paper Trader Farm Import", True)
        print("  ✅ paper_trader_farm imports")
    except Exception as e:
        results.add("Paper Trader Farm Import", False, str(e))
        print(f"  ❌ paper_trader_farm imports: {e}")
    
    try:
        from src.trading.trader_config_generator import (
            generate_all_configs, get_config_summary, get_total_combinations,
            PARAM_RANGES, STRATEGY_IV_RANGES
        )
        results.add("Config Generator Import", True)
        print("  ✅ trader_config_generator imports")
    except Exception as e:
        results.add("Config Generator Import", False, str(e))
        print(f"  ❌ trader_config_generator imports: {e}")
    
    # =========================================================================
    # SECTION 2: TraderConfig Tests
    # =========================================================================
    print("\n⚙️ SECTION 2: TraderConfig Tests")
    print("-" * 40)
    
    from src.trading.paper_trader import TraderConfig, StrategyType, PaperTrader
    
    # Test config creation
    try:
        config = TraderConfig(
            trader_id="TEST-001",
            strategy_type=StrategyType.BULL_PUT,
            iv_min=55.0,
            iv_max=90.0,
            warmth_min=6,
            pop_min=65.0,
            dte_target=45,
            dte_min=30,
            dte_max=60,
            gap_tolerance_pct=10.0,
            profit_target_pct=50.0,
            stop_loss_pct=200.0,
            position_size_pct=9.0,
            max_risk_dollars=500.0,
        )
        assert config.trader_id == "TEST-001"
        assert config.strategy_type == StrategyType.BULL_PUT
        results.add("TraderConfig Creation", True)
        print("  ✅ TraderConfig creation")
    except Exception as e:
        results.add("TraderConfig Creation", False, str(e))
        print(f"  ❌ TraderConfig creation: {e}")
    
    # Test config serialization
    try:
        config_dict = config.to_dict()
        assert config_dict['trader_id'] == "TEST-001"
        assert config_dict['strategy_type'] == "bull_put"
        restored = TraderConfig.from_dict(config_dict)
        assert restored.trader_id == config.trader_id
        results.add("Config Serialization", True)
        print("  ✅ Config serialization/deserialization")
    except Exception as e:
        results.add("Config Serialization", False, str(e))
        print(f"  ❌ Config serialization: {e}")
    
    # =========================================================================
    # SECTION 3: Entry Condition Tests (6 conditions)
    # =========================================================================
    print("\n🚪 SECTION 3: Entry Condition Tests")
    print("-" * 40)
    
    trader = PaperTrader(config, db=None)
    
    # Test 1: IV bounds
    try:
        # IV too low
        result = trader.should_enter(
            symbol="IBIT", current_iv=50.0, warmth_score=7,
            dte=45, pop=70.0, gap_risk_pct=5.0, market_direction="bullish"
        )
        assert result == False, "Should reject low IV"
        
        # IV in range
        result = trader.should_enter(
            symbol="IBIT", current_iv=60.0, warmth_score=7,
            dte=45, pop=70.0, gap_risk_pct=5.0, market_direction="bullish"
        )
        assert result == True, "Should accept valid IV"
        
        results.add("Entry: IV Bounds Check", True)
        print("  ✅ IV bounds check")
    except Exception as e:
        results.add("Entry: IV Bounds Check", False, str(e))
        print(f"  ❌ IV bounds check: {e}")
    
    # Test 2: Warmth check
    try:
        result = trader.should_enter(
            symbol="IBIT", current_iv=60.0, warmth_score=4,  # Below min of 6
            dte=45, pop=70.0, gap_risk_pct=5.0, market_direction="bullish"
        )
        assert result == False, "Should reject low warmth"
        results.add("Entry: Warmth Check", True)
        print("  ✅ Warmth check")
    except Exception as e:
        results.add("Entry: Warmth Check", False, str(e))
        print(f"  ❌ Warmth check: {e}")
    
    # Test 3: DTE check
    try:
        result = trader.should_enter(
            symbol="IBIT", current_iv=60.0, warmth_score=7,
            dte=20,  # Below min of 30
            pop=70.0, gap_risk_pct=5.0, market_direction="bullish"
        )
        assert result == False, "Should reject low DTE"
        results.add("Entry: DTE Check", True)
        print("  ✅ DTE check")
    except Exception as e:
        results.add("Entry: DTE Check", False, str(e))
        print(f"  ❌ DTE check: {e}")
    
    # Test 4: PoP check
    try:
        result = trader.should_enter(
            symbol="IBIT", current_iv=60.0, warmth_score=7,
            dte=45, pop=55.0,  # Below min of 65
            gap_risk_pct=5.0, market_direction="bullish"
        )
        assert result == False, "Should reject low PoP"
        results.add("Entry: PoP Check", True)
        print("  ✅ PoP check")
    except Exception as e:
        results.add("Entry: PoP Check", False, str(e))
        print(f"  ❌ PoP check: {e}")
    
    # Test 5: Gap risk check
    try:
        result = trader.should_enter(
            symbol="IBIT", current_iv=60.0, warmth_score=7,
            dte=45, pop=70.0,
            gap_risk_pct=15.0,  # Above max of 10
            market_direction="bullish"
        )
        assert result == False, "Should reject high gap risk"
        results.add("Entry: Gap Risk Check", True)
        print("  ✅ Gap risk check")
    except Exception as e:
        results.add("Entry: Gap Risk Check", False, str(e))
        print(f"  ❌ Gap risk check: {e}")
    
    # Test 6: Strategy-direction alignment
    try:
        # Bull put should reject bearish market
        result = trader.should_enter(
            symbol="IBIT", current_iv=60.0, warmth_score=7,
            dte=45, pop=70.0, gap_risk_pct=5.0,
            market_direction="bearish"  # Mismatch for bull put
        )
        assert result == False, "Bull put should reject bearish"
        results.add("Entry: Strategy-Direction Alignment", True)
        print("  ✅ Strategy-direction alignment")
    except Exception as e:
        results.add("Entry: Strategy-Direction Alignment", False, str(e))
        print(f"  ❌ Strategy-direction alignment: {e}")
    
    # =========================================================================
    # SECTION 4: Trade Entry Tests
    # =========================================================================
    print("\n📈 SECTION 4: Trade Entry Tests")
    print("-" * 40)
    
    try:
        trade = trader.enter_trade(
            symbol="IBIT",
            strikes="$48/$44",
            expiry="2026-03-15",
            entry_credit=1.25,
            contracts=2,
            market_conditions={'iv': 60.0, 'warmth': 7}
        )
        assert trade.id is not None
        assert trade.symbol == "IBIT"
        assert trade.strikes == "$48/$44"
        assert trade.entry_credit == 1.25
        assert trade.contracts == 2
        assert trade.status == "open"
        assert len(trader.open_positions) == 1
        results.add("Trade Entry", True)
        print("  ✅ Trade entry creates valid trade")
    except Exception as e:
        results.add("Trade Entry", False, str(e))
        print(f"  ❌ Trade entry: {e}")
    
    # =========================================================================
    # SECTION 5: Exit Strategy Tests (6 strategies)
    # =========================================================================
    print("\n🚫 SECTION 5: Exit Strategy Tests")
    print("-" * 40)
    
    # Create fresh trader for exit tests
    exit_config = TraderConfig(
        trader_id="EXIT-001",
        strategy_type=StrategyType.BULL_PUT,
        profit_target_pct=50.0,      # Close at 50% profit
        stop_loss_pct=200.0,         # Close at 200% loss
        trailing_stop_pct=25.0,      # Trailing stop if drops 25% from peak
        dte_exit=7,                  # Close at 7 DTE
        time_exit_days=30,           # Close after 30 days
        iv_exit_drop_pct=20.0,       # Close if IV drops 20%
    )
    exit_trader = PaperTrader(exit_config, db=None)
    
    # Test 1: Profit target exit
    try:
        trade = exit_trader.enter_trade(
            symbol="IBIT", strikes="$48/$44", expiry="2026-03-15",
            entry_credit=1.00, contracts=1, market_conditions={'iv': 60.0}
        )
        
        # Simulate profit: price drops to 0.40 (60% profit on credit)
        closed = exit_trader.check_exits(
            current_prices={trade.id: 0.40}  # Profit > 50% target
        )
        assert len(closed) == 1, "Should close on profit target"
        assert closed[0].market_conditions.get('close_reason') == 'PROFIT_TARGET'
        results.add("Exit: Profit Target", True)
        print("  ✅ Profit target exit")
    except Exception as e:
        results.add("Exit: Profit Target", False, str(e))
        print(f"  ❌ Profit target exit: {e}")
    
    # Test 2: Stop loss exit
    try:
        trade = exit_trader.enter_trade(
            symbol="IBIT", strikes="$48/$44", expiry="2026-03-15",
            entry_credit=1.00, contracts=1, market_conditions={'iv': 60.0}
        )
        
        # Simulate loss: price rises to 3.10 (210% of credit = stop loss)
        closed = exit_trader.check_exits(
            current_prices={trade.id: 3.10}
        )
        assert len(closed) == 1, "Should close on stop loss"
        assert closed[0].market_conditions.get('close_reason') == 'STOP_LOSS'
        results.add("Exit: Stop Loss", True)
        print("  ✅ Stop loss exit")
    except Exception as e:
        results.add("Exit: Stop Loss", False, str(e))
        print(f"  ❌ Stop loss exit: {e}")
    
    # Test 3: DTE exit
    try:
        # Set expiry 5 days from now (below dte_exit threshold of 7)
        near_expiry = (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%Y-%m-%d")
        trade = exit_trader.enter_trade(
            symbol="IBIT", strikes="$48/$44", expiry=near_expiry,
            entry_credit=1.00, contracts=1, market_conditions={'iv': 60.0}
        )
        
        # Price is neutral but DTE should trigger
        closed = exit_trader.check_exits(
            current_prices={trade.id: 0.90}  # Small profit, but DTE triggers
        )
        assert len(closed) == 1, "Should close on DTE"
        assert 'DTE_EXIT' in closed[0].market_conditions.get('close_reason', '')
        results.add("Exit: DTE Exit", True)
        print("  ✅ DTE exit")
    except Exception as e:
        results.add("Exit: DTE Exit", False, str(e))
        print(f"  ❌ DTE exit: {e}")
    
    # Test 4: Time exit
    try:
        trade = exit_trader.enter_trade(
            symbol="IBIT", strikes="$48/$44", expiry="2026-04-15",
            entry_credit=1.00, contracts=1, market_conditions={'iv': 60.0}
        )
        # Backdating entry to trigger time exit (use naive datetime to match internal format)
        trade.timestamp = (datetime.now() - timedelta(days=35)).isoformat()
        
        closed = exit_trader.check_exits(
            current_prices={trade.id: 0.90}
        )
        assert len(closed) == 1, "Should close on time limit"
        assert 'TIME_EXIT' in closed[0].market_conditions.get('close_reason', '')
        results.add("Exit: Time Exit", True)
        print("  ✅ Time exit")
    except Exception as e:
        results.add("Exit: Time Exit", False, str(e))
        print(f"  ❌ Time exit: {e}")
    
    # Test 5: IV exit
    try:
        trade = exit_trader.enter_trade(
            symbol="IBIT", strikes="$48/$44", expiry="2026-04-15",
            entry_credit=1.00, contracts=1, market_conditions={'iv': 60.0}  # Entry IV 60%
        )
        
        # IV dropped from 60 to 45 = 25% drop, above threshold of 20%
        closed = exit_trader.check_exits(
            current_prices={trade.id: 0.90},
            current_iv=45.0  # 25% drop from entry
        )
        assert len(closed) == 1, "Should close on IV drop"
        assert 'IV_EXIT' in closed[0].market_conditions.get('close_reason', '')
        results.add("Exit: IV Exit", True)
        print("  ✅ IV exit")
    except Exception as e:
        results.add("Exit: IV Exit", False, str(e))
        print(f"  ❌ IV exit: {e}")
    
    # =========================================================================
    # SECTION 6: Config Generator Tests
    # =========================================================================
    print("\n🔧 SECTION 6: Config Generator Tests")
    print("-" * 40)
    
    try:
        from src.trading.trader_config_generator import (
            generate_all_configs, get_total_combinations
        )
        
        totals = get_total_combinations()
        assert 'total' in totals
        assert totals['total'] > 40000  # Should be ~388K+ with budget tiers
        results.add("Config Generator: Total Combinations", True)
        print(f"  ✅ Total combinations: {totals['total']:,}")
    except Exception as e:
        results.add("Config Generator: Total Combinations", False, str(e))
        print(f"  ❌ Total combinations: {e}")
    
    # Test generating a small subset
    try:
        configs = generate_all_configs(total_traders=100, full_coverage=False)
        assert len(configs) >= 100
        # Verify they have unique IDs
        ids = [c.trader_id for c in configs]
        assert len(ids) == len(set(ids)), "IDs should be unique"
        results.add("Config Generator: Subset Generation", True)
        print(f"  ✅ Subset generation: {len(configs)} configs")
    except Exception as e:
        results.add("Config Generator: Subset Generation", False, str(e))
        print(f"  ❌ Subset generation: {e}")
    
    # =========================================================================
    # SECTION 7: Paper Trader Farm Tests
    # =========================================================================
    print("\n🏭 SECTION 7: Paper Trader Farm Tests")
    print("-" * 40)
    
    try:
        from src.trading.paper_trader_farm import PaperTraderFarm
        
        # Create small farm for testing (use total_traders and full_coverage=False)
        farm = PaperTraderFarm(db=None, total_traders=10, full_coverage=False)
        # Farm requires async initialize() to create traders
        await farm.initialize()
        assert len(farm.trader_configs) >= 10  # Changed from farm.traders
        assert farm.trader_tensors is not None # GPU tensor created
        results.add("Farm: Initialization", True)
        print(f"  ✅ Farm initialized with {len(farm.trader_configs)} configs")
    except Exception as e:
        results.add("Farm: Initialization", False, str(e))
        print(f"  ❌ Farm initialization: {e}")
    
    # Test leaderboard
    try:
        leaderboard = farm.get_leaderboard(top_n=5)
        assert isinstance(leaderboard, list)
        results.add("Farm: Leaderboard", True)
        print("  ✅ Leaderboard retrieval")
    except Exception as e:
        results.add("Farm: Leaderboard", False, str(e))
        print(f"  ❌ Leaderboard: {e}")
    
    # Test strategy breakdown
    try:
        breakdown = farm.get_strategy_breakdown()
        assert isinstance(breakdown, dict)
        results.add("Farm: Strategy Breakdown", True)
        print("  ✅ Strategy breakdown")
    except Exception as e:
        results.add("Farm: Strategy Breakdown", False, str(e))
        print(f"  ❌ Strategy breakdown: {e}")
    
    # Test aggregate P&L
    try:
        aggregate = farm.get_aggregate_pnl()
        assert 'total_traders' in aggregate
        assert 'realized_pnl' in aggregate
        results.add("Farm: Aggregate P&L", True)
        print("  ✅ Aggregate P&L")
    except Exception as e:
        results.add("Farm: Aggregate P&L", False, str(e))
        print(f"  ❌ Aggregate P&L: {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    passed, total = results.summary()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if results.failed:
        print("\n❌ FAILURES:")
        for name, error in results.failed:
            print(f"  • {name}: {error[:60]}...")
    
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL PAPER TRADING TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_paper_trader_tests())
    sys.exit(exit_code)
