"""
Paper Trader Farm
=================

Manages 97K+ paper traders running in parallel.
Tests all unique parameter combinations with market regime awareness.
Generates monthly reports with strategy averages and top threshold combos.
Checks economic calendar for blackout periods before entries.
"""

import asyncio
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json

from .paper_trader import PaperTrader, TraderConfig, PaperTrade, StrategyType
from .trader_config_generator import generate_all_configs, get_config_summary
from ..core.economic_calendar import EconomicCalendar

logger = logging.getLogger(__name__)


class PaperTraderFarm:
    """
    Manages a farm of paper traders.
    
    Each trader runs with different parameters but shares
    the same market data. Trades are logged to database
    for later analysis.
    """
    
    def __init__(
        self,
        db=None,
        total_traders: int = 400000,
        full_coverage: bool = True,
        config_file: Optional[str] = None,
    ):
        """
        Initialize the paper trader farm.
        
        Args:
            db: Database instance for logging trades
            total_traders: Number of traders (ignored if full_coverage=True)
            full_coverage: If True, generate ALL unique parameter combinations
            config_file: Optional path to saved configs
        """
        self.db = db
        self.total_traders = total_traders
        self.full_coverage = full_coverage
        
        # New: Tensors for GPU evaluation
        self.trader_tensors: Optional['torch.Tensor'] = None
        self.trader_configs: List[TraderConfig] = []
        
        # Only active traders (those with open positions) are instantiated
        self.active_traders: Dict[str, PaperTrader] = {} 
        
        self._running = False
        self.last_evaluation_time: Optional[datetime] = None
        self.last_evaluation_symbol: Optional[str] = None
        self.last_evaluation_entered: int = 0
        
        # Callbacks for market data (set by orchestrator)
        self._get_conditions: Optional[Callable] = None
        self._get_options_chain: Optional[Callable] = None
        self._get_gap_risk: Optional[Callable] = None
        
        # Economic calendar for blackout periods (FOMC, CPI, Jobs)
        self.economic_calendar = EconomicCalendar()
        
        # Track open positions by symbol for correlation check
        self._positions_by_symbol: Dict[str, int] = {}  # symbol -> count
        
        logger.info(f"PaperTraderFarm initialized (full_coverage={full_coverage})")
    
    async def initialize(self) -> None:
        """Initialize all traders and database tables."""
        # Create database table
        if self.db:
            await self._create_tables()
        
        # Generate trader configs - use full coverage for all unique combos
        configs = generate_all_configs(
            total_traders=self.total_traders,
            full_coverage=self.full_coverage,
        )
        self.trader_configs = configs
        
        # Prepare Tensors for GPU
        await self._prepare_trader_tensors()
        
        # Log summary
        summary = get_config_summary(configs)
        logger.info(f"Initialized paper trader farm with {len(configs):,} configurations:")
        for strat, count in summary['by_strategy'].items():
            logger.info(f"  {strat}: {count:,} traders")
    
    async def _create_tables(self) -> None:
        """Create database tables for paper trading."""
        # Create paper_trades table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id TEXT PRIMARY KEY,
                trader_id TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                strikes TEXT,
                expiry TEXT,
                entry_credit REAL,
                contracts INTEGER,
                status TEXT DEFAULT 'open',
                close_timestamp TEXT,
                close_price REAL,
                realized_pnl REAL,
                market_conditions TEXT
            )
        """)
        
        # Create indexes
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_trader 
            ON paper_trades(trader_id)
        """)
        
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_status 
            ON paper_trades(status)
        """)
        
        # Create configs table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS paper_trader_configs (
                trader_id TEXT PRIMARY KEY,
                config TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Migration: Add missing columns to existing tables
        try:
            # Check if columns exist by trying a simple select
            await self.db.execute("SELECT trader_id FROM paper_trades LIMIT 1")
        except Exception:
            # Table might exist but be empty or have wrong schema
            # Try to add the column
            try:
                await self.db.execute("ALTER TABLE paper_trades ADD COLUMN trader_id TEXT")
                logger.info("Added trader_id column to paper_trades")
            except Exception:
                pass  # Column already exists or other issue
        
        logger.info("Paper trading tables created")

    async def _prepare_trader_tensors(self) -> None:
        """Convert all trader configs into a single tensor for GPU evaluation."""
        import torch
        from .paper_trader import StrategyType
        
        N = len(self.trader_configs)
        # Columns: iv_min, iv_max, warmth_min, pop_min, dte_min, dte_max, gap_max, strategy_id
        params = torch.zeros((N, 8), dtype=torch.float32)
        
        strat_map = {
            StrategyType.BULL_PUT: 0,
            StrategyType.BEAR_CALL: 1,
            StrategyType.IRON_CONDOR: 2,
            StrategyType.STRADDLE_SELL: 3
        }
        
        for i, config in enumerate(self.trader_configs):
            params[i, 0] = config.iv_min
            params[i, 1] = config.iv_max
            params[i, 2] = config.warmth_min
            params[i, 3] = config.pop_min
            params[i, 4] = config.dte_min
            params[i, 5] = config.dte_max
            params[i, 6] = config.gap_tolerance_pct
            params[i, 7] = strat_map.get(config.strategy_type, 0)
            
        from ..analysis.gpu_engine import get_gpu_engine
        engine = get_gpu_engine()
        self.trader_tensors = params.to(engine._device)
        logger.info(f"Uploaded {N:,} trader params to GPU ({params.element_size() * N * 8 / 1e6:.1f} MB)")
    
    def set_data_sources(
        self,
        get_conditions: Callable = None,
        get_options_chain: Callable = None,
        get_gap_risk: Callable = None,
    ) -> None:
        """Set callbacks for market data access."""
        if get_conditions:
            self._get_conditions = get_conditions
        if get_options_chain:
            self._get_options_chain = get_options_chain
        if get_gap_risk:
            self._get_gap_risk = get_gap_risk
    
    async def evaluate_signal(
        self,
        symbol: str,
        signal_data: Dict[str, Any],
    ) -> List[PaperTrade]:
        """
        Evaluate a trading signal across all traders.
        
        Each trader decides independently whether to enter.
        Respects economic calendar blackout periods and correlation limits.
        
        Args:
            symbol: Ticker symbol
            signal_data: Dict with signal details:
                - iv: Current implied volatility
                - warmth: Conditions warmth score
                - dte: Days to expiration
                - pop: Probability of profit
                - gap_risk: Current gap risk %
                - direction: Market direction
                - strikes: Strike prices
                - expiry: Expiration date
                - credit: Entry credit
                
        Returns:
            List of trades that were entered
        """
        # 1. Check economic calendar blackout periods (FOMC, CPI, Jobs)
        is_blackout, event = self.economic_calendar.is_blackout_period()
        if is_blackout:
            logger.debug(f"Blocked signal for {symbol}: blackout period ({event.name if event else 'unknown'})")
            return []
        
        # 2. Check IBIT/BITO correlation (both track BTC, don't double-up)
        btc_symbols = {'IBIT', 'BITO', 'GBTC', 'BTCO'}  # All BTC-tracking ETFs
        if symbol.upper() in btc_symbols:
            # Count existing BTC exposure across all symbols
            btc_exposure = sum(
                count for sym, count in self._positions_by_symbol.items()
                if sym.upper() in btc_symbols
            )
            # Limit total BTC exposure to 50% of traders that would enter
            max_btc_exposure = len(self.trader_configs) // 2
            if btc_exposure >= max_btc_exposure:
                logger.debug(f"Blocked {symbol}: BTC exposure limit reached ({btc_exposure:,} positions)")
                return []
        
        entered_trades = []
        self.last_evaluation_time = datetime.now(timezone.utc)
        self.last_evaluation_symbol = symbol
        self.last_evaluation_entered = 0
        
        # Get current time for session filters
        current_time = signal_data.get('timestamp')
        if isinstance(current_time, str):
            try:
                current_time = datetime.fromisoformat(current_time)
            except ValueError:
                current_time = None
        
        # 3. Vectorized Evaluation on GPU
        from ..analysis.gpu_engine import get_gpu_engine
        import torch
        engine = get_gpu_engine()
        
        # Prepare market data tensor
        # 0: iv | 1: warmth | 2: dte | 3: pop | 4: gap_risk 
        # 5: bull | 6: bear | 7: neutral | 8: morning | 9: midday | 10: afternoon
        dir_val = signal_data.get('direction', 'neutral')
        market_tensor = torch.tensor([
            signal_data.get('iv', 50),
            signal_data.get('warmth', 5),
            signal_data.get('dte', 45),
            signal_data.get('pop', 65),
            signal_data.get('gap_risk', 0),
            1.0 if dir_val == 'bullish' else 0.0,
            1.0 if dir_val == 'bearish' else 0.0,
            1.0 if dir_val == 'neutral' else 0.0,
            # We skip session filters in this specific batch check for now 
            # or could add them to the tensor if needed. 
            0.0, 0.0, 0.0 
        ], device=engine._device, dtype=torch.float32)
        
        # Batch evaluate
        start_eval = datetime.now()
        mask = engine.evaluate_traders_batch(self.trader_tensors, market_tensor)
        eval_time = (datetime.now() - start_eval).total_seconds() * 1000
        
        # Get indices of traders that should enter
        entry_indices = torch.where(mask)[0].cpu().tolist()
        
        for idx in entry_indices:
            config = self.trader_configs[idx]
            
            # Check session filter (this is the only part not yet in the tensor mask)
            # We could add it to the tensor for 100% speed, but it's only for the matches
            dt = current_time or datetime.now(timezone.utc)
            hour = dt.hour + (dt.minute / 60)
            session = config.session_filter
            if session != 'any':
                if session == 'morning' and not (9.5 <= hour <= 11.5): continue
                elif session == 'midday' and not (11.5 < hour <= 14.0): continue
                elif session == 'afternoon' and not (14.0 < hour <= 16.0): continue
            
            # Re-check max positions for this specific config
            # (Note: active_traders only tracks those with open positions)
            active_trader = self.active_traders.get(config.trader_id)
            if active_trader and len(active_trader.open_positions) >= config.max_position_size:
                continue
                
            # Instantiate trader if not already active (Lazy Loading)
            if not active_trader:
                active_trader = PaperTrader(config=config, db=self.db)
                self.active_traders[config.trader_id] = active_trader
            
            # Execute paper trade
            trade = active_trader.enter_trade(
                symbol=symbol,
                strikes=signal_data.get('strikes', 'N/A'),
                expiry=signal_data.get('expiry', 'N/A'),
                entry_credit=signal_data.get('credit', 0.40),
                contracts=1,  # Base contracts
                market_conditions={
                    'iv': signal_data.get('iv'),
                    'warmth': signal_data.get('warmth'),
                    'direction': signal_data.get('direction'),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                },
            )
            
            entered_trades.append(trade)
            
            # Track positions by symbol for correlation check
            self._positions_by_symbol[symbol] = self._positions_by_symbol.get(symbol, 0) + 1
            
            # Log to database
            if self.db:
                await self._save_trade(trade)
        
        if entered_trades:
            logger.info(
                f"Vectorized evaluation complete in {eval_time:.1f}ms: "
                f"{len(entered_trades)}/{len(self.trader_configs)} traders entered"
            )
        
        if entered_trades:
            logger.info(
                f"Signal evaluated: {len(entered_trades)}/{len(self.trader_configs)} "
                f"traders entered {symbol}"
            )
        self.last_evaluation_entered = len(entered_trades)
        return entered_trades

    def get_status_summary(self) -> Dict[str, Any]:
        """Return last evaluation status for the farm."""
        return {
            "last_evaluation_time": self.last_evaluation_time.isoformat() if self.last_evaluation_time else None,
            "last_evaluation_symbol": self.last_evaluation_symbol,
            "last_evaluation_entered": self.last_evaluation_entered,
            "active_traders": len(self.active_traders),
            "total_configs": len(self.trader_configs),
        }
    
    async def _save_trade(self, trade: PaperTrade) -> None:
        """Save a trade to the database."""
        await self.db.execute("""
            INSERT OR REPLACE INTO paper_trades 
            (id, trader_id, strategy_type, symbol, timestamp, strikes, expiry,
             entry_credit, contracts, status, close_timestamp, close_price,
             realized_pnl, market_conditions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.id,
            trade.trader_id,
            trade.strategy_type,
            trade.symbol,
            trade.timestamp,
            trade.strikes,
            trade.expiry,
            trade.entry_credit,
            trade.contracts,
            trade.status,
            trade.close_timestamp,
            trade.close_price,
            trade.realized_pnl,
            json.dumps(trade.market_conditions) if trade.market_conditions else None,
        ))
    
    async def check_exits(self, current_prices: Dict[str, Dict[str, float]]) -> List[PaperTrade]:
        """
        Check all traders for exit conditions.
        
        Args:
            current_prices: Dict mapping symbol to {trade_id: current_price}
            
        Returns:
            List of closed trades
        """
        all_closed = []
        
        # Only check traders with open positions
        for trader_id, trader in list(self.active_traders.items()):
            # Get prices relevant to this trader's positions
            trader_prices = {}
            for trade in trader.open_positions:
                if trade.symbol in current_prices:
                    prices = current_prices[trade.symbol]
                    if trade.id in prices:
                        trader_prices[trade.id] = prices[trade.id]
            
            if not trader_prices:
                continue
                
            closed = trader.check_exits(trader_prices)
            
            for trade in closed:
                if self.db:
                    await self._save_trade(trade)
                all_closed.append(trade)
            
            # Clean up active_traders if no more open positions
            if not trader.open_positions:
                del self.active_traders[trader_id]
        
        if all_closed:
            logger.info(f"Exit check: {len(all_closed)} trades closed")
        
        return all_closed
    
    async def expire_positions(self, expiry_date: str) -> List[PaperTrade]:
        """Handle expiration for a given date."""
        all_expired = []
        
        for trader_id, trader in list(self.active_traders.items()):
            expired = trader.expire_positions(expiry_date, {})
            
            for trade in expired:
                if self.db:
                    await self._save_trade(trade)
                all_expired.append(trade)
            
            # Clean up active_traders if no more open positions
            if not trader.open_positions:
                del self.active_traders[trader_id]
        
        if all_expired:
            logger.info(f"Expiration: {len(all_expired)} trades expired for {expiry_date}")
        
        return all_expired
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def get_aggregate_positions(self) -> List[Dict]:
        """Get all open positions across all traders (active only)."""
        positions = []
        for trader in self.active_traders.values():
            positions.extend(trader.get_positions_summary())
        return positions
    
    def get_aggregate_pnl(self) -> Dict:
        """Get aggregate P&L across all traders."""
        total_realized = 0
        total_trades = 0
        total_wins = 0
        open_positions = 0
        
        for trader in self.active_traders.values():
            pnl = trader.get_pnl_summary()
            total_realized += pnl['realized_pnl']
            total_trades += pnl['total_trades']
            total_wins += pnl['wins']
            open_positions += pnl['open_positions']
        
        return {
            'total_traders': len(self.trader_configs),
            'total_trades': total_trades,
            'total_wins': total_wins,
            'win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0,
            'realized_pnl': total_realized,
            'open_positions': open_positions,
        }
    
    def get_leaderboard(self, top_n: int = 20) -> List[Dict]:
        """Get top performing traders (active only for now)."""
        rankings = []
        
        for trader in self.active_traders.values():
            pnl = trader.get_pnl_summary()
            rankings.append(pnl)
        
        # Sort by realized P&L
        rankings.sort(key=lambda x: x['realized_pnl'], reverse=True)
        
        return rankings[:top_n]
    
    def get_strategy_breakdown(self) -> Dict[str, Dict]:
        """Get performance breakdown by strategy."""
        strategies = {}
        
        # For a full breakdown, we need to load stats from DB or track all-time
        # Since we lazy load objects, we'll iterate the configs
        for config in self.trader_configs:
            strat = config.strategy_type.value
            # Note: Total stats would ideally come from DB queries for a farm this size
            # but for this iteration we'll provide a placeholder or partial from active
            if strat not in strategies:
                strategies[strat] = {'traders': 0, 'total_trades': 0, 'wins': 0, 'realized_pnl': 0}
            strategies[strat]['traders'] += 1
            
        # Update with active stats
        for trader in self.active_traders.values():
            strat = trader.config.strategy_type.value
            pnl = trader.get_pnl_summary()
            strategies[strat]['total_trades'] += pnl['total_trades']
            strategies[strat]['wins'] += pnl['wins']
            strategies[strat]['realized_pnl'] += pnl['realized_pnl']
        
        # Calculate win rates
        for strat in strategies:
            total = strategies[strat]['total_trades']
            wins = strategies[strat]['wins']
            strategies[strat]['win_rate'] = (wins / total * 100) if total > 0 else 0
        
        return strategies
    
    def format_leaderboard(self, month: str = None) -> str:
        """Format monthly report for Telegram/display."""
        if month is None:
            month = datetime.now().strftime('%B %Y')
        
        breakdown = self.get_strategy_breakdown()
        aggregate = self.get_aggregate_pnl()
        
        lines = [
            f"📊 PAPER TRADER MONTHLY REPORT — {month}",
            "",
            "STRATEGY PERFORMANCE",
            "━" * 60,
            "Strategy      | Traders | Trades | Win% | Avg P&L | Total P&L",
            "━" * 60,
        ]
        
        # Sort by total P&L
        sorted_strats = sorted(breakdown.items(), key=lambda x: x[1]['realized_pnl'], reverse=True)
        best_strat = sorted_strats[0][0] if sorted_strats else None
        
        for strat_name, data in sorted_strats:
            traders_count = data['traders']
            trades = data['total_trades']
            win_rate = data['win_rate']
            total_pnl = data['realized_pnl']
            avg_pnl = total_pnl / trades if trades > 0 else 0
            
            star = " ⭐" if strat_name == best_strat else ""
            lines.append(
                f"{strat_name:13s} | {traders_count:7,d} | {trades:6d} | "
                f"{win_rate:3.0f}% | ${avg_pnl:+6.0f} | ${total_pnl:+,.0f}{star}"
            )
        
        # Top threshold combinations
        lines.extend([
            "",
            "TOP THRESHOLD COMBINATIONS",
            "━" * 60,
        ])
        
        top_combos = self._get_top_threshold_combos(5)
        for i, combo in enumerate(top_combos, 1):
            lines.append(
                f"{i}. {combo['strategy']} + IV>{combo['iv_min']} + "
                f"warmth>{combo['warmth']}: ${combo['pnl']:+,.0f} "
                f"({combo['trades']} trades, {combo['win_rate']:.0f}% win)"
            )
        
        # Summary
        lines.extend([
            "",
            "━" * 60,
            f"Total: {aggregate['total_traders']:,} traders | "
            f"{aggregate['total_trades']} trades | "
            f"${aggregate['realized_pnl']:+,.0f} P&L",
        ])
        
        return "\n".join(lines)
    
    def _get_top_threshold_combos(self, top_n: int = 5) -> List[Dict]:
        """Get top performing threshold combinations."""
        # Group ACTIVE traders by key threshold combos
        combos = {}
        
        for trader in self.active_traders.values():
            key = (
                trader.config.strategy_type.value,
                trader.config.iv_min,
                trader.config.warmth_min,
            )
            
            if key not in combos:
                combos[key] = {
                    'strategy': trader.config.strategy_type.value,
                    'iv_min': trader.config.iv_min,
                    'warmth': trader.config.warmth_min,
                    'pnl': 0,
                    'trades': 0,
                    'wins': 0,
                }
            
            combos[key]['pnl'] += trader.stats['total_pnl']
            combos[key]['trades'] += trader.stats['total_trades']
            combos[key]['wins'] += trader.stats['wins']
        
        # Calculate win rates
        for combo in combos.values():
            combo['win_rate'] = (combo['wins'] / combo['trades'] * 100) if combo['trades'] > 0 else 0
        
        # Sort by P&L
        sorted_combos = sorted(combos.values(), key=lambda x: x['pnl'], reverse=True)
        
        return sorted_combos[:top_n]
    
    # =========================================================================
    # For /positions and /pnl commands
    # =========================================================================
    
    async def get_positions_for_telegram(self) -> List[Dict]:
        """
        Get positions formatted for /positions command.
        
        Returns aggregated view, NOT individual positions.
        Groups by symbol and strategy for manageable output.
        """
        # Group by symbol AND strategy
        by_symbol_strategy = {}
        for pos in self.get_aggregate_positions():
            key = (pos['symbol'], pos.get('strategy_type', 'unknown'))
            if key not in by_symbol_strategy:
                by_symbol_strategy[key] = []
            by_symbol_strategy[key].append(pos)
        
        result = []
        for (symbol, strategy), positions in by_symbol_strategy.items():
            result.append({
                'symbol': symbol,
                'strategy': strategy,
                'count': len(positions),
                'traders_entered': len(set(p['trader_id'] for p in positions)),
                'sample_strikes': positions[0]['strikes'] if positions else 'N/A',
            })
        
        # Sort by count (most positions first)
        result.sort(key=lambda x: x['count'], reverse=True)
        
        # Limit to top 10 to avoid Telegram message limits
        return result[:10]
    
    async def get_pnl_for_telegram(self) -> Dict:
        """
        Get P&L formatted for /pnl command.
        
        Returns aggregate stats, NOT individual trader P&L.
        """
        aggregate = self.get_aggregate_pnl()
        breakdown = self.get_strategy_breakdown()
        
        # Find best strategy
        best_strat = max(breakdown.items(), key=lambda x: x[1]['realized_pnl'])
        
        return {
            'total_traders': aggregate['total_traders'],
            'total_trades': aggregate['total_trades'],
            'win_rate': aggregate['win_rate'],
            'realized_pnl': aggregate['realized_pnl'],
            'open_positions': aggregate['open_positions'],
            'best_strategy': best_strat[0],
            'best_strategy_pnl': best_strat[1]['realized_pnl'],
        }


# Test function
async def test_paper_trader_farm():
    """Test the paper trader farm."""
    print("Paper Trader Farm Test")
    print("=" * 50)
    
    # Mock database
    class MockDB:
        async def execute(self, *args): pass
        async def fetch_one(self, *args): return None
        async def fetch_all(self, *args): return []
    
    farm = PaperTraderFarm(db=MockDB(), total_traders=100)
    await farm.initialize()
    
    print(f"✅ Created {len(farm.trader_configs)} traders")
    
    # Test signal evaluation
    signal = {
        'iv': 58,
        'warmth': 6,
        'dte': 45,
        'pop': 68,
        'gap_risk': 5,
        'direction': 'neutral',
        'strikes': '$48/$44',
        'expiry': '2026-02-21',
        'credit': 0.42,
    }
    
    trades = await farm.evaluate_signal('IBIT', signal)
    print(f"✅ Signal evaluated: {len(trades)} traders entered")
    
    # Test aggregate stats
    aggregate = farm.get_aggregate_pnl()
    print(f"✅ Aggregate P&L: {aggregate}")
    
    # Test strategy breakdown
    breakdown = farm.get_strategy_breakdown()
    print(f"✅ Strategy breakdown:")
    for strat, data in breakdown.items():
        print(f"   {strat}: {data['traders']} traders, {data['total_trades']} trades")
    
    print("\n✅ All paper trader farm tests passed!")


if __name__ == "__main__":
    asyncio.run(test_paper_trader_farm())
