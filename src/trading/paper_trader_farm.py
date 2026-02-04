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
from datetime import datetime, date, timedelta, time, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any, Callable, Set
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
        starting_balance: float = 5000.0,
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
        self.starting_balance = starting_balance
        
        # New: Tensors for GPU evaluation
        self.trader_tensors: Optional['torch.Tensor'] = None
        self.trader_configs: List[TraderConfig] = []
        
        # Only active traders (those with open positions) are instantiated
        self.active_traders: Dict[str, PaperTrader] = {} 
        
        self._running = False
        self.last_evaluation_time: Optional[datetime] = None
        self.last_evaluation_symbol: Optional[str] = None
        self.last_evaluation_entered: int = 0
        self._promoted_trader_ids: Optional[Set[str]] = None
        self._promoted_indices: Optional[Set[int]] = None
        
        # Callbacks for market data (set by orchestrator)
        self._get_conditions: Optional[Callable] = None
        self._get_options_chain: Optional[Callable] = None
        self._get_gap_risk: Optional[Callable] = None
        
        # Economic calendar for blackout periods (FOMC, CPI, Jobs)
        self.economic_calendar = EconomicCalendar()
        
        # Track open positions by symbol for correlation check
        self._positions_by_symbol: Dict[str, int] = {}  # symbol -> count

        # P3: Drawdown circuit breaker
        self._peak_balance: float = starting_balance
        self._drawdown_breaker_pct: float = 20.0  # halt at 20% drawdown
        self._drawdown_halted: bool = False

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
        # P3: Drawdown circuit breaker check
        if self._drawdown_halted:
            logger.debug(f"Blocked signal for {symbol}: drawdown circuit breaker active")
            return []

        # 1. Check economic calendar blackout periods (FOMC, CPI, Jobs)
        is_blackout, event = self.economic_calendar.is_blackout_period()
        if is_blackout:
            logger.debug(f"Blocked signal for {symbol}: blackout period ({event.name if event else 'unknown'})")
            return []

        # 2. Check IBIT/BITO correlation (both track BTC, don't double-up)
        btc_symbols = {'IBIT', 'BITO', 'GBTC', 'BTCO'}  # All BTC-tracking ETFs
        if symbol.upper() in btc_symbols:
            btc_exposure = sum(
                count for sym, count in self._positions_by_symbol.items()
                if sym.upper() in btc_symbols
            )
            max_btc_exposure = len(self.trader_configs) // 2
            if btc_exposure >= max_btc_exposure:
                logger.debug(f"Blocked {symbol}: BTC exposure limit reached ({btc_exposure:,} positions)")
                return []

        entered_trades = []
        db_batch = []
        self.last_evaluation_time = datetime.now(timezone.utc)
        self.last_evaluation_symbol = symbol
        self.last_evaluation_entered = 0

        # P1 fix: Convert timestamp to Eastern Time for session filters
        eastern = ZoneInfo("America/New_York")
        current_time = signal_data.get('timestamp')
        if isinstance(current_time, str):
            try:
                current_time = datetime.fromisoformat(current_time)
            except ValueError:
                current_time = None
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        # Ensure timezone-aware
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        current_time_et = current_time.astimezone(eastern)

        # 3. Vectorized Evaluation on GPU
        from ..analysis.gpu_engine import get_gpu_engine
        import torch
        engine = get_gpu_engine()

        # Prepare market data tensor
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
            0.0, 0.0, 0.0
        ], device=engine._device, dtype=torch.float32)

        # Batch evaluate
        start_eval = datetime.now()
        mask = engine.evaluate_traders_batch(self.trader_tensors, market_tensor)
        eval_time = (datetime.now() - start_eval).total_seconds() * 1000

        # Get indices of traders that should enter
        entry_indices = torch.where(mask)[0].cpu().tolist()

        if self._promoted_indices is not None:
            entry_indices = [i for i in entry_indices if i in self._promoted_indices]

        # P1 fix: Use Eastern Time hour for session filter
        et_hour = current_time_et.hour + (current_time_et.minute / 60)

        # P1: Base credit from signal, apply realistic fill slippage
        raw_credit = signal_data.get('credit', 0.40)
        # Apply 5% bid-ask slippage to simulate realistic fills
        slippage_factor = 0.95
        realistic_credit = raw_credit * slippage_factor

        for idx in entry_indices:
            config = self.trader_configs[idx]

            # P1 fix: Session filter uses Eastern Time
            session = config.session_filter
            if session != 'any':
                if session == 'morning' and not (9.5 <= et_hour <= 11.5): continue
                elif session == 'midday' and not (11.5 < et_hour <= 14.0): continue
                elif session == 'afternoon' and not (14.0 < et_hour <= 16.0): continue

            # Re-check max positions for this specific config
            active_trader = self.active_traders.get(config.trader_id)
            if active_trader and len(active_trader.open_positions) >= config.max_position_size:
                continue

            # Instantiate trader if not already active (Lazy Loading)
            if not active_trader:
                active_trader = PaperTrader(config=config, db=self.db)
                self.active_traders[config.trader_id] = active_trader

            # P1: Differentiate trade construction per config
            # Compute contracts from position_size_pct and max_risk_dollars
            base_spread_width = 4.0  # Default $4 spread width
            try:
                strikes_str = signal_data.get('strikes', '')
                if '/' in strikes_str:
                    parts = strikes_str.replace('$', '').split('/')
                    base_spread_width = abs(float(parts[0]) - float(parts[1]))
            except (ValueError, IndexError):
                pass

            max_risk_per_contract = (base_spread_width - realistic_credit) * 100
            if max_risk_per_contract > 0:
                risk_budget = min(
                    self.starting_balance * (config.position_size_pct / 100),
                    config.max_risk_dollars,
                )
                contracts = max(1, int(risk_budget / max_risk_per_contract))
            else:
                contracts = 1

            # Execute paper trade
            trade = active_trader.enter_trade(
                symbol=symbol,
                strikes=signal_data.get('strikes', 'N/A'),
                expiry=signal_data.get('expiry', 'N/A'),
                entry_credit=realistic_credit,
                contracts=contracts,
                market_conditions={
                    'iv': signal_data.get('iv'),
                    'warmth': signal_data.get('warmth'),
                    'direction': signal_data.get('direction'),
                    'pop': signal_data.get('pop'),
                    'pot': signal_data.get('pot'),
                    'iv_rank': signal_data.get('iv_rank'),
                    'dte': signal_data.get('dte'),
                    'credit': signal_data.get('credit'),
                    'realistic_credit': realistic_credit,
                    'contracts': contracts,
                    'conditions_label': signal_data.get('conditions_label'),
                    'btc_change_pct': signal_data.get('btc_change_pct'),
                    'ibit_change_pct': signal_data.get('ibit_change_pct'),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                },
            )

            entered_trades.append(trade)
            db_batch.append(trade)

            # Track positions by symbol for correlation check
            self._positions_by_symbol[symbol] = self._positions_by_symbol.get(symbol, 0) + 1

        # P2: Batch database writes for entries
        if db_batch and self.db:
            await self._save_trades_batch(db_batch)

        if entered_trades:
            logger.info(
                f"Signal evaluated in {eval_time:.1f}ms: "
                f"{len(entered_trades)}/{len(self.trader_configs)} "
                f"traders entered {symbol} (credit=${realistic_credit:.3f}, slippage={1-slippage_factor:.0%})"
            )
        self.last_evaluation_entered = len(entered_trades)

        # P3: Update drawdown circuit breaker
        self._update_drawdown_check()

        return entered_trades

    def _update_drawdown_check(self) -> None:
        """P3: Check portfolio drawdown and halt if exceeded."""
        total_pnl = sum(
            t.stats['total_pnl'] for t in self.active_traders.values()
        )
        current_balance = self.starting_balance + total_pnl
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance
        if self._peak_balance > 0:
            drawdown_pct = ((self._peak_balance - current_balance) / self._peak_balance) * 100
            if drawdown_pct >= self._drawdown_breaker_pct:
                if not self._drawdown_halted:
                    self._drawdown_halted = True
                    logger.warning(
                        f"DRAWDOWN CIRCUIT BREAKER: {drawdown_pct:.1f}% drawdown "
                        f"(peak=${self._peak_balance:,.0f}, current=${current_balance:,.0f}). "
                        f"New entries halted."
                    )
            elif self._drawdown_halted and drawdown_pct < self._drawdown_breaker_pct * 0.5:
                # Reset breaker when drawdown recovers below half threshold
                self._drawdown_halted = False
                logger.info(f"Drawdown circuit breaker reset (drawdown={drawdown_pct:.1f}%)")

    def get_status_summary(self) -> Dict[str, Any]:
        """Return last evaluation status for the farm."""
        return {
            "last_evaluation_time": self.last_evaluation_time.isoformat() if self.last_evaluation_time else None,
            "last_evaluation_symbol": self.last_evaluation_symbol,
            "last_evaluation_entered": self.last_evaluation_entered,
            "active_traders": len(self.active_traders),
            "total_configs": len(self.trader_configs),
            "promoted_traders": len(self._promoted_trader_ids) if self._promoted_trader_ids else 0,
        }

    def set_promoted_traders(self, trader_ids: List[str]) -> None:
        """Restrict evaluations to promoted trader IDs."""
        promoted = set(trader_ids)
        indices = {
            idx for idx, config in enumerate(self.trader_configs)
            if config.trader_id in promoted
        }
        self._promoted_trader_ids = promoted
        self._promoted_indices = indices
    
    async def _save_trade(self, trade: PaperTrade) -> None:
        """Save a single trade to the database."""
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

    async def _save_trades_batch(self, trades: List[PaperTrade]) -> None:
        """Save multiple trades in a single batch transaction (P2)."""
        if not trades:
            return
        rows = [
            (
                t.id, t.trader_id, t.strategy_type, t.symbol, t.timestamp,
                t.strikes, t.expiry, t.entry_credit, t.contracts, t.status,
                t.close_timestamp, t.close_price, t.realized_pnl,
                json.dumps(t.market_conditions) if t.market_conditions else None,
            )
            for t in trades
        ]
        await self.db.execute_many("""
            INSERT OR REPLACE INTO paper_trades
            (id, trader_id, strategy_type, symbol, timestamp, strikes, expiry,
             entry_credit, contracts, status, close_timestamp, close_price,
             realized_pnl, market_conditions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    
    async def check_exits(self, current_prices: Dict[str, Dict[str, float]]) -> List[PaperTrade]:
        """
        Check all traders for exit conditions.

        Args:
            current_prices: Dict mapping symbol to {trade_id: current_price}

        Returns:
            List of closed trades
        """
        all_closed = []
        db_batch = []

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
                db_batch.append(trade)
                all_closed.append(trade)
                # P0 fix: Decrement position counter
                if trade.symbol in self._positions_by_symbol:
                    self._positions_by_symbol[trade.symbol] = max(
                        0, self._positions_by_symbol[trade.symbol] - 1
                    )

            # Clean up active_traders if no more open positions
            if not trader.open_positions:
                del self.active_traders[trader_id]

        # P2: Batch database writes
        if db_batch and self.db:
            await self._save_trades_batch(db_batch)

        if all_closed:
            logger.info(f"Exit check: {len(all_closed)} trades closed")

        return all_closed

    async def expire_positions(self, expiry_date: str) -> List[PaperTrade]:
        """Handle expiration for a given date."""
        all_expired = []
        db_batch = []

        for trader_id, trader in list(self.active_traders.items()):
            expired = trader.expire_positions(expiry_date, {})

            for trade in expired:
                db_batch.append(trade)
                all_expired.append(trade)
                # P0 fix: Decrement position counter
                if trade.symbol in self._positions_by_symbol:
                    self._positions_by_symbol[trade.symbol] = max(
                        0, self._positions_by_symbol[trade.symbol] - 1
                    )

            # Clean up active_traders if no more open positions
            if not trader.open_positions:
                del self.active_traders[trader_id]

        # P2: Batch database writes
        if db_batch and self.db:
            await self._save_trades_batch(db_batch)

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

    async def get_positions_for_review(self) -> List[Dict]:
        """Get detailed open positions for daily review."""
        return self.get_aggregate_positions()
    
    def _et_day_bounds(self, target_date: date) -> tuple[datetime, datetime]:
        """Return UTC start/end for a given ET date."""
        eastern = ZoneInfo("America/New_York")
        start_et = datetime.combine(target_date, time.min, tzinfo=eastern)
        end_et = start_et + timedelta(days=1)
        return start_et.astimezone(timezone.utc), end_et.astimezone(timezone.utc)

    async def _fetch_realized_stats(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Fetch realized trade stats between start/end timestamps."""
        if not self.db:
            return {'closed_trades': 0, 'winners': 0, 'realized_pnl': 0.0}
        row = await self.db.fetch_one(
            """
            SELECT
                COUNT(*) as closed_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winners,
                SUM(realized_pnl) as realized_pnl
            FROM paper_trades
            WHERE status != 'open'
              AND close_timestamp >= ?
              AND close_timestamp < ?
            """,
            (start.isoformat(), end.isoformat()),
        )
        if not row:
            return {'closed_trades': 0, 'winners': 0, 'realized_pnl': 0.0}
        return {
            'closed_trades': row['closed_trades'] or 0,
            'winners': row['winners'] or 0,
            'realized_pnl': row['realized_pnl'] or 0.0,
        }

    async def _fetch_opened_count(self, start: datetime, end: datetime) -> int:
        """Fetch count of trades opened between start/end timestamps."""
        if not self.db:
            return 0
        row = await self.db.fetch_one(
            """
            SELECT COUNT(*) as opened_trades
            FROM paper_trades
            WHERE timestamp >= ?
              AND timestamp < ?
            """,
            (start.isoformat(), end.isoformat()),
        )
        return row['opened_trades'] if row else 0

    async def _fetch_open_positions_count(self) -> int:
        """Fetch count of open positions."""
        if not self.db:
            return len(self.get_aggregate_positions())
        row = await self.db.fetch_one(
            """
            SELECT COUNT(*) as open_trades
            FROM paper_trades
            WHERE status = 'open'
            """
        )
        return row['open_trades'] if row else 0

    async def get_trade_activity_summary(self) -> Dict[str, Any]:
        """Return realized P&L and activity stats for daily review and Telegram."""
        today_et = datetime.now(ZoneInfo("America/New_York")).date()
        month_start = today_et.replace(day=1)
        year_start = date(today_et.year, 1, 1)

        today_start, today_end = self._et_day_bounds(today_et)
        month_start_utc, _ = self._et_day_bounds(month_start)
        year_start_utc, _ = self._et_day_bounds(year_start)

        today_stats = await self._fetch_realized_stats(today_start, today_end)
        mtd_stats = await self._fetch_realized_stats(month_start_utc, today_end)
        ytd_stats = await self._fetch_realized_stats(year_start_utc, today_end)

        opened_today = await self._fetch_opened_count(today_start, today_end)
        open_positions = await self._fetch_open_positions_count()

        win_rate_mtd = (
            (mtd_stats['winners'] / mtd_stats['closed_trades'] * 100)
            if mtd_stats['closed_trades'] > 0 else 0
        )
        win_rate_ytd = (
            (ytd_stats['winners'] / ytd_stats['closed_trades'] * 100)
            if ytd_stats['closed_trades'] > 0 else 0
        )

        return {
            'today_pnl': today_stats['realized_pnl'],
            'month_pnl': mtd_stats['realized_pnl'],
            'year_pnl': ytd_stats['realized_pnl'],
            'trades_today': today_stats['closed_trades'],
            'trades_mtd': mtd_stats['closed_trades'],
            'trades_ytd': ytd_stats['closed_trades'],
            'opened_today': opened_today,
            'open_positions': open_positions,
            'win_rate_mtd': win_rate_mtd,
            'win_rate_ytd': win_rate_ytd,
            'account_value': self.starting_balance + ytd_stats['realized_pnl'],
        }

    async def get_pnl_for_telegram(self) -> Dict:
        """
        Get P&L formatted for /pnl command.
        
        Returns aggregate stats, NOT individual trader P&L.
        """
        activity = await self.get_trade_activity_summary()
        today_pct = (activity['today_pnl'] / self.starting_balance * 100) if self.starting_balance else 0
        month_pct = (activity['month_pnl'] / self.starting_balance * 100) if self.starting_balance else 0
        year_pct = (activity['year_pnl'] / self.starting_balance * 100) if self.starting_balance else 0

        return {
            'today_pnl': activity['today_pnl'],
            'today_pct': today_pct,
            'month_pnl': activity['month_pnl'],
            'month_pct': month_pct,
            'year_pnl': activity['year_pnl'],
            'year_pct': year_pct,
            'trades_today': activity['trades_today'],
            'trades_mtd': activity['trades_mtd'],
            'win_rate_mtd': activity['win_rate_mtd'],
            'opened_today': activity['opened_today'],
            'open_positions': activity['open_positions'],
            'account_value': activity['account_value'],
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
