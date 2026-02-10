"""
Deterministic Replay Harness
=============================

Replays persisted ``market_bars`` and ``bar_outcomes`` from the database
in strict chronological order, advancing a virtual clock bar-by-bar.

Lookahead barrier (the "Time Guard")
-------------------------------------
The **invariant**: a strategy must never observe an outcome whose
``window_end_ms`` is greater than the current simulation time.

This is enforced structurally:
1. Bars are yielded in ascending timestamp order.
2. Outcomes are only made available when
   ``sim_time >= outcome.window_end_ms``.
3. The harness never calls ``strategy.evaluate()`` with data it
   could not have observed at that point in real time.

Replay loop
-----------
::

    for each bar in chronological order:
        sim_time = bar.timestamp_ms + bar_duration_ms
        strategy.on_bar(bar, visible_outcomes)
        intents = strategy.generate_intents(sim_time)
        for intent in intents:
            fill = execution_model.attempt_fill(...)
            if fill.filled:
                portfolio.apply(fill)
        portfolio.mark_to_market(bar)

Usage
-----
::

    harness = ReplayHarness(
        bars=bars,
        outcomes=outcomes,
        strategy=my_strategy,
        execution_model=ExecutionModel(),
    )
    result = harness.run()
    print(result.summary())
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from src.core.outcome_engine import BarData, OutcomeResult
from src.core.sessions import get_session_regime

logger = logging.getLogger("argus.replay_harness")


# ═══════════════════════════════════════════════════════════════════════════
# Strategy interface for replay
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TradeIntent:
    """A proposed trade emitted by a replay strategy.

    The harness passes this to the ExecutionModel for fill simulation.
    """
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    intent_type: Literal["OPEN", "CLOSE"]  # opening vs closing a position
    strike: Optional[float] = None
    expiry_ms: Optional[int] = None
    option_type: Optional[str] = None    # "call" or "put"
    limit_price: Optional[float] = None  # unused by market-order model
    tag: str = ""                        # free-form label for analytics
    meta: Dict[str, Any] = field(default_factory=dict)


class ReplayStrategy(ABC):
    """Interface that replay-compatible strategies must implement.

    Unlike the live ``BaseStrategy`` which is event-bus driven, the
    replay strategy is *pull-based*: the harness calls methods in a
    strict sequence and the strategy never sees future data.
    """

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique identifier (e.g. ``"OVERNIGHT_MOMENTUM_V1"``)."""
        ...

    @abstractmethod
    def on_bar(
        self,
        bar: BarData,
        sim_ts_ms: int,
        session_regime: str,
        visible_outcomes: Dict[int, OutcomeResult],
    ) -> None:
        """Feed one bar to the strategy.

        ``visible_outcomes`` only contains outcomes whose
        ``window_end_ms <= sim_ts_ms`` (the lookahead barrier).
        """
        ...

    @abstractmethod
    def generate_intents(self, sim_ts_ms: int) -> List[TradeIntent]:
        """Return zero or more trade intents for the current bar.

        Called immediately after ``on_bar``.  The harness will attempt
        to fill each intent via the ExecutionModel.
        """
        ...

    def on_fill(self, intent: TradeIntent, fill: Any) -> None:
        """Optional callback when a fill is executed."""

    def on_reject(self, intent: TradeIntent, fill: Any) -> None:
        """Optional callback when an intent is rejected."""

    def finalize(self) -> Dict[str, Any]:
        """Called once at the end of replay. Return any internal state."""
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# Virtual Portfolio
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """A single tracked position."""
    symbol: str
    side: Literal["LONG", "SHORT"]
    quantity: int
    entry_price: float
    entry_ts_ms: int
    tag: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    # Mutable state
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    closed: bool = False
    exit_price: float = 0.0
    exit_ts_ms: int = 0


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""
    ts_ms: int
    equity: float
    cash: float
    open_positions: int
    unrealized_pnl: float
    realized_pnl: float
    session_regime: str = ""


class VirtualPortfolio:
    """Tracks positions, PnL, and equity curve during replay."""

    def __init__(self, starting_cash: float = 10_000.0) -> None:
        self._starting_cash = starting_cash
        self._cash = starting_cash
        self._positions: List[Position] = []
        self._closed_positions: List[Position] = []
        self._equity_curve: List[PortfolioSnapshot] = []
        self._peak_equity: float = starting_cash
        self._max_drawdown: float = 0.0
        self._total_commission: float = 0.0

    def open_position(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        quantity: int,
        fill_price: float,
        ts_ms: int,
        commission: float = 0.0,
        multiplier: int = 100,
        tag: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Position:
        """Record a new position."""
        notional = fill_price * quantity * multiplier
        if side == "SHORT":
            self._cash += notional  # credit received
        else:
            self._cash -= notional  # debit paid
        self._cash -= commission
        self._total_commission += commission

        pos = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=fill_price,
            entry_ts_ms=ts_ms,
            tag=tag,
            meta=meta or {},
        )
        self._positions.append(pos)
        return pos

    def close_position(
        self,
        position: Position,
        fill_price: float,
        ts_ms: int,
        commission: float = 0.0,
        multiplier: int = 100,
    ) -> float:
        """Close a position and return realized PnL (notional)."""
        notional = fill_price * position.quantity * multiplier
        if position.side == "SHORT":
            # Buy to close: pay debit
            self._cash -= notional
            pnl = (position.entry_price - fill_price) * position.quantity * multiplier
        else:
            # Sell to close: receive credit
            self._cash += notional
            pnl = (fill_price - position.entry_price) * position.quantity * multiplier

        self._cash -= commission
        self._total_commission += commission

        position.realized_pnl = pnl
        position.closed = True
        position.exit_price = fill_price
        position.exit_ts_ms = ts_ms
        self._positions.remove(position)
        self._closed_positions.append(position)
        return pnl

    def mark_to_market(
        self,
        prices: Dict[str, float],
        ts_ms: int,
        session_regime: str = "",
        multiplier: int = 100,
    ) -> PortfolioSnapshot:
        """Update unrealized PnL and record an equity curve point."""
        unrealized = 0.0
        for pos in self._positions:
            price = prices.get(pos.symbol, pos.entry_price)
            if pos.side == "SHORT":
                pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity * multiplier
            else:
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity * multiplier
            unrealized += pos.unrealized_pnl

        realized = sum(p.realized_pnl for p in self._closed_positions)
        equity = self._cash + unrealized

        # Drawdown tracking
        if equity > self._peak_equity:
            self._peak_equity = equity
        dd = self._peak_equity - equity
        if dd > self._max_drawdown:
            self._max_drawdown = dd

        snap = PortfolioSnapshot(
            ts_ms=ts_ms,
            equity=round(equity, 2),
            cash=round(self._cash, 2),
            open_positions=len(self._positions),
            unrealized_pnl=round(unrealized, 2),
            realized_pnl=round(realized, 2),
            session_regime=session_regime,
        )
        self._equity_curve.append(snap)
        return snap

    @property
    def open_positions(self) -> List[Position]:
        return list(self._positions)

    @property
    def closed_positions(self) -> List[Position]:
        return list(self._closed_positions)

    @property
    def equity_curve(self) -> List[PortfolioSnapshot]:
        return list(self._equity_curve)

    @property
    def max_drawdown(self) -> float:
        return self._max_drawdown

    @property
    def total_commission(self) -> float:
        return self._total_commission

    def summary(self) -> Dict[str, Any]:
        realized = sum(p.realized_pnl for p in self._closed_positions)
        total_trades = len(self._closed_positions)
        winners = sum(1 for p in self._closed_positions if p.realized_pnl > 0)

        return {
            "starting_cash": self._starting_cash,
            "final_equity": round(self._cash, 2),
            "total_realized_pnl": round(realized, 2),
            "total_return_pct": round(
                (realized / self._starting_cash) * 100, 2
            ) if self._starting_cash > 0 else 0.0,
            "total_trades": total_trades,
            "winners": winners,
            "losers": total_trades - winners,
            "win_rate": round(winners / total_trades * 100, 1) if total_trades > 0 else 0.0,
            "max_drawdown": round(self._max_drawdown, 2),
            "max_drawdown_pct": round(
                (self._max_drawdown / self._starting_cash) * 100, 2
            ) if self._starting_cash > 0 else 0.0,
            "total_commission": round(self._total_commission, 2),
            "open_positions": len(self._positions),
            "equity_curve_points": len(self._equity_curve),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Replay Harness
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReplayConfig:
    """Configuration for the replay harness."""
    starting_cash: float = 10_000.0
    bar_duration_seconds: int = 60
    market: str = "EQUITIES"
    # Contract multiplier for options
    multiplier: int = 100


@dataclass
class ReplayResult:
    """Complete result of a replay run."""
    strategy_id: str
    config: ReplayConfig
    portfolio_summary: Dict[str, Any]
    execution_summary: Dict[str, Any]
    strategy_state: Dict[str, Any]
    bars_replayed: int
    outcomes_used: int
    session_distribution: Dict[str, int]

    def summary(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "bars_replayed": self.bars_replayed,
            "outcomes_used": self.outcomes_used,
            "portfolio": self.portfolio_summary,
            "execution": self.execution_summary,
            "sessions": self.session_distribution,
        }


class ReplayHarness:
    """Deterministic replay engine with strict lookahead barrier.

    Parameters
    ----------
    bars : list of BarData
        Market bars in ascending timestamp order.
    outcomes : list of dict
        Pre-computed bar outcomes (from ``bar_outcomes`` table).
        Each dict must have at least ``timestamp_ms`` and ``window_end_ms``.
    strategy : ReplayStrategy
        The strategy under test.
    execution_model : ExecutionModel
        Fill simulator (from ``execution_model.py``).
    config : ReplayConfig
        Harness-level settings.
    """

    def __init__(
        self,
        bars: List[BarData],
        outcomes: List[Dict[str, Any]],
        strategy: ReplayStrategy,
        execution_model: Any,  # ExecutionModel — avoid circular import
        config: Optional[ReplayConfig] = None,
    ) -> None:
        self._bars = sorted(bars, key=lambda b: b.timestamp_ms)
        self._cfg = config or ReplayConfig()
        self._strategy = strategy
        self._exec = execution_model
        self._portfolio = VirtualPortfolio(starting_cash=self._cfg.starting_cash)

        # Pre-index outcomes by bar timestamp for O(1) lookup
        self._outcomes_by_ts: Dict[int, Dict[str, Any]] = {}
        for o in outcomes:
            ts = o.get("timestamp_ms", 0)
            if ts > 0:
                self._outcomes_by_ts[ts] = o

        # Build sorted list of (window_end_ms, outcome) for the barrier
        self._outcomes_sorted: List[tuple] = sorted(
            [(o.get("window_end_ms", 0), o) for o in outcomes if o.get("window_end_ms")],
            key=lambda x: x[0],
        )

        self._session_counts: Dict[str, int] = {}

    def run(self) -> ReplayResult:
        """Execute the full replay loop.

        This is the core of the backtester.  It enforces:
        1. Bars are processed in strict chronological order.
        2. Outcomes are only visible when ``sim_time >= window_end_ms``.
        3. Strategy only sees data it could have observed at that point.
        """
        bar_duration_ms = self._cfg.bar_duration_seconds * 1000
        visible_outcomes: Dict[int, OutcomeResult] = {}
        outcome_cursor = 0  # pointer into _outcomes_sorted
        outcomes_used = 0

        logger.info(
            "Replay starting: %d bars, %d outcomes, strategy=%s",
            len(self._bars), len(self._outcomes_sorted), self._strategy.strategy_id,
        )

        for bar in self._bars:
            # sim_time is the *close* of this bar (bar open + duration)
            sim_ts_ms = bar.timestamp_ms + bar_duration_ms

            # ── Lookahead Barrier: release outcomes whose window has closed ──
            while outcome_cursor < len(self._outcomes_sorted):
                window_end_ms, outcome_dict = self._outcomes_sorted[outcome_cursor]
                if window_end_ms > sim_ts_ms:
                    break  # This outcome is still in the future
                # Safe to release
                ots = outcome_dict.get("timestamp_ms", 0)
                visible_outcomes[ots] = self._dict_to_outcome(outcome_dict)
                outcome_cursor += 1
                outcomes_used += 1

            # ── Session regime ───────────────────────────────────────────
            session = get_session_regime(self._cfg.market, sim_ts_ms)
            self._session_counts[session] = self._session_counts.get(session, 0) + 1

            # ── Strategy evaluation ──────────────────────────────────────
            self._strategy.on_bar(bar, sim_ts_ms, session, visible_outcomes)
            intents = self._strategy.generate_intents(sim_ts_ms)

            # ── Execution ────────────────────────────────────────────────
            for intent in intents:
                self._execute_intent(intent, bar, sim_ts_ms)

            # ── Mark to market ───────────────────────────────────────────
            self._portfolio.mark_to_market(
                prices={bar.symbol: bar.close} if hasattr(bar, "symbol") else {},
                ts_ms=sim_ts_ms,
                session_regime=session,
            )

        # ── Finalize ─────────────────────────────────────────────────────
        strategy_state = self._strategy.finalize()

        result = ReplayResult(
            strategy_id=self._strategy.strategy_id,
            config=self._cfg,
            portfolio_summary=self._portfolio.summary(),
            execution_summary=self._exec.ledger.summary(),
            strategy_state=strategy_state,
            bars_replayed=len(self._bars),
            outcomes_used=outcomes_used,
            session_distribution=dict(self._session_counts),
        )

        logger.info(
            "Replay complete: %d bars, %d fills, %d rejects, PnL=%.2f",
            result.bars_replayed,
            self._exec.ledger.fills_count,
            self._exec.ledger.rejects_count,
            result.portfolio_summary.get("total_realized_pnl", 0),
        )
        return result

    @property
    def portfolio(self) -> VirtualPortfolio:
        return self._portfolio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_intent(
        self,
        intent: TradeIntent,
        bar: BarData,
        sim_ts_ms: int,
    ) -> None:
        """Attempt to fill a TradeIntent via the execution model."""
        from .execution_model import Quote

        # Build a conservative quote from the bar
        # (In a full implementation, actual option quotes from the DB
        #  would be used.  For equity-only replay, we derive from OHLCV.)
        quote = Quote(
            bid=bar.low,       # worst case bid = bar low
            ask=bar.high,      # worst case ask = bar high
            bid_size=0,        # unknown → skip size check
            ask_size=0,
            quote_ts_ms=sim_ts_ms,
            symbol=intent.symbol,
        )

        # If the intent carries explicit quote data, use it
        if "quote" in intent.meta:
            q = intent.meta["quote"]
            quote = Quote(
                bid=q.get("bid", bar.low),
                ask=q.get("ask", bar.high),
                bid_size=q.get("bid_size", 0),
                ask_size=q.get("ask_size", 0),
                quote_ts_ms=q.get("quote_ts_ms", sim_ts_ms),
                symbol=intent.symbol,
            )

        fill = self._exec.attempt_fill(
            quote=quote,
            side=intent.side,
            quantity=intent.quantity,
            sim_ts_ms=sim_ts_ms,
            multiplier=self._cfg.multiplier,
        )

        if fill.filled:
            self._strategy.on_fill(intent, fill)
            if intent.intent_type == "OPEN":
                side = "SHORT" if intent.side == "SELL" else "LONG"
                self._portfolio.open_position(
                    symbol=intent.symbol,
                    side=side,
                    quantity=intent.quantity,
                    fill_price=fill.fill_price,
                    ts_ms=sim_ts_ms,
                    commission=fill.commission,
                    multiplier=self._cfg.multiplier,
                    tag=intent.tag,
                    meta=intent.meta,
                )
            elif intent.intent_type == "CLOSE":
                # Find the matching open position
                for pos in self._portfolio.open_positions:
                    if pos.symbol == intent.symbol and not pos.closed:
                        self._portfolio.close_position(
                            position=pos,
                            fill_price=fill.fill_price,
                            ts_ms=sim_ts_ms,
                            commission=fill.commission,
                            multiplier=self._cfg.multiplier,
                        )
                        break
        else:
            self._strategy.on_reject(intent, fill)

    @staticmethod
    def _dict_to_outcome(d: Dict[str, Any]) -> OutcomeResult:
        """Convert a DB dict row into an OutcomeResult."""
        return OutcomeResult(
            provider=d.get("provider", ""),
            symbol=d.get("symbol", ""),
            bar_duration_seconds=d.get("bar_duration_seconds", 60),
            timestamp_ms=d.get("timestamp_ms", 0),
            horizon_seconds=d.get("horizon_seconds", 0),
            outcome_version=d.get("outcome_version", ""),
            close_now=d.get("close_now", 0.0),
            close_at_horizon=d.get("close_at_horizon"),
            fwd_return=d.get("fwd_return"),
            max_runup=d.get("max_runup"),
            max_drawdown=d.get("max_drawdown"),
            realized_vol=d.get("realized_vol"),
            max_high_in_window=d.get("max_high_in_window"),
            min_low_in_window=d.get("min_low_in_window"),
            max_runup_ts_ms=d.get("max_runup_ts_ms"),
            max_drawdown_ts_ms=d.get("max_drawdown_ts_ms"),
            time_to_max_runup_ms=d.get("time_to_max_runup_ms"),
            time_to_max_drawdown_ms=d.get("time_to_max_drawdown_ms"),
            status=d.get("status", "UNKNOWN"),
            close_ref_ms=d.get("close_ref_ms", 0),
            window_start_ms=d.get("window_start_ms", 0),
            window_end_ms=d.get("window_end_ms", 0),
            bars_expected=d.get("bars_expected", 0),
            bars_found=d.get("bars_found", 0),
            gap_count=d.get("gap_count", 0),
            computed_at_ms=d.get("computed_at_ms"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Helper: load bars + outcomes from DB (async → sync bridge)
# ═══════════════════════════════════════════════════════════════════════════

async def load_replay_data(
    db: Any,
    provider: str,
    symbol: str,
    bar_duration: int,
    start_ms: int,
    end_ms: int,
    horizons: Optional[List[int]] = None,
) -> tuple:
    """Load bars and outcomes from the database for replay.

    Returns ``(bars: List[BarData], outcomes: List[dict])``.
    """
    from src.core.outcome_engine import BarData, _timestamp_to_ms

    bars_raw = await db.get_bars_for_outcome_computation(
        source=provider,
        symbol=symbol,
        bar_duration=bar_duration,
        start_ms=start_ms,
        end_ms=end_ms,
    )

    bars = []
    for row in bars_raw:
        ts = row.get("timestamp")
        if isinstance(ts, str):
            ts_ms = _timestamp_to_ms(ts)
        elif isinstance(ts, (int, float)):
            ts_ms = int(ts * 1000) if ts < 1e12 else int(ts)
        else:
            continue
        bars.append(BarData(
            timestamp_ms=ts_ms,
            open=float(row.get("open", 0)),
            high=float(row.get("high", 0)),
            low=float(row.get("low", 0)),
            close=float(row.get("close", 0)),
            volume=float(row.get("volume", 0) or 0),
        ))

    outcomes = await db.get_bar_outcomes(
        provider=provider,
        symbol=symbol,
        bar_duration_seconds=bar_duration,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=500_000,
    )

    return bars, outcomes
