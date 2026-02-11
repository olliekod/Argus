"""
Argus Regime Detector (Phase 2)
===============================

Deterministic regime classification from BarEvents.

GUARANTEES:
- Same tape → same regimes → same downstream signals
- No wall-clock dependence
- No randomness
- Arrival-faithful processing

ARCHITECTURE:
- Subscribes to market.bars (not metrics)
- Per-symbol incremental indicator state
- Emits SymbolRegimeEvent and MarketRegimeEvent
- Supports warmup_from_db for restart reconstruction
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from .bus import EventBus
from .events import (
    BarEvent,
    ComponentHeartbeatEvent,
    TOPIC_MARKET_BARS,
    TOPIC_REGIMES_SYMBOL,
    TOPIC_REGIMES_MARKET,
    TOPIC_SYSTEM_COMPONENT_HEARTBEAT,
)
from .indicators import ATRState, EMAState, RSIState, RollingVolState
from .regimes import (
    SymbolRegimeEvent,
    MarketRegimeEvent,
    DQ_NONE,
    DQ_REPAIRED_INPUT,
    DQ_GAP_WINDOW,
    DQ_STALE_INPUT,
    DEFAULT_REGIME_THRESHOLDS,
    compute_config_hash,
    get_market_for_symbol,
    symbol_regime_to_dict,
    market_regime_to_dict,
)

logger = logging.getLogger("argus.regime_detector")


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

ATR_EPSILON = 1e-8

# Indicator periods
EMA_FAST_PERIOD = 12
EMA_SLOW_PERIOD = 26
RSI_PERIOD = 14
ATR_PERIOD = 14
VOL_WINDOW = 20

# Annualization factor for 1-minute bars (sqrt of bars per year)
# ~525600 minutes per year → sqrt ≈ 725
ANNUALIZE_1M = 725.0

from .sessions import get_session_regime


# ═══════════════════════════════════════════════════════════════════════════
# Per-Symbol State
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SymbolState:
    """Per-symbol indicator state for regime detection."""
    symbol: str
    timeframe: int

    # Indicators
    ema_fast: EMAState
    ema_slow: EMAState
    rsi: RSIState
    atr: ATRState
    vol: RollingVolState

    # Previous values for slope calculation
    prev_ema_fast: Optional[float] = None
    prev_close: Optional[float] = None

    # Gap detection
    last_bar_ts_ms: Optional[int] = None
    gap_flag_remaining: int = 0

    # Volatility z-score tracking
    vol_history: Deque[float] = None

    # Liquidity tracking
    spread_history: Deque[float] = None   # recent spread % values
    volume_history: Deque[float] = None   # recent volume values

    # Warmup tracking
    bars_processed: int = 0

    def __post_init__(self):
        if self.vol_history is None:
            self.vol_history = deque(maxlen=50)
        if self.spread_history is None:
            self.spread_history = deque(maxlen=50)
        if self.volume_history is None:
            self.volume_history = deque(maxlen=50)


def create_symbol_state(symbol: str, timeframe: int = 60) -> SymbolState:
    """Factory for creating new symbol state with default periods."""
    return SymbolState(
        symbol=symbol,
        timeframe=timeframe,
        ema_fast=EMAState(EMA_FAST_PERIOD),
        ema_slow=EMAState(EMA_SLOW_PERIOD),
        rsi=RSIState(RSI_PERIOD),
        atr=ATRState(ATR_PERIOD),
        vol=RollingVolState(VOL_WINDOW, ANNUALIZE_1M),
        vol_history=deque(maxlen=50),
        spread_history=deque(maxlen=50),
        volume_history=deque(maxlen=50),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Regime Detector
# ═══════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    Deterministic regime classifier from BarEvents.
    
    Computes volatility and trend regimes per symbol,
    session regimes per market. All computation is driven
    by bar arrival (feature clock pattern).
    """

    def __init__(
        self,
        bus: EventBus,
        db: Optional[Any] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._bus = bus
        self._db = db
        self._lock = threading.Lock()
        self._start_time = time.time()
        
        # Config
        self._thresholds = thresholds or DEFAULT_REGIME_THRESHOLDS
        self._config_hash = compute_config_hash(self._thresholds)
        
        # Per-symbol state
        self._symbol_states: Dict[str, SymbolState] = {}
        
        # Market-level state
        self._last_market_regimes: Dict[str, MarketRegimeEvent] = {}
        
        # Telemetry
        self._bars_received = 0
        self._symbol_events_emitted = 0
        self._market_events_emitted = 0
        self._warmup_skips = 0
        self._gaps_detected = 0
        
        # Subscribe to bars
        bus.subscribe(TOPIC_MARKET_BARS, self._on_bar)
        logger.info(
            "RegimeDetector initialized — config_hash=%s, subscribed to %s",
            self._config_hash, TOPIC_MARKET_BARS
        )

    # ─── Warmup ───────────────────────────────────────────────────────────

    async def warmup_from_db(self, n_bars: int = 60) -> Dict[str, int]:
        """
        Reconstruct indicator state from persisted bars.
        
        For each symbol with bars in DB, replays the last n_bars
        through the same indicator logic as live processing.
        
        Returns dict of {symbol: bars_replayed}.
        """
        if self._db is None:
            logger.warning("warmup_from_db called but no db reference")
            return {}
        
        result = {}
        
        # Get list of symbols from recent bars
        cursor = await self._db.fetch_all("""
            SELECT DISTINCT symbol, source, bar_duration 
            FROM market_bars 
            ORDER BY timestamp DESC 
            LIMIT 100
        """)
        
        seen = set()
        for row in cursor:
            key = (row['symbol'], row['source'], row['bar_duration'])
            if key in seen:
                continue
            seen.add(key)
            
            symbol = row['symbol']
            source = row['source']
            timeframe = row['bar_duration']
            
            # Get recent bars for this symbol
            bars = await self._db.get_recent_bars(source, symbol, timeframe, n_bars)
            
            if not bars:
                continue
            
            logger.debug("Warming up %s with %d bars", symbol, len(bars))
            
            # Create state and replay
            state = self._get_or_create_state(symbol, timeframe)
            
            for bar_dict in bars:
                self._update_indicators_from_bar_dict(state, bar_dict)
            
            result[symbol] = len(bars)
        
        logger.info("Warmup complete: %d symbols initialized", len(result))
        return result

    def _update_indicators_from_bar_dict(self, state: SymbolState, bar: Dict) -> None:
        """Update indicators from a bar dict (for warmup replay)."""
        close = bar['close']
        high = bar['high']
        low = bar['low']
        
        # Update indicators
        state.ema_fast.update(close)
        state.ema_slow.update(close)
        state.rsi.update(close)
        state.atr.update(high, low, close)
        
        # Log return for volatility
        if state.prev_close is not None and state.prev_close > 0:
            log_ret = math.log(close / state.prev_close)
            vol = state.vol.update(log_ret)
            if vol is not None:
                state.vol_history.append(vol)
        
        state.prev_close = close
        state.bars_processed += 1

    # ─── Bar Handler ──────────────────────────────────────────────────────

    def _on_bar(self, event: BarEvent) -> None:
        """Handle incoming bar event."""
        with self._lock:
            self._bars_received += 1
            
            symbol = event.symbol
            timeframe = event.bar_duration
            timestamp_ms = int(event.timestamp * 1000)
            
            # Get or create state
            state = self._get_or_create_state(symbol, timeframe)
            
            # Check for gaps
            dq_flags = DQ_NONE
            if event.repaired:
                dq_flags |= DQ_REPAIRED_INPUT
            
            dq_flags |= self._check_gap(state, timestamp_ms, timeframe)
            
            # Update indicators
            self._update_indicators(state, event)
            
            # Classify and emit symbol regime
            self._emit_symbol_regime(state, timestamp_ms, dq_flags)
            
            # Update market regime
            market = get_market_for_symbol(symbol)
            self._emit_market_regime(market, timeframe, timestamp_ms, dq_flags)

    def _get_or_create_state(self, symbol: str, timeframe: int) -> SymbolState:
        """Get existing state or create new one."""
        key = f"{symbol}:{timeframe}"
        if key not in self._symbol_states:
            self._symbol_states[key] = create_symbol_state(symbol, timeframe)
        return self._symbol_states[key]

    def _check_gap(self, state: SymbolState, ts_ms: int, timeframe: int) -> int:
        """Check for gap in bar sequence. Returns DQ flags."""
        flags = DQ_NONE
        
        # Decrement any existing gap flag
        if state.gap_flag_remaining > 0:
            state.gap_flag_remaining -= 1
            flags |= DQ_GAP_WINDOW
        
        if state.last_bar_ts_ms is not None:
            expected_ts = state.last_bar_ts_ms + (timeframe * 1000)
            tolerance = self._thresholds.get("gap_tolerance_bars", 1) * timeframe * 1000
            
            if ts_ms > expected_ts + tolerance:
                # Gap detected
                self._gaps_detected += 1
                flags |= DQ_GAP_WINDOW
                state.gap_flag_remaining = self._thresholds.get("gap_flag_duration_bars", 2)
                logger.debug(
                    "Gap detected for %s: expected=%d, got=%d",
                    state.symbol, expected_ts, ts_ms
                )
        
        state.last_bar_ts_ms = ts_ms
        return flags

    def _update_indicators(self, state: SymbolState, bar: BarEvent) -> None:
        """Update all indicators from bar event."""
        close = bar.close
        high = bar.high
        low = bar.low

        # Store previous EMA for slope calculation
        if state.ema_fast._ema is not None:
            state.prev_ema_fast = state.ema_fast._ema

        # Update indicators
        state.ema_fast.update(close)
        state.ema_slow.update(close)
        state.rsi.update(close)
        state.atr.update(high, low, close)

        # Log return for volatility
        if state.prev_close is not None and state.prev_close > 0:
            log_ret = math.log(close / state.prev_close)
            vol = state.vol.update(log_ret)
            if vol is not None:
                state.vol_history.append(vol)

        # Track spread (high-low as proxy for intra-bar spread)
        mid = (high + low) / 2.0
        if mid > 0:
            spread_pct = (high - low) / mid
            state.spread_history.append(spread_pct)
        else:
            state.spread_history.append(0.0)

        # Track volume
        volume = getattr(bar, "volume", 0) or 0
        state.volume_history.append(float(volume))

        state.prev_close = close
        state.bars_processed += 1

    # ─── Regime Classification ────────────────────────────────────────────

    def _emit_symbol_regime(
        self, state: SymbolState, ts_ms: int, dq_flags: int
    ) -> None:
        """Classify and emit symbol regime event."""
        warmup_bars = self._thresholds.get("warmup_bars", 30)
        
        # Check warmup
        ema_fast = state.ema_fast._ema
        ema_slow = state.ema_slow._ema
        atr = state.atr._atr
        rsi_val = state.rsi._avg_gain  # Check if RSI is warm
        
        is_warm = (
            state.bars_processed >= warmup_bars
            and ema_fast is not None
            and ema_slow is not None
            and atr is not None
            and atr > ATR_EPSILON
            and rsi_val is not None
            and len(state.vol_history) >= 5
        )
        
        if not is_warm:
            self._warmup_skips += 1
            dq_flags |= DQ_STALE_INPUT
        
        # Get current indicator values (with defaults for warmup)
        atr = atr if atr and atr > ATR_EPSILON else 1.0
        ema_fast = ema_fast if ema_fast else state.prev_close or 0.0
        ema_slow = ema_slow if ema_slow else state.prev_close or 0.0
        close = state.prev_close or 1.0
        
        # Compute RSI
        rsi = 50.0
        if state.rsi._avg_gain is not None and state.rsi._avg_loss is not None:
            if state.rsi._avg_loss == 0:
                rsi = 100.0
            else:
                rs = state.rsi._avg_gain / state.rsi._avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Compute ATR percentage
        atr_pct = atr / close if close > 0 else 0.0
        
        # Compute EMA slope (normalized by ATR)
        ema_slope = 0.0
        if state.prev_ema_fast is not None and atr > ATR_EPSILON:
            ema_slope = (ema_fast - state.prev_ema_fast) / atr
        
        # Compute trend strength
        trend_strength = abs(ema_fast - ema_slow) / atr if atr > ATR_EPSILON else 0.0
        
        # Compute vol z-score
        vol_z = 0.0
        if len(state.vol_history) >= 5:
            vol_list = list(state.vol_history)
            current_vol = vol_list[-1] if vol_list else 0.0
            mean_vol = sum(vol_list) / len(vol_list)
            if len(vol_list) > 1:
                var = sum((v - mean_vol) ** 2 for v in vol_list) / (len(vol_list) - 1)
                std_vol = math.sqrt(var) if var > 0 else 1.0
                vol_z = (current_vol - mean_vol) / std_vol if std_vol > 0 else 0.0
        
        # Classify volatility regime
        vol_regime = self._classify_vol_regime(vol_z)

        # Classify trend regime
        trend_regime = self._classify_trend_regime(ema_slope, trend_strength, is_warm)

        # Classify liquidity regime
        current_spread = state.spread_history[-1] if state.spread_history else 0.0
        current_volume = state.volume_history[-1] if state.volume_history else 0.0

        # Compute volume percentile from history
        volume_pctile = 50.0  # default
        if len(state.volume_history) >= 5 and current_volume > 0:
            vol_list = sorted(state.volume_history)
            rank = sum(1 for v in vol_list if v <= current_volume)
            volume_pctile = (rank / len(vol_list)) * 100.0

        liq_regime, spread_pct_val, vol_pctile_val = self._classify_liquidity_regime(
            current_spread, volume_pctile
        )

        # Compute confidence
        confidence = 1.0 if is_warm else 0.5
        if dq_flags & DQ_REPAIRED_INPUT:
            confidence *= 0.9
        if dq_flags & DQ_GAP_WINDOW:
            confidence *= 0.8

        # Create and emit event
        event = SymbolRegimeEvent(
            symbol=state.symbol,
            timeframe=state.timeframe,
            timestamp_ms=ts_ms,
            vol_regime=vol_regime,
            trend_regime=trend_regime,
            liquidity_regime=liq_regime,
            atr=atr,
            atr_pct=atr_pct,
            vol_z=vol_z,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_slope=ema_slope,
            rsi=rsi,
            spread_pct=current_spread,
            volume_pctile=vol_pctile_val,
            confidence=confidence,
            is_warm=is_warm,
            data_quality_flags=dq_flags,
            config_hash=self._config_hash,
        )

        self._bus.publish(TOPIC_REGIMES_SYMBOL, event)
        self._symbol_events_emitted += 1

    def _classify_vol_regime(self, vol_z: float) -> str:
        """Classify volatility regime from z-score."""
        spike_z = self._thresholds.get("vol_spike_z", 2.5)
        high_z = self._thresholds.get("vol_high_z", 1.0)
        low_z = self._thresholds.get("vol_low_z", -0.5)
        
        if vol_z > spike_z:
            return "VOL_SPIKE"
        elif vol_z > high_z:
            return "VOL_HIGH"
        elif vol_z < low_z:
            return "VOL_LOW"
        else:
            return "VOL_NORMAL"

    def _classify_trend_regime(
        self, ema_slope: float, trend_strength: float, is_warm: bool
    ) -> str:
        """Classify trend regime."""
        if not is_warm:
            return "RANGE"

        slope_thresh = self._thresholds.get("trend_slope_threshold", 0.5)
        strength_thresh = self._thresholds.get("trend_strength_threshold", 1.0)

        if trend_strength > strength_thresh:
            if ema_slope > slope_thresh:
                return "TREND_UP"
            elif ema_slope < -slope_thresh:
                return "TREND_DOWN"

        return "RANGE"

    def _classify_liquidity_regime(
        self, spread_pct: float, volume_pctile: float
    ) -> Tuple[str, float, float]:
        """Classify liquidity regime from spread and volume.

        Returns (regime, spread_pct, volume_pctile).
        """
        dried_pct = self._thresholds.get("liq_spread_dried_pct", 0.50)
        low_pct = self._thresholds.get("liq_spread_low_pct", 0.20)
        high_pct = self._thresholds.get("liq_spread_high_pct", 0.05)

        if spread_pct > dried_pct:
            return "LIQ_DRIED", spread_pct, volume_pctile
        if spread_pct > low_pct or volume_pctile < self._thresholds.get("liq_volume_low_pctile", 25):
            return "LIQ_LOW", spread_pct, volume_pctile
        if spread_pct < high_pct and volume_pctile > self._thresholds.get("liq_volume_high_pctile", 75):
            return "LIQ_HIGH", spread_pct, volume_pctile
        return "LIQ_NORMAL", spread_pct, volume_pctile

    def _emit_market_regime(
        self, market: str, timeframe: int, ts_ms: int, dq_flags: int
    ) -> None:
        """Emit market regime event."""
        # Session regime from timestamp
        session = self._get_session_regime(market, ts_ms)
        
        # Create event
        event = MarketRegimeEvent(
            market=market,
            timeframe=timeframe,
            timestamp_ms=ts_ms,
            session_regime=session,
            risk_regime="UNKNOWN",  # Stub for future
            confidence=1.0,
            data_quality_flags=dq_flags,
            config_hash=self._config_hash,
        )
        
        self._bus.publish(TOPIC_REGIMES_MARKET, event)
        self._market_events_emitted += 1
        self._last_market_regimes[market] = event

    def _get_session_regime(self, market: str, ts_ms: int) -> str:
        """Determine session from timestamp (no wall clock)."""
        return get_session_regime(market, ts_ms)

    # ─── Public API ───────────────────────────────────────────────────────

    def get_symbol_regime(self, symbol: str, timeframe: int = 60) -> Optional[str]:
        """Get current volatility regime for symbol."""
        key = f"{symbol}:{timeframe}"
        with self._lock:
            state = self._symbol_states.get(key)
            if state is None:
                return None
            # Return combined regime string
            vol_z = 0.0
            if len(state.vol_history) >= 5:
                vol_list = list(state.vol_history)
                current_vol = vol_list[-1]
                mean_vol = sum(vol_list) / len(vol_list)
                var = sum((v - mean_vol) ** 2 for v in vol_list) / max(1, len(vol_list) - 1)
                std_vol = math.sqrt(var) if var > 0 else 1.0
                vol_z = (current_vol - mean_vol) / std_vol if std_vol > 0 else 0.0
            return self._classify_vol_regime(vol_z)

    def get_all_regimes(self) -> Dict[str, Dict[str, Any]]:
        """Get current regime status for all symbols."""
        with self._lock:
            result = {}
            for key, state in self._symbol_states.items():
                result[key] = {
                    "bars_processed": state.bars_processed,
                    "is_warm": state.bars_processed >= self._thresholds.get("warmup_bars", 30),
                    "last_ts_ms": state.last_bar_ts_ms,
                }
            return result

    def get_market_regime(self, market: str) -> Optional[MarketRegimeEvent]:
        """Get last market regime event."""
        with self._lock:
            return self._last_market_regimes.get(market)

    # ─── Telemetry ────────────────────────────────────────────────────────

    def emit_heartbeat(self) -> ComponentHeartbeatEvent:
        """Emit component heartbeat."""
        now = time.time()
        with self._lock:
            bars = self._bars_received
            symbol_events = self._symbol_events_emitted
            market_events = self._market_events_emitted
            warmup_skips = self._warmup_skips
            gaps = self._gaps_detected
        
        uptime = now - self._start_time
        health = "ok" if bars > 0 or uptime < 120 else "down"
        
        hb = ComponentHeartbeatEvent(
            component="regime_detector",
            uptime_seconds=round(uptime, 1),
            events_processed=bars,
            health=health,
            extra={
                "symbol_events_emitted": symbol_events,
                "market_events_emitted": market_events,
                "warmup_skips_total": warmup_skips,
                "gaps_detected_total": gaps,
                "config_hash": self._config_hash,
            },
        )
        self._bus.publish(TOPIC_SYSTEM_COMPONENT_HEARTBEAT, hb)
        return hb

    def get_status(self) -> Dict[str, Any]:
        """Get status for dashboard."""
        from .status import build_status
        
        now = time.time()
        with self._lock:
            bars = self._bars_received
            symbol_events = self._symbol_events_emitted
            market_events = self._market_events_emitted
            warmup_skips = self._warmup_skips
            gaps = self._gaps_detected
            symbols = list(self._symbol_states.keys())
        
        uptime = now - self._start_time
        status = "ok" if bars > 0 else ("ok" if uptime < 120 else "down")
        
        return build_status(
            name="regime_detector",
            type="internal",
            status=status,
            request_count=bars,
            extras={
                "bars_received": bars,
                "symbol_events_emitted": symbol_events,
                "market_events_emitted": market_events,
                "warmup_skips_total": warmup_skips,
                "gaps_detected_total": gaps,
                "config_hash": self._config_hash,
                "symbols_tracked": len(symbols),
                "uptime_seconds": round(uptime, 1),
            },
        )
