"""
Argus Regime Detector (Stream 4)
================================

Consumes rolling metrics from ``market.metrics`` (specifically
``realized_vol`` and ``log_return``) and classifies the current
market regime for each symbol.

Regimes
-------
* ``UNKNOWN``       — insufficient data
* ``LO_VOL_TREND``  — low volatility, trending (returns biased)
* ``HI_VOL_RANGE``  — high volatility, mean-reverting
* ``LO_VOL_RANGE``  — low volatility, range-bound
* ``HI_VOL_TREND``  — high volatility with directional momentum
* ``CRASH``         — extreme downside move detected

Emits ``RegimeChangeEvent`` to ``signals.regime`` on transitions.

Safety constraints
------------------
* Downstream-only: subscribes to ``market.metrics``, publishes to
  ``signals.regime``.  Never touches upstream bar/quote state.
* Uses deterministic threshold logic (no ML, no randomness).
* Bounded state per symbol.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from .bus import EventBus
from .events import (
    ComponentHeartbeatEvent,
    MetricEvent,
    RegimeChangeEvent,
    TOPIC_MARKET_METRICS,
    TOPIC_SIGNALS_REGIME,
    TOPIC_SYSTEM_COMPONENT_HEARTBEAT,
)

logger = logging.getLogger("argus.regime_detector")

# ── Regime constants ─────────────────────────────────────────
UNKNOWN = "UNKNOWN"
LO_VOL_TREND = "LO_VOL_TREND"
HI_VOL_RANGE = "HI_VOL_RANGE"
LO_VOL_RANGE = "LO_VOL_RANGE"
HI_VOL_TREND = "HI_VOL_TREND"
CRASH = "CRASH"

# Thresholds (annualised realised vol)
_VOL_LOW = 0.30       # < 30% annualised = low vol
_VOL_HIGH = 0.80      # > 80% annualised = high vol
_TREND_BIAS = 0.003   # |mean return| > 0.3% per bar = trending
_CRASH_RET = -0.05    # single-bar return < -5% = crash signal

# Minimum observations before classifying
_MIN_OBS = 15

# Rolling window for return-bias calculation
_BIAS_WINDOW = 30


class RegimeDetector:
    """Deterministic regime classifier from rolling metrics.

    Parameters
    ----------
    bus : EventBus
        Shared event bus.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Per-symbol state
        self._current_regime: Dict[str, str] = {}
        self._last_vol: Dict[str, float] = {}
        self._returns_buf: Dict[str, Deque[float]] = {}
        self._events_processed = 0
        self._regime_changes = 0

        bus.subscribe(TOPIC_MARKET_METRICS, self._on_metric)
        logger.info("RegimeDetector initialised — subscribed to %s", TOPIC_MARKET_METRICS)

    # ── metric handler ───────────────────────────────────────

    def _on_metric(self, event: MetricEvent) -> None:
        """Ingest realized_vol and log_return to classify regime."""
        # Only process feature_builder metrics
        if not event.source.startswith("feature_builder:"):
            return

        symbol = event.symbol
        metric = event.metric

        with self._lock:
            self._events_processed += 1

            if metric == "realized_vol":
                self._last_vol[symbol] = event.value

            elif metric == "log_return":
                if symbol not in self._returns_buf:
                    self._returns_buf[symbol] = deque(maxlen=_BIAS_WINDOW)
                self._returns_buf[symbol].append(event.value)

                # Check for crash
                if event.value < _CRASH_RET:
                    self._transition(symbol, CRASH, event.timestamp, {
                        "trigger_return": event.value,
                    })
                    return

            # Attempt classification after every metric update
            self._classify(symbol, event.timestamp)

    def _classify(self, symbol: str, ts: float) -> None:
        """Classify regime from accumulated vol + return data."""
        vol = self._last_vol.get(symbol)
        ret_buf = self._returns_buf.get(symbol)

        if vol is None or ret_buf is None or len(ret_buf) < _MIN_OBS:
            return  # not enough data yet

        mean_ret = sum(ret_buf) / len(ret_buf)
        trending = abs(mean_ret) > _TREND_BIAS

        if vol < _VOL_LOW:
            regime = LO_VOL_TREND if trending else LO_VOL_RANGE
        elif vol > _VOL_HIGH:
            regime = HI_VOL_TREND if trending else HI_VOL_RANGE
        else:
            # Medium vol — use trending flag to disambiguate
            regime = LO_VOL_TREND if trending else LO_VOL_RANGE

        self._transition(symbol, regime, ts, {
            "realized_vol": vol,
            "mean_return": mean_ret,
            "obs_count": len(ret_buf),
        })

    def _transition(self, symbol: str, new_regime: str, ts: float,
                    data: Dict[str, Any]) -> None:
        """Emit a RegimeChangeEvent if the regime has changed."""
        old = self._current_regime.get(symbol, UNKNOWN)
        if old == new_regime:
            return

        self._current_regime[symbol] = new_regime
        self._regime_changes += 1

        event = RegimeChangeEvent(
            symbol=symbol,
            old_regime=old,
            new_regime=new_regime,
            confidence=1.0,  # deterministic = full confidence
            data=data,
            timestamp=ts,
        )
        self._bus.publish(TOPIC_SIGNALS_REGIME, event)
        logger.info("Regime change %s: %s → %s", symbol, old, new_regime)

    # ── public API ───────────────────────────────────────────

    def get_current_regime(self, symbol: str) -> str:
        """Return current regime for *symbol* (thread-safe)."""
        with self._lock:
            return self._current_regime.get(symbol, UNKNOWN)

    def get_all_regimes(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._current_regime)

    # ── heartbeat / status ───────────────────────────────────

    def emit_heartbeat(self) -> ComponentHeartbeatEvent:
        now = time.time()
        with self._lock:
            processed = self._events_processed
            changes = self._regime_changes
            regimes = dict(self._current_regime)

        health = "ok"
        if not processed and (now - self._start_time) > 120:
            health = "down"

        hb = ComponentHeartbeatEvent(
            component="regime_detector",
            uptime_seconds=round(now - self._start_time, 1),
            events_processed=processed,
            health=health,
            extra={
                "regime_changes_total": changes,
                "current_regimes": regimes,
            },
        )
        self._bus.publish(TOPIC_SYSTEM_COMPONENT_HEARTBEAT, hb)
        return hb

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": "regime_detector",
                "events_processed": self._events_processed,
                "regime_changes_total": self._regime_changes,
                "current_regimes": dict(self._current_regime),
                "uptime_seconds": round(time.time() - self._start_time, 1),
            }
