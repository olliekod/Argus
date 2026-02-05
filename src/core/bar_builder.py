"""
Argus Bar Builder
=================

Subscribes to ``market.quotes`` and aggregates tick data into
1-minute OHLCV bars aligned to UTC minute boundaries.

Rules
-----
* Use exchange ``timestamp`` when available; fall back to
  ``receive_time`` otherwise.
* Bars are aligned to the **start** of each UTC minute
  (e.g. 12:03:00.000 – 12:03:59.999 → bar timestamp 12:03:00).
* When a new minute begins, the completed bar is published to
  ``market.bars`` via the event bus.

Volume handling
---------------
``QuoteEvent.volume_24h`` is **cumulative** exchange volume.  Summing
it directly would inflate bar volume by orders of magnitude.  Instead
we track the last-seen cumulative value per symbol and only add the
*delta* (current − previous).  A negative delta (exchange reset /
rollover) is treated as zero to avoid corrupting the bar.

Late-tick policy
----------------
A tick whose minute-floor falls **before** the active bar's open
timestamp is silently discarded.  Once a bar is emitted it is
immutable.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict

from .bus import EventBus
from .events import (
    BarEvent,
    MinuteTickEvent,
    QuoteEvent,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_QUOTES,
    TOPIC_SYSTEM_MINUTE_TICK,
)

logger = logging.getLogger("argus.bar_builder")


class _BarAccumulator:
    """Mutable accumulator for a single in-progress bar."""

    __slots__ = ("open", "high", "low", "close", "volume", "ts_open", "source", "tick_count")

    def __init__(self, price: float, volume_delta: float, ts_open: float, source: str) -> None:
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = volume_delta
        self.ts_open = ts_open
        self.source = source
        self.tick_count = 1

    def update(self, price: float, volume_delta: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume_delta
        self.tick_count += 1


def _minute_floor(epoch: float) -> float:
    """Round *epoch* down to the start of its UTC minute."""
    return float(int(epoch) // 60 * 60)


class BarBuilder:
    """Aggregates :class:`QuoteEvent` ticks into 1-minute :class:`BarEvent`.

    Parameters
    ----------
    bus : EventBus
        The shared event bus.  BarBuilder will subscribe to
        ``market.quotes`` and publish completed bars on ``market.bars``.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._bars: Dict[str, _BarAccumulator] = {}   # symbol → accumulator
        self._last_cum_vol: Dict[str, float] = {}      # symbol → last cumulative volume_24h
        self._lock = threading.Lock()
        self._bars_emitted_total = 0
        self._bars_emitted_by_symbol: Dict[str, int] = {}
        self._last_bar_ts_by_symbol: Dict[str, float] = {}
        self._late_ticks_dropped_total = 0
        self._late_ticks_dropped_by_symbol: Dict[str, int] = {}
        self._quotes_received_by_symbol: Dict[str, int] = {}
        self._bars_emitted_window_minutes = 60
        self._bars_emitted_timestamps_by_symbol: Dict[str, Deque[float]] = {}
        bus.subscribe(TOPIC_MARKET_QUOTES, self._on_quote)
        bus.subscribe(TOPIC_SYSTEM_MINUTE_TICK, self._on_minute_tick)
        logger.info("BarBuilder initialised — subscribed to %s", TOPIC_MARKET_QUOTES)

    # ── volume delta helper ─────────────────────────────────

    def _volume_delta(self, symbol: str, cum_vol: float) -> float:
        """Compute the volume delta since last tick for *symbol*.

        * First tick for a symbol → delta = 0 (no prior reference).
        * Negative delta (exchange reset / rollover) → 0.
        * Otherwise → ``cum_vol - last_cum_vol``.
        """
        prev = self._last_cum_vol.get(symbol)
        self._last_cum_vol[symbol] = cum_vol

        if prev is None:
            return 0.0

        delta = cum_vol - prev
        if delta < 0:
            # Exchange reset / rollover — ignore this tick's volume
            return 0.0
        return delta

    # ── handler (called from the bus worker thread) ─────────

    def _on_quote(self, event: QuoteEvent) -> None:
        """Ingest a quote and build / emit bars."""
        # Prefer exchange timestamp; fall back to receive_time
        ts = event.timestamp if event.timestamp and event.timestamp > 0 else event.receive_time
        minute = _minute_floor(ts)
        price = event.last if event.last else event.mid
        if price <= 0:
            return

        self._quotes_received_by_symbol[event.symbol] = (
            self._quotes_received_by_symbol.get(event.symbol, 0) + 1
        )
        vol_delta = self._volume_delta(event.symbol, event.volume_24h)

        with self._lock:
            acc = self._bars.get(event.symbol)

            if acc is None:
                # First tick for this symbol — start a new bar
                self._bars[event.symbol] = _BarAccumulator(
                    price, vol_delta, minute, event.source
                )
                return

            # ── Late-tick guard ─────────────────────────────
            # Discard ticks older than the active bar window.
            # Once a bar is emitted it must never change.
            if minute < acc.ts_open:
                self._late_ticks_dropped_total += 1
                self._late_ticks_dropped_by_symbol[event.symbol] = (
                    self._late_ticks_dropped_by_symbol.get(event.symbol, 0) + 1
                )
                return

            if minute > acc.ts_open:
                # New minute — emit the completed bar and start fresh
                self._emit_bar(event.symbol, acc)

                # Reset accumulator for the new minute
                self._bars[event.symbol] = _BarAccumulator(
                    price, vol_delta, minute, event.source
                )
            else:
                # Same minute — update accumulator
                acc.update(price, vol_delta)

    def _on_minute_tick(self, event: MinuteTickEvent) -> None:
        """Flush any bars whose minute has closed at a boundary tick."""
        tick_minute = _minute_floor(event.timestamp)
        with self._lock:
            to_remove = []
            for symbol, acc in self._bars.items():
                if tick_minute > acc.ts_open:
                    self._emit_bar(symbol, acc)
                    to_remove.append(symbol)
            for symbol in to_remove:
                self._bars.pop(symbol, None)

    # ── utility ─────────────────────────────────────────────

    def _emit_bar(self, symbol: str, acc: _BarAccumulator) -> BarEvent:
        bar = BarEvent(
            symbol=symbol,
            open=acc.open,
            high=acc.high,
            low=acc.low,
            close=acc.close,
            volume=acc.volume,
            timestamp=acc.ts_open,
            source=acc.source,
            bar_duration=60,
            tick_count=acc.tick_count,
        )
        self._bus.publish(TOPIC_MARKET_BARS, bar)
        self._bars_emitted_total += 1
        self._bars_emitted_by_symbol[symbol] = self._bars_emitted_by_symbol.get(symbol, 0) + 1
        self._last_bar_ts_by_symbol[symbol] = acc.ts_open

        now = time.time()
        window_seconds = self._bars_emitted_window_minutes * 60
        ts_deque = self._bars_emitted_timestamps_by_symbol.setdefault(symbol, deque())
        ts_deque.append(acc.ts_open)
        while ts_deque and (now - ts_deque[0]) > window_seconds:
            ts_deque.popleft()
        return bar

    def flush(self) -> list[BarEvent]:
        """Flush all in-progress bars (e.g. on shutdown).

        Returns the list of emitted bars.
        """
        emitted: list[BarEvent] = []
        with self._lock:
            for symbol, acc in self._bars.items():
                emitted.append(self._emit_bar(symbol, acc))
            self._bars.clear()
        logger.info("BarBuilder flushed %d partial bars", len(emitted))
        return emitted

    def get_status(self) -> Dict[str, object]:
        now = time.time()
        with self._lock:
            last_bar_ts = dict(self._last_bar_ts_by_symbol)
            active_symbols = list(self._bars.keys())
            bars_emitted_total = self._bars_emitted_total
            bars_emitted_by_symbol = dict(self._bars_emitted_by_symbol)
            late_ticks_total = self._late_ticks_dropped_total
            late_ticks_by_symbol = dict(self._late_ticks_dropped_by_symbol)
            quotes_received_by_symbol = dict(self._quotes_received_by_symbol)
            bars_emitted_recent_by_symbol = {
                symbol: sum(1 for ts in timestamps if (now - ts) <= (self._bars_emitted_window_minutes * 60))
                for symbol, timestamps in self._bars_emitted_timestamps_by_symbol.items()
            }

        ages = {
            symbol: round(now - ts, 1)
            for symbol, ts in last_bar_ts.items()
            if ts is not None
        }
        max_age = max(ages.values()) if ages else None

        if not last_bar_ts:
            status = "unknown"
        elif max_age is not None and max_age > 300:
            status = "degraded"
        else:
            status = "ok"

        from .status import build_status

        return build_status(
            name="bar_builder",
            type="internal",
            status=status,
            last_success_ts=max(last_bar_ts.values()) if last_bar_ts else None,
            consecutive_failures=0,
            request_count=bars_emitted_total,
            error_count=late_ticks_total,
            last_message_ts=max(last_bar_ts.values()) if last_bar_ts else None,
            age_seconds=max_age,
            extras={
                "bars_emitted_total": bars_emitted_total,
                "bars_emitted_by_symbol": bars_emitted_by_symbol,
                "bars_emitted_recent_by_symbol": bars_emitted_recent_by_symbol,
                "bars_emitted_recent_window_minutes": self._bars_emitted_window_minutes,
                "last_bar_ts_by_symbol": {
                    symbol: datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    for symbol, ts in last_bar_ts.items()
                },
                "last_bar_ts_by_symbol_epoch": dict(last_bar_ts),
                "late_ticks_dropped_total": late_ticks_total,
                "late_ticks_dropped_by_symbol": late_ticks_by_symbol,
                "quotes_received_by_symbol": quotes_received_by_symbol,
                "active_symbols_count": len(active_symbols),
            },
        )
