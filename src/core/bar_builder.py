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
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Dict, Optional

from .bus import EventBus
from .events import (
    BarEvent,
    QuoteEvent,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_QUOTES,
)

logger = logging.getLogger("argus.bar_builder")


class _BarAccumulator:
    """Mutable accumulator for a single in-progress bar."""

    __slots__ = ("open", "high", "low", "close", "volume", "ts_open", "source", "tick_count")

    def __init__(self, price: float, volume: float, ts_open: float, source: str) -> None:
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = volume
        self.ts_open = ts_open
        self.source = source
        self.tick_count = 1

    def update(self, price: float, volume: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
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
        self._bars: Dict[str, _BarAccumulator] = {}  # symbol → accumulator
        self._lock = threading.Lock()
        bus.subscribe(TOPIC_MARKET_QUOTES, self._on_quote)
        logger.info("BarBuilder initialised — subscribed to %s", TOPIC_MARKET_QUOTES)

    # ── handler (called from the bus worker thread) ─────────

    def _on_quote(self, event: QuoteEvent) -> None:
        """Ingest a quote and build / emit bars."""
        # Prefer exchange timestamp; fall back to receive_time
        ts = event.timestamp if event.timestamp and event.timestamp > 0 else event.receive_time
        minute = _minute_floor(ts)
        price = event.last if event.last else event.mid
        if price <= 0:
            return

        volume = event.volume_24h  # cumulative; delta not available

        with self._lock:
            acc = self._bars.get(event.symbol)

            if acc is None:
                # First tick for this symbol
                self._bars[event.symbol] = _BarAccumulator(
                    price, volume, minute, event.source
                )
                return

            if minute > acc.ts_open:
                # New minute — emit the completed bar and start fresh
                bar = BarEvent(
                    symbol=event.symbol,
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

                # Reset accumulator for the new minute
                self._bars[event.symbol] = _BarAccumulator(
                    price, volume, minute, event.source
                )
            else:
                acc.update(price, volume)

    # ── utility ─────────────────────────────────────────────

    def flush(self) -> list[BarEvent]:
        """Flush all in-progress bars (e.g. on shutdown).

        Returns the list of emitted bars.
        """
        emitted: list[BarEvent] = []
        with self._lock:
            for symbol, acc in self._bars.items():
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
                emitted.append(bar)
            self._bars.clear()
        logger.info("BarBuilder flushed %d partial bars", len(emitted))
        return emitted
