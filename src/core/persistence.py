"""
Argus Persistence Module
========================

Subscribes to bus topics and writes data to the database.

Priority rules
--------------
* **Market bars** are batched and flushed every 1 second (or on
  heartbeat / shutdown) to reduce SQLite write amplification.
* **SignalEvents** are persisted **immediately** (no batching).

Flush triggers
--------------
1. Periodic 1-second timer inside the bar-batch writer.
2. ``system.heartbeat`` events (flush on heartbeat boundaries).
3. ``SIGINT / Ctrl-C`` — the orchestrator calls :meth:`shutdown` which
   flushes all remaining buffered bars.

Storage optimisation
--------------------
Only 1-minute bars are logged.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .bus import EventBus
from .events import (
    BarEvent,
    HeartbeatEvent,
    MetricEvent,
    SignalEvent,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_METRICS,
    TOPIC_SIGNALS,
    TOPIC_SYSTEM_HEARTBEAT,
)

logger = logging.getLogger("argus.persistence")

# How often the bar-batch writer flushes (seconds)
_FLUSH_INTERVAL = 1.0


class PersistenceManager:
    """Async-safe persistence subscriber for the event bus.

    Parameters
    ----------
    bus : EventBus
        Shared event bus.
    db : Database
        The Argus async SQLite database instance.
    loop : asyncio.AbstractEventLoop
        The running asyncio loop (needed to bridge bus worker threads
        into the async database layer).
    """

    def __init__(
        self,
        bus: EventBus,
        db: Any,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._bus = bus
        self._db = db
        self._loop = loop

        # Bar buffer (thread-safe via its own lock)
        self._bar_buffer: deque[BarEvent] = deque(maxlen=100_000)
        self._bar_lock = threading.Lock()
        self._flush_thread: Optional[threading.Thread] = None
        self._running = False

        # Subscribe to relevant topics
        bus.subscribe(TOPIC_MARKET_BARS, self._on_bar)
        bus.subscribe(TOPIC_SIGNALS, self._on_signal)
        bus.subscribe(TOPIC_SYSTEM_HEARTBEAT, self._on_heartbeat)
        bus.subscribe(TOPIC_MARKET_METRICS, self._on_metric)

        logger.info("PersistenceManager initialised")

    # ── lifecycle ───────────────────────────────────────────

    def start(self) -> None:
        """Start the background flush thread."""
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="persistence-flush",
            daemon=True,
        )
        self._flush_thread.start()
        logger.info("PersistenceManager flush thread started")

    def shutdown(self) -> None:
        """Flush all remaining bars and stop the flush thread.

        Called on SIGINT / Ctrl-C from the orchestrator.
        """
        self._running = False
        # Final flush
        self._do_flush()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)
        logger.info("PersistenceManager shut down (all bars flushed)")

    # ── handlers (run on bus worker threads) ────────────────

    def _on_bar(self, event: BarEvent) -> None:
        """Buffer bar for batched write.  O(1)."""
        with self._bar_lock:
            self._bar_buffer.append(event)

    def _on_signal(self, event: SignalEvent) -> None:
        """Persist signal immediately (no batching)."""
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._write_signal(event), self._loop
            )
            future.result(timeout=10.0)
        except Exception:
            logger.exception("Failed to persist signal event %s", event.detector)

    def _on_heartbeat(self, event: HeartbeatEvent) -> None:
        """Flush buffered bars on heartbeat boundary."""
        self._do_flush()

    def _on_metric(self, event: MetricEvent) -> None:
        """Persist market metrics immediately."""
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._write_metric(event), self._loop
            )
            # Result check optional for metrics, but helps catch schema errors early
            future.result(timeout=10.0)
        except Exception:
            logger.exception("Failed to persist metric %s", event.metric)

    # ── flush logic ─────────────────────────────────────────

    def _flush_loop(self) -> None:
        """Background thread: flush bar buffer every _FLUSH_INTERVAL seconds."""
        while self._running:
            time.sleep(_FLUSH_INTERVAL)
            self._do_flush()

    def _do_flush(self) -> None:
        """Drain the bar buffer and write to DB."""
        with self._bar_lock:
            if not self._bar_buffer:
                return
            batch = list(self._bar_buffer)
            self._bar_buffer.clear()

        if not batch:
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._write_bars(batch), self._loop
            )
            future.result(timeout=30.0)
        except Exception:
            logger.exception("Failed to flush %d bars to DB", len(batch))

    # ── async DB writers ────────────────────────────────────

    async def _write_bars(self, bars: List[BarEvent]) -> None:
        """Batch-insert 1-minute bars into the ``market_bars`` table."""
        rows = [
            (
                datetime.fromtimestamp(b.timestamp, tz=timezone.utc).isoformat(),
                b.symbol,
                b.source,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                b.tick_count,
            )
            for b in bars
            if b.bar_duration == 60  # only 1m bars
        ]
        if not rows:
            return
        await self._db.execute_many(
            """INSERT OR IGNORE INTO market_bars
               (timestamp, symbol, source, open, high, low, close, volume, tick_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        logger.debug("Flushed %d bars to market_bars", len(rows))

    async def _write_signal(self, event: SignalEvent) -> None:
        """Write a signal event immediately."""
        await self._db.execute(
            """INSERT INTO signal_events
               (timestamp, detector, symbol, signal_type, priority, data)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat(),
                event.detector,
                event.symbol,
                event.signal_type,
                int(event.priority),
                str(event.data),
            ),
        )
        logger.debug("Persisted signal: %s %s", event.detector, event.signal_type)

    async def _write_metric(self, event: MetricEvent) -> None:
        """Generic DB writer for market metrics."""
        import json
        ts_iso = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat()
        
        # metadata_json handling
        meta = None
        if event.extra:
            try:
                meta = json.dumps(event.extra)
            except (TypeError, ValueError):
                meta = str(event.extra)

        await self._db.insert_market_metric(
            timestamp=ts_iso,
            source=event.source,
            symbol=event.symbol,
            metric=event.metric,
            value=event.value,
            metadata_json=meta
        )
        logger.debug("Persisted metric: %s:%s=%s", event.symbol, event.metric, event.value)
