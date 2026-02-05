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
        self._status_lock = threading.Lock()
        self._last_flush_ts: Optional[float] = None
        self._last_flush_ms: Optional[float] = None
        self._flush_latency_ema: Optional[float] = None
        self._last_write_ts: Optional[float] = None
        self._db_write_errors: int = 0
        self._last_error: Optional[str] = None
        self._metrics_writes_total: int = 0
        self._bars_writes_total: int = 0
        self._signals_writes_total: int = 0
        self._consecutive_failures: int = 0

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
            with self._status_lock:
                self._db_write_errors += 1
                self._consecutive_failures += 1
                self._last_error = "signal_write_failed"
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
            with self._status_lock:
                self._db_write_errors += 1
                self._consecutive_failures += 1
                self._last_error = "metric_write_failed"
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

        start = time.perf_counter()
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._write_bars(batch), self._loop
            )
            future.result(timeout=30.0)
            duration_ms = (time.perf_counter() - start) * 1000
            now = time.time()
            with self._status_lock:
                self._last_flush_ts = now
                self._last_flush_ms = duration_ms
                self._flush_latency_ema = (
                    duration_ms if self._flush_latency_ema is None
                    else (duration_ms * 0.2) + (self._flush_latency_ema * 0.8)
                )
                self._last_write_ts = now
                self._consecutive_failures = 0
        except Exception:
            with self._status_lock:
                self._db_write_errors += 1
                self._consecutive_failures += 1
                self._last_error = "bar_flush_failed"
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
        now = time.time()
        with self._status_lock:
            self._bars_writes_total += len(rows)
            self._last_write_ts = now
            self._consecutive_failures = 0
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
        now = time.time()
        with self._status_lock:
            self._signals_writes_total += 1
            self._last_write_ts = now
            self._consecutive_failures = 0
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
        now = time.time()
        with self._status_lock:
            self._metrics_writes_total += 1
            self._last_write_ts = now
            self._consecutive_failures = 0
        logger.debug("Persisted metric: %s:%s=%s", event.symbol, event.metric, event.value)

    def get_status(self) -> Dict[str, Any]:
        with self._bar_lock:
            bar_buffer_size = len(self._bar_buffer)
        with self._status_lock:
            last_flush_ts = self._last_flush_ts
            last_flush_ms = self._last_flush_ms
            flush_ema = self._flush_latency_ema
            last_write_ts = self._last_write_ts
            db_write_errors = self._db_write_errors
            last_error = self._last_error
            metrics_writes_total = self._metrics_writes_total
            bars_writes_total = self._bars_writes_total
            signals_writes_total = self._signals_writes_total
            consecutive_failures = self._consecutive_failures

        now = time.time()
        age_seconds = (now - last_write_ts) if last_write_ts else None
        status = "ok"
        if consecutive_failures > 0 or db_write_errors > 0:
            status = "degraded"
        if last_write_ts is None:
            status = "unknown"

        from .status import build_status

        return build_status(
            name="persistence",
            type="internal",
            status=status,
            last_success_ts=last_write_ts,
            last_error=last_error,
            consecutive_failures=consecutive_failures,
            request_count=metrics_writes_total + bars_writes_total + signals_writes_total,
            error_count=db_write_errors,
            avg_latency_ms=round(flush_ema, 2) if flush_ema is not None else None,
            last_latency_ms=round(last_flush_ms, 2) if last_flush_ms is not None else None,
            last_poll_ts=last_flush_ts,
            age_seconds=round(age_seconds, 1) if age_seconds is not None else None,
            extras={
                "bar_buffer_size": bar_buffer_size,
                "last_flush_ts": (
                    datetime.fromtimestamp(last_flush_ts, tz=timezone.utc).isoformat()
                    if last_flush_ts
                    else None
                ),
                "metrics_writes_total": metrics_writes_total,
                "bars_writes_total": bars_writes_total,
                "signals_writes_total": signals_writes_total,
            },
        )
