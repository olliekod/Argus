"""
Argus Persistence Module
========================

Subscribes to bus topics and writes data to the database.

Priority rules
--------------
* **Market bars** (HIGH priority) are batched and flushed every 1 second
  (or on heartbeat / shutdown).  Bars are **never dropped**: on flush
  failure they are returned to the buffer for retry.
* **SignalEvents** (MEDIUM priority) are persisted via a dedicated queue.
* **Metrics / Heartbeats** (LOW priority) share a separate queue and
  are the first to be dropped under overload.

Flush triggers
--------------
1. Periodic 1-second timer inside the bar-batch writer.
2. ``system.heartbeat`` events (flush on heartbeat boundaries).
3. ``SIGINT / Ctrl-C`` — the orchestrator calls :meth:`shutdown` which
   flushes all remaining buffered bars.

Storage optimisation
--------------------
Only 1-minute bars are logged.

Lag tracking (Stream 2.1)
-------------------------
Computes ``persist_lag_ms = now - source_ts`` for every bar flush and
exposes it via status / heartbeat telemetry.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional

from .bus import EventBus
from .events import (
    BarEvent,
    ComponentHeartbeatEvent,
    HeartbeatEvent,
    MetricEvent,
    SignalEvent,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_METRICS,
    TOPIC_SIGNALS,
    TOPIC_SYSTEM_HEARTBEAT,
    TOPIC_SYSTEM_COMPONENT_HEARTBEAT,
)

logger = logging.getLogger("argus.persistence")

# How often the bar-batch writer flushes (seconds)
_FLUSH_INTERVAL = 1.0
_SIGNAL_QUEUE_MAXSIZE = 5_000     # Medium priority (signals)
_TELEMETRY_QUEUE_MAXSIZE = 5_000  # Low priority (metrics / heartbeats)
_BAR_FLUSH_MAX_RETRIES = 3        # Retry failed bar flushes
_WRITE_STOP = object()


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

        # Bar buffer — unbounded list guarded by lock.
        # Bars are NEVER dropped: on flush failure they're returned here.
        self._bar_buffer: List[BarEvent] = []
        self._bar_lock = threading.Lock()
        self._flush_thread: Optional[threading.Thread] = None
        self._signal_write_thread: Optional[threading.Thread] = None
        self._telemetry_write_thread: Optional[threading.Thread] = None

        # Priority queues: signals (medium) vs metrics/heartbeats (low)
        self._signal_queue: Queue = Queue(maxsize=_SIGNAL_QUEUE_MAXSIZE)
        self._telemetry_queue: Queue = Queue(maxsize=_TELEMETRY_QUEUE_MAXSIZE)

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
        self._start_time = time.time()
        self._signals_dropped_total: int = 0
        self._metrics_dropped_total: int = 0
        self._heartbeats_dropped_total: int = 0

        # Bars must never be dropped — this counter must stay zero
        self._bars_dropped_total: int = 0
        self._bar_flush_success_total: int = 0
        self._bar_flush_failure_total: int = 0
        self._bar_retry_count: int = 0

        # Lag tracking (Stream 2.1)
        self._last_persist_lag_ms: Optional[float] = None
        self._persist_lag_ema: Optional[float] = None

        # Subscribe to relevant topics
        bus.subscribe(TOPIC_MARKET_BARS, self._on_bar)
        bus.subscribe(TOPIC_SIGNALS, self._on_signal)
        bus.subscribe(TOPIC_SYSTEM_HEARTBEAT, self._on_heartbeat)
        bus.subscribe(TOPIC_MARKET_METRICS, self._on_metric)
        bus.subscribe(TOPIC_SYSTEM_COMPONENT_HEARTBEAT, self._on_component_heartbeat)

        logger.info("PersistenceManager initialised")

    # ── lifecycle ───────────────────────────────────────────

    def start(self) -> None:
        """Start the background flush and write threads."""
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="persistence-flush",
            daemon=True,
        )
        self._flush_thread.start()
        self._signal_write_thread = threading.Thread(
            target=self._write_loop,
            args=(self._signal_queue, "signal"),
            name="persistence-signal-writes",
            daemon=True,
        )
        self._signal_write_thread.start()
        self._telemetry_write_thread = threading.Thread(
            target=self._write_loop,
            args=(self._telemetry_queue, "telemetry"),
            name="persistence-telemetry-writes",
            daemon=True,
        )
        self._telemetry_write_thread.start()
        logger.info("PersistenceManager flush + write threads started")

    def shutdown(self) -> None:
        """Flush all remaining bars and stop the flush thread.

        Called on SIGINT / Ctrl-C from the orchestrator.
        """
        self._running = False
        # Final flush
        self._do_flush()
        for q in (self._signal_queue, self._telemetry_queue):
            try:
                q.put_nowait(_WRITE_STOP)
            except Exception:
                pass
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)
        if self._signal_write_thread and self._signal_write_thread.is_alive():
            self._signal_write_thread.join(timeout=5.0)
        if self._telemetry_write_thread and self._telemetry_write_thread.is_alive():
            self._telemetry_write_thread.join(timeout=5.0)
        logger.info("PersistenceManager shut down (all bars flushed)")

    # ── handlers (run on bus worker threads) ────────────────

    def _on_bar(self, event: BarEvent) -> None:
        """Buffer bar for batched write.  O(1), never drops."""
        with self._bar_lock:
            self._bar_buffer.append(event)

    def _on_signal(self, event: SignalEvent) -> None:
        """Persist signal via medium-priority queue."""
        try:
            self._signal_queue.put_nowait(("signal", event))
        except Full:
            with self._status_lock:
                self._signals_dropped_total += 1
            logger.warning("Signal queue full — dropping signal event")

    def _on_heartbeat(self, event: HeartbeatEvent) -> None:
        """Flush buffered bars on heartbeat boundary."""
        self._do_flush()

    def _on_metric(self, event: MetricEvent) -> None:
        """Persist market metric via low-priority queue."""
        try:
            self._telemetry_queue.put_nowait(("metric", event))
        except Full:
            with self._status_lock:
                self._metrics_dropped_total += 1
            logger.debug("Telemetry queue full — dropping metric event")

    def _on_component_heartbeat(self, event: ComponentHeartbeatEvent) -> None:
        """Persist component heartbeat via low-priority queue."""
        try:
            self._telemetry_queue.put_nowait(("heartbeat", event))
        except Full:
            with self._status_lock:
                self._heartbeats_dropped_total += 1
            logger.debug("Telemetry queue full — dropping heartbeat event")

    # ── flush logic ─────────────────────────────────────────

    def _flush_loop(self) -> None:
        """Background thread: flush bar buffer every _FLUSH_INTERVAL seconds."""
        while self._running:
            time.sleep(_FLUSH_INTERVAL)
            self._do_flush()

    def _write_loop(self, queue: Queue, label: str) -> None:
        """Background thread: serialize async DB writes for one priority queue."""
        while self._running or not queue.empty():
            try:
                item = queue.get(timeout=0.5)
            except Empty:
                continue
            if item is _WRITE_STOP:
                return
            kind, event = item
            try:
                if kind == "signal":
                    future = asyncio.run_coroutine_threadsafe(
                        self._write_signal(event), self._loop
                    )
                elif kind == "metric":
                    future = asyncio.run_coroutine_threadsafe(
                        self._write_metric(event), self._loop
                    )
                else:
                    future = asyncio.run_coroutine_threadsafe(
                        self._write_component_heartbeat(event), self._loop
                    )
                future.result(timeout=10.0)
            except Exception:
                with self._status_lock:
                    self._db_write_errors += 1
                    self._consecutive_failures += 1
                    self._last_error = f"{kind}_write_failed"
                logger.exception("Failed to persist %s event via %s queue", kind, label)

    def _do_flush(self) -> None:
        """Drain the bar buffer and write to DB.

        On failure, bars are returned to the buffer for retry so that
        bars are **never silently dropped**.
        """
        with self._bar_lock:
            if not self._bar_buffer:
                return
            batch = list(self._bar_buffer)
            self._bar_buffer.clear()

        if not batch:
            return

        start = time.perf_counter()
        success = False
        for attempt in range(_BAR_FLUSH_MAX_RETRIES):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._write_bars(batch), self._loop
                )
                future.result(timeout=30.0)
                success = True
                break
            except Exception:
                logger.warning(
                    "Bar flush attempt %d/%d failed for %d bars",
                    attempt + 1, _BAR_FLUSH_MAX_RETRIES, len(batch),
                    exc_info=True,
                )
                if attempt < _BAR_FLUSH_MAX_RETRIES - 1:
                    with self._status_lock:
                        self._bar_retry_count += 1
                    time.sleep(0.5 * (attempt + 1))  # brief backoff

        if success:
            duration_ms = (time.perf_counter() - start) * 1000
            now = time.time()

            # Compute persist_lag_ms from source timestamps
            source_ts_values = [
                b.source_ts for b in batch
                if hasattr(b, 'source_ts') and b.source_ts > 0
            ]
            if source_ts_values:
                avg_source_ts = sum(source_ts_values) / len(source_ts_values)
                persist_lag = (now - avg_source_ts) * 1000
            else:
                persist_lag = None

            with self._status_lock:
                self._last_flush_ts = now
                self._last_flush_ms = duration_ms
                self._flush_latency_ema = (
                    duration_ms if self._flush_latency_ema is None
                    else (duration_ms * 0.2) + (self._flush_latency_ema * 0.8)
                )
                self._last_write_ts = now
                self._bar_flush_success_total += 1
                self._consecutive_failures = 0
                if persist_lag is not None:
                    self._last_persist_lag_ms = persist_lag
                    self._persist_lag_ema = (
                        persist_lag if self._persist_lag_ema is None
                        else (persist_lag * 0.2) + (self._persist_lag_ema * 0.8)
                    )
        else:
            # All retries failed — return bars to buffer (never drop)
            with self._bar_lock:
                self._bar_buffer = batch + self._bar_buffer
            with self._status_lock:
                self._db_write_errors += 1
                self._consecutive_failures += 1
                self._bar_flush_failure_total += 1
                self._last_error = "bar_flush_failed"
            logger.error(
                "All %d bar flush retries exhausted for %d bars — "
                "returned to buffer for next cycle",
                _BAR_FLUSH_MAX_RETRIES, len(batch),
            )

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
                getattr(b, 'n_ticks', b.tick_count),
                getattr(b, 'first_source_ts', None),
                getattr(b, 'last_source_ts', None),
                getattr(b, 'late_ticks_dropped', 0),
                getattr(b, 'close_reason', 0),
            )
            for b in bars
            if b.bar_duration == 60  # only 1m bars
        ]
        if not rows:
            return
        await self._db.execute_many(
            """INSERT OR IGNORE INTO market_bars
               (timestamp, symbol, source, open, high, low, close, volume,
                tick_count, n_ticks, first_source_ts, last_source_ts,
                late_ticks_dropped, close_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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

    async def _write_component_heartbeat(self, event: ComponentHeartbeatEvent) -> None:
        """Write a structured component heartbeat to the DB."""
        import json
        ts_iso = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat()
        extra_json = json.dumps(event.extra) if event.extra else None
        await self._db.execute(
            """INSERT INTO component_heartbeats
               (timestamp, component, uptime_seconds, events_processed,
                latest_lag_ms, health, extra_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                ts_iso,
                event.component,
                event.uptime_seconds,
                event.events_processed,
                event.latest_lag_ms,
                event.health,
                extra_json,
            ),
        )

    def emit_heartbeat(self) -> ComponentHeartbeatEvent:
        """Create and publish a structured heartbeat for PersistenceManager."""
        now = time.time()
        with self._status_lock:
            total_writes = self._bars_writes_total + self._metrics_writes_total + self._signals_writes_total
            lag_ms = self._last_persist_lag_ms

        health = "ok"
        if self._consecutive_failures > 0:
            health = "degraded"
        if self._db_write_errors > 5:
            health = "down"

        hb = ComponentHeartbeatEvent(
            component="persistence",
            uptime_seconds=round(now - self._start_time, 1),
            events_processed=total_writes,
            latest_lag_ms=round(lag_ms, 1) if lag_ms is not None else None,
            health=health,
            extra={
                "bars_writes_total": self._bars_writes_total,
                "bars_dropped_total": self._bars_dropped_total,
                "bar_flush_success_total": self._bar_flush_success_total,
                "bar_flush_failure_total": self._bar_flush_failure_total,
                "bar_retry_count": self._bar_retry_count,
                "metrics_writes_total": self._metrics_writes_total,
                "signals_writes_total": self._signals_writes_total,
                "signals_dropped_total": self._signals_dropped_total,
                "metrics_dropped_total": self._metrics_dropped_total,
                "heartbeats_dropped_total": self._heartbeats_dropped_total,
                "signal_queue_depth": self._signal_queue.qsize(),
                "telemetry_queue_depth": self._telemetry_queue.qsize(),
                "persist_lag_ema_ms": round(self._persist_lag_ema, 1) if self._persist_lag_ema else None,
            },
        )
        self._bus.publish(TOPIC_SYSTEM_COMPONENT_HEARTBEAT, hb)
        return hb

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
            persist_lag_ms = self._last_persist_lag_ms
            persist_lag_ema = self._persist_lag_ema
            signals_dropped_total = self._signals_dropped_total
            metrics_dropped_total = self._metrics_dropped_total
            heartbeats_dropped_total = self._heartbeats_dropped_total
            bars_dropped_total = self._bars_dropped_total
            bar_flush_success = self._bar_flush_success_total
            bar_flush_failure = self._bar_flush_failure_total
            bar_retry_count = self._bar_retry_count

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
                "signals_dropped_total": signals_dropped_total,
                "metrics_dropped_total": metrics_dropped_total,
                "heartbeats_dropped_total": heartbeats_dropped_total,
                "bars_dropped_total": bars_dropped_total,
                "bar_flush_success_total": bar_flush_success,
                "bar_flush_failure_total": bar_flush_failure,
                "bar_retry_count": bar_retry_count,
                "signal_queue_depth": self._signal_queue.qsize(),
                "telemetry_queue_depth": self._telemetry_queue.qsize(),
                "write_queue_depth": self._signal_queue.qsize() + self._telemetry_queue.qsize(),
                "persist_lag_ms": round(persist_lag_ms, 1) if persist_lag_ms is not None else None,
                "persist_lag_ema_ms": round(persist_lag_ema, 1) if persist_lag_ema is not None else None,
            },
        )
