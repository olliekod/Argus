"""
Tape Recorder — Optional Determinism Artifact Capture
=====================================================

Captures a bounded rolling window of QuoteEvents and MinuteTickEvents
for a configurable subset of symbols.  The tape is replayable through
the BarBuilder for determinism proof.

Disabled by default; enabled via ``soak.tape.enabled: true`` in config.

Storage is bounded:
* In-memory deque with configurable maxlen (default 100K events).
* Export to JSONL file for offline replay.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from ..core.events import (
    MinuteTickEvent,
    QuoteEvent,
    TOPIC_MARKET_QUOTES,
    TOPIC_SYSTEM_MINUTE_TICK,
)

logger = logging.getLogger("argus.soak.tape")


def _quote_to_dict(q: QuoteEvent) -> Dict[str, Any]:
    """Serialize a QuoteEvent to a replayable dict."""
    return {
        "type": "quote",
        "symbol": q.symbol,
        "bid": q.bid,
        "ask": q.ask,
        "mid": q.mid,
        "last": q.last,
        "timestamp": q.timestamp,
        "source": q.source,
        "volume_24h": q.volume_24h,
        "source_ts": q.source_ts,
        "event_ts": q.event_ts,
        "receive_time": q.receive_time,
    }


def _tick_to_dict(t: MinuteTickEvent) -> Dict[str, Any]:
    """Serialize a MinuteTickEvent to a replayable dict."""
    return {
        "type": "minute_tick",
        "timestamp": t.timestamp,
    }


def _dict_to_quote(d: Dict[str, Any]) -> QuoteEvent:
    """Deserialize a dict back to a QuoteEvent."""
    return QuoteEvent(
        symbol=d["symbol"],
        bid=d["bid"],
        ask=d["ask"],
        mid=d["mid"],
        last=d["last"],
        timestamp=d["timestamp"],
        source=d["source"],
        volume_24h=d.get("volume_24h", 0.0),
        source_ts=d.get("source_ts", 0.0),
        event_ts=d.get("event_ts", d["timestamp"]),
        receive_time=d.get("receive_time", d["timestamp"]),
    )


def _dict_to_event(d: Dict[str, Any]):
    """Deserialize a dict to the appropriate event type."""
    if d["type"] == "quote":
        return _dict_to_quote(d)
    elif d["type"] == "minute_tick":
        return MinuteTickEvent(timestamp=d["timestamp"])
    raise ValueError(f"Unknown event type: {d['type']}")


class TapeRecorder:
    """Bounded rolling tape capture for determinism proof.

    Parameters
    ----------
    enabled : bool
        If False, all operations are no-ops.
    symbols : set of str or None
        Symbols to capture.  None = capture all.
    maxlen : int
        Maximum events in the rolling buffer.
    """

    def __init__(
        self,
        enabled: bool = False,
        symbols: Optional[Set[str]] = None,
        maxlen: int = 100_000,
    ) -> None:
        self._enabled = enabled
        self._symbols = symbols
        self._maxlen = maxlen
        self._tape: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._events_captured: int = 0
        self._events_evicted: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def attach(self, bus) -> None:
        """Subscribe to event bus topics if enabled."""
        if not self._enabled:
            logger.info("TapeRecorder disabled — not subscribing")
            return
        bus.subscribe(TOPIC_MARKET_QUOTES, self._on_quote)
        bus.subscribe(TOPIC_SYSTEM_MINUTE_TICK, self._on_minute_tick)
        logger.info(
            "TapeRecorder attached (maxlen=%d, symbols=%s)",
            self._maxlen,
            self._symbols or "ALL",
        )

    def _on_quote(self, event: QuoteEvent) -> None:
        if not self._enabled:
            return
        if self._symbols and event.symbol not in self._symbols:
            return
        d = _quote_to_dict(event)
        with self._lock:
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def _on_minute_tick(self, event: MinuteTickEvent) -> None:
        if not self._enabled:
            return
        d = _tick_to_dict(event)
        with self._lock:
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def get_status(self) -> Dict[str, Any]:
        """Status snapshot for soak summary."""
        with self._lock:
            size = len(self._tape)
        return {
            "enabled": self._enabled,
            "tape_size": size,
            "maxlen": self._maxlen,
            "events_captured": self._events_captured,
            "events_evicted": self._events_evicted,
            "symbols": sorted(self._symbols) if self._symbols else None,
        }

    def export_jsonl(self, path: str, last_n_minutes: Optional[int] = None) -> int:
        """Export tape to a JSONL file.

        Parameters
        ----------
        path : str
            Output file path.
        last_n_minutes : int or None
            If set, only export events from the last N minutes.

        Returns
        -------
        int
            Number of events written.
        """
        if not self._enabled:
            logger.warning("TapeRecorder is disabled — nothing to export")
            return 0

        now = time.time()
        cutoff = now - (last_n_minutes * 60) if last_n_minutes else 0

        with self._lock:
            snapshot = list(self._tape)

        count = 0
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for entry in snapshot:
                ts = entry.get("timestamp", 0)
                if ts >= cutoff:
                    f.write(json.dumps(entry) + "\n")
                    count += 1
        logger.info("Exported %d events to %s", count, path)
        return count

    @staticmethod
    def load_tape(path: str) -> List[Dict[str, Any]]:
        """Load a JSONL tape file."""
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    @staticmethod
    def replay_tape(tape: List[Dict[str, Any]]) -> List:
        """Replay a tape through a fresh BarBuilder and return emitted bars.

        Reuses the same BarBuilder + EventBus machinery as the replay harness.
        """
        from ..core.bar_builder import BarBuilder
        from ..core.bus import EventBus
        from ..core.events import BarEvent, TOPIC_MARKET_BARS

        bus = EventBus()
        emitted: list = []
        bus.subscribe(TOPIC_MARKET_BARS, lambda bar: emitted.append(bar))
        bb = BarBuilder(bus)
        bus.start()

        try:
            for entry in tape:
                event = _dict_to_event(entry)
                if isinstance(event, QuoteEvent):
                    bb._on_quote(event)
                elif isinstance(event, MinuteTickEvent):
                    bb._on_minute_tick(event)

            flushed = bb.flush()
            # Drain bus
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                depths = bus.get_queue_depths()
                if all(d == 0 for d in depths.values()):
                    break
                time.sleep(0.01)
        finally:
            bus.stop()

        return emitted + flushed
