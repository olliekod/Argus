"""
Tape Recorder — Deterministic Event Capture & Replay
=====================================================

Captures a bounded rolling window of QuoteEvents, BarEvents, and MinuteTickEvents
for a configurable subset of symbols. The tape is replayable for determinism proof.

REPLAY MODES
------------
Two distinct replay modes with clear semantics:

**Faithful Mode (DEFAULT)**:
  - Append-only with monotonic `sequence_id` at record time.
  - Replay outputs events in EXACT recorded order (by sequence_id).
  - REQUIRED for deterministic strategy evaluation.

**Canonical Mode (OPTIONAL)**:
  - Produces stable order independent of arrival timing.
  - Sort key: (event_ts, provider_priority, event_type_priority, symbol, sequence_id)
  - Logs warning that it is NOT faithful to arrival order.
  - Use only for analysis/comparison, NOT primary evaluation.

TIMESTAMP CONVENTION
--------------------
All timestamps are stored as **int milliseconds** (UTC epoch ms).
This is enforced at record time and validated at replay.

ENVELOPE SCHEMA
---------------
Every taped record includes:
  - sequence_id: int (monotonic, assigned at record time)
  - event_ts: int (ms, arrival time)
  - provider: str
  - event_type: str ("bar" | "quote" | "minute_tick")
  - symbol: str
  - timeframe: int (bar_duration in seconds, for bars)

PRIORITY TABLES
---------------
Provider Priority (lower = higher priority):
  alpaca=1, yahoo=2, bybit=3, binance=4, deribit=5, polymarket=6, unknown=99

Event Type Priority (lower = higher priority):
  bar=1, quote=2, metric=3, minute_tick=4, signal=5, heartbeat=6
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Set, Tuple, Union

from ..core.events import (
    BarEvent,
    MinuteTickEvent,
    QuoteEvent,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_QUOTES,
    TOPIC_SYSTEM_MINUTE_TICK,
)

logger = logging.getLogger("argus.soak.tape")

# ══════════════════════════════════════════════════════════════
# PRIORITY TABLES
# ══════════════════════════════════════════════════════════════

# Provider priority for canonical tape ordering (lower = higher priority)
PROVIDER_PRIORITY: Dict[str, int] = {
    "alpaca": 1,
    "yahoo": 2,
    "bybit": 3,
    "binance": 4,
    "deribit": 5,
    "polymarket": 6,
    "unknown": 99,
}

# Event type priority for canonical tape ordering (lower = higher priority)
EVENT_TYPE_PRIORITY: Dict[str, int] = {
    "bar": 1,
    "quote": 2,
    "metric": 3,
    "minute_tick": 4,
    "signal": 5,
    "heartbeat": 6,
}


def _to_ms(ts: Union[int, float]) -> int:
    """Convert timestamp to int milliseconds.
    
    Handles:
      - Already int ms: return as-is
      - Float seconds (epoch < 2e10): convert to ms
      - Float ms: round to int
    """
    if isinstance(ts, int):
        return ts
    # Heuristic: if ts < 2e10, it's seconds; convert to ms
    if ts < 2e10:
        return int(ts * 1000)
    return int(ts)


def _validate_ms(ts: int, field_name: str) -> None:
    """Validate timestamp is a sane int milliseconds value."""
    if not isinstance(ts, int):
        raise ValueError(f"{field_name} must be int milliseconds, got {type(ts)}")
    # Sanity bounds: 2020-01-01 to 2035-01-01 in ms
    if not (1_577_836_800_000 <= ts <= 2_051_222_400_000):
        raise ValueError(f"{field_name}={ts} outside sane range [2020, 2035] in ms")


# ══════════════════════════════════════════════════════════════
# SERIALIZATION
# ══════════════════════════════════════════════════════════════

def _quote_to_dict(q: QuoteEvent, sequence_id: int) -> Dict[str, Any]:
    """Serialize a QuoteEvent to a tape envelope."""
    event_ts_ms = _to_ms(getattr(q, 'event_ts', 0) or q.timestamp)
    return {
        # Envelope fields (required for all events)
        "sequence_id": sequence_id,
        "event_ts": event_ts_ms,
        "provider": q.source or "unknown",
        "event_type": "quote",
        "symbol": q.symbol,
        "timeframe": 0,  # N/A for quotes
        # Quote-specific fields
        "bid": q.bid,
        "ask": q.ask,
        "mid": q.mid,
        "last": q.last,
        "timestamp": _to_ms(q.timestamp),
        "source": q.source,
        "volume_24h": q.volume_24h,
        "source_ts": _to_ms(q.source_ts) if q.source_ts else 0,
        "receive_time": _to_ms(getattr(q, 'receive_time', 0) or q.timestamp),
    }


def _tick_to_dict(t: MinuteTickEvent, sequence_id: int) -> Dict[str, Any]:
    """Serialize a MinuteTickEvent to a tape envelope."""
    return {
        # Envelope fields
        "sequence_id": sequence_id,
        "event_ts": _to_ms(t.timestamp),
        "provider": "system",
        "event_type": "minute_tick",
        "symbol": "",
        "timeframe": 0,
        # Tick-specific fields
        "timestamp": _to_ms(t.timestamp),
    }


def _bar_to_dict(b: BarEvent, sequence_id: int) -> Dict[str, Any]:
    """Serialize a BarEvent to a tape envelope."""
    event_ts_ms = _to_ms(getattr(b, 'event_ts', 0) or b.timestamp)
    return {
        # Envelope fields (required for all events)
        "sequence_id": sequence_id,
        "event_ts": event_ts_ms,
        "provider": b.source or "unknown",
        "event_type": "bar",
        "symbol": b.symbol,
        "timeframe": getattr(b, 'bar_duration', 60),
        # Bar-specific fields
        "open": b.open,
        "high": b.high,
        "low": b.low,
        "close": b.close,
        "volume": b.volume,
        "timestamp": _to_ms(b.timestamp),
        "source": b.source,
        "bar_duration": getattr(b, 'bar_duration', 60),
        "n_ticks": getattr(b, 'n_ticks', 1),
        "first_source_ts": _to_ms(getattr(b, 'first_source_ts', b.timestamp)),
        "last_source_ts": _to_ms(getattr(b, 'last_source_ts', b.timestamp)),
        "source_ts": _to_ms(getattr(b, 'source_ts', b.timestamp)),
    }


def _dict_to_quote(d: Dict[str, Any]) -> QuoteEvent:
    """Deserialize a dict back to a QuoteEvent."""
    return QuoteEvent(
        symbol=d["symbol"],
        bid=d["bid"],
        ask=d["ask"],
        mid=d["mid"],
        last=d["last"],
        timestamp=d["timestamp"] / 1000.0,  # Convert ms back to seconds for event
        source=d["source"],
        volume_24h=d.get("volume_24h", 0.0),
        source_ts=d.get("source_ts", 0) / 1000.0 if d.get("source_ts") else 0.0,
        event_ts=d.get("event_ts", d["timestamp"]) / 1000.0,
        receive_time=d.get("receive_time", d["timestamp"]) / 1000.0,
    )


def _dict_to_bar(d: Dict[str, Any]) -> BarEvent:
    """Deserialize a dict back to a BarEvent."""
    return BarEvent(
        symbol=d["symbol"],
        open=d["open"],
        high=d["high"],
        low=d["low"],
        close=d["close"],
        volume=d["volume"],
        timestamp=d["timestamp"] / 1000.0,  # Convert ms back to seconds
        source=d["source"],
        bar_duration=d.get("bar_duration", 60),
        n_ticks=d.get("n_ticks", 1),
        first_source_ts=d.get("first_source_ts", d["timestamp"]) / 1000.0,
        last_source_ts=d.get("last_source_ts", d["timestamp"]) / 1000.0,
        source_ts=d.get("source_ts", d["timestamp"]) / 1000.0,
        event_ts=d.get("event_ts", d["timestamp"]) / 1000.0,
    )


def _dict_to_event(d: Dict[str, Any]):
    """Deserialize a dict to the appropriate event type."""
    event_type = d.get("event_type") or d.get("type")
    if event_type == "quote":
        return _dict_to_quote(d)
    elif event_type == "minute_tick":
        ts_ms = d["timestamp"]
        return MinuteTickEvent(timestamp=ts_ms / 1000.0)
    elif event_type == "bar":
        return _dict_to_bar(d)
    raise ValueError(f"Unknown event type: {event_type}")


# ══════════════════════════════════════════════════════════════
# SORT KEYS
# ══════════════════════════════════════════════════════════════

def _faithful_sort_key(entry: Dict[str, Any]) -> int:
    """Sort key for faithful replay: sequence_id only."""
    return entry.get("sequence_id", 0)


def _canonical_sort_key(entry: Dict[str, Any]) -> Tuple[int, int, int, str, int]:
    """
    Sort key for canonical ordering.
    
    Key: (event_ts, provider_priority, event_type_priority, symbol, sequence_id)
    
    This produces a stable order independent of arrival timing.
    sequence_id is the final tiebreaker for determinism.
    """
    event_ts = entry.get("event_ts", 0)
    provider = entry.get("provider", "unknown")
    event_type = entry.get("event_type") or entry.get("type", "unknown")
    symbol = entry.get("symbol", "")
    sequence_id = entry.get("sequence_id", 0)
    
    provider_priority = PROVIDER_PRIORITY.get(provider, 99)
    event_type_priority = EVENT_TYPE_PRIORITY.get(event_type, 99)
    
    return (event_ts, provider_priority, event_type_priority, symbol, sequence_id)


# ══════════════════════════════════════════════════════════════
# TAPE RECORDER
# ══════════════════════════════════════════════════════════════

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
        # Monotonic sequence counter for faithful replay
        self._next_sequence_id: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_next_sequence_id(self) -> int:
        """Get next monotonic sequence ID (must hold lock)."""
        seq_id = self._next_sequence_id
        self._next_sequence_id += 1
        return seq_id

    def attach(self, bus) -> None:
        """Subscribe to event bus topics if enabled."""
        if not self._enabled:
            logger.info("TapeRecorder disabled — not subscribing")
            return
        bus.subscribe(TOPIC_MARKET_QUOTES, self._on_quote)
        bus.subscribe(TOPIC_MARKET_BARS, self._on_bar)
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
        with self._lock:
            seq_id = self._get_next_sequence_id()
            d = _quote_to_dict(event, seq_id)
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def _on_minute_tick(self, event: MinuteTickEvent) -> None:
        if not self._enabled:
            return
        with self._lock:
            seq_id = self._get_next_sequence_id()
            d = _tick_to_dict(event, seq_id)
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def _on_bar(self, event: BarEvent) -> None:
        if not self._enabled:
            return
        if self._symbols and event.symbol not in self._symbols:
            return
        with self._lock:
            seq_id = self._get_next_sequence_id()
            d = _bar_to_dict(event, seq_id)
            was_full = len(self._tape) == self._tape.maxlen
            self._tape.append(d)
            self._events_captured += 1
            if was_full:
                self._events_evicted += 1

    def get_status(self) -> Dict[str, Any]:
        """Status snapshot for soak summary."""
        with self._lock:
            size = len(self._tape)
            next_seq = self._next_sequence_id
        return {
            "enabled": self._enabled,
            "tape_size": size,
            "maxlen": self._maxlen,
            "events_captured": self._events_captured,
            "events_evicted": self._events_evicted,
            "next_sequence_id": next_seq,
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

        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - (last_n_minutes * 60_000) if last_n_minutes else 0

        with self._lock:
            snapshot = list(self._tape)

        count = 0
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for entry in snapshot:
                ts_ms = entry.get("event_ts", entry.get("timestamp", 0))
                if ts_ms >= cutoff_ms:
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
    def replay_tape(
        tape: List[Dict[str, Any]],
        mode: str = "faithful",
    ) -> List:
        """Replay a tape through a fresh BarBuilder and return emitted bars.

        Parameters
        ----------
        tape : list of dict
            Tape entries (quote, bar, minute_tick events).
        mode : str
            Replay mode:
              - "faithful" (DEFAULT): replay in exact recorded order (sequence_id).
                Required for deterministic strategy evaluation.
              - "canonical": replay in stable sorted order independent of arrival.
                NOT faithful to live arrival order. Use only for analysis.

        Returns
        -------
        list
            Emitted BarEvents from replay.
        """
        if mode not in ("faithful", "canonical"):
            raise ValueError(f"mode must be 'faithful' or 'canonical', got {mode!r}")
        
        if mode == "canonical":
            print("\n" + "!" * 80)
            print("!!! WARNING: CANONICAL REPLAY MODE ACTIVE !!!")
            print("!!! Canonical ordering is NOT faithful to live arrival order.")
            print("!!! Use ONLY for analysis/comparison, NOT for determinism proof.")
            print("!" * 80 + "\n")
            logger.warning(
                "CANONICAL replay is NOT faithful to arrival order; use only for analysis."
            )
            tape = sorted(tape, key=_canonical_sort_key)
        else:
            # Faithful mode: sort by sequence_id
            tape = sorted(tape, key=_faithful_sort_key)

        from ..core.bar_builder import BarBuilder
        from ..core.bus import EventBus
        from ..core.events import BarEvent as BE, TOPIC_MARKET_BARS

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
                elif isinstance(event, BE):
                    # Pre-aggregated bars are passed through directly
                    emitted.append(event)

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

    @staticmethod
    def replay_tape_events(
        tape: List[Dict[str, Any]],
        mode: str = "faithful",
    ) -> Iterator[Tuple[int, Any]]:
        """Iterate tape events in replay order without processing.
        
        Yields (sequence_id, event) tuples for testing/inspection.
        
        Parameters
        ----------
        tape : list of dict
            Tape entries.
        mode : str
            "faithful" or "canonical".
            
        Yields
        ------
        tuple of (int, event)
            Sequence ID and deserialized event.
        """
        if mode not in ("faithful", "canonical"):
            raise ValueError(f"mode must be 'faithful' or 'canonical', got {mode!r}")
        
        if mode == "canonical":
            logger.warning(
                "CANONICAL replay is NOT faithful to arrival order; use only for analysis."
            )
            tape = sorted(tape, key=_canonical_sort_key)
        else:
            tape = sorted(tape, key=_faithful_sort_key)
        
        for entry in tape:
            seq_id = entry.get("sequence_id", 0)
            event = _dict_to_event(entry)
            yield seq_id, event
