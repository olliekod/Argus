"""
Argus Event Types
=================

Dataclass-based events for the Pub/Sub event bus.
All events are immutable after creation.
"""

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional


class Priority(IntEnum):
    """Signal priority / severity levels."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(frozen=True, slots=True)
class QuoteEvent:
    """Real-time price quote from a connector.

    Published to: market.quotes
    Contains price-related fields only.  Non-price metrics
    (IV, funding, open interest) belong in :class:`MetricEvent`.
    """
    symbol: str
    bid: float
    ask: float
    mid: float
    last: float
    timestamp: float          # exchange epoch seconds (UTC)
    source: str               # 'bybit', 'deribit', 'yahoo'
    volume_24h: float = 0.0
    receive_time: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class MetricEvent:
    """Non-price market metric from a connector.

    Published to: market.metrics
    Carries funding rates, open interest, IV, or other
    auxiliary data that should not pollute the price path.
    """
    symbol: str
    metric: str               # 'funding_rate', 'open_interest', 'atm_iv', …
    value: float
    timestamp: float
    source: str
    extra: Dict[str, Any] = field(default_factory=dict)
    receive_time: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class BarEvent:
    """One-minute OHLCV bar aligned to UTC minute boundaries.

    Published to: market.bars
    """
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float          # UTC epoch of the bar open (minute-aligned)
    source: str               # originating connector
    bar_duration: int = 60    # seconds
    tick_count: int = 0       # quotes aggregated
    receive_time: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """Detection / trading signal from a detector.

    Published to: signals.detections
    MUST include priority / severity.
    """
    detector: str             # 'ibit', 'bito', 'options_iv', 'volatility'
    symbol: str               # tradeable instrument (IBIT / BITO only)
    signal_type: str          # 'put_spread', 'iv_spike', 'regime_change'
    priority: Priority
    timestamp: float          # UTC epoch
    data: Dict[str, Any] = field(default_factory=dict)
    receive_time: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class StatusEvent:
    """Component health / status beacon.

    Published to: system.status
    """
    component: str
    status: str               # 'ok', 'degraded', 'error'
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HeartbeatEvent:
    """Periodic heartbeat for flush / bookkeeping.

    Published to: system.heartbeat
    """
    sequence: int = 0
    timestamp: float = field(default_factory=time.time)


# ─── Topic constants ────────────────────────────────────────
TOPIC_MARKET_QUOTES = "market.quotes"
TOPIC_MARKET_BARS = "market.bars"
TOPIC_MARKET_METRICS = "market.metrics"
TOPIC_SIGNALS = "signals.detections"
TOPIC_SYSTEM_STATUS = "system.status"
TOPIC_SYSTEM_HEARTBEAT = "system.heartbeat"
