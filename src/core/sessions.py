"""
Session schedule helpers.

Provides deterministic session classification and timing utilities
based solely on timestamps (no wall clock).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# Session time boundaries (UTC hour)
# Equities
EQUITIES_PRE_START = 9      # 4 AM ET = 9 UTC
EQUITIES_RTH_START = 14     # 9:30 AM ET ≈ 14 UTC (simplified)
EQUITIES_RTH_END = 21       # 4 PM ET = 21 UTC
EQUITIES_POST_END = 1       # 8 PM ET = 1 UTC next day

# Crypto
CRYPTO_ASIA_START = 0       # 00:00 UTC
CRYPTO_ASIA_END = 8         # 08:00 UTC
CRYPTO_EU_START = 8         # 08:00 UTC
CRYPTO_EU_END = 14          # 14:00 UTC
CRYPTO_US_START = 14        # 14:00 UTC
CRYPTO_US_END = 22          # 22:00 UTC


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """Session window bounds in minutes from UTC day start."""

    start_minute: int
    end_minute: int


def _hour_from_ts_ms(ts_ms: int) -> int:
    seconds = ts_ms // 1000
    return (seconds // 3600) % 24


def _minute_of_day_from_ts_ms(ts_ms: int) -> int:
    seconds = ts_ms // 1000
    return (seconds // 60) % (24 * 60)


def get_session_regime(market: str, ts_ms: int) -> str:
    """Determine session regime from timestamp (no wall clock)."""
    hour = _hour_from_ts_ms(ts_ms)

    if market == "EQUITIES":
        if EQUITIES_RTH_START <= hour < EQUITIES_RTH_END:
            return "RTH"
        if EQUITIES_PRE_START <= hour < EQUITIES_RTH_START:
            return "PRE"
        if hour >= EQUITIES_RTH_END or hour < EQUITIES_POST_END:
            return "POST"
        return "CLOSED"

    # CRYPTO
    if CRYPTO_ASIA_START <= hour < CRYPTO_ASIA_END:
        return "ASIA"
    if CRYPTO_EU_START <= hour < CRYPTO_EU_END:
        return "EU"
    if CRYPTO_US_START <= hour < CRYPTO_US_END:
        return "US"
    return "OFFPEAK"


def get_equities_rth_window_minutes() -> SessionWindow:
    """Return RTH window in minutes from UTC day start."""
    return SessionWindow(
        start_minute=EQUITIES_RTH_START * 60,
        end_minute=EQUITIES_RTH_END * 60,
    )


def minutes_from_session_end(ts_ms: int, window: SessionWindow) -> int:
    """
    Minutes from timestamp to session end. Returns negative if after end.
    Assumes window does not cross midnight.
    """
    minute_of_day = _minute_of_day_from_ts_ms(ts_ms)
    return window.end_minute - minute_of_day


def is_within_last_n_minutes(ts_ms: int, window: SessionWindow, n_minutes: int) -> bool:
    """Return True if timestamp is within the last N minutes of the window."""
    if n_minutes <= 0:
        return False
    remaining = minutes_from_session_end(ts_ms, window)
    return 0 <= remaining <= n_minutes

