"""
Argus Query Layer
=================

Unified command interface that pulls latest state from the event bus
and historical data from the database.

Commands
--------
/status  — Health / lag for every component
/market  — Current regime, prices, IV
/signals — Last 10 signal events
/db      — Storage size, retention, row counts
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import logging

from .bus import EventBus
from .events import (
    TOPIC_MARKET_QUOTES,
    TOPIC_MARKET_BARS,
    TOPIC_SIGNALS,
    TOPIC_SYSTEM_STATUS,
    TOPIC_SYSTEM_HEARTBEAT,
)

logger = logging.getLogger("argus.query")


class QueryLayer:
    """Provides a unified read interface over bus state + DB history.

    Parameters
    ----------
    bus : EventBus
        The running event bus (used for live queue depth / stats).
    db : Database
        Argus async database handle.
    detectors : dict
        Name → detector instance mapping (for regime info).
    connectors : dict
        Name → connector instance mapping (for health info).
    """

    def __init__(
        self,
        bus: EventBus,
        db: Any,
        detectors: Optional[Dict[str, Any]] = None,
        connectors: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._bus = bus
        self._db = db
        self._detectors = detectors or {}
        self._connectors = connectors or {}

    # ── /status ─────────────────────────────────────────────

    async def status(self) -> Dict[str, Any]:
        """Health and lag for every component.

        Returns a dict with:
        - ``bus``: per-topic queue depth and publish/process stats
        - ``connectors``: per-connector health snapshot
        - ``db``: connection status
        """
        bus_stats = self._bus.get_stats()
        queue_depths = self._bus.get_queue_depths()

        connector_health: Dict[str, Any] = {}
        for name, conn in self._connectors.items():
            health = {"status": "unknown"}
            if hasattr(conn, "get_health"):
                try:
                    health = conn.get_health()
                except Exception as exc:
                    health = {"status": "error", "error": str(exc)}
            elif hasattr(conn, "is_connected"):
                health = {"status": "ok" if conn.is_connected else "disconnected"}
            connector_health[name] = health

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bus": {
                "queue_depths": queue_depths,
                "stats": bus_stats,
            },
            "connectors": connector_health,
            "db": {"connected": self._db._connection is not None},
        }

    # ── /market ─────────────────────────────────────────────

    async def market(self) -> Dict[str, Any]:
        """Current regime, latest prices, and IV.

        Pulls live price caches from connectors and regime info
        from volatility / conditions detectors.
        """
        prices: Dict[str, Any] = {}

        # Bybit tickers
        bybit = self._connectors.get("bybit")
        if bybit and hasattr(bybit, "tickers"):
            for sym, data in bybit.tickers.items():
                prices[sym] = {
                    "last": data.get("last_price"),
                    "bid": data.get("bid_price"),
                    "ask": data.get("ask_price"),
                    "source": "bybit",
                }

        # Yahoo prices
        yahoo = self._connectors.get("yahoo")
        if yahoo and hasattr(yahoo, "prices"):
            for sym, data in yahoo.prices.items():
                prices[sym] = {
                    "last": data.get("price"),
                    "change_pct": data.get("price_change_pct"),
                    "source": "yahoo",
                }

        # Regime from volatility detector
        regime: Dict[str, str] = {}
        vol_det = self._detectors.get("volatility")
        if vol_det and hasattr(vol_det, "get_current_regime"):
            for sym in ("BTCUSDT", "ETHUSDT", "IBIT", "BITO"):
                r = vol_det.get_current_regime(sym)
                if r and r != "unknown":
                    regime[sym] = r

        # IV from options_iv detector
        iv_info: Dict[str, float] = {}
        iv_det = self._detectors.get("options_iv")
        if iv_det and hasattr(iv_det, "get_current_iv"):
            for cur in ("BTC", "ETH"):
                iv = iv_det.get_current_iv(cur)
                if iv is not None:
                    iv_info[cur] = iv

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prices": prices,
            "regime": regime,
            "iv": iv_info,
        }

    # ── /signals ────────────────────────────────────────────

    async def signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Last *limit* signal events from the database."""
        try:
            rows = await self._db.fetch_all(
                "SELECT * FROM signal_events ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("signals query failed: %s", exc)
            return []

    # ── /db ─────────────────────────────────────────────────

    async def db(self) -> Dict[str, Any]:
        """Storage size, retention policy, and row counts."""
        stats = await self._db.get_db_stats()

        # Latest timestamps per key table
        tables_to_check = [
            "market_bars",
            "signal_events",
            "detections",
            "price_snapshots",
            "system_health",
        ]
        latest = await self._db.get_latest_timestamps(tables_to_check)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "size_mb": stats.get("db_size_mb"),
            "row_counts": stats.get("row_counts"),
            "latest_timestamps": latest,
        }
