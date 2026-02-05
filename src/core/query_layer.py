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
        bar_builder: Optional[Any] = None,
        persistence: Optional[Any] = None,
        provider_names: Optional[List[str]] = None,
    ) -> None:
        self._bus = bus
        self._db = db
        self._detectors = detectors or {}
        self._connectors = connectors or {}
        self._bar_builder = bar_builder
        self._persistence = persistence
        self._provider_names = provider_names or [
            "bybit",
            "deribit",
            "yahoo",
            "binance",
            "okx",
            "coinglass",
            "coinbase",
            "ibit_options",
        ]

    # ── /status ─────────────────────────────────────────────

    async def status(self) -> Dict[str, Any]:
        """Health and lag for every component.

        Returns a dict with:
        - ``bus``: per-topic queue depth and publish/process stats
        - ``providers``: per-connector health snapshot
        - ``internal``: BarBuilder/Persistence status
        - ``db``: connection status and size
        """
        bus_stats = self._bus.get_status_summary()

        connector_health: Dict[str, Any] = {}
        provider_types = {
            "bybit": "ws",
            "binance": "ws",
            "deribit": "rest",
            "yahoo": "rest",
            "okx": "rest",
            "coinglass": "rest",
            "coinbase": "rest",
            "ibit_options": "batch",
        }
        from .status import build_status
        provider_names = list(dict.fromkeys(self._provider_names + list(self._connectors.keys())))
        for name in provider_names:
            conn = self._connectors.get(name)
            health: Dict[str, Any]
            if conn is None:
                health = build_status(
                    name=name,
                    type=provider_types.get(name, "rest"),
                    status="unknown",
                    last_error="not_configured",
                    extras={"configured": False},
                )
            elif hasattr(conn, "get_health_status"):
                try:
                    health = conn.get_health_status()
                except Exception as exc:
                    health = build_status(
                        name=name,
                        type=provider_types.get(name, "rest"),
                        status="unknown",
                        last_error=str(exc),
                    )
            elif hasattr(conn, "get_health"):
                try:
                    health = conn.get_health()
                except Exception as exc:
                    health = build_status(
                        name=name,
                        type=provider_types.get(name, "rest"),
                        status="unknown",
                        last_error=str(exc),
                    )
            elif hasattr(conn, "is_connected"):
                health = build_status(
                    name=name,
                    type=provider_types.get(name, "rest"),
                    status="ok" if conn.is_connected else "down",
                )
            else:
                health = build_status(
                    name=name,
                    type=provider_types.get(name, "rest"),
                    status="unknown",
                )
            connector_health[name] = health

        internal_status: Dict[str, Any] = {}
        if self._bar_builder and hasattr(self._bar_builder, "get_status"):
            internal_status["bar_builder"] = self._bar_builder.get_status()
        if self._persistence and hasattr(self._persistence, "get_status"):
            internal_status["persistence"] = self._persistence.get_status()

        last_write_ts = None
        if self._persistence and hasattr(self._persistence, "get_status"):
            last_write_ts = internal_status.get("persistence", {}).get("last_success_ts")

        bar_continuity: Dict[str, Any] = {}
        if "bar_builder" in internal_status:
            bar_builder_status = internal_status["bar_builder"]
            extras = bar_builder_status.get("extras", {})
            last_bar_ts_epoch = extras.get("last_bar_ts_by_symbol_epoch", {}) or {}
            recent_counts = extras.get("bars_emitted_recent_by_symbol", {}) or {}
            window_minutes = extras.get("bars_emitted_recent_window_minutes", 60)
            stale_threshold_seconds = 300
            now = time.time()
            running = self._bus.is_running() if hasattr(self._bus, "is_running") else True
            for symbol, ts in last_bar_ts_epoch.items():
                if ts is None:
                    continue
                bar_age_seconds = round(now - ts, 1)
                expected_minutes = window_minutes
                observed_minutes = recent_counts.get(symbol, 0)
                missing_estimate = max(expected_minutes - observed_minutes, 0) if expected_minutes else None
                if not running:
                    state = "stopped"
                elif bar_age_seconds > stale_threshold_seconds:
                    state = "stale"
                else:
                    state = "fresh"
                bar_continuity[symbol] = {
                    "bar_age_seconds": bar_age_seconds,
                    "missing_bar_estimate": missing_estimate,
                    "status": state,
                }
            bar_builder_status["continuity"] = {
                "running": running,
                "stale_threshold_seconds": stale_threshold_seconds,
                "window_minutes": window_minutes,
                "symbols": bar_continuity,
            }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bus": {
                "stats": bus_stats,
            },
            "providers": connector_health,
            "internal": internal_status,
            "db": {
                "connected": self._db._connection is not None,
                "db_size_mb": self._db.get_size_mb(),
                "last_write_ts": last_write_ts,
            },
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
            "market_metrics",
        ]
        latest = await self._db.get_latest_timestamps(tables_to_check)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "size_mb": stats.get("db_size_mb"),
            "row_counts": stats.get("row_counts"),
            "latest_timestamps": latest,
        }
