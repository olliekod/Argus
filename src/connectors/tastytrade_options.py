"""
Tastytrade Options Snapshot Connector
======================================

REST-based snapshot polling for option chain data via Tastytrade API.
Uses TastytradeRestClient for authentication and nested chain fetching,
then normalizes into OptionChainSnapshotEvents compatible with the
existing schema used by ReplayHarness.

This does NOT replace DXLink streaming — it is for snapshot polling only.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..connectors.tastytrade_rest import (
    RetryConfig,
    TastytradeError,
    TastytradeRestClient,
)
from ..core.options_normalize import normalize_tastytrade_nested_chain
from ..core.option_events import (
    OptionQuoteEvent,
    OptionChainSnapshotEvent,
    compute_snapshot_id,
)

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    """Current time as int milliseconds."""
    return int(time.time() * 1000)


def _date_to_ms(date_str: str) -> int:
    """Convert date string (YYYY-MM-DD) to UTC midnight milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _compute_contract_id(option_symbol: str) -> str:
    """Compute deterministic contract ID from option symbol."""
    return hashlib.sha256(option_symbol.encode()).hexdigest()[:16]


@dataclass
class TastytradeOptionsConfig:
    """Configuration for Tastytrade options snapshot connector."""
    username: str = ""
    password: str = ""
    environment: str = "live"
    timeout_seconds: float = 20.0
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    symbols: Optional[List[str]] = None
    min_dte: int = 7
    max_dte: int = 21
    poll_interval_seconds: int = 60


class TastytradeOptionsConnector:
    """Tastytrade options snapshot connector.

    Fetches nested option chains via REST, normalizes them, and builds
    OptionChainSnapshotEvents compatible with option_chain_snapshots table
    and ReplayHarness.

    Thread-safety: all methods are synchronous (the REST client is sync).
    The orchestrator wraps calls in ``asyncio.to_thread`` or similar.
    """

    PROVIDER = "tastytrade"

    def __init__(self, config: TastytradeOptionsConfig) -> None:
        self._config = config
        self._client: Optional[TastytradeRestClient] = None
        self._sequence_id = 0
        self._authenticated = False

        # Health metrics
        self._request_count = 0
        self._error_count = 0
        self._last_request_ms = 0
        self._last_latency_ms = 0.0

    def _next_sequence_id(self) -> int:
        """Get next monotonic sequence ID."""
        self._sequence_id += 1
        return self._sequence_id

    def _ensure_client(self) -> TastytradeRestClient:
        """Create and authenticate client if needed."""
        if self._client is not None and self._authenticated:
            return self._client

        retry = RetryConfig(
            max_attempts=self._config.max_attempts,
            backoff_seconds=self._config.backoff_seconds,
            backoff_multiplier=self._config.backoff_multiplier,
        )
        self._client = TastytradeRestClient(
            username=self._config.username,
            password=self._config.password,
            environment=self._config.environment,
            timeout_seconds=self._config.timeout_seconds,
            retries=retry,
        )
        try:
            self._client.login()
            self._authenticated = True
            logger.info("Tastytrade options connector authenticated (env=%s)", self._config.environment)
        except TastytradeError as exc:
            self._error_count += 1
            logger.error("Tastytrade login failed: %s", exc)
            self._authenticated = False
            raise

        return self._client

    def fetch_nested_chain(self, symbol: str) -> Dict[str, Any]:
        """Fetch raw nested option chain for a symbol.

        Returns raw API response dict (or empty dict on failure).
        """
        start_ms = _now_ms()
        self._request_count += 1
        try:
            client = self._ensure_client()
            data = client.get_nested_option_chains(symbol)
            self._last_latency_ms = _now_ms() - start_ms
            self._last_request_ms = _now_ms()
            return data
        except TastytradeError as exc:
            self._error_count += 1
            self._last_latency_ms = _now_ms() - start_ms
            logger.error(
                "Tastytrade nested chain fetch failed for %s: %s",
                symbol, exc,
            )
            # Mark as unauthenticated so next call re-logins
            self._authenticated = False
            return {}
        except Exception as exc:
            self._error_count += 1
            logger.error(
                "Unexpected error fetching Tastytrade chain for %s: %s",
                symbol, exc,
            )
            self._authenticated = False
            return {}

    def get_expirations_in_range(
        self,
        normalized: List[Dict[str, Any]],
        min_dte: int = 7,
        max_dte: int = 21,
    ) -> List[Tuple[str, int]]:
        """Filter normalized contracts to expirations within DTE range.

        Args:
            normalized: Output from normalize_tastytrade_nested_chain.
            min_dte: Minimum days to expiration.
            max_dte: Maximum days to expiration.

        Returns:
            List of (expiry_date_str, dte) tuples.
        """
        today = datetime.now(timezone.utc).date()
        expirations = sorted({c["expiry"] for c in normalized if c.get("expiry")})
        results = []
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                results.append((exp_str, dte))
        return results

    def build_chain_snapshot(
        self,
        symbol: str,
        expiration: str,
        normalized: List[Dict[str, Any]],
        underlying_price: float = 0.0,
    ) -> Optional[OptionChainSnapshotEvent]:
        """Build an OptionChainSnapshotEvent from normalized contracts.

        Args:
            symbol: Underlying symbol (e.g. "SPY").
            expiration: Expiration date string (YYYY-MM-DD).
            normalized: Full normalized chain from normalize_tastytrade_nested_chain.
            underlying_price: Current underlying price (0.0 if unavailable).

        Returns:
            OptionChainSnapshotEvent or None if chain is empty.
        """
        now_ms = _now_ms()
        recv_ts_ms = now_ms
        expiration_ms = _date_to_ms(expiration)

        # Filter to this expiration
        exp_contracts = [
            c for c in normalized
            if c.get("expiry") == expiration
        ]

        if not exp_contracts:
            logger.debug(
                "No contracts for %s exp=%s after filtering (provider=%s)",
                symbol, expiration, self.PROVIDER,
            )
            return None

        puts: List[OptionQuoteEvent] = []
        calls: List[OptionQuoteEvent] = []

        for contract in exp_contracts:
            option_symbol = contract.get("option_symbol", "")
            if not option_symbol:
                continue

            strike = contract.get("strike")
            if strike is None:
                continue

            right = contract.get("right", "")
            option_type = "CALL" if right == "C" else "PUT"

            contract_id = _compute_contract_id(str(option_symbol))

            # Tastytrade nested chains don't include live quotes
            # (quotes come from DXLink streaming). We create zero-quote
            # entries so the snapshot structure is valid for replay.
            quote = OptionQuoteEvent(
                contract_id=contract_id,
                symbol=symbol,
                strike=float(strike),
                expiration_ms=expiration_ms,
                option_type=option_type,
                bid=0.0,
                ask=0.0,
                last=0.0,
                mid=0.0,
                volume=0,
                open_interest=0,
                iv=None,
                delta=None,
                gamma=None,
                theta=None,
                vega=None,
                timestamp_ms=now_ms,
                source_ts_ms=now_ms,
                recv_ts_ms=recv_ts_ms,
                provider=self.PROVIDER,
                sequence_id=self._next_sequence_id(),
            )

            if option_type == "PUT":
                puts.append(quote)
            else:
                calls.append(quote)

        # Sort by strike for determinism
        puts.sort(key=lambda q: q.strike)
        calls.sort(key=lambda q: q.strike)

        if not puts and not calls:
            logger.warning(
                "Chain for %s exp=%s has no valid puts or calls (provider=%s)",
                symbol, expiration, self.PROVIDER,
            )
            return None

        # Compute ATM IV (from put closest to underlying price, if available)
        atm_iv = None
        if puts and underlying_price > 0:
            atm_put = min(puts, key=lambda q: abs(q.strike - underlying_price))
            atm_iv = atm_put.iv

        snapshot_id = compute_snapshot_id(symbol, expiration_ms, now_ms)

        return OptionChainSnapshotEvent(
            symbol=symbol,
            expiration_ms=expiration_ms,
            underlying_price=underlying_price,
            underlying_bid=0.0,
            underlying_ask=0.0,
            puts=tuple(puts),
            calls=tuple(calls),
            n_strikes=len(puts),
            atm_iv=atm_iv,
            timestamp_ms=now_ms,
            source_ts_ms=now_ms,
            recv_ts_ms=recv_ts_ms,
            provider=self.PROVIDER,
            snapshot_id=snapshot_id,
            sequence_id=self._next_sequence_id(),
        )

    def build_snapshots_for_symbol(
        self,
        symbol: str,
        min_dte: int = 7,
        max_dte: int = 21,
        underlying_price: float = 0.0,
    ) -> List[OptionChainSnapshotEvent]:
        """Fetch chain and build all snapshots within DTE range for a symbol.

        This is the main entry point for the orchestrator.

        Args:
            symbol: Underlying symbol.
            min_dte: Minimum days to expiration.
            max_dte: Maximum days to expiration.
            underlying_price: Current underlying price.

        Returns:
            List of OptionChainSnapshotEvents (may be empty on failure).
        """
        raw = self.fetch_nested_chain(symbol)
        if not raw:
            logger.warning(
                "Empty response from Tastytrade for %s — skipping snapshot build",
                symbol,
            )
            return []

        normalized = normalize_tastytrade_nested_chain(raw)
        if not normalized:
            logger.warning(
                "Normalization returned empty list for %s (provider=%s)",
                symbol, self.PROVIDER,
            )
            return []

        expirations = self.get_expirations_in_range(normalized, min_dte, max_dte)
        if not expirations:
            logger.debug(
                "No expirations in DTE range [%d, %d] for %s (provider=%s)",
                min_dte, max_dte, symbol, self.PROVIDER,
            )
            return []

        snapshots = []
        for exp_date, dte in expirations:
            snapshot = self.build_chain_snapshot(
                symbol, exp_date, normalized, underlying_price
            )
            if snapshot:
                snapshots.append(snapshot)
                logger.debug(
                    "Tastytrade snapshot: %s exp=%s DTE=%d puts=%d calls=%d",
                    symbol, exp_date, dte,
                    len(snapshot.puts), len(snapshot.calls),
                )

        return snapshots

    def get_health_status(self) -> Dict[str, Any]:
        """Get connector health metrics."""
        return {
            "provider": self.PROVIDER,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
            "last_request_ms": self._last_request_ms,
            "last_latency_ms": self._last_latency_ms,
            "sequence_id": self._sequence_id,
            "authenticated": self._authenticated,
            "health": "ok" if self._error_count < max(1, self._request_count) * 0.3 else "degraded",
        }

    def close(self) -> None:
        """Close the connector and release resources."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:
                logger.debug("Error closing Tastytrade client: %s", exc)
            self._client = None
            self._authenticated = False
