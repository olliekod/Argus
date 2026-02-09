"""
Tastytrade REST API Client
==========================

Thin REST client with session auth, retries, and timestamp parsing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

TASTYTRADE_LIVE_URL = "https://api.tastytrade.com"
TASTYTRADE_SANDBOX_URL = "https://api.cert.tastytrade.com"


class TastytradeError(RuntimeError):
    """Raised when Tastytrade REST calls fail."""


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0


def parse_rfc3339_nano(timestamp: str) -> datetime:
    """Parse RFC3339 timestamps with optional nanoseconds into UTC datetime."""
    if not timestamp:
        raise ValueError("Timestamp is empty")
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"
    if "." in timestamp:
        prefix, rest = timestamp.split(".", 1)
        if "+" in rest or "-" in rest:
            frac, offset = rest.split("+", 1) if "+" in rest else rest.split("-", 1)
            sign = "+" if "+" in rest else "-"
            frac = (frac + "000000")[:6]
            timestamp = f"{prefix}.{frac}{sign}{offset}"
        else:
            frac = (rest + "000000")[:6]
            timestamp = f"{prefix}.{frac}"
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _attach_nested_chain_timestamps(payload: Dict[str, Any]) -> None:
    expirations = (
        payload.get("expirations")
        or payload.get("items")
        or payload.get("option-chains")
        or []
    )
    for expiration in expirations:
        expiry_raw = (
            expiration.get("expiration-date")
            or expiration.get("expiration-date-time")
            or expiration.get("expiration")
        )
        if isinstance(expiry_raw, str) and ("T" in expiry_raw or "Z" in expiry_raw or "+" in expiry_raw):
            try:
                expiration["expiration-datetime"] = parse_rfc3339_nano(expiry_raw)
            except ValueError:
                continue


class TastytradeRestClient:
    """Synchronous REST client for Tastytrade."""

    def __init__(
        self,
        username: str,
        password: str,
        environment: str = "live",
        timeout_seconds: float = 20.0,
        retries: Optional[RetryConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._username = username
        self._password = password
        self._base_url = self._resolve_base_url(environment)
        self._timeout = timeout_seconds
        self._retry = retries or RetryConfig()
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._token: Optional[str] = None

    @property
    def base_url(self) -> str:
        return self._base_url

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def login(self) -> str:
        payload = {"login": self._username, "password": self._password}
        data = self._request("POST", "/sessions", json=payload, auth=False)
        token = (
            data.get("data", {}).get("session-token")
            or data.get("session-token")
        )
        if not token:
            raise TastytradeError("No session token returned from login.")
        self._token = token
        self._session.headers["Authorization"] = token
        return token

    def get_accounts(self) -> Any:
        data = self._request("GET", "/customers/me/accounts")
        return data.get("data", {}).get("items", data.get("data"))

    def get_balances(self, account_number: str) -> Any:
        data = self._request("GET", f"/accounts/{account_number}/balances")
        return data.get("data", data)

    def get_positions(self, account_number: str) -> Any:
        data = self._request("GET", f"/accounts/{account_number}/positions")
        return data.get("data", {}).get("items", data.get("data"))

    def get_option_chain(self, underlying: str) -> Any:
        data = self._request("GET", f"/option-chains/{underlying}")
        return data.get("data", data)

    def list_nested_option_chains(self, underlying: str, **params: Any) -> Any:
        return self.get_nested_option_chains(underlying, **params)

    def get_nested_option_chains(self, symbol: str, **params: Any) -> Any:
        data = self._request(
            "GET",
            f"/option-chains/{symbol}/nested",
            params=params or None,
            error_excerpt_limit=500,
        )
        payload = data.get("data", data)
        if isinstance(payload, dict):
            _attach_nested_chain_timestamps(payload)
        return data

    def get_quotes(self, symbols: list[str]) -> Any:
        logger.warning("Quotes endpoint not yet implemented.")
        return {"symbols": symbols, "data": []}

    def _resolve_base_url(self, environment: str) -> str:
        env = (environment or "live").lower()
        if env == "sandbox":
            return TASTYTRADE_SANDBOX_URL
        return TASTYTRADE_LIVE_URL

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        error_excerpt_limit: int = 200,
    ) -> Dict[str, Any]:
        if auth and not self._token:
            raise TastytradeError("Not authenticated. Call login() first.")

        url = f"{self._base_url}{path}"
        attempts = max(1, self._retry.max_attempts)
        for attempt in range(attempts):
            try:
                response = self._session.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                if attempt < attempts - 1:
                    self._sleep(attempt)
                    continue
                raise TastytradeError(f"Request failed: {exc}") from exc

            if response.status_code in (429,) or response.status_code >= 500:
                if attempt < attempts - 1:
                    self._sleep(attempt)
                    continue
            if not response.ok:
                raise TastytradeError(
                    f"Tastytrade HTTP {response.status_code}: {response.text[:error_excerpt_limit]}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise TastytradeError("Invalid JSON response") from exc
            return payload

        raise TastytradeError("Request failed after retries.")

    def _sleep(self, attempt: int) -> None:
        delay = self._retry.backoff_seconds * (
            self._retry.backoff_multiplier ** attempt
        )
        time.sleep(delay)


class TastytradeClient(TastytradeRestClient):
    """Backward-compatible alias for TastytradeRestClient."""
