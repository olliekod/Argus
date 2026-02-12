"""Public.com API client for options greeks."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:  # pragma: no cover - tested via orchestrator guards
    aiohttp = None

logger = logging.getLogger(__name__)


class PublicAPIError(RuntimeError):
    """Raised on Public API request failures."""


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class PublicAPIConfig:
    api_secret: str
    account_id: str = ""
    base_url: str = "https://api.public.com"
    timeout_seconds: float = 20.0


class PublicAPIClient:
    """Minimal async client for Public options greeks endpoint."""

    MAX_GREEKS_SYMBOLS = 250

    def __init__(self, config: PublicAPIConfig) -> None:
        if not config.api_secret:
            raise PublicAPIError("public.api_secret is required")
        self._config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._resolved_account_id: str = config.account_id or ""

    async def _get_session(self) -> aiohttp.ClientSession:
        if aiohttp is None:
            raise ImportError("aiohttp required: pip install aiohttp")
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
            headers = {
                "Authorization": f"Bearer {self._config.api_secret}",
                "Accept": "application/json",
            }
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None, retry_401: bool = True) -> Dict[str, Any]:
        session = await self._get_session()
        url = f"{self._config.base_url.rstrip('/')}{path}"
        async with session.request(method, url, params=params) as resp:
            if resp.status == 401 and retry_401:
                logger.warning("Public API 401 for %s; retrying once", path)
                return await self._request(method, path, params=params, retry_401=False)
            if resp.status >= 400:
                text = await resp.text()
                raise PublicAPIError(f"Public API error {resp.status}: {text[:300]}")
            data = await resp.json(content_type=None)
            return data if isinstance(data, dict) else {"data": data}

    async def get_account_id(self) -> str:
        """Resolve account_id from config or account listing endpoint."""
        if self._resolved_account_id:
            return self._resolved_account_id

        payload = await self._request("GET", "/userapigateway/accounts")
        accounts = payload.get("accounts") or payload.get("data") or []
        if isinstance(accounts, dict):
            accounts = [accounts]
        if not accounts:
            raise PublicAPIError("Unable to resolve Public account id; set public.account_id")
        first = accounts[0] if isinstance(accounts[0], dict) else {}
        account_id = str(first.get("accountId") or first.get("id") or "").strip()
        if not account_id:
            raise PublicAPIError("Unable to parse account id from Public accounts response")
        self._resolved_account_id = account_id
        return account_id

    async def get_option_greeks(self, osi_symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch greeks for up to 250 OSI symbols."""
        if len(osi_symbols) > self.MAX_GREEKS_SYMBOLS:
            raise PublicAPIError(f"Public greeks limit exceeded: {len(osi_symbols)} > {self.MAX_GREEKS_SYMBOLS}")
        if not osi_symbols:
            return []
        account_id = await self.get_account_id()
        payload = await self._request(
            "GET",
            f"/userapigateway/option-details/{account_id}/greeks",
            params={"osiSymbols": osi_symbols},
        )
        greeks = payload.get("greeks") or payload.get("data") or []
        if isinstance(greeks, dict):
            greeks = [greeks]
        return [g for g in greeks if isinstance(g, dict)]

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
