"""News sentiment external metric updater.

Fetches a lightweight sentiment score from a third-party endpoint and publishes
``ExternalMetricEvent(key="news_sentiment", value=...)``.

Value contract (JSON-serialisable dict):

    {
        "score": float,        # range [-1.0, +1.0]
        "label": str,          # "bullish" | "neutral" | "bearish" | "stub"
        "n_headlines": int,    # number of headlines used
    }

On API/rate-limit failures, ``update()`` returns ``None`` and does not publish.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp

from .bus import EventBus
from .events import ExternalMetricEvent, TOPIC_EXTERNAL_METRICS

logger = logging.getLogger("argus.news_sentiment_updater")


class NewsSentimentUpdater:
    """Collect and publish a ``news_sentiment`` external metric."""

    def __init__(self, bus: EventBus, config: Dict[str, Any]) -> None:
        self._bus = bus

        ns_cfg = config.get("news_sentiment") or {}
        self._enabled: bool = bool(ns_cfg.get("enabled", False))
        self._api_url: str = str(
            ns_cfg.get("api_url") or "https://www.alphavantage.co/query"
        )

        api_key_raw = ns_cfg.get("api_key") or ""
        if isinstance(api_key_raw, str) and api_key_raw.startswith("${") and api_key_raw.endswith("}"):
            env_var = api_key_raw[2:-1]
            api_key_raw = os.getenv(env_var, "")
        self._api_key = str(api_key_raw or "")

        self._session: Optional[aiohttp.ClientSession] = None
        self._last_call_ts: float = 0.0
        self._min_call_interval_s = 12.5

        if self._enabled:
            logger.info("NewsSentimentUpdater enabled (source=alphavantage_news_sentiment)")
        else:
            logger.info("NewsSentimentUpdater disabled (news_sentiment.enabled=false)")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call_ts
        if elapsed < self._min_call_interval_s:
            await asyncio.sleep(self._min_call_interval_s - elapsed)
        self._last_call_ts = time.monotonic()

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score > 0.15:
            return "bullish"
        if score < -0.15:
            return "bearish"
        return "neutral"

    @staticmethod
    def _extract_feed_score(item: Dict[str, Any]) -> Optional[float]:
        # AlphaVantage NEWS_SENTIMENT returns sentiment_score and
        # ticker_sentiment entries with ticker_sentiment_score.
        candidates: List[float] = []

        raw = item.get("overall_sentiment_score")
        if raw is not None:
            try:
                candidates.append(float(raw))
            except (TypeError, ValueError):
                pass

        for ts in item.get("ticker_sentiment") or []:
            raw_ticker = ts.get("ticker_sentiment_score")
            if raw_ticker is None:
                continue
            try:
                candidates.append(float(raw_ticker))
            except (TypeError, ValueError):
                continue

        if not candidates:
            return None
        return sum(candidates) / len(candidates)

    @staticmethod
    def _stub_payload() -> Dict[str, Any]:
        return {"score": 0.0, "label": "stub", "n_headlines": 0}

    async def update(self) -> Optional[Dict[str, Any]]:
        """Fetch and publish the latest news sentiment payload.

        Returns:
            Dict payload on success, ``None`` on transient API failure or when
            disabled. If enabled but no API key configured, emits deterministic
            stub payload.
        """
        if not self._enabled:
            return None

        now_ms = int(time.time() * 1000)

        if not self._api_key:
            payload = self._stub_payload()
            self._bus.publish(
                TOPIC_EXTERNAL_METRICS,
                ExternalMetricEvent(key="news_sentiment", value=payload, timestamp_ms=now_ms),
            )
            logger.info(
                "NewsSentiment updated: score=%.4f label=%s source=stub n_headlines=%d",
                payload["score"], payload["label"], payload["n_headlines"],
            )
            return payload

        params = {
            "function": "NEWS_SENTIMENT",
            "topics": "financial_markets",
            "sort": "LATEST",
            "limit": "50",
            "apikey": self._api_key,
        }

        try:
            await self._throttle()
            session = await self._get_session()
            async with session.get(self._api_url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("NewsSentiment HTTP %d: %s", resp.status, text[:200])
                    return None
                data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("NewsSentiment fetch failed: %s", exc)
            return None

        rate_note = data.get("Note") or data.get("Information")
        if rate_note:
            logger.warning("NewsSentiment rate limited: %s", rate_note)
            return None

        feed = data.get("feed") or []
        scores: List[float] = []
        for item in feed:
            if not isinstance(item, dict):
                continue
            score = self._extract_feed_score(item)
            if score is None:
                continue
            # keep contract strict in [-1, 1]
            scores.append(max(-1.0, min(1.0, float(score))))

        if not scores:
            logger.info("NewsSentiment unavailable: no scoreable headlines")
            return None

        avg_score = sum(scores) / len(scores)
        payload = {
            "score": round(max(-1.0, min(1.0, avg_score)), 6),
            "label": self._label_from_score(avg_score),
            "n_headlines": len(scores),
        }

        self._bus.publish(
            TOPIC_EXTERNAL_METRICS,
            ExternalMetricEvent(key="news_sentiment", value=payload, timestamp_ms=now_ms),
        )
        logger.info(
            "NewsSentiment updated: score=%.4f label=%s source=alphavantage_news_sentiment n_headlines=%d",
            payload["score"], payload["label"], payload["n_headlines"],
        )
        return payload
