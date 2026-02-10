#!/usr/bin/env python3
"""Provider benchmark across the Liquid ETF Universe.

Scoring formula (conservative defaults):
score = w1*(1 - missing_rate) + w2*clamp(1 - stale_p95/stale_thresh)
      + w3*clamp(1 - spread_p95/spread_thresh) + w4*clamp(1 - latency_p95/lat_thresh)
Weights: w1=0.45, w2=0.2, w3=0.2, w4=0.15
Thresholds: stale=5s, spread=150bps, latency=3000ms
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.tastytrade_health_audit import run_quotes_probe, _load_tasty_client, audit_oauth
from src.connectors.alpaca_client import AlpacaDataClient
from src.connectors.yahoo_client import YahooFinanceClient
from src.core.bus import EventBus
from src.core.config import load_config, load_secrets
from src.core.liquid_etf_universe import LIQUID_ETF_UNIVERSE

W1, W2, W3, W4 = 0.45, 0.20, 0.20, 0.15
STALE_THRESH = 5.0
SPREAD_THRESH = 150.0
LAT_THRESH = 3000.0


def clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def score_row(missing_rate: float, stale_p95: float | None, spread_p95: float | None, latency_p95: float | None) -> float:
    stale = clamp(1 - ((stale_p95 or STALE_THRESH) / STALE_THRESH))
    spread = clamp(1 - ((spread_p95 or SPREAD_THRESH) / SPREAD_THRESH))
    latency = clamp(1 - ((latency_p95 or LAT_THRESH) / LAT_THRESH))
    return round(W1 * (1 - missing_rate) + W2 * stale + W3 * spread + W4 * latency, 4)


async def bars_benchmark(config: dict[str, Any], secrets: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Yahoo
    y = YahooFinanceClient(symbols=list(LIQUID_ETF_UNIVERSE))
    for sym in LIQUID_ETF_UNIVERSE:
        st = time.perf_counter()
        ok = False
        age = None
        try:
            q = await y.get_quote(sym)
            ok = bool(q and q.get("price"))
            source_ts = (q or {}).get("source_ts")
            if source_ts:
                age = max(0.0, time.time() - float(source_ts))
        except Exception:
            ok = False
        rows.append({"provider": "yahoo", "symbol": sym, "request_latency_ms": int((time.perf_counter() - st) * 1000), "bar_timestamp_age_sec": age, "success": ok})
    await y.close()

    # Alpaca
    key = secrets.get("alpaca", {}).get("api_key", "")
    sec = secrets.get("alpaca", {}).get("api_secret", "")
    if key and sec and not key.startswith("PASTE_"):
        a = AlpacaDataClient(api_key=key, api_secret=sec, symbols=list(LIQUID_ETF_UNIVERSE), event_bus=EventBus(), poll_interval=60)
        for sym in LIQUID_ETF_UNIVERSE:
            st = time.perf_counter()
            ok = False
            age = None
            try:
                bars = await a.fetch_bars(sym, limit=1)
                ok = bool(bars)
                if bars:
                    ts = bars[0].get("t")
                    if ts:
                        if isinstance(ts, str):
                            ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            age = max(0.0, (datetime.now(timezone.utc) - ts_dt).total_seconds())
                
            except Exception:
                ok = False
            rows.append({"provider": "alpaca", "symbol": sym, "request_latency_ms": int((time.perf_counter() - st) * 1000), "bar_timestamp_age_sec": age, "success": ok})
        await a.close()
    return rows


async def options_benchmark(config: dict[str, Any], secrets: dict[str, Any], duration: int) -> list[dict[str, Any]]:
    client = _load_tasty_client(config, secrets)
    oauth = audit_oauth(secrets)
    rows: list[dict[str, Any]] = []
    try:
        for sym in LIQUID_ETF_UNIVERSE:
            chain = client.get_nested_option_chains(sym)
            from src.core.options_normalize import normalize_tastytrade_nested_chain
            normalized = normalize_tastytrade_nested_chain(chain)
            probe = await run_quotes_probe(sym, normalized, oauth["access_token"], duration=duration)
            rows.append({"provider": "tastytrade_dxlink", **probe, "greeks_presence_rate": None})
    finally:
        client.close()
    return rows


def summarize(payload: dict[str, Any]) -> None:
    print("\nProvider summary:")
    for p in payload["provider_scores"]:
        print(f"  {p['provider']}: score={p['score']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=10)
    args = parser.parse_args()

    config = load_config()
    secrets = load_secrets()

    bars = asyncio.run(bars_benchmark(config, secrets))
    options = asyncio.run(options_benchmark(config, secrets, args.duration))

    provider_rows = []
    for p in {row["provider"] for row in bars}:
        p_rows = [r for r in bars if r["provider"] == p]
        missing = 1 - (sum(1 for r in p_rows if r["success"]) / max(1, len(p_rows)))
        lat = sorted([r["request_latency_ms"] for r in p_rows])
        lat_p95 = lat[min(len(lat)-1, int(0.95*(len(lat)-1)))] if lat else None
        stale = [r["bar_timestamp_age_sec"] for r in p_rows if r.get("bar_timestamp_age_sec") is not None]
        stale_p95 = sorted(stale)[min(len(stale)-1, int(0.95*(len(stale)-1)))] if stale else None
        provider_rows.append({"provider": p, "score": score_row(missing, stale_p95, None, lat_p95)})

    opt_missing = [r["missing_quote_rate"] for r in options]
    opt_stale = [r["stale_age_sec_p95"] for r in options if r.get("stale_age_sec_p95") is not None]
    opt_spread = [r["spread_bps_p95"] for r in options if r.get("spread_bps_p95") is not None]
    opt_lat = [r["handshake_latency_ms"] for r in options if r.get("handshake_latency_ms") is not None]
    provider_rows.append({
        "provider": "tastytrade_dxlink",
        "score": score_row(
            sum(opt_missing)/max(1,len(opt_missing)) if opt_missing else 1.0,
            sorted(opt_stale)[min(len(opt_stale)-1, int(0.95*(len(opt_stale)-1)))] if opt_stale else None,
            sorted(opt_spread)[min(len(opt_spread)-1, int(0.95*(len(opt_spread)-1)))] if opt_spread else None,
            sorted(opt_lat)[min(len(opt_lat)-1, int(0.95*(len(opt_lat)-1)))] if opt_lat else None,
        ),
    })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": list(LIQUID_ETF_UNIVERSE),
        "weights": {"w1": W1, "w2": W2, "w3": W3, "w4": W4},
        "thresholds": {"stale_sec": STALE_THRESH, "spread_bps": SPREAD_THRESH, "latency_ms": LAT_THRESH},
        "bars": bars,
        "options": options,
        "provider_scores": sorted(provider_rows, key=lambda r: (-r["score"], r["provider"])),
    }
    out = Path("logs") / f"provider_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summarize(payload)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
