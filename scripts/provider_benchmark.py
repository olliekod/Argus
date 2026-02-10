#!/usr/bin/env python3
"""Benchmark providers for the liquid ETF universe with separate scorecards."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

from scripts.tastytrade_health_audit import _load_tasty_client, audit_oauth, run_quotes_probe
from src.connectors.alpaca_client import AlpacaDataClient
from src.connectors.yahoo_client import YahooFinanceClient
from src.core.bus import EventBus
from src.core.config import ConfigurationError, load_config, load_secrets
from src.core.liquid_etf_universe import LIQUID_ETF_UNIVERSE
from src.core.options_normalize import normalize_tastytrade_nested_chain

BARS_WEIGHTS = {"success_rate": 0.5, "latency_p95": 0.25, "bar_age_p95": 0.25}
BARS_THRESHOLDS = {"latency_ms": 3000.0, "bar_age_sec": 300.0}
OPTIONS_WEIGHTS = {"missing_rate": 0.45, "latency": 0.25, "stale": 0.15, "spread": 0.15}
OPTIONS_THRESHOLDS = {"latency_ms": 5000.0, "stale_sec": 5.0, "spread_bps": 150.0}
GREEKS_WEIGHTS = {"presence": 0.7, "stale": 0.3}
GREEKS_THRESHOLDS = {"stale_sec": 5.0}


def _is_placeholder(v: str) -> bool:
    return (not v) or v.startswith("PASTE_") or v.startswith("YOUR_")


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int((p / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def bars_score(success_rate: float, latency_p95: float | None, bar_age_p95: float | None) -> float | None:
    if latency_p95 is None or bar_age_p95 is None:
        return None
    return round(
        BARS_WEIGHTS["success_rate"] * success_rate
        + BARS_WEIGHTS["latency_p95"] * _clamp(1 - latency_p95 / BARS_THRESHOLDS["latency_ms"])
        + BARS_WEIGHTS["bar_age_p95"] * _clamp(1 - bar_age_p95 / BARS_THRESHOLDS["bar_age_sec"]),
        4,
    )


def options_score(missing_rate: float, latency_p95: float | None, stale_p95: float | None, spread_p95: float | None) -> float | None:
    if latency_p95 is None or stale_p95 is None or spread_p95 is None:
        return None
    return round(
        OPTIONS_WEIGHTS["missing_rate"] * (1 - missing_rate)
        + OPTIONS_WEIGHTS["latency"] * _clamp(1 - latency_p95 / OPTIONS_THRESHOLDS["latency_ms"])
        + OPTIONS_WEIGHTS["stale"] * _clamp(1 - stale_p95 / OPTIONS_THRESHOLDS["stale_sec"])
        + OPTIONS_WEIGHTS["spread"] * _clamp(1 - spread_p95 / OPTIONS_THRESHOLDS["spread_bps"]),
        4,
    )


def greeks_score(presence_rate: float, stale_p95: float | None) -> float | None:
    if stale_p95 is None:
        return None
    return round(
        GREEKS_WEIGHTS["presence"] * presence_rate
        + GREEKS_WEIGHTS["stale"] * _clamp(1 - stale_p95 / GREEKS_THRESHOLDS["stale_sec"]),
        4,
    )


async def _bars_rows(secrets: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skips: list[str] = []
    universe = list(LIQUID_ETF_UNIVERSE)

    yahoo = YahooFinanceClient(symbols=universe)
    for sym in universe:
        start = time.perf_counter()
        success = False
        age = None
        try:
            quote = await yahoo.get_quote(sym)
            success = bool(quote and quote.get("price"))
            source_ts = (quote or {}).get("source_ts")
            if source_ts is not None:
                age = max(0.0, time.time() - float(source_ts))
        except Exception:
            success = False
        rows.append({"provider": "yahoo", "symbol": sym, "request_latency_ms": (time.perf_counter() - start) * 1000, "bar_age_sec": age, "success": success})
    await yahoo.close()

    key = secrets.get("alpaca", {}).get("api_key", "")
    sec = secrets.get("alpaca", {}).get("api_secret", "")
    if _is_placeholder(key) or _is_placeholder(sec):
        skips.append("alpaca bars skipped: credentials missing")
    else:
        alpaca = AlpacaDataClient(api_key=key, api_secret=sec, symbols=universe, event_bus=EventBus(), poll_interval=60)
        for sym in universe:
            start = time.perf_counter()
            success = False
            age = None
            try:
                bars = await alpaca.fetch_bars(sym, limit=1)
                success = bool(bars)
                if bars and isinstance(bars[0].get("t"), str):
                    ts = datetime.fromisoformat(bars[0]["t"].replace("Z", "+00:00"))
                    age = (datetime.now(timezone.utc) - ts).total_seconds()
            except Exception:
                success = False
            rows.append({"provider": "alpaca", "symbol": sym, "request_latency_ms": (time.perf_counter() - start) * 1000, "bar_age_sec": age, "success": success})
        await alpaca.close()

    return rows, skips


async def _options_rows(config: dict[str, Any], secrets: dict[str, Any], duration: int, greeks: bool) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skips: list[str] = []
    try:
        client = _load_tasty_client(config, secrets)
        oauth = audit_oauth(secrets)
    except Exception as exc:
        skips.append(f"tastytrade options skipped: {exc}")
        return rows, skips

    try:
        for sym in LIQUID_ETF_UNIVERSE:
            chain = client.get_nested_option_chains(sym)
            normalized = normalize_tastytrade_nested_chain(chain)
            probe = await run_quotes_probe(sym, normalized, oauth["access_token"], duration=duration, greeks=greeks)
            rows.append({"provider": "tastytrade_dxlink", **probe})
    except Exception as exc:
        skips.append(f"tastytrade options error: {exc}")
    finally:
        client.close()
    return rows, skips


def _build_scorecards(bar_rows: list[dict[str, Any]], option_rows: list[dict[str, Any]], greeks_enabled: bool) -> dict[str, Any]:
    bars_card: list[dict[str, Any]] = []
    for provider in sorted({r["provider"] for r in bar_rows}):
        subset = [r for r in bar_rows if r["provider"] == provider]
        success_rate = sum(1 for r in subset if r["success"]) / max(1, len(subset))
        lat = [float(r["request_latency_ms"]) for r in subset]
        age = [float(r["bar_age_sec"]) for r in subset if r.get("bar_age_sec") is not None]
        row = {
            "provider": provider,
            "request_latency_ms_p50": _pct(lat, 50),
            "request_latency_ms_p95": _pct(lat, 95),
            "bar_age_sec_p50": _pct(age, 50),
            "bar_age_sec_p95": _pct(age, 95),
            "success_rate": round(success_rate, 4),
        }
        row["score"] = bars_score(row["success_rate"], row["request_latency_ms_p95"], row["bar_age_sec_p95"])
        bars_card.append(row)

    options_card: list[dict[str, Any]] = []
    if option_rows:
        missing = [float(r["missing_quote_rate"]) for r in option_rows]
        lat = [float(r["time_to_first_option_quote_ms"]) for r in option_rows if r.get("time_to_first_option_quote_ms") is not None]
        stale = [float(r["stale_age_sec_p95"]) for r in option_rows if r.get("stale_age_sec_p95") is not None]
        spread = [float(r["spread_bps_p95"]) for r in option_rows if r.get("spread_bps_p95") is not None]
        row = {
            "provider": "tastytrade_dxlink",
            "time_to_first_quote_ms_p50": _pct(lat, 50),
            "time_to_first_quote_ms_p95": _pct(lat, 95),
            "missing_quote_rate": round(sum(missing) / len(missing), 4) if missing else None,
            "stale_p95": _pct(stale, 95),
            "spread_bps_p50": _pct(spread, 50),
            "spread_bps_p95": _pct(spread, 95),
        }
        row["score"] = options_score(row["missing_quote_rate"] or 1.0, row["time_to_first_quote_ms_p95"], row["stale_p95"], row["spread_bps_p95"])
        options_card.append(row)

    greeks_card: list[dict[str, Any]] = []
    if greeks_enabled and option_rows:
        presence = [float(r.get("greeks_presence_rate", 0.0)) for r in option_rows]
        stale = [float(r["stale_age_sec_p95"]) for r in option_rows if r.get("stale_age_sec_p95") is not None]
        row = {
            "provider": "tastytrade_dxlink",
            "greeks_presence_rate": round(sum(presence) / len(presence), 4) if presence else 0.0,
            "stale_p95": _pct(stale, 95),
        }
        row["score"] = greeks_score(row["greeks_presence_rate"], row["stale_p95"])
        greeks_card.append(row)

    composite_required = bool(bars_card) and bool(options_card) and (bool(greeks_card) if greeks_enabled else True)
    composite = {
        "status": "complete" if composite_required else "partial",
        "score": None,
    }
    if composite_required:
        parts = [r["score"] for r in bars_card if r.get("score") is not None]
        parts += [r["score"] for r in options_card if r.get("score") is not None]
        if greeks_enabled:
            parts += [r["score"] for r in greeks_card if r.get("score") is not None]
        if parts:
            composite["score"] = round(sum(parts) / len(parts), 4)

    return {
        "BarsScorecard": {"weights": BARS_WEIGHTS, "thresholds": BARS_THRESHOLDS, "rows": bars_card},
        "OptionsQuoteScorecard": {"weights": OPTIONS_WEIGHTS, "thresholds": OPTIONS_THRESHOLDS, "rows": options_card},
        "OptionsGreeksScorecard": {"weights": GREEKS_WEIGHTS, "thresholds": GREEKS_THRESHOLDS, "rows": greeks_card},
        "Composite": composite,
    }


def _market_hours_flag() -> str:
    # Simple deterministic flag from UTC weekday and US RTH hour window proxy.
    now = datetime.now(timezone.utc)
    return "likely_rth" if now.weekday() < 5 and 14 <= now.hour <= 21 else "likely_off_hours"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--greeks", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": list(LIQUID_ETF_UNIVERSE),
        "market_hours_flag": _market_hours_flag(),
        "skips": [],
    }

    try:
        config = load_config()
        secrets = load_secrets()
    except ConfigurationError as exc:
        payload["skips"].append(f"config/secrets unavailable: {exc}")
        payload.update(_build_scorecards([], [], args.greeks))
        out = Path(args.json_out) if args.json_out else Path("logs") / f"provider_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"SKIP: not configured ({exc})")
        print(f"Wrote {out}")
        return 0

    bar_rows, bar_skips = asyncio.run(_bars_rows(secrets))
    opt_rows, opt_skips = asyncio.run(_options_rows(config, secrets, args.duration, args.greeks))
    payload["skips"].extend(bar_skips + opt_skips)
    payload["bars_raw"] = bar_rows
    payload["options_raw"] = opt_rows
    payload.update(_build_scorecards(bar_rows, opt_rows, args.greeks))

    print("BarsScorecard:")
    for row in payload["BarsScorecard"]["rows"]:
        print(f"  {row['provider']}: success={row['success_rate']:.2%} score={row['score']}")
    print("OptionsQuoteScorecard:")
    for row in payload["OptionsQuoteScorecard"]["rows"]:
        print(f"  {row['provider']}: missing={row['missing_quote_rate']} score={row['score']}")
    if args.greeks:
        print("OptionsGreeksScorecard:")
        for row in payload["OptionsGreeksScorecard"]["rows"]:
            print(f"  {row['provider']}: presence={row['greeks_presence_rate']} score={row['score']}")
    print(f"Composite: {payload['Composite']}")

    out = Path(args.json_out) if args.json_out else Path("logs") / f"provider_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
