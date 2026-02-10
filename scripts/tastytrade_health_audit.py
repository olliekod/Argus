#!/usr/bin/env python3
"""Tastytrade health audit and option chain sanity checks."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import statistics
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests
import websockets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import ConfigurationError, load_config, load_secrets
from src.core.liquid_etf_universe import LIQUID_ETF_UNIVERSE
from src.core.options_normalize import normalize_tastytrade_nested_chain
from src.connectors.tastytrade_oauth import TastytradeOAuthClient
from src.connectors.tastytrade_rest import (
    RetryConfig,
    TastytradeError,
    TastytradeRestClient,
)


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("PASTE_") or value.startswith("YOUR_")


def _parse_expiry(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        raw = value
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            try:
                return datetime.fromisoformat(raw.split("T")[0]).date()
            except ValueError:
                return None
        return parsed.date()
    return None


def _extract_bid_ask(option: Any) -> tuple[float | None, float | None]:
    if not isinstance(option, dict):
        return None, None
    bid = option.get("bid") or option.get("bid-price") or option.get("bid_price")
    ask = option.get("ask") or option.get("ask-price") or option.get("ask_price")
    try:
        bid_val = float(bid) if bid is not None else None
    except (TypeError, ValueError):
        bid_val = None
    try:
        ask_val = float(ask) if ask is not None else None
    except (TypeError, ValueError):
        ask_val = None
    return bid_val, ask_val


def _compute_spread_bps(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    return (ask - bid) / mid * 10000


def _extract_chain_expirations(raw: dict[str, Any]) -> list[dict[str, Any]]:
    data = raw.get("data")
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        chains = data.get("items", [])
    elif isinstance(data, dict) and isinstance(data.get("expirations"), list):
        chains = [data]
    elif isinstance(raw.get("expirations"), list):
        chains = [raw]
    else:
        chains = []

    expirations: list[dict[str, Any]] = []
    for chain in chains:
        if not isinstance(chain, dict):
            continue
        expirations.extend(chain.get("expirations") or [])
    return [exp for exp in expirations if isinstance(exp, dict)]


def audit_nested_chain(raw: dict[str, Any]) -> dict[str, Any]:
    expirations = _extract_chain_expirations(raw)
    today = date.today()
    expired = 0
    missing_strikes_total = 0
    strike_counts = []
    spread_values: list[float] = []
    strike_side_counts = Counter()

    for exp in expirations:
        expiry = _parse_expiry(exp.get("expiration-date") or exp.get("expiration") or exp.get("date"))
        if expiry and expiry < today:
            expired += 1
        strikes = exp.get("strikes") or exp.get("strike-prices") or []
        strike_counts.append(len(strikes))
        for strike in strikes:
            if not isinstance(strike, dict):
                continue
            call, put = strike.get("call"), strike.get("put")
            if call and not put:
                missing_strikes_total += 1
            if put and not call:
                missing_strikes_total += 1
            if call:
                strike_side_counts["call"] += 1
            if put:
                strike_side_counts["put"] += 1

    return {
        "expirations_total": len(expirations),
        "expired_chains": expired,
        "strike_counts": strike_counts,
        "missing_strikes": missing_strikes_total,
        "side_counts": dict(strike_side_counts),
        "spread_samples": len(spread_values),
        "spread_p95_bps": None,
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    items = sorted(values)
    idx = min(len(items) - 1, int(round((pct / 100.0) * (len(items) - 1))))
    return items[idx]


def _sample_option_symbols(contracts: list[dict[str, Any]], now_utc: datetime, max_contracts: int = 80) -> list[dict[str, Any]]:
    active = [
        c for c in contracts
        if c.get("expiry") and c.get("strike") is not None and c.get("right") in {"C", "P"} and c.get("option_symbol")
    ]
    expiries = sorted({c["expiry"] for c in active if _parse_expiry(c["expiry"]) and _parse_expiry(c["expiry"]) >= now_utc.date()})[:2]
    filtered = [c for c in active if c.get("expiry") in expiries]
    strikes = sorted({float(c["strike"]) for c in filtered})
    if not strikes:
        return []
    spot_proxy = statistics.median(strikes)
    nearby = sorted(strikes, key=lambda s: (abs(s - spot_proxy), s))[:11]
    selected = [c for c in filtered if float(c["strike"]) in set(nearby)]
    selected.sort(key=lambda c: (c.get("expiry"), float(c.get("strike", 0.0)), c.get("right"), c.get("option_symbol")))
    return selected[:max_contracts]


def _extract_event_ts(quote: dict[str, Any]) -> float | None:
    ts = quote.get("eventTime") or quote.get("time") or quote.get("event_ts")
    if ts is None:
        return None
    try:
        return float(ts)
    except (TypeError, ValueError):
        return None


def _ensure_snapshot_table(sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS option_quote_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                provider TEXT NOT NULL,
                underlying TEXT NOT NULL,
                option_symbol TEXT NOT NULL,
                expiry TEXT,
                strike REAL,
                right TEXT,
                bid REAL,
                ask REAL,
                mid REAL,
                event_ts REAL,
                recv_ts REAL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_option_quote_snapshots_lookup ON option_quote_snapshots(underlying, ts_utc)"
        )
        conn.commit()
    finally:
        conn.close()


def _append_snapshots(sqlite_path: Path, rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.executemany(
            """
            INSERT INTO option_quote_snapshots (
                ts_utc, provider, underlying, option_symbol, expiry, strike, right, bid, ask, mid, event_ts, recv_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def audit_oauth(secrets: dict[str, Any]) -> dict[str, Any]:
    oauth_cfg = secrets.get("tastytrade_oauth2", {}) or {}
    client_id = oauth_cfg.get("client_id", "")
    client_secret = oauth_cfg.get("client_secret", "")
    refresh_token = oauth_cfg.get("refresh_token", "")
    if _is_placeholder(client_id) or _is_placeholder(client_secret) or _is_placeholder(refresh_token):
        raise RuntimeError("Missing tastytrade_oauth2 client_id/client_secret/refresh_token.")
    client = TastytradeOAuthClient(client_id=client_id, client_secret=client_secret, refresh_token=refresh_token)
    start = time.perf_counter()
    token_result = client.refresh_access_token()
    return {
        "access_token": token_result.access_token,
        "latency_s": round(time.perf_counter() - start, 3),
    }


async def run_quotes_probe(symbol: str, chain_contracts: list[dict[str, Any]], access_token: str, duration: int, persist_path: Path | None = None) -> dict[str, Any]:
    quote_resp = requests.get(
        "https://api.tastytrade.com/api-quote-tokens",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    quote_resp.raise_for_status()
    quote_data = quote_resp.json().get("data", {})
    dxlink_url = quote_data.get("dxlink-url")
    dxlink_token = quote_data.get("token")

    now_utc = datetime.now(timezone.utc)
    selected = _sample_option_symbols(chain_contracts, now_utc)
    symbols = [c.get("option_symbol") for c in selected if c.get("option_symbol")]
    subscribed = len(symbols)

    quote_seen: dict[str, dict[str, Any]] = {}
    underlying_ok = False
    spreads: list[float] = []
    stale_ages: list[float] = []
    first_underlying_ms = None
    first_option_ms = None

    rows: list[tuple[Any, ...]] = []
    start = time.perf_counter()
    handshake_latency = None

    async with websockets.connect(dxlink_url) as ws:
        async def send(payload: dict[str, Any]) -> None:
            await ws.send(json.dumps(payload))

        await send({"type": "SETUP", "channel": 0, "keepaliveTimeout": 60, "acceptDataFormat": "json", "version": "0.1"})

        feed_channel = 1
        while handshake_latency is None:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "AUTH_STATE" and msg.get("state") == "UNAUTHORIZED":
                await send({"type": "AUTH", "channel": 0, "token": dxlink_token})
            elif msg.get("type") == "AUTH_STATE" and msg.get("state") == "AUTHORIZED":
                await send({"type": "CHANNEL_REQUEST", "channel": feed_channel, "service": "FEED", "parameters": {"contract": "AUTO"}})
            elif msg.get("type") == "CHANNEL_OPENED":
                feed_channel = msg.get("channel", 1)
                await send({"type": "FEED_SETUP", "channel": feed_channel, "acceptEventTypes": ["Quote"]})
                adds = [{"type": "Quote", "symbol": symbol}] + [{"type": "Quote", "symbol": s} for s in symbols]
                await send({"type": "FEED_SUBSCRIPTION", "channel": feed_channel, "add": adds})
                handshake_latency = int((time.perf_counter() - start) * 1000)
                break

        deadline = time.perf_counter() + duration
        while time.perf_counter() < deadline:
            timeout = max(0.1, deadline - time.perf_counter())
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
            except asyncio.TimeoutError:
                continue
            if msg.get("type") == "KEEPALIVE":
                await send({"type": "KEEPALIVE", "channel": 0})
                continue
            if msg.get("type") != "FEED_DATA":
                continue
            data = msg.get("data") or []
            recv_ts = time.time()
            for item in data:
                sym = item.get("eventSymbol") or item.get("symbol")
                if not sym:
                    continue
                if sym == symbol:
                    underlying_ok = True
                    if first_underlying_ms is None:
                        first_underlying_ms = int((time.perf_counter() - start) * 1000)
                    continue
                if sym not in symbols:
                    continue
                quote_seen[sym] = item
                if first_option_ms is None:
                    first_option_ms = int((time.perf_counter() - start) * 1000)
                bid = item.get("bidPrice")
                ask = item.get("askPrice")
                try:
                    bid_f = float(bid) if bid is not None else None
                    ask_f = float(ask) if ask is not None else None
                except (TypeError, ValueError):
                    bid_f = ask_f = None
                spread = _compute_spread_bps(bid_f, ask_f)
                if spread is not None:
                    spreads.append(spread)
                event_ts = _extract_event_ts(item)
                if event_ts is not None:
                    stale_ages.append(max(0.0, recv_ts - event_ts))
                if persist_path and bid_f is not None and ask_f is not None:
                    mid = (bid_f + ask_f) / 2.0 if bid_f > 0 and ask_f > 0 else None
                    meta = next((c for c in selected if c.get("option_symbol") == sym), {})
                    rows.append((
                        datetime.now(timezone.utc).isoformat(),
                        "tastytrade-dxlink",
                        symbol,
                        sym,
                        meta.get("expiry"),
                        meta.get("strike"),
                        meta.get("right"),
                        bid_f,
                        ask_f,
                        mid,
                        event_ts,
                        recv_ts,
                    ))

    if persist_path and rows:
        _ensure_snapshot_table(persist_path)
        _append_snapshots(persist_path, rows)

    received = len(quote_seen)
    missing_rate = 1.0 - (received / subscribed) if subscribed else 1.0
    return {
        "symbol": symbol,
        "underlying_quote_ok": underlying_ok,
        "option_quotes_received": received,
        "subscribed": subscribed,
        "missing_quote_rate": round(missing_rate, 4),
        "stale_age_sec_p95": _percentile(stale_ages, 95),
        "spread_bps_p50": _percentile(spreads, 50),
        "spread_bps_p95": _percentile(spreads, 95),
        "handshake_latency_ms": handshake_latency,
        "time_to_first_underlying_quote_ms": first_underlying_ms,
        "time_to_first_option_quote_ms": first_option_ms,
    }


def _load_tasty_client(config: dict[str, Any], secrets: dict[str, Any]) -> TastytradeRestClient:
    tasty_secrets = secrets.get("tastytrade", {})
    username = tasty_secrets.get("username", "")
    password = tasty_secrets.get("password", "")
    if _is_placeholder(username) or _is_placeholder(password):
        raise RuntimeError("Tastytrade credentials missing; update config/secrets.yaml.")
    tt_config = config.get("tastytrade", {})
    retry_cfg = tt_config.get("retries", {})
    retry = RetryConfig(
        max_attempts=retry_cfg.get("max_attempts", 3),
        backoff_seconds=retry_cfg.get("backoff_seconds", 1.0),
        backoff_multiplier=retry_cfg.get("backoff_multiplier", 2.0),
    )
    client = TastytradeRestClient(
        username=username,
        password=password,
        environment=tt_config.get("environment", "live"),
        timeout_seconds=tt_config.get("timeout_seconds", 20),
        retries=retry,
    )
    client.login()
    return client


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Tastytrade health audit.")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol (default: SPY)")
    parser.add_argument("--quotes", action="store_true", help="Run DXLink quote probe")
    parser.add_argument("--duration", type=int, default=10, help="Quote probe duration in seconds")
    parser.add_argument("--universe", action="store_true", help="Audit the liquid ETF universe")
    parser.add_argument("--max-failures", type=int, default=2, help="Fail if more than this many tickers fail")
    parser.add_argument("--persist-snapshots", action="store_true", help="Persist sampled option quote snapshots")
    parser.add_argument("--snapshots-db", default="data/argus.db", help="SQLite path for snapshot storage")
    args = parser.parse_args()

    try:
        config = load_config()
        secrets = load_secrets()
    except ConfigurationError as exc:
        print(f"Config error: {exc}")
        return 1

    symbols = list(LIQUID_ETF_UNIVERSE) if args.universe else [args.symbol.upper()]

    try:
        client = _load_tasty_client(config, secrets)
    except Exception as exc:
        print(str(exc))
        return 1

    oauth = None
    if args.quotes:
        try:
            oauth = audit_oauth(secrets)
        except Exception as exc:
            print(f"OAuth refresh failed: {exc}")
            client.close()
            return 1

    failures = 0
    results: list[dict[str, Any]] = []
    try:
        for symbol in symbols:
            print(f"\n=== {symbol} ===")
            chain = client.get_nested_option_chains(symbol)
            normalized = normalize_tastytrade_nested_chain(chain)
            summary = audit_nested_chain(chain)
            print(f"Nested chain OK. contracts={len(normalized)} expirations={summary['expirations_total']}")

            row = {
                "symbol": symbol,
                "underlying_quote_ok": False,
                "option_quotes_received": 0,
                "subscribed": 0,
                "missing_quote_rate": 1.0,
                "stale_age_sec_p95": None,
                "spread_bps_p50": None,
                "spread_bps_p95": None,
                "handshake_latency_ms": None,
            }
            if args.quotes and oauth:
                probe = await run_quotes_probe(
                    symbol,
                    normalized,
                    oauth["access_token"],
                    duration=args.duration,
                    persist_path=(Path(args.snapshots_db) if args.persist_snapshots else None),
                )
                row.update(probe)
                print(
                    "Quotes: underlying_ok={underlying_quote_ok} received={option_quotes_received}/{subscribed} "
                    "missing_rate={missing_quote_rate:.2%}".format(**row)
                )

            if not row["underlying_quote_ok"] and args.quotes:
                failures += 1
            results.append(row)
    except TastytradeError as exc:
        print(f"Tastytrade request failed: {exc}")
        failures += 1
    finally:
        client.close()

    if args.universe and results:
        ranked = sorted(results, key=lambda r: (r.get("missing_quote_rate", 1.0), -(r.get("option_quotes_received") or 0), r["symbol"]))
        print("\nRanked summary:")
        print("symbol  underlying_quote_ok  option_quotes_received/subscribed  missing_quote_rate  stale_age_sec_p95  spread_bps_p50/p95  handshake_latency_ms")
        for r in ranked:
            spread_text = (
                f"{round(r['spread_bps_p50'], 1)}/{round(r['spread_bps_p95'], 1)}"
                if r['spread_bps_p95'] is not None
                else "n/a"
            )
            stale_text = round(r['stale_age_sec_p95'], 3) if r['stale_age_sec_p95'] is not None else "n/a"
            print(
                f"{r['symbol']:5}  {str(r['underlying_quote_ok']):18}  "
                f"{r['option_quotes_received']:>4}/{r['subscribed']:<4}  "
                f"{r['missing_quote_rate']:.2%:>16}  "
                f"{stale_text:>16}  "
                f"{spread_text:>18}  "
                f"{r['handshake_latency_ms']}"
            )

    if failures > args.max_failures:
        print(f"\nFailures {failures} exceeded --max-failures={args.max_failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
