#!/usr/bin/env python3
"""Tastytrade health audit and deterministic DXLink quote probes."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
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

from src.connectors.tastytrade_oauth import TastytradeOAuthClient
from src.connectors.tastytrade_rest import RetryConfig, TastytradeError, TastytradeRestClient
from src.core.config import ConfigurationError, load_config, load_secrets
from src.core.liquid_etf_universe import LIQUID_ETF_UNIVERSE
from src.core.options_normalize import normalize_tastytrade_nested_chain


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("PASTE_") or value.startswith("YOUR_")


def _parse_expiry(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        raw = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            try:
                return datetime.fromisoformat(raw.split("T")[0]).date()
            except ValueError:
                return None
        return parsed.date()
    return None


def _compute_spread_bps(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid * 10000


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def _extract_event_ts(quote: dict[str, Any]) -> float | None:
    ts = quote.get("eventTime") or quote.get("time") or quote.get("event_ts")
    try:
        return float(ts) if ts is not None else None
    except (TypeError, ValueError):
        return None


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
        if isinstance(chain, dict):
            expirations.extend(chain.get("expirations") or [])
    return [item for item in expirations if isinstance(item, dict)]


def audit_nested_chain(raw: dict[str, Any]) -> dict[str, Any]:
    expirations = _extract_chain_expirations(raw)
    today = date.today()
    expired = 0
    missing_strikes = 0
    strike_counts: list[int] = []
    side_counts = Counter()

    for exp in expirations:
        expiry = _parse_expiry(exp.get("expiration-date") or exp.get("expiration") or exp.get("date"))
        if expiry and expiry < today:
            expired += 1
        strikes = exp.get("strikes") or exp.get("strike-prices") or []
        strike_counts.append(len(strikes))
        for strike in strikes:
            if not isinstance(strike, dict):
                continue
            call = strike.get("call")
            put = strike.get("put")
            if call and not put:
                missing_strikes += 1
            if put and not call:
                missing_strikes += 1
            if call:
                side_counts["call"] += 1
            if put:
                side_counts["put"] += 1

    return {
        "expirations_total": len(expirations),
        "expired_chains": expired,
        "strike_counts": strike_counts,
        "missing_strikes": missing_strikes,
        "side_counts": dict(side_counts),
    }


def select_spot(
    dxlink_underlying_quote: dict[str, Any] | None,
    cli_spot: float | None,
    strike_values: list[float],
) -> dict[str, Any]:
    if dxlink_underlying_quote:
        bid = dxlink_underlying_quote.get("bidPrice")
        ask = dxlink_underlying_quote.get("askPrice")
        last = dxlink_underlying_quote.get("price") or dxlink_underlying_quote.get("lastPrice")
        mid = None
        try:
            if bid is not None and ask is not None:
                bid_f, ask_f = float(bid), float(ask)
                if bid_f > 0 and ask_f > 0:
                    mid = (bid_f + ask_f) / 2.0
        except (TypeError, ValueError):
            mid = None
        if mid is None and last is not None:
            try:
                mid = float(last)
            except (TypeError, ValueError):
                mid = None
        if mid is not None:
            return {
                "spot_source": "dxlink",
                "spot_value": mid,
                "spot_event_ts": _extract_event_ts(dxlink_underlying_quote),
                "spot_recv_ts": dxlink_underlying_quote.get("_recv_ts"),
                "warning": None,
            }

    if cli_spot is not None:
        return {
            "spot_source": "cli",
            "spot_value": float(cli_spot),
            "spot_event_ts": None,
            "spot_recv_ts": None,
            "warning": None,
        }

    median_strike = sorted(strike_values)[len(strike_values) // 2] if strike_values else None
    return {
        "spot_source": "median_strike",
        "spot_value": median_strike,
        "spot_event_ts": None,
        "spot_recv_ts": None,
        "warning": "WARNING: spot fallback used median strike because DXLink/CLI spot were unavailable.",
    }


def _select_sampled_contracts(
    contracts: list[dict[str, Any]],
    spot_value: float | None,
    now_utc: datetime,
    expiry_count: int = 2,
    strike_window: int = 5,
    max_contracts: int = 80,
) -> list[dict[str, Any]]:
    active = [
        c for c in contracts
        if c.get("expiry") and c.get("strike") is not None and c.get("right") in {"C", "P"} and c.get("option_symbol")
    ]
    valid_expiries = sorted(
        {
            c["expiry"]
            for c in active
            if _parse_expiry(c["expiry"]) is not None and _parse_expiry(c["expiry"]) >= now_utc.date()
        }
    )[:expiry_count]
    eligible = [c for c in active if c.get("expiry") in valid_expiries]
    strikes = sorted({float(c["strike"]) for c in eligible})
    if not strikes:
        return []

    if spot_value is None:
        spot_value = strikes[len(strikes) // 2]
    center = min(strikes, key=lambda s: (abs(s - spot_value), s))
    center_idx = strikes.index(center)
    lo = max(0, center_idx - strike_window)
    hi = min(len(strikes), center_idx + strike_window + 1)
    selected_strikes = set(strikes[lo:hi])

    sampled = [c for c in eligible if float(c["strike"]) in selected_strikes]
    sampled.sort(key=lambda c: (c.get("expiry") or "", float(c.get("strike", 0.0)), c.get("right") or "", c.get("option_symbol") or ""))
    return sampled[:max_contracts]


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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_option_quote_snapshots_underlying_ts ON option_quote_snapshots(underlying, ts_utc)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_option_quote_snapshots_symbol_ts ON option_quote_snapshots(option_symbol, ts_utc)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_option_quote_snapshots_provider ON option_quote_snapshots(provider)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_option_quote_snapshots_contract ON option_quote_snapshots(expiry, strike, right)")
        conn.commit()
    finally:
        conn.close()


def _append_snapshots(sqlite_path: Path, rows: list[tuple[Any, ...]]) -> None:
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


def _prune_snapshots_sql(days: int) -> tuple[str, tuple[Any, ...]]:
    sql = "DELETE FROM option_quote_snapshots WHERE ts_utc < datetime('now', ?)"
    return sql, (f"-{int(days)} days",)


def audit_oauth(secrets: dict[str, Any]) -> dict[str, Any]:
    oauth_cfg = secrets.get("tastytrade_oauth2", {}) or {}
    client_id = oauth_cfg.get("client_id", "")
    client_secret = oauth_cfg.get("client_secret", "")
    refresh_token = oauth_cfg.get("refresh_token", "")
    if _is_placeholder(client_id) or _is_placeholder(client_secret) or _is_placeholder(refresh_token):
        raise RuntimeError("SKIP: not configured (tastytrade_oauth2 client_id/client_secret/refresh_token).")
    client = TastytradeOAuthClient(client_id=client_id, client_secret=client_secret, refresh_token=refresh_token)
    start = time.perf_counter()
    token_result = client.refresh_access_token()
    return {
        "access_token": token_result.access_token,
        "latency_s": round(time.perf_counter() - start, 3),
    }


async def run_quotes_probe(
    symbol: str,
    chain_contracts: list[dict[str, Any]],
    access_token: str,
    duration: int,
    persist_path: Path | None = None,
    cli_spot: float | None = None,
    expiry_count: int = 2,
    strike_window: int = 5,
    max_contracts: int = 80,
    max_snapshots_per_underlying_per_run: int = 200,
    force: bool = False,
    greeks: bool = False,
) -> dict[str, Any]:
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
    initial_sample = _select_sampled_contracts(chain_contracts, None, now_utc, expiry_count, strike_window, max_contracts)
    candidate_symbols = [c["option_symbol"] for c in initial_sample]

    quote_seen: dict[str, dict[str, Any]] = {}
    greeks_seen: set[str] = set()
    spreads: list[float] = []
    stale_ages: list[float] = []
    first_underlying_ms: int | None = None
    first_option_ms: int | None = None
    handshake_latency_ms: int | None = None
    underlying_quote: dict[str, Any] | None = None
    underlying_quote_ok = False
    greeks_supported = greeks

    rows: list[tuple[Any, ...]] = []
    start = time.perf_counter()

    async with websockets.connect(dxlink_url) as ws:
        async def send(payload: dict[str, Any]) -> None:
            await ws.send(json.dumps(payload))

        await send({"type": "SETUP", "channel": 0, "keepaliveTimeout": 60, "acceptDataFormat": "json", "version": "0.1"})
        feed_channel = 1
        event_types = ["Quote"] + (["Greeks"] if greeks else [])

        while handshake_latency_ms is None:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "AUTH_STATE" and msg.get("state") == "UNAUTHORIZED":
                await send({"type": "AUTH", "channel": 0, "token": dxlink_token})
            elif msg.get("type") == "AUTH_STATE" and msg.get("state") == "AUTHORIZED":
                await send({"type": "CHANNEL_REQUEST", "channel": feed_channel, "service": "FEED", "parameters": {"contract": "AUTO"}})
            elif msg.get("type") == "CHANNEL_OPENED":
                feed_channel = msg.get("channel", 1)
                await send({"type": "FEED_SETUP", "channel": feed_channel, "acceptEventTypes": event_types})
                sub = [{"type": "Quote", "symbol": symbol}] + [{"type": "Quote", "symbol": s} for s in candidate_symbols]
                if greeks:
                    sub += [{"type": "Greeks", "symbol": s} for s in candidate_symbols]
                await send({"type": "FEED_SUBSCRIPTION", "channel": feed_channel, "add": sub})
                handshake_latency_ms = int((time.perf_counter() - start) * 1000)
            elif msg.get("type") == "ERROR" and greeks:
                greeks_supported = False

        deadline = time.perf_counter() + duration
        while time.perf_counter() < deadline:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.perf_counter())))
            except asyncio.TimeoutError:
                continue
            if msg.get("type") == "KEEPALIVE":
                await send({"type": "KEEPALIVE", "channel": 0})
                continue
            if msg.get("type") == "ERROR" and greeks:
                greeks_supported = False
                continue
            if msg.get("type") != "FEED_DATA":
                continue
            recv_ts = time.time()
            for item in msg.get("data") or []:
                sym = item.get("eventSymbol") or item.get("symbol")
                kind = item.get("eventType")
                if not sym:
                    continue
                if sym == symbol and (kind in (None, "Quote")):
                    underlying_quote_ok = True
                    item["_recv_ts"] = recv_ts
                    underlying_quote = item
                    if first_underlying_ms is None:
                        first_underlying_ms = int((time.perf_counter() - start) * 1000)
                    continue
                if sym not in candidate_symbols:
                    continue
                if kind == "Greeks":
                    greeks_seen.add(sym)
                    continue
                quote_seen[sym] = item
                if first_option_ms is None:
                    first_option_ms = int((time.perf_counter() - start) * 1000)
                try:
                    bid_f = float(item.get("bidPrice")) if item.get("bidPrice") is not None else None
                    ask_f = float(item.get("askPrice")) if item.get("askPrice") is not None else None
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
                    meta = next((c for c in initial_sample if c.get("option_symbol") == sym), {})
                    rows.append((datetime.now(timezone.utc).isoformat(), "tastytrade-dxlink", symbol, sym, meta.get("expiry"), meta.get("strike"), meta.get("right"), bid_f, ask_f, mid, event_ts, recv_ts))

    strikes = sorted({float(c["strike"]) for c in chain_contracts if c.get("strike") is not None})
    spot_meta = select_spot(underlying_quote, cli_spot, strikes)
    sampled_contracts = _select_sampled_contracts(chain_contracts, spot_meta["spot_value"], now_utc, expiry_count, strike_window, max_contracts)
    sampled_symbols = [c["option_symbol"] for c in sampled_contracts]

    if persist_path and rows:
        if len(rows) > max_snapshots_per_underlying_per_run and not force:
            raise RuntimeError(
                f"Refusing snapshot persistence: {len(rows)} rows exceeds max_snapshots_per_underlying_per_run={max_snapshots_per_underlying_per_run}. Use --force to override."
            )
        _ensure_snapshot_table(persist_path)
        _append_snapshots(persist_path, rows)

    received = len([s for s in sampled_symbols if s in quote_seen])
    subscribed = len(sampled_symbols)
    missing_rate = 1.0 - (received / subscribed) if subscribed else 1.0
    greeks_presence = (len([s for s in sampled_symbols if s in greeks_seen]) / subscribed) if subscribed else 0.0

    out = {
        "symbol": symbol,
        "underlying_quote_ok": underlying_quote_ok,
        "option_quotes_received": received,
        "subscribed": subscribed,
        "missing_quote_rate": round(missing_rate, 4),
        "stale_age_sec_p95": _percentile(stale_ages, 95),
        "spread_bps_p50": _percentile(spreads, 50),
        "spread_bps_p95": _percentile(spreads, 95),
        "handshake_latency_ms": handshake_latency_ms,
        "time_to_first_underlying_quote_ms": first_underlying_ms,
        "time_to_first_option_quote_ms": first_option_ms,
        "sampled_contract_count": subscribed,
        "sampled_symbols_preview": sampled_symbols[:3],
        "spot_source": spot_meta["spot_source"],
        "spot_value": spot_meta["spot_value"],
        "spot_event_ts": spot_meta["spot_event_ts"],
        "spot_recv_ts": spot_meta["spot_recv_ts"],
        "greeks_supported": greeks_supported,
        "greeks_presence_rate": round(greeks_presence, 4),
        "greeks_todo": (
            "Greeks missing; fallback path is IV solve + Black–Scholes (derived greeks)"
            if greeks and greeks_presence == 0.0
            else None
        ),
    }
    if spot_meta.get("warning"):
        out["warning"] = spot_meta["warning"]
    return out


def _load_tasty_client(config: dict[str, Any], secrets: dict[str, Any]) -> TastytradeRestClient:
    tasty_secrets = secrets.get("tastytrade", {})
    username = tasty_secrets.get("username", "")
    password = tasty_secrets.get("password", "")
    if _is_placeholder(username) or _is_placeholder(password):
        raise RuntimeError("SKIP: not configured (tastytrade.username/password).")

    tt_config = config.get("tastytrade", {})
    retry_cfg = tt_config.get("retries", {})
    client = TastytradeRestClient(
        username=username,
        password=password,
        environment=tt_config.get("environment", "live"),
        timeout_seconds=tt_config.get("timeout_seconds", 20),
        retries=RetryConfig(
            max_attempts=retry_cfg.get("max_attempts", 3),
            backoff_seconds=retry_cfg.get("backoff_seconds", 1.0),
            backoff_multiplier=retry_cfg.get("backoff_multiplier", 2.0),
        ),
    )
    client.login()
    return client


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Tastytrade health audit.")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol (default: SPY)")
    parser.add_argument("--universe", action="store_true", help="Audit the liquid ETF universe")
    parser.add_argument("--quotes", action="store_true", help="Run DXLink quote probe")
    parser.add_argument("--greeks", action="store_true", help="Attempt Greeks events")
    parser.add_argument("--require-greeks", action="store_true", help="Fail when greeks are unavailable")
    parser.add_argument("--duration", type=int, default=10, help="Probe duration (seconds)")
    parser.add_argument("--spot", type=float, default=None, help="User-provided spot fallback")
    parser.add_argument("--expiry-count", type=int, default=2)
    parser.add_argument("--strike-window", type=int, default=5)
    parser.add_argument("--max-contracts", type=int, default=80)
    parser.add_argument("--max-failures", type=int, default=2)
    parser.add_argument("--persist-snapshots", action="store_true")
    parser.add_argument("--snapshots-db", default="data/argus.db")
    parser.add_argument("--max-snapshots-per-underlying-per-run", type=int, default=200)
    parser.add_argument("--force", action="store_true", help="Override snapshot guardrails")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    live_probe_requested = args.quotes or args.greeks

    try:
        config = load_config()
        secrets = load_secrets()
    except ConfigurationError as exc:
        print(f"SKIP: not configured ({exc})")
        return 1 if live_probe_requested else 0

    symbols = list(LIQUID_ETF_UNIVERSE) if args.universe else [args.symbol.upper()]

    try:
        client = _load_tasty_client(config, secrets)
    except Exception as exc:
        print(str(exc))
        return 1 if live_probe_requested else 0

    oauth = None
    if args.quotes or args.greeks:
        try:
            oauth = audit_oauth(secrets)
        except Exception as exc:
            print(str(exc))
            client.close()
            return 1 if live_probe_requested else 0

    failures = 0
    results: list[dict[str, Any]] = []
    try:
        for symbol in symbols:
            chain = client.get_nested_option_chains(symbol)
            normalized = normalize_tastytrade_nested_chain(chain)
            chain_audit = audit_nested_chain(chain)
            row: dict[str, Any] = {"symbol": symbol, "chain_audit": chain_audit, "normalized_contracts": len(normalized)}
            if args.quotes or args.greeks:
                probe = await run_quotes_probe(
                    symbol=symbol,
                    chain_contracts=normalized,
                    access_token=oauth["access_token"],
                    duration=args.duration,
                    persist_path=Path(args.snapshots_db) if args.persist_snapshots else None,
                    cli_spot=args.spot,
                    expiry_count=args.expiry_count,
                    strike_window=args.strike_window,
                    max_contracts=args.max_contracts,
                    max_snapshots_per_underlying_per_run=args.max_snapshots_per_underlying_per_run,
                    force=args.force,
                    greeks=args.greeks,
                )
                row.update(probe)
                if not row.get("underlying_quote_ok"):
                    failures += 1
                if args.require_greeks and row.get("greeks_presence_rate", 0.0) == 0.0:
                    failures += 1
            results.append(row)
    except (TastytradeError, RuntimeError, requests.RequestException) as exc:
        print(f"Audit failed: {exc}")
        failures += 1
    finally:
        client.close()

    if args.universe and (args.quotes or args.greeks):
        ranked = sorted(results, key=lambda r: (r.get("missing_quote_rate", 1.0), -(r.get("option_quotes_received", 0)), r["symbol"]))
        print("symbol  underlying_quote_ok  received/subscribed  missing_rate  stale_p95  spread_p95  handshake_ms")
        for r in ranked:
            print(
                f"{r['symbol']:5}  {str(r.get('underlying_quote_ok', False)):18}  "
                f"{r.get('option_quotes_received', 0):>4}/{r.get('subscribed', 0):<4}  "
                f"{r.get('missing_quote_rate', 1.0):>11.2%}  "
                f"{str(round(r['stale_age_sec_p95'], 3)) if r.get('stale_age_sec_p95') is not None else 'n/a':>8}  "
                f"{str(round(r['spread_bps_p95'], 1)) if r.get('spread_bps_p95') is not None else 'n/a':>10}  "
                f"{r.get('handshake_latency_ms')}"
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "args": vars(args),
        "results": results,
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    if failures > args.max_failures:
        print(f"Failures {failures} exceeded --max-failures={args.max_failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
