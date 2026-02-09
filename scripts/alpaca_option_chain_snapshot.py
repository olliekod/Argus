#!/usr/bin/env python3
"""
Options Data Quality Probe for Alpaca.

How to run:
  python scripts/alpaca_option_chain_snapshot.py --underlying IBIT --limit 200
  python scripts/alpaca_option_chain_snapshot.py --underlying BITO --limit 200 --batch-size 50
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import time
from typing import Iterable

import requests
import yaml


RATE_LIMIT_HEADER = "x-ratelimit-remaining"


@dataclass
class AuditThresholds:
    quote_age_p99: float
    spread_bps_p95: float


@dataclass
class QuoteMetrics:
    quote_age_seconds: float | None
    trade_age_seconds: float | None
    spread_abs: float | None
    spread_bps: float | None
    bid_zero: bool
    ask_zero: bool
    missing_latest_quote: bool


def parse_rfc3339_to_datetime(ts_str: str) -> datetime:
    """Parse RFC3339 timestamps with nanoseconds and Z suffix."""
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    if "." in ts_str:
        head, frac = ts_str.split(".", 1)
        frac, tail = frac.split("+", 1) if "+" in frac else (frac, "")
        frac = frac[:6].ljust(6, "0")
        ts_str = f"{head}.{frac}+{tail}" if tail else f"{head}.{frac}"
    return datetime.fromisoformat(ts_str)


def quote_age_seconds(ts_str: str, now: datetime) -> float:
    return (now - parse_rfc3339_to_datetime(ts_str)).total_seconds()


def compute_spread_bps(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    return (ask - bid) / mid * 10000


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def find_secrets_file() -> Path:
    override = os.getenv("ARGUS_SECRETS")
    if override:
        p = Path(override).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"ARGUS_SECRETS set but not found: {p}")

    here = Path(__file__).resolve()
    markers = {"pyproject.toml", "requirements.txt", "main.py", ".git"}
    root = here.parent
    for _ in range(10):
        if any((root / m).exists() for m in markers):
            break
        root = root.parent

    candidates = [
        root / "config" / "secrets.yaml",
        root / "argus" / "config" / "secrets.yaml",
        root / "scripts" / "secrets.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Could not find secrets.yaml. Checked:\n" + "\n".join(str(c) for c in candidates)
    )


def load_alpaca_creds(secrets_path: str) -> dict:
    data = yaml.safe_load(Path(secrets_path).read_text(encoding="utf-8")) or {}

    def grab(section_name: str) -> tuple[str, str] | None:
        sec = data.get(section_name) or {}
        k = sec.get("api_key")
        s = sec.get("api_secret")
        if k and s:
            return k, s
        return None

    return {
        "data": grab("alpaca"),
        "paper": grab("alpaca_paper_trading"),
        "live": grab("alpaca_live_trading") or grab("alpaca_trading"),
    }


def request_get_with_retries(
    url: str,
    headers: dict,
    params: dict | None = None,
    timeout: int = 30,
    max_retries: int = 3,
    log_rate_limit_once: bool = False,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            if log_rate_limit_once and RATE_LIMIT_HEADER in response.headers:
                print(f"{RATE_LIMIT_HEADER}: {response.headers.get(RATE_LIMIT_HEADER)}")
                log_rate_limit_once = False
            if response.status_code in {429, 500, 502, 503, 504}:
                wait = 2 ** (attempt - 1)
                print(f"HTTP {response.status_code} {response.reason}; retrying in {wait}s")
                time.sleep(wait)
                continue
            return response
        except requests.RequestException as exc:
            last_exc = exc
            wait = 2 ** (attempt - 1)
            print(f"Request error: {exc}; retrying in {wait}s")
            time.sleep(wait)

    if last_exc:
        raise last_exc
    raise RuntimeError("Request failed after retries")


def fetch_option_contracts_v2(
    underlying: str,
    key: str,
    secret: str,
    paper: bool = True,
    limit: int = 100,
) -> dict:
    base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    url = f"{base}/v2/options/contracts"
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {"underlying_symbols": underlying, "limit": limit}
    r = request_get_with_retries(url, headers=headers, params=params, log_rate_limit_once=True)
    print(f"\n[CONTRACTS v2] HTTP {r.status_code} {r.reason}  ({url})")
    if r.status_code != 200:
        print("Contracts response (first 2000 chars):")
        print(r.text[:2000])
        r.raise_for_status()
    return r.json()


def fetch_option_snapshots_by_symbols(
    symbols: list[str],
    key: str,
    secret: str,
    log_rate_limit_once: bool = False,
) -> dict:
    url = "https://data.alpaca.markets/v1beta1/options/snapshots"
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {"symbols": ",".join(symbols)}
    r = request_get_with_retries(url, headers=headers, params=params, log_rate_limit_once=log_rate_limit_once)
    print(f"[SNAPSHOTS] HTTP {r.status_code} {r.reason} (symbols={len(symbols)})")
    if r.status_code != 200:
        print("Response (first 2000 chars):")
        print(r.text[:2000])
        r.raise_for_status()
    return r.json()


def fetch_underlying_last_price(symbol: str, key: str, secret: str) -> float | None:
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/snapshot"
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    r = request_get_with_retries(url, headers=headers)
    if r.status_code != 200:
        print("Underlying snapshot response (first 2000 chars):")
        print(r.text[:2000])
        return None
    payload = r.json()
    latest_trade = payload.get("latestTrade") or {}
    if "p" in latest_trade:
        return latest_trade["p"]
    latest_quote = payload.get("latestQuote") or {}
    if "ap" in latest_quote and latest_quote.get("ap"):
        return latest_quote["ap"]
    return None


def preview_snapshots(payload: dict, max_contracts: int = 3) -> None:
    snapshots = payload.get("snapshots") if isinstance(payload, dict) else None
    if not isinstance(snapshots, dict):
        snapshots = payload if isinstance(payload, dict) else {}

    contract_keys = list(snapshots.keys())
    print(f"\nContracts returned: {len(contract_keys)}")
    if not contract_keys:
        print("No contracts returned. This could mean no data access, symbol issue, or plan limitation.")
        print("Top-level keys:", list(payload.keys()) if isinstance(payload, dict) else type(payload))
        return

    print("\nPreviewing a few contracts:\n")
    for k in contract_keys[:max_contracts]:
        snap = snapshots.get(k, {})
        greeks = snap.get("greeks")
        iv = snap.get("impliedVolatility") or snap.get("implied_volatility")
        latest_quote = snap.get("latestQuote") or snap.get("latest_quote")
        latest_trade = snap.get("latestTrade") or snap.get("latest_trade")

        print("=" * 90)
        print("Contract:", k)
        print("Has greeks:", bool(greeks))
        if greeks:
            print("Greeks:", json.dumps(greeks, indent=2)[:800])
        if iv is not None:
            print("IV:", iv)

        if latest_quote:
            print("LatestQuote (trim):", json.dumps(latest_quote, indent=2)[:800])
        else:
            print("LatestQuote: MISSING")

        if latest_trade:
            print("LatestTrade (trim):", json.dumps(latest_trade, indent=2)[:800])
        else:
            print("LatestTrade: MISSING")

    print("=" * 90)


def select_contracts(
    contracts: list[dict],
    limit: int,
    underlying_price: float | None,
) -> list[dict]:
    filtered = [
        c
        for c in contracts
        if c.get("tradable") is True and c.get("status") == "active"
    ]
    if not filtered:
        filtered = contracts

    if underlying_price is None:
        return filtered[:limit]

    def strike_distance(contract: dict) -> float:
        strike = contract.get("strike_price")
        if strike is None:
            return float("inf")
        return abs(float(strike) - underlying_price)

    filtered.sort(key=strike_distance)
    return filtered[:limit]


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def build_metrics_for_snapshot(snapshot: dict, now: datetime) -> QuoteMetrics:
    latest_quote = snapshot.get("latestQuote") or snapshot.get("latest_quote")
    latest_trade = snapshot.get("latestTrade") or snapshot.get("latest_trade")

    if latest_quote and latest_quote.get("t"):
        quote_age = quote_age_seconds(latest_quote["t"], now)
        bid = latest_quote.get("bp")
        ask = latest_quote.get("ap")
    else:
        quote_age = None
        bid = None
        ask = None

    trade_age = None
    if latest_trade and latest_trade.get("t"):
        trade_age = quote_age_seconds(latest_trade["t"], now)

    spread_abs = None
    spread_bps = None
    if bid is not None and ask is not None:
        spread_abs = ask - bid
        spread_bps = compute_spread_bps(bid, ask)

    return QuoteMetrics(
        quote_age_seconds=quote_age,
        trade_age_seconds=trade_age,
        spread_abs=spread_abs,
        spread_bps=spread_bps,
        bid_zero=(bid == 0),
        ask_zero=(ask == 0),
        missing_latest_quote=(latest_quote is None),
    )


def audit_options_data(
    underlying: str,
    contracts: list[dict],
    data_key: str,
    data_secret: str,
    limit: int,
    batch_size: int,
    spread_bps_buckets: list[float],
    thresholds: AuditThresholds,
) -> None:
    print("\n[Audit] Starting options data quality probe...")
    underlying_price = fetch_underlying_last_price(underlying, data_key, data_secret)
    if underlying_price is not None:
        print(f"Underlying last price: {underlying_price}")
    else:
        print("Underlying last price unavailable; using first N contracts.")

    selected_contracts = select_contracts(contracts, limit, underlying_price)
    symbols = [c["symbol"] for c in selected_contracts]

    print(f"Selected contracts: {len(symbols)} (limit={limit})")

    snapshots: dict[str, dict] = {}
    for idx, batch in enumerate(chunked(symbols, batch_size), start=1):
        payload = fetch_option_snapshots_by_symbols(
            batch,
            data_key,
            data_secret,
            log_rate_limit_once=(idx == 1),
        )
        batch_snaps = payload.get("snapshots") if isinstance(payload, dict) else None
        if isinstance(batch_snaps, dict):
            snapshots.update(batch_snaps)

    now = datetime.now(timezone.utc)
    quote_ages: list[float] = []
    spread_bps_values: list[float] = []
    worst_spreads: list[tuple[str, float]] = []
    worst_quotes: list[tuple[str, float]] = []

    missing_snapshots = 0
    missing_quotes = 0
    bid_zero_count = 0
    ask_zero_count = 0

    for symbol in symbols:
        snapshot = snapshots.get(symbol)
        if snapshot is None:
            missing_snapshots += 1
            continue

        metrics = build_metrics_for_snapshot(snapshot, now)
        if metrics.missing_latest_quote:
            missing_quotes += 1
        if metrics.bid_zero:
            bid_zero_count += 1
        if metrics.ask_zero:
            ask_zero_count += 1

        if metrics.quote_age_seconds is not None:
            quote_ages.append(metrics.quote_age_seconds)
            worst_quotes.append((symbol, metrics.quote_age_seconds))
        if metrics.spread_bps is not None:
            spread_bps_values.append(metrics.spread_bps)
            worst_spreads.append((symbol, metrics.spread_bps))

    worst_spreads.sort(key=lambda x: x[1], reverse=True)
    worst_quotes.sort(key=lambda x: x[1], reverse=True)

    def percentile_report(values: list[float], label: str) -> dict[str, float | None]:
        pcts = {"p50": 50, "p90": 90, "p95": 95, "p99": 99}
        report = {k: percentile(values, v) for k, v in pcts.items()}
        report["max"] = max(values) if values else None
        print(f"\n{label} percentiles:")
        for k in ["p50", "p90", "p95", "p99", "max"]:
            val = report[k]
            display = f"{val:.4f}" if isinstance(val, (float, int)) else "n/a"
            print(f"  {k}: {display}")
        return report

    quote_report = percentile_report(quote_ages, "Quote age (seconds)")
    spread_report = percentile_report(spread_bps_values, "Spread (bps)")

    print("\nCounts:")
    print(f"  Total contracts sampled: {len(symbols)}")
    print(f"  Snapshots returned: {len(snapshots)}")
    print(f"  Missing snapshots: {missing_snapshots}")
    print(f"  Missing latestQuote: {missing_quotes}")
    print(f"  bid==0 count: {bid_zero_count}")
    print(f"  ask==0 count: {ask_zero_count}")

    print("\nLiquidity sanity (% of contracts with spread_bps <= threshold):")
    for bucket in spread_bps_buckets:
        if spread_bps_values:
            pct = sum(1 for v in spread_bps_values if v <= bucket) / len(spread_bps_values) * 100
            print(f"  <= {bucket:.0f} bps: {pct:.2f}%")
        else:
            print(f"  <= {bucket:.0f} bps: n/a")

    print("\nWorst 10 by spread_bps:")
    for symbol, value in worst_spreads[:10]:
        print(f"  {symbol}: {value:.2f} bps")

    print("\nWorst 10 by quote_age_seconds:")
    for symbol, value in worst_quotes[:10]:
        print(f"  {symbol}: {value:.3f} s")

    reasons: list[str] = []
    p99_quote = quote_report["p99"]
    p95_spread = spread_report["p95"]
    if p99_quote is None:
        reasons.append("p99 quote age unavailable")
    elif p99_quote > thresholds.quote_age_p99:
        reasons.append(f"p99 quote age {p99_quote:.3f}s > {thresholds.quote_age_p99:.3f}s")

    if p95_spread is None:
        reasons.append("p95 spread bps unavailable")
    elif p95_spread > thresholds.spread_bps_p95:
        reasons.append(f"p95 spread {p95_spread:.2f} bps > {thresholds.spread_bps_p95:.2f} bps")

    if reasons:
        print("\nConclusion: FAIL")
        for reason in reasons:
            print(f"  - {reason}")
    else:
        print("\nConclusion: PASS")


def parse_spread_bps_buckets(raw: str) -> list[float]:
    try:
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid spread bucket list: {raw}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Options Data Quality Probe for Alpaca")
    parser.add_argument("--underlying", default="IBIT")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--use-paper-contracts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quote-age-p99-threshold", type=float, default=3.0)
    parser.add_argument("--spread-bps-p95-threshold", type=float, default=200)
    parser.add_argument(
        "--spread-bps-buckets",
        type=parse_spread_bps_buckets,
        default=[25.0, 50.0, 100.0],
    )
    args = parser.parse_args()

    secrets_path = find_secrets_file()
    creds = load_alpaca_creds(str(secrets_path))

    if not creds.get("data"):
        raise RuntimeError(f"Missing 'alpaca:' api_key/api_secret in {secrets_path}")
    data_key, data_secret = creds["data"]

    if args.use_paper_contracts:
        if not creds.get("paper"):
            raise RuntimeError(f"Missing 'alpaca_paper_trading:' api_key/api_secret in {secrets_path}")
        trade_key, trade_secret = creds["paper"]
    else:
        if not creds.get("live"):
            raise RuntimeError(f"Missing live trading creds in {secrets_path}")
        trade_key, trade_secret = creds["live"]

    print(f"Loaded Alpaca creds from: {secrets_path}")

    contracts_payload = fetch_option_contracts_v2(
        args.underlying,
        trade_key,
        trade_secret,
        paper=args.use_paper_contracts,
        limit=max(args.limit, 1),
    )
    contracts = contracts_payload.get("option_contracts", [])
    if not contracts:
        raise RuntimeError("No option contracts returned; check API access and underlying symbol.")

    example_symbols = [c["symbol"] for c in contracts[:5]]
    example_payload = fetch_option_snapshots_by_symbols(example_symbols, data_key, data_secret, log_rate_limit_once=True)
    preview_snapshots(example_payload, max_contracts=3)

    thresholds = AuditThresholds(
        quote_age_p99=args.quote_age_p99_threshold,
        spread_bps_p95=args.spread_bps_p95_threshold,
    )

    audit_options_data(
        underlying=args.underlying,
        contracts=contracts,
        data_key=data_key,
        data_secret=data_secret,
        limit=args.limit,
        batch_size=args.batch_size,
        spread_bps_buckets=args.spread_bps_buckets,
        thresholds=thresholds,
    )


if __name__ == "__main__":
    main()
