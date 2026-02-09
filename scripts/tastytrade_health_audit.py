#!/usr/bin/env python3
"""Tastytrade health audit and option chain sanity checks."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import ConfigurationError, load_config, load_secrets
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
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
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
        expiry_value = exp.get("expiration-date") or exp.get("expiration") or exp.get("date")
        expiry = _parse_expiry(expiry_value)
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
                missing_strikes_total += 1
            if put and not call:
                missing_strikes_total += 1

            if call:
                strike_side_counts["call"] += 1
                bid, ask = _extract_bid_ask(call)
                spread = _compute_spread_bps(bid, ask)
                if spread is not None:
                    spread_values.append(spread)
            if put:
                strike_side_counts["put"] += 1
                bid, ask = _extract_bid_ask(put)
                spread = _compute_spread_bps(bid, ask)
                if spread is not None:
                    spread_values.append(spread)

    spread_values.sort()
    spread_p95 = None
    if spread_values:
        idx = int(len(spread_values) * 0.95)
        spread_p95 = spread_values[min(idx, len(spread_values) - 1)]

    return {
        "expirations_total": len(expirations),
        "expired_chains": expired,
        "strike_counts": strike_counts,
        "missing_strikes": missing_strikes_total,
        "side_counts": dict(strike_side_counts),
        "spread_samples": len(spread_values),
        "spread_p95_bps": spread_p95,
    }


def audit_oauth(secrets: dict[str, Any]) -> dict[str, Any]:
    oauth_cfg = secrets.get("tastytrade_oauth2", {}) or {}
    client_id = oauth_cfg.get("client_id", "")
    client_secret = oauth_cfg.get("client_secret", "")
    refresh_token = oauth_cfg.get("refresh_token", "")

    if _is_placeholder(client_id) or _is_placeholder(client_secret) or _is_placeholder(refresh_token):
        raise RuntimeError("Missing tastytrade_oauth2 client_id/client_secret/refresh_token.")

    client = TastytradeOAuthClient(
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=refresh_token,
    )
    start = time.perf_counter()
    token_result = client.refresh_access_token()
    latency = time.perf_counter() - start
    return {
        "access_token_present": True,
        "refresh_token_present": bool(token_result.refresh_token),
        "expires_in": token_result.expires_in,
        "token_type": token_result.token_type,
        "latency_s": round(latency, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Tastytrade health audit.")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol (default: SPY)")
    args = parser.parse_args()

    try:
        config = load_config()
        secrets = load_secrets()
    except ConfigurationError as exc:
        print(f"Config error: {exc}")
        return 1

    tasty_secrets = secrets.get("tastytrade", {})
    username = tasty_secrets.get("username", "")
    password = tasty_secrets.get("password", "")
    if _is_placeholder(username) or _is_placeholder(password):
        print("Tastytrade credentials missing; update config/secrets.yaml.")
        return 1

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

    try:
        print(f"Logging into Tastytrade ({args.symbol})...")
        login_start = time.perf_counter()
        client.login()
        print(f"Login OK ({time.perf_counter() - login_start:.2f}s)")

        chain_start = time.perf_counter()
        chain = client.get_nested_option_chains(args.symbol)
        latency = time.perf_counter() - chain_start
        normalized = normalize_tastytrade_nested_chain(chain)
        print(f"Nested chain fetched ({latency:.2f}s). Normalized contracts: {len(normalized)}")

        audit = audit_nested_chain(chain)
        print("\n--- Chain Summary ---")
        print(f"Expirations: {audit['expirations_total']}")
        print(f"Expired chains: {audit['expired_chains']}")
        if audit["strike_counts"]:
            print(f"Strike counts (min/avg/max): {min(audit['strike_counts'])}/"
                  f"{sum(audit['strike_counts'])/len(audit['strike_counts']):.1f}/"
                  f"{max(audit['strike_counts'])}")
        print(f"Missing strikes (call/put gaps): {audit['missing_strikes']}")
        print(f"Call contracts: {audit['side_counts'].get('call', 0)}")
        print(f"Put contracts: {audit['side_counts'].get('put', 0)}")
        if audit["spread_samples"]:
            print(f"Spread p95 (bps): {audit['spread_p95_bps']:.1f}")
        else:
            print("Spread metrics: no bid/ask samples found in response.")

        print("\n--- OAuth Refresh ---")
        try:
            oauth_result = audit_oauth(secrets)
            ttl = oauth_result.get("expires_in")
            ttl_text = f"{ttl}s" if ttl is not None else "unknown"
            print(f"Access token refresh OK (TTL {ttl_text}, {oauth_result['latency_s']}s)")
        except Exception as exc:
            print(f"OAuth refresh failed: {exc}")
            return 1

    except TastytradeError as exc:
        print(f"Tastytrade request failed: {exc}")
        return 1
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
