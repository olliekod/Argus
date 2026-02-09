"""Probe Tastytrade nested option chains.

Usage:
  python scripts/tastytrade_option_chain_probe.py --underlying IBIT
"""

import argparse
import sys
from typing import Any

from src.core.config import ConfigurationError, load_config, load_secrets
from src.core.options_normalize import normalize_tastytrade_nested_chain
from src.connectors.tastytrade_rest import (
    RetryConfig,
    TastytradeError,
    TastytradeRestClient,
)


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("PASTE_") or value.startswith("YOUR_")


def _extract_underlying_price(payload: dict[str, Any]) -> float | None:
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        return None
    for key in (
        "underlying-price",
        "underlying-price-value",
        "underlying-mark",
        "underlying_price",
    ):
        value = data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _sample_contracts(contracts: list[dict[str, Any]], underlying_price: float | None) -> list[dict[str, Any]]:
    if not contracts:
        return []
    if underlying_price is None:
        return contracts[:10]

    def distance(contract: dict[str, Any]) -> float:
        strike = contract.get("strike")
        if strike is None:
            return float("inf")
        return abs(strike - underlying_price)

    ranked = sorted(contracts, key=distance)
    return ranked[:10]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--underlying", default="IBIT")
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
        client.login()
        chain = client.get_nested_option_chains(args.underlying)
    except TastytradeError as exc:
        client.close()
        print(f"Tastytrade request failed: {exc}")
        return 1

    normalized = normalize_tastytrade_nested_chain(chain)
    expirations = sorted({item["expiry"] for item in normalized if item.get("expiry")})
    strikes = {item["strike"] for item in normalized if item.get("strike") is not None}
    calls = sum(1 for item in normalized if item.get("right") == "C")
    puts = sum(1 for item in normalized if item.get("right") == "P")

    print(f"Underlying: {args.underlying}")
    print(f"Expirations count: {len(expirations)}")
    print(f"Strikes count: {len(strikes)}")
    print(f"Contracts count: {len(normalized)} (calls={calls}, puts={puts})")

    if expirations:
        print(f"First expiry: {expirations[0]}")
        print(f"Last expiry: {expirations[-1]}")
    else:
        print("First expiry: n/a")
        print("Last expiry: n/a")

    underlying_price = _extract_underlying_price(chain)
    if underlying_price is not None:
        print(f"Underlying price: {underlying_price:.2f}")

    samples = _sample_contracts(normalized, underlying_price)
    print("Sample contracts:")
    for contract in samples:
        symbol = contract.get("option_symbol") or "?"
        expiry = contract.get("expiry") or "?"
        right = contract.get("right") or "?"
        strike = contract.get("strike")
        strike_display = f"{strike:.2f}" if strike is not None else "?"
        print(f"  {symbol} {expiry} {right}{strike_display}")

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
