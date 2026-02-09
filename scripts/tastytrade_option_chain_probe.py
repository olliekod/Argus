"""Probe Tastytrade nested option chains.

Usage:
  python scripts/tastytrade_option_chain_probe.py --underlying IBIT
"""

import argparse
import sys

from src.core.config import ConfigurationError, load_config, load_secrets
from src.connectors.tastytrade_rest import (
    RetryConfig,
    TastytradeError,
    TastytradeRestClient,
    normalize_nested_option_chain,
    parse_rfc3339_nano,
)


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("PASTE_") or value.startswith("YOUR_")


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
        print("Tastytrade credentials missing; skipping probe.")
        return 0

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
        chain = client.list_nested_option_chains(args.underlying)
    except TastytradeError as exc:
        client.close()
        print(f"Tastytrade auth failed: {exc}")
        return 1

    normalized = normalize_nested_option_chain(chain, args.underlying)
    expirations = {item["expiration"] for item in normalized if item["expiration"]}
    calls = sum(1 for item in normalized if item["option_type"] == "call")
    puts = sum(1 for item in normalized if item["option_type"] == "put")
    print(f"Underlying: {args.underlying}")
    print(f"Expirations: {len(expirations)}")
    print(f"Contracts: {len(normalized)} (calls={calls}, puts={puts})")

    parsed_samples = 0
    for item in normalized:
        raw = item["expiration"]
        if isinstance(raw, str) and ("T" in raw or "Z" in raw or "+" in raw):
            parse_rfc3339_nano(raw)
            parsed_samples += 1
            if parsed_samples >= 1:
                break
    print(f"Timestamp parse checks: {parsed_samples} sample(s)")

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
