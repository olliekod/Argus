from src.core.config import ConfigurationError, load_config, load_secrets
from src.connectors.tastytrade_rest import RetryConfig, TastytradeRestClient


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("PASTE_") or value.startswith("YOUR_")


def main() -> None:
    try:
        config = load_config()
        secrets = load_secrets()
    except ConfigurationError as exc:
        print(f"Config error: {exc}")
        return

    tasty_secrets = secrets.get("tastytrade", {})
    username = tasty_secrets.get("username", "")
    password = tasty_secrets.get("password", "")
    if _is_placeholder(username) or _is_placeholder(password):
        print("Tastytrade credentials missing; skipping smoke test.")
        return

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
        accounts = client.get_accounts()
        print(f"Accounts returned: {len(accounts) if accounts else 0}")
        for symbol in ("IBIT", "BITO"):
            chain = client.get_option_chain(symbol)
            print(f"Chain {symbol}: {bool(chain)}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
