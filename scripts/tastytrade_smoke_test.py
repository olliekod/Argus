import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import ConfigurationError, load_config, load_secrets
from src.connectors.tastytrade_rest import (
    RetryConfig,
    TastytradeError,
    TastytradeRestClient,
)


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
        accounts = client.get_accounts() or []
    except TastytradeError as exc:
        client.close()
        print(f"Tastytrade auth failed: {exc}")
        sys.exit(1)

    masked_accounts = []
    for account in accounts:
        account_id = (
            account.get("account-number")
            or account.get("account_number")
            or account.get("id")
            or ""
        )
        if account_id:
            masked_accounts.append(f"****{str(account_id)[-4:]}")
        else:
            masked_accounts.append("****")
    print(f"Accounts returned: {len(accounts)}")
    if masked_accounts:
        print(f"Account IDs: {', '.join(masked_accounts)}")

    for symbol in ("IBIT", "BITO"):
        chain = client.get_option_chain(symbol)
        print(f"Chain {symbol}: {bool(chain)}")

    client.close()


if __name__ == "__main__":
    main()
