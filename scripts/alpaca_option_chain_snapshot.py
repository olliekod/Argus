#!/usr/bin/env python3
from pathlib import Path
import os
import requests
import json
import yaml
from datetime import datetime, timezone

def age_seconds(ts_str: str) -> float:
    # handles nanoseconds by truncating to microseconds
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    # python can't parse 9-digit fractional seconds directly; trim to 6
    if "." in ts_str:
        head, frac = ts_str.split(".", 1)
        frac, tail = frac.split("+", 1) if "+" in frac else (frac, "")
        frac = frac[:6].ljust(6, "0")
        ts_str = f"{head}.{frac}+{tail}" if tail else f"{head}.{frac}"
    dt = datetime.fromisoformat(ts_str)
    return (datetime.now(timezone.utc) - dt).total_seconds()

def find_secrets_file() -> Path:
    # 1) optional override
    override = os.getenv("ARGUS_SECRETS")
    if override:
        p = Path(override).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"ARGUS_SECRETS set but not found: {p}")

    # 2) find repo root by walking up until we see a marker
    here = Path(__file__).resolve()
    markers = {"pyproject.toml", "requirements.txt", "main.py", ".git"}
    root = here.parent
    for _ in range(10):
        if any((root / m).exists() for m in markers):
            break
        root = root.parent

    # 3) common locations
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

    creds = {
        "data": grab("alpaca"),  # what you used successfully for data endpoint
        "paper": grab("alpaca_paper_trading"),
        "live": grab("alpaca_live_trading") or grab("alpaca_trading"),  # optional if you ever add it
    }
    return creds




def fetch_option_snapshots(symbol: str, key: str, secret: str) -> dict:
    url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    r = requests.get(url, headers=headers, timeout=30)
    print(f"HTTP {r.status_code} {r.reason}")
    # Helpful rate-limit headers if present
    for h in ["x-ratelimit-remaining", "x-ratelimit-limit", "x-ratelimit-reset"]:
        if h in r.headers:
            print(f"{h}: {r.headers.get(h)}")

    if r.status_code != 200:
        # Print first chunk so you can see auth/plan errors
        print("Response (first 2000 chars):")
        print(r.text[:2000])
        r.raise_for_status()

    return r.json()


def preview_snapshots(payload: dict, max_contracts: int = 3) -> None:
    # Alpaca returns a dict keyed by option symbol, or a structure containing that dict.
    # We handle both.
    if isinstance(payload, dict) and "snapshots" in payload and isinstance(payload["snapshots"], dict):
        snapshots = payload["snapshots"]
    else:
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
        iv = snap.get("impliedVolatility") or snap.get("implied_volatility")  # just in case
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


def fetch_option_chain(symbol: str, key: str, secret: str) -> dict:
    # Option chain endpoint (different from snapshots)
    url = "https://data.alpaca.markets/v1beta1/options/chain"
    params = {"underlying_symbol": symbol}
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    print(f"\n[CHAIN] HTTP {r.status_code} {r.reason}")
    if r.status_code != 200:
        print("Chain response (first 2000 chars):")
        print(r.text[:2000])
        r.raise_for_status()
    return r.json()

def fetch_option_contracts(underlying: str, key: str, secret: str, limit: int = 100) -> dict:
    url = "https://data.alpaca.markets/v1beta1/options/contracts"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    params = {
        "underlying_symbols": underlying,
        "limit": limit,
        # Optional filters you can add later:
        # "status": "active",
        # "expiration_date": "2026-02-09",
        # "type": "call",  # or "put"
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    print(f"\n[CONTRACTS] HTTP {r.status_code} {r.reason}")
    if r.status_code != 200:
        print("Contracts response (first 2000 chars):")
        print(r.text[:2000])
        r.raise_for_status()
    return r.json()

def fetch_option_contracts_v2(
    underlying: str,
    key: str,
    secret: str,
    paper: bool = True,
    limit: int = 100,
) -> dict:
    # Contracts endpoint is on the TRADING host, not the DATA host
    base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    url = f"{base}/v2/options/contracts"

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    params = {
        "underlying_symbols": underlying,
        "limit": limit,
        # Optional later:
        # "status": "active",
        # "type": "call",  # or "put"
        # "expiration_date": "2026-02-09",
    }

    r = requests.get(url, headers=headers, params=params, timeout=30)
    print(f"\n[CONTRACTS v2] HTTP {r.status_code} {r.reason}  ({url})")
    if r.status_code != 200:
        print("Contracts response (first 2000 chars):")
        print(r.text[:2000])
        r.raise_for_status()
    return r.json()

def fetch_option_snapshots_by_symbols(symbols: list[str], key: str, secret: str) -> dict:
    url = "https://data.alpaca.markets/v1beta1/options/snapshots"
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {"symbols": ",".join(symbols)}  # <-- comma-separated
    r = requests.get(url, headers=headers, params=params, timeout=30)
    print(f"\n[SNAPSHOTS by symbols] HTTP {r.status_code} {r.reason}")
    if r.status_code != 200:
        print("Response (first 2000 chars):")
        print(r.text[:2000])
        r.raise_for_status()
    return r.json()



def find_any_greeks(payload: dict) -> tuple[int, int]:
    """Return (contracts_checked, contracts_with_greeks_like_fields)."""
    if not isinstance(payload, dict):
        return 0, 0

    # Heuristic: scan nested dicts for common greek keys
    greek_keys = {"delta", "gamma", "theta", "vega", "rho"}
    checked = 0
    hits = 0

    def scan_obj(obj):
        nonlocal checked, hits
        if isinstance(obj, dict):
            # direct greeks object
            if "greeks" in obj and isinstance(obj["greeks"], dict) and obj["greeks"]:
                hits += 1
            # greeks flattened
            if greek_keys.intersection(obj.keys()):
                hits += 1
            for v in obj.values():
                scan_obj(v)
        elif isinstance(obj, list):
            for v in obj:
                scan_obj(v)

    # Try to locate contracts dict/list
    scan_obj(payload)
    # This overcounts hits in nested structures, but is good enough as a probe.
    return checked, hits


def main():
    secrets_path = find_secrets_file()
    creds = load_alpaca_creds(str(secrets_path))

    # Data creds (used for data.alpaca.markets snapshots)
    if not creds.get("data"):
        raise RuntimeError(f"Missing 'alpaca:' api_key/api_secret in {secrets_path}")
    data_key, data_secret = creds["data"]

    # Paper creds (used for paper-api.alpaca.markets trading endpoints)
    if not creds.get("paper"):
        raise RuntimeError(f"Missing 'alpaca_paper_trading:' api_key/api_secret in {secrets_path}")
    paper_key, paper_secret = creds["paper"]

    print(f"Loaded Alpaca creds from: {secrets_path}")

    # 1) Option chain snapshots (DATA host)
    payload = fetch_option_snapshots("IBIT", data_key, data_secret)
    preview_snapshots(payload, max_contracts=3)

    # 2) Try contracts list (PAPER TRADING host)
    contracts = fetch_option_contracts_v2("IBIT", paper_key, paper_secret, paper=True, limit=25)
    print("\nContracts payload preview:")
    print(json.dumps(contracts, indent=2)[:2000])

    symbols = [c["symbol"] for c in contracts["option_contracts"][:5]]
    snap2 = fetch_option_snapshots_by_symbols(symbols, data_key, data_secret)
    print(json.dumps(snap2, indent=2)[:2000])

    snapshots = snap2["snapshots"]  # dict keyed by option symbol

    for sym, snap in list(snapshots.items())[:5]:
        q = snap.get("latestQuote")
        t = q.get("t") if q else None
        if not t:
            print(sym, "no latestQuote timestamp")
            continue
        print(sym, "quote_age_s:", age_seconds(t), "quote_ts:", t)




if __name__ == "__main__":
    main()
