"""Options & Greeks stream health: Public vs Tastytrade.

Shows a per-provider report so you can confirm every aspect of each
options/greeks stream is working before running the comparison script.

- Tastytrade: REST option chain snapshots + DXLink Greeks (IV in snapshots
  when Greeks cache is populated).
- Public: REST option chain snapshots with IV/Greeks from Public API.

Usage:
  python scripts/options_providers_health.py
  python scripts/options_providers_health.py --hours 6 --db data/argus.db
  python scripts/options_providers_health.py --symbol SPY
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def _ms_to_iso(ms: int | None) -> str:
    if ms is None:
        return "N/A"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_report(db_path: Path, hours: int, symbol: str | None) -> int:
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return 2

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    window_ms = hours * 60 * 60 * 1000
    start_ms = now_ms - window_ms

    # Only Public and Tastytrade
    providers = ("public", "tastytrade")
    symbol_filter = " AND symbol = ?" if symbol else ""
    extra_params: list = [symbol] if symbol else []

    print("=" * 72)
    print("Options & Greeks stream health (Public vs Tastytrade)")
    print("=" * 72)
    print(f"  Window: last {hours} hour(s)  |  DB: {db_path}")
    if symbol:
        print(f"  Symbol filter: {symbol}")
    print()

    for provider in providers:
        # Total rows in window
        cursor.execute(
            f"""
            SELECT COUNT(*) as cnt,
                   COUNT(CASE WHEN atm_iv IS NOT NULL AND atm_iv > 0 THEN 1 END) as with_iv,
                   MIN(recv_ts_ms) as first_recv,
                   MAX(recv_ts_ms) as last_recv,
                   COUNT(DISTINCT symbol) as distinct_symbols
            FROM option_chain_snapshots
            WHERE provider = ? AND timestamp_ms >= ? AND timestamp_ms <= ?{symbol_filter}
            """,
            [provider, start_ms, now_ms] + extra_params,
        )
        row = cursor.fetchone()
        if not row or row["cnt"] == 0:
            print(f"[{provider.upper()}]")
            print("  Status: NO DATA in window")
            print("  → Start Argus with this provider enabled and wait for at least one poll cycle.")
            print()
            continue

        total = row["cnt"]
        with_iv = row["with_iv"]
        first_recv = row["first_recv"]
        last_recv = row["last_recv"]
        distinct_symbols = row["distinct_symbols"]
        pct_iv = (100.0 * with_iv / total) if total else 0
        age_sec = (now_ms - last_recv) / 1000 if last_recv else None

        # Sample symbols
        cursor.execute(
            f"""
            SELECT symbol, COUNT(*) as cnt, MAX(atm_iv) as max_iv
            FROM option_chain_snapshots
            WHERE provider = ? AND timestamp_ms >= ? AND timestamp_ms <= ?{symbol_filter}
            GROUP BY symbol
            ORDER BY cnt DESC
            LIMIT 10
            """,
            [provider, start_ms, now_ms] + extra_params,
        )
        symbol_rows = cursor.fetchall()

        print(f"[{provider.upper()}]")
        print("  Snapshots (in window):", total)
        print("  With atm_iv (Greeks/IV):", with_iv, f"({pct_iv:.1f}%)")
        print("  Distinct symbols:", distinct_symbols)
        print("  First recv_ts:", _ms_to_iso(first_recv))
        print("  Last recv_ts: ", _ms_to_iso(last_recv), end="")
        if age_sec is not None:
            print(f"  ({age_sec:.0f}s ago)" if age_sec >= 0 else " (future)")
        else:
            print()

        if symbol_rows:
            print("  Top symbols (by snapshot count):")
            for r in symbol_rows:
                iv_str = f", atm_iv sample: {r['max_iv']:.4f}" if r["max_iv"] is not None else ""
                print(f"    {r['symbol']}: {r['cnt']} snapshots{iv_str}")

        # Verdict
        if total == 0:
            status = "NO DATA"
        elif with_iv == 0:
            status = "WARN (no IV in snapshots — check Greeks source)"
        elif age_sec is not None and age_sec > 600:
            status = f"WARN (stale: last data {age_sec/60:.0f} min ago)"
        else:
            status = "OK (stream and IV look good)"
        print("  Status:", status)
        print()

    # Ready for comparison?
    cursor.execute(
        """
        SELECT provider, COUNT(*) as cnt
        FROM option_chain_snapshots
        WHERE provider IN ('public', 'tastytrade')
          AND timestamp_ms >= ? AND timestamp_ms <= ?
        GROUP BY provider
        """,
        (start_ms, now_ms),
    )
    counts = {r[0]: r[1] for r in cursor.fetchall()}
    conn.close()

    public_ok = counts.get("public", 0) >= 10
    tasty_ok = counts.get("tastytrade", 0) >= 10
    if public_ok and tasty_ok:
        print("Ready for comparison: both providers have data in window.")
        print("  Run: python scripts/compare_options_snapshot_providers.py --symbol SPY --start <date> --end <date>")
    elif public_ok or tasty_ok:
        print("Only one provider has data so far. Let Argus run longer, then re-run this script.")
    else:
        print("No Public or Tastytrade data in window. Ensure both are enabled and Argus has been running.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Options & Greeks stream health for Public and Tastytrade (pre-check before comparison)."
    )
    parser.add_argument("--db", default="data/argus.db", help="Path to argus.db")
    parser.add_argument("--hours", type=int, default=24, help="Look at last N hours (default: 24)")
    parser.add_argument("--symbol", default=None, help="Filter to one symbol (e.g. SPY)")
    args = parser.parse_args()
    return run_report(Path(args.db), args.hours, args.symbol)


if __name__ == "__main__":
    sys.exit(main())
