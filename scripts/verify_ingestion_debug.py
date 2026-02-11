"""Verify option chain snapshot ingestion from all providers.

Shows recent snapshots grouped by provider, with stats per symbol.

Usage:
  python scripts/verify_ingestion_debug.py
  python scripts/verify_ingestion_debug.py --limit 50
"""

import argparse
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


async def verify_ingestion(limit: int = 30):
    db_path = Path("data/argus.db")
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=" * 90)
    print("Option Chain Snapshot Ingestion Report")
    print("=" * 90)

    # ── Provider summary ──────────────────────────────────────────────
    print("\n[1] Provider Summary")
    print("-" * 60)
    summary_query = """
    SELECT provider, COUNT(*) as cnt,
           MIN(timestamp_ms) as first_ts,
           MAX(timestamp_ms) as last_ts
    FROM option_chain_snapshots
    GROUP BY provider
    ORDER BY provider;
    """
    cursor.execute(summary_query)
    rows = cursor.fetchall()

    if not rows:
        print("  No snapshots found in database.")
        conn.close()
        return

    for provider, cnt, first_ts, last_ts in rows:
        first_dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"  {provider or '(empty)':<15} | {cnt:>6} snapshots | first: {first_dt} | last: {last_dt}")

    # ── Per-symbol breakdown ──────────────────────────────────────────
    print(f"\n[2] Per-Symbol Breakdown")
    print("-" * 70)
    symbol_query = """
    SELECT provider, symbol, COUNT(*) as cnt,
           MAX(timestamp_ms) as last_ts,
           AVG(n_strikes) as avg_strikes
    FROM option_chain_snapshots
    WHERE symbol IN ('SPY', 'QQQ', 'IBIT', 'BITO')
    GROUP BY provider, symbol
    ORDER BY symbol, provider;
    """
    cursor.execute(symbol_query)
    rows = cursor.fetchall()

    if rows:
        print(f"  {'Provider':<15} | {'Symbol':<8} | {'Count':>6} | {'Avg Strikes':>11} | {'Last Snapshot':<20}")
        print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*6}-+-{'-'*11}-+-{'-'*20}")
        for provider, symbol, cnt, last_ts, avg_strikes in rows:
            last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(f"  {provider or '(empty)':<15} | {symbol:<8} | {cnt:>6} | {avg_strikes:>11.1f} | {last_dt}")
    else:
        print("  No snapshots for SPY, QQQ, IBIT, or BITO.")

    # ── Recent snapshots ──────────────────────────────────────────────
    print(f"\n[3] Recent Snapshots (last {limit})")
    print("-" * 100)
    recent_query = f"""
    SELECT provider, symbol, timestamp_ms, recv_ts_ms, n_strikes, underlying_price, atm_iv
    FROM option_chain_snapshots
    WHERE symbol IN ('SPY', 'QQQ', 'IBIT', 'BITO')
    ORDER BY timestamp_ms DESC
    LIMIT {limit};
    """
    cursor.execute(recent_query)
    rows = cursor.fetchall()

    if rows:
        print(f"  {'Provider':<12} | {'Symbol':<6} | {'Timestamp (UTC)':<20} | {'Lag(ms)':>8} | {'Strikes':>7} | {'Price':>9} | {'ATM IV':>8}")
        print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*20}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}-+-{'-'*8}")
        for provider, symbol, ts_ms, recv_ms, n_strikes, price, atm_iv in rows:
            ts_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            lag = (recv_ms - ts_ms) if recv_ms and ts_ms else 0
            price_str = f"${price:.2f}" if price else "N/A"
            iv_str = f"{atm_iv:.4f}" if atm_iv else "N/A"
            print(f"  {provider or '?':<12} | {symbol:<6} | {ts_dt:<20} | {lag:>8} | {n_strikes:>7} | {price_str:>9} | {iv_str:>8}")
    else:
        print("  No recent snapshots found.")

    # ── Multi-provider overlap check ──────────────────────────────────
    print(f"\n[4] Multi-Provider Overlap (same symbol, same minute)")
    print("-" * 60)
    overlap_query = """
    SELECT symbol,
           datetime(timestamp_ms/1000, 'unixepoch') as ts_min,
           GROUP_CONCAT(DISTINCT provider) as providers,
           COUNT(*) as cnt
    FROM option_chain_snapshots
    WHERE symbol IN ('SPY', 'QQQ', 'IBIT', 'BITO')
    GROUP BY symbol, timestamp_ms / 60000
    HAVING COUNT(DISTINCT provider) > 1
    ORDER BY timestamp_ms DESC
    LIMIT 10;
    """
    cursor.execute(overlap_query)
    rows = cursor.fetchall()

    if rows:
        print(f"  {'Symbol':<8} | {'Minute':<20} | {'Providers':<30} | {'Count':>5}")
        print(f"  {'-'*8}-+-{'-'*20}-+-{'-'*30}-+-{'-'*5}")
        for symbol, ts_min, providers, cnt in rows:
            print(f"  {symbol:<8} | {ts_min:<20} | {providers:<30} | {cnt:>5}")
    else:
        print("  No multi-provider overlaps found (this is normal if only one provider is active).")

    conn.close()
    print(f"\n{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(description="Verify option chain snapshot ingestion")
    parser.add_argument("--limit", type=int, default=30, help="Number of recent snapshots to show")
    args = parser.parse_args()
    asyncio.run(verify_ingestion(limit=args.limit))


if __name__ == "__main__":
    main()
