"""
Argus Database Audit
====================

Audit database freshness and bar continuity without fabricating data.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        ts_norm = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_norm)
    except ValueError:
        return None


def _format_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    return str(timedelta(seconds=int(seconds)))


def _is_market_minute(dt_utc: datetime) -> bool:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_et = dt_utc.astimezone(ET)
    if dt_et.weekday() >= 5:
        return False
    market_open = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= dt_et < market_close


def _classify_symbol(symbol: str, equities_override: Iterable[str]) -> str:
    symbol_upper = symbol.upper()
    override = {sym.upper() for sym in equities_override}
    if symbol_upper in override:
        return "equity"
    if symbol_upper.endswith(("USDT", "USD", "PERP")):
        return "crypto"
    if symbol_upper.isalpha() and len(symbol_upper) <= 5:
        return "equity"
    return "crypto"


def _load_status_snapshot(path: Optional[str]) -> Tuple[Optional[Dict[str, object]], bool]:
    if not path:
        return None, False
    if not os.path.exists(path):
        return None, False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, False
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return data, False
    age_seconds = max(0.0, time.time() - mtime)
    return data, age_seconds <= 120


def _get_table_timestamp(conn: sqlite3.Connection, table: str) -> Optional[datetime]:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    if "timestamp" not in columns:
        return None
    cursor = conn.execute(f"SELECT MAX(timestamp) FROM {table}")
    row = cursor.fetchone()
    if row and row[0]:
        return _parse_ts(row[0])
    return None


def _load_bar_rows(
    conn: sqlite3.Connection,
    lookback_hours: int,
    symbols: Optional[List[str]] = None,
) -> Dict[str, List[datetime]]:
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    params = [since.isoformat()]
    symbol_filter = ""
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        symbol_filter = f"AND symbol IN ({placeholders})"
        params.extend(symbols)
    query = f"""
        SELECT symbol, timestamp
        FROM market_bars
        WHERE timestamp >= ?
        {symbol_filter}
        ORDER BY symbol, timestamp
    """
    rows = conn.execute(query, params).fetchall()
    data: Dict[str, List[datetime]] = defaultdict(list)
    for symbol, ts in rows:
        dt = _parse_ts(ts)
        if dt:
            data[symbol].append(dt)
    return data


def _find_missing_ranges(
    timestamps: List[datetime],
    symbol_type: str,
    max_ranges: int = 10,
) -> List[Tuple[datetime, datetime]]:
    missing: List[Tuple[datetime, datetime]] = []
    if len(timestamps) < 2:
        return missing
    for prev, nxt in zip(timestamps, timestamps[1:]):
        gap_seconds = (nxt - prev).total_seconds()
        if gap_seconds <= 60:
            continue
        missing_start = prev + timedelta(minutes=1)
        missing_end = nxt - timedelta(minutes=1)
        if symbol_type == "crypto":
            missing.append((missing_start, missing_end))
        else:
            current = missing_start
            range_start: Optional[datetime] = None
            while current <= missing_end:
                if _is_market_minute(current):
                    if range_start is None:
                        range_start = current
                else:
                    if range_start is not None:
                        missing.append((range_start, current - timedelta(minutes=1)))
                        range_start = None
                current += timedelta(minutes=1)
            if range_start is not None:
                missing.append((range_start, missing_end))
        if len(missing) >= max_ranges:
            return missing[:max_ranges]
    return missing[:max_ranges]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Argus database continuity.")
    parser.add_argument("--db", default="data/argus.db", help="Path to Argus SQLite DB.")
    parser.add_argument("--lookback-hours", type=int, default=24, help="Lookback window for continuity.")
    parser.add_argument("--symbols", help="Comma-separated symbol filter.")
    parser.add_argument("--equities", help="Comma-separated equity symbols override.")
    parser.add_argument("--status-json", help="Optional path to status snapshot JSON.")
    args = parser.parse_args()

    db_path = args.db
    if not os.path.exists(db_path):
        raise SystemExit(f"Database not found: {db_path}")

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    equities_override = [s.strip() for s in args.equities.split(",")] if args.equities else []

    status_snapshot, is_running = _load_status_snapshot(args.status_json)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("=" * 72)
    print("ARGUS DB AUDIT")
    print("=" * 72)
    print(f"DB: {db_path}")
    print(f"Lookback: {args.lookback_hours}h")
    if symbols:
        print(f"Symbols filter: {', '.join(symbols)}")
    if equities_override:
        print(f"Equities override: {', '.join(equities_override)}")
    if args.status_json:
        print(f"Status snapshot: {args.status_json}")
        print(f"Argus running: {'yes' if is_running else 'no'}")
    print()

    print("TABLE FRESHNESS")
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    for table in tables:
        last_ts = _get_table_timestamp(conn, table)
        if last_ts:
            age_seconds = (datetime.now(timezone.utc) - last_ts.replace(tzinfo=timezone.utc)).total_seconds()
            age_text = _format_age(age_seconds)
            if not is_running:
                status = f"{age_text} (Argus not running)"
            else:
                status = "STALE" if age_seconds > 300 else "OK"
                status = f"{status} — age {age_text}"
            print(f"  {table:<24} {status}")
        else:
            print(f"  {table:<24} n/a (no timestamp)")

    print()
    print("BAR CONTINUITY")
    bars = _load_bar_rows(conn, args.lookback_hours, symbols)
    if not bars:
        print("  No market_bars found in lookback window.")
    else:
        for symbol, timestamps in bars.items():
            symbol_type = _classify_symbol(symbol, equities_override)
            missing_ranges = _find_missing_ranges(timestamps, symbol_type)
            print(f"  {symbol} ({symbol_type})")
            if missing_ranges:
                for start, end in missing_ranges[:10]:
                    print(f"    missing: {start.isoformat()} -> {end.isoformat()}")
            else:
                print("    missing: none")

    if status_snapshot:
        print()
        print("DB COUNTS VS BAR BUILDER COUNTERS")
        bar_builder = status_snapshot.get("internal", {}).get("bar_builder", {})
        extras = bar_builder.get("extras", {}) if isinstance(bar_builder, dict) else {}
        runtime_counts = extras.get("bars_emitted_by_symbol", {})
        if runtime_counts:
            for symbol, runtime_count in runtime_counts.items():
                row = conn.execute(
                    "SELECT COUNT(*) FROM market_bars WHERE symbol = ?",
                    (symbol,),
                ).fetchone()
                db_count = row[0] if row else 0
                print(f"  {symbol:<10} db={db_count:<8} runtime={runtime_count}")
        else:
            print("  No runtime bar builder counters found in snapshot.")
    else:
        print()
        print("DB COUNTS VS BAR BUILDER COUNTERS")
        print("  Status snapshot not provided; skipping runtime comparison.")

    conn.close()


if __name__ == "__main__":
    main()
