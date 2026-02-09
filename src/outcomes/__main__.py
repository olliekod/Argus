"""
Argus Outcomes CLI
==================

Command-line interface for outcome computation and coverage reporting.

Usage::

    # Backfill outcomes for a date range
    python -m src.outcomes backfill --provider bybit --symbol BTC/USDT:USDT --bar 60 --start 2025-01-01 --end 2025-01-31

    # Show bar and outcome coverage stats
    python -m src.outcomes coverage --provider bybit --symbol BTC/USDT:USDT

    # Backfill all configured symbols
    python -m src.outcomes backfill-all --start 2025-01-01 --end 2025-01-31
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.core.database import Database
from src.core.outcome_engine import OutcomeEngine


def _parse_date(s: str) -> int:
    """Parse date string to epoch milliseconds."""
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        # Try ISO format
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)


def _format_duration(ms: int) -> str:
    """Format milliseconds as human readable duration."""
    seconds = ms // 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    years, days = divmod(days, 365)
    months, days = divmod(days, 30)
    
    parts = []
    if years:
        parts.append(f"{years}y")
    if months:
        parts.append(f"{months}mo")
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds:
        parts.append(f"{seconds}s")
    
    return " ".join(parts) if parts else "0s"


async def _cmd_backfill(args):
    """Backfill outcomes for a date range."""
    config = load_config(args.config)
    outcomes_config = config.get("outcomes", {})
    
    db_path = config.get("database", {}).get("path", "data/argus.db")
    db = Database(db_path)
    await db.connect()
    
    try:
        engine = OutcomeEngine(db, outcomes_config)
        
        start_ms = _parse_date(args.start)
        end_ms = _parse_date(args.end) + 86400 * 1000 - 1  # End of day
        
        print(f"Backfilling outcomes:")
        print(f"  Provider: {args.provider}")
        print(f"  Symbol: {args.symbol}")
        print(f"  Bar duration: {args.bar}s")
        print(f"  Range: {args.start} to {args.end}")
        print(f"  Version: {engine.outcome_version}")
        print()
        
        bars, outcomes, upserted = await engine.compute_outcomes_for_range(
            provider=args.provider,
            symbol=args.symbol,
            bar_duration_seconds=args.bar,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        
        print(f"Results:")
        print(f"  Bars processed: {bars:,}")
        print(f"  Outcomes computed: {outcomes:,}")
        print(f"  Rows upserted: {upserted:,}")
        
    finally:
        await db.close()


async def _cmd_backfill_all(args):
    """Backfill outcomes for all configured symbols."""
    config = load_config(args.config)
    outcomes_config = config.get("outcomes", {})
    
    if not outcomes_config.get("enabled", True):
        print("Outcomes are disabled in config")
        return
    
    db_path = config.get("database", {}).get("path", "data/argus.db")
    db = Database(db_path)
    await db.connect()
    
    try:
        engine = OutcomeEngine(db, outcomes_config)
        
        start_ms = _parse_date(args.start)
        end_ms = _parse_date(args.end) + 86400 * 1000 - 1
        
        # Get unique provider/symbol combos from market_bars
        cursor = await db._connection.execute("""
            SELECT DISTINCT source, symbol, bar_duration FROM market_bars
            ORDER BY source, symbol, bar_duration
        """)
        rows = await cursor.fetchall()
        
        total_bars = 0
        total_outcomes = 0
        total_upserted = 0
        
        for row in rows:
            provider = row[0]
            symbol = row[1]
            bar_duration = row[2]
            
            # Skip if not in configured durations
            if bar_duration not in engine.horizons_by_bar:
                continue
            
            print(f"Processing {provider}/{symbol} ({bar_duration}s)...")
            
            bars, outcomes, upserted = await engine.compute_outcomes_for_range(
                provider=provider,
                symbol=symbol,
                bar_duration_seconds=bar_duration,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            
            total_bars += bars
            total_outcomes += outcomes
            total_upserted += upserted
            
            print(f"  → {bars:,} bars, {outcomes:,} outcomes, {upserted:,} upserted")
        
        print()
        print(f"Total: {total_bars:,} bars, {total_outcomes:,} outcomes, {total_upserted:,} upserted")
        
    finally:
        await db.close()


async def _cmd_coverage(args):
    """Show bar and outcome coverage stats."""
    config = load_config(args.config)
    
    db_path = config.get("database", {}).get("path", "data/argus.db")
    db = Database(db_path)
    await db.connect()
    
    try:
        print("=" * 60)
        print("  ARGUS DATA COVERAGE REPORT")
        print("=" * 60)
        
        # Bar coverage
        bar_stats = await db.get_bar_coverage_stats(
            source=args.provider,
            symbol=args.symbol,
        )
        
        print("\n  MARKET BARS")
        print("-" * 60)
        
        if bar_stats.get("total_bars"):
            min_ts = bar_stats.get("min_ts")
            max_ts = bar_stats.get("max_ts")
            total = bar_stats.get("total_bars", 0)
            
            # Parse timestamps
            if isinstance(min_ts, str):
                min_dt = datetime.fromisoformat(min_ts.replace("Z", "+00:00"))
                max_dt = datetime.fromisoformat(max_ts.replace("Z", "+00:00"))
            else:
                min_dt = datetime.fromtimestamp(min_ts, tz=timezone.utc) if min_ts else None
                max_dt = datetime.fromtimestamp(max_ts, tz=timezone.utc) if max_ts else None
            
            span_ms = 0
            if min_dt and max_dt:
                span_ms = int((max_dt - min_dt).total_seconds() * 1000)
            
            print(f"  Total bars: {total:,}")
            if min_dt:
                print(f"  Min timestamp: {min_dt.isoformat()}")
            if max_dt:
                print(f"  Max timestamp: {max_dt.isoformat()}")
            if span_ms:
                print(f"  Span: {_format_duration(span_ms)} ({span_ms / 1000:,.0f}s)")
        else:
            print("  No bars found")
        
        # Outcome coverage
        outcome_stats = await db.get_outcome_coverage_stats(
            provider=args.provider,
            symbol=args.symbol,
        )
        
        print("\n  BAR OUTCOMES")
        print("-" * 60)
        
        if outcome_stats.get("total_outcomes"):
            total = outcome_stats.get("total_outcomes", 0)
            ok_count = outcome_stats.get("ok_count", 0)
            incomplete = outcome_stats.get("incomplete_count", 0)
            gap_count = outcome_stats.get("gap_count", 0)
            min_ts_ms = outcome_stats.get("min_ts_ms")
            max_ts_ms = outcome_stats.get("max_ts_ms")
            
            print(f"  Total outcomes: {total:,}")
            print(f"  Status breakdown:")
            print(f"    OK: {ok_count:,} ({100*ok_count/total:.1f}%)" if total else "    OK: 0")
            print(f"    INCOMPLETE: {incomplete:,} ({100*incomplete/total:.1f}%)" if total else "    INCOMPLETE: 0")
            print(f"    GAP: {gap_count:,} ({100*gap_count/total:.1f}%)" if total else "    GAP: 0")
            
            if min_ts_ms:
                min_dt = datetime.fromtimestamp(min_ts_ms / 1000, tz=timezone.utc)
                max_dt = datetime.fromtimestamp(max_ts_ms / 1000, tz=timezone.utc)
                span_ms = max_ts_ms - min_ts_ms
                
                print(f"  Min timestamp: {min_dt.isoformat()}")
                print(f"  Max timestamp: {max_dt.isoformat()}")
                print(f"  Span: {_format_duration(span_ms)} ({span_ms / 1000:,.0f}s)")
        else:
            print("  No outcomes found")
        
        print()
        print("=" * 60)
        
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(
        prog="python -m src.outcomes",
        description="Argus Outcome Engine CLI — compute forward returns for backtesting",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    
    sub = parser.add_subparsers(dest="command")
    
    # Backfill sub-command
    backfill_p = sub.add_parser("backfill", help="Backfill outcomes for a date range")
    backfill_p.add_argument("--provider", required=True, help="Data provider (e.g., bybit, alpaca)")
    backfill_p.add_argument("--symbol", required=True, help="Symbol (e.g., BTC/USDT:USDT, IBIT)")
    backfill_p.add_argument("--bar", type=int, default=60, help="Bar duration in seconds (default: 60)")
    backfill_p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    backfill_p.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    
    # Backfill-all sub-command
    backfill_all_p = sub.add_parser("backfill-all", help="Backfill all configured symbols")
    backfill_all_p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    backfill_all_p.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    
    # Coverage sub-command
    coverage_p = sub.add_parser("coverage", help="Show bar and outcome coverage stats")
    coverage_p.add_argument("--provider", default=None, help="Filter by provider")
    coverage_p.add_argument("--symbol", default=None, help="Filter by symbol")
    
    args = parser.parse_args()
    
    if args.command == "backfill":
        asyncio.run(_cmd_backfill(args))
    elif args.command == "backfill-all":
        asyncio.run(_cmd_backfill_all(args))
    elif args.command == "coverage":
        asyncio.run(_cmd_coverage(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
