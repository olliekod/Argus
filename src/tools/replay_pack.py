"""
Daily Replay Pack Tool
=======================

Slices market_bars, bar_outcomes, and regimes from the database for a
specific symbol and time window, and saves them to a JSON file for
deterministic offline replay.
"""

import asyncio
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.core.db import Database
from src.core.outcome_engine import BarData

async def create_replay_pack(
    symbol: str,
    start_date: str, # YYYY-MM-DD
    end_date: str,   # YYYY-MM-DD
    output_path: str,
    provider: str = "tastytrade",
    bar_duration: int = 60,
):
    db = Database()
    await db.connect()
    
    try:
        # 1. Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000) + (24 * 3600 * 1000) - 1
        
        print(f"Packing data for {symbol} ({provider}) from {start_date} to {end_date}...")
        
        # 2. Fetch Bars
        bars_raw = await db.get_bars_for_outcome_computation(
            source=provider,
            symbol=symbol,
            bar_duration=bar_duration,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        print(f"Fetched {len(bars_raw)} bars.")
        
        # 3. Fetch Outcomes
        outcomes = await db.get_bar_outcomes(
            provider=provider,
            symbol=symbol,
            bar_duration_seconds=bar_duration,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        print(f"Fetched {len(outcomes)} outcomes.")
        
        # 4. Fetch Regimes (Symbol and Market)
        # We need market regimes too (EQUITIES)
        market = "EQUITIES" # Assuming equities for now
        
        symbol_regimes = await db.get_regimes(scope=symbol, start_ms=start_ms, end_ms=end_ms)
        market_regimes = await db.get_regimes(scope=market, start_ms=start_ms, end_ms=end_ms)
        all_regimes = symbol_regimes + market_regimes
        print(f"Fetched {len(all_regimes)} regimes ({len(symbol_regimes)} symbol, {len(market_regimes)} market).")
        
        # 5. Fetch Snapshots if available (STUB - snapshots table might be large)
        snapshots = []
        
        # 6. Save Pack
        pack = {
            "metadata": {
                "symbol": symbol,
                "provider": provider,
                "start_date": start_date,
                "end_date": end_date,
                "packed_at": datetime.utcnow().isoformat(),
            },
            "bars": bars_raw,
            "outcomes": outcomes,
            "regimes": all_regimes,
            "snapshots": snapshots,
        }
        
        with open(output_path, "w") as f:
            json.dump(pack, f, indent=2)
            
        print(f"Pack saved to {output_path}")
        
    finally:
        await db.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Replay Pack from DB.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", required=True)
    parser.add_argument("--provider", default="tastytrade")
    
    args = parser.parse_args()
    
    asyncio.run(create_replay_pack(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        output_path=args.out,
        provider=args.provider,
    ))
