
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

async def verify_ingestion():
    db_path = Path("data/argus.db")
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Checking option_chain_snapshots for SPY and QQQ...")
    
    query = """
    SELECT symbol, timestamp_ms, recv_ts_ms, n_strikes
    FROM option_chain_snapshots
    WHERE symbol IN ('SPY', 'QQQ', 'IBIT', 'BITO')
    ORDER BY timestamp_ms DESC
    LIMIT 20;
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    if not rows:
        print("No recent snapshots found for SPY, QQQ, IBIT, or BITO.")
    else:
        print(f"{'Symbol':<10} | {'Timestamp (UTC)':<25} | {'Recv Lag (ms)':<15} | {'Strikes':<10}")
        print("-" * 70)
        for symbol, ts_ms, recv_ms, n_strikes in rows:
            ts_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            lag = (recv_ms - ts_ms) if recv_ms and ts_ms else "N/A"
            print(f"{symbol:<10} | {ts_dt.isoformat():<25} | {lag:<15} | {n_strikes:<10}")

    conn.close()

if __name__ == "__main__":
    asyncio.run(verify_ingestion())
