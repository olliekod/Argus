# Option Chain Snapshots — Retention and Pruning

The `option_chain_snapshots` table stores Alpaca options chain snapshots used for replay packs (SPY, QQQ, IBIT, BITO, etc.). Rows are keyed by `timestamp_ms` and include `recv_ts_ms` and `provider` for data-availability gating in the replay harness.

## Config-driven retention

In `config.yaml`:

- **`data_retention.option_chain_snapshots_days`** (default: 30) — How many days of snapshots to keep. The orchestrator runs DB maintenance hourly and prunes this table using this value.

## How pruning runs

1. **Automatically (recommended)**  
   With the main Argus process running, `_run_db_maintenance` runs every hour and calls `Database.cleanup_old_data()` with the retention map. If `option_chain_snapshots_days` is set, `option_chain_snapshots` is pruned by `timestamp_ms`.

2. **Manually**  
   To prune without the orchestrator or to use a custom retention window:

   ```bash
   python scripts/prune_option_chain_snapshots.py --days 30
   python scripts/prune_option_chain_snapshots.py --days 14 --db data/argus.db
   ```

   This deletes rows where `timestamp_ms` is older than the given number of days.

## Safe defaults

- `--days` must be greater than 0.
- Pruning uses only `timestamp_ms < cutoff_ms`; no other tables or columns are modified.
- The script skips cleanly if the database or table is missing.
