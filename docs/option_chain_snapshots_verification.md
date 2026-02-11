# Option Chain Snapshots — Verification Summary

## 1. DB uniqueness and timestamp normalization

**Constraint:** `UNIQUE(provider, symbol, timestamp_ms)` with `ON CONFLICT(provider, symbol, timestamp_ms) DO NOTHING`.

- **Intent:** Prevents duplicate rows for the same provider/symbol/time; allows both Alpaca and Tastytrade to store a row at the same logical time (different `provider`).
- **Normalization:** Both connectors use **poll time floored to the minute** for `timestamp_ms`:
  - Alpaca: `_poll_time_ms()` in `alpaca_options.py` → `(now_ms // 60_000) * 60_000`
  - Tastytrade: same in `tastytrade_options.py`
- **Receipt time:** `recv_ts_ms` is set to actual receipt (`_now_ms()`) in both connectors and is used for replay gating (data availability barrier).
- **Result:** Same granularity across providers; no surprise collisions; replay still sees accurate receipt order via `recv_ts_ms`.

## 2. Polling frequency and backoff

- **Two separate tasks:** `_poll_options_chains` (Alpaca) and `_poll_tastytrade_options_chains` (Tastytrade) are started as independent asyncio tasks. Neither shares state with the other.
- **Interval:** Each uses its own `interval` (default 60s) and sleeps `await asyncio.sleep(interval)` after each full pass.
- **Backoff:** Each loop has its own `consecutive_errors`; on 3+ consecutive errors it backs off with `min(interval * consecutive_errors, 300)` then continues. After a successful pass, `consecutive_errors` is reset to 0 and the loop resumes the normal interval.
- **RTH:** If `off_hours_disable_options_snapshots` is true, both loops skip work when the US market is closed but still sleep their interval, so they don’t starve when market reopens.

## 3. Replay pack — providers

- **Default:** Replay pack loads option chain snapshots via `get_option_chain_snapshots(symbol, start_ms, end_ms)`, which returns **all providers** (no provider filter in the DB query). So by default packs include both Alpaca and Tastytrade snapshots for cross-validation.
- **Optional filter:** `--snapshot-provider` (e.g. `alpaca` or `tastytrade`) restricts snapshots to that provider when building the pack.
- **Bars/outcomes:** `--provider` still controls only bars and outcomes (e.g. `tastytrade`); snapshot inclusion is controlled by `--snapshot-provider` when set.

**Examples:**

```bash
# Include all providers (default)
python -m src.tools.replay_pack --symbol SPY --start 2024-01-01 --end 2024-01-31

# Only Alpaca snapshots
python -m src.tools.replay_pack --symbol SPY --start 2024-01-01 --end 2024-01-31 --snapshot-provider alpaca

# Only Tastytrade snapshots
python -m src.tools.replay_pack --symbol SPY --start 2024-01-01 --end 2024-01-31 --snapshot-provider tastytrade
```
