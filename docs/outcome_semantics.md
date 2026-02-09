# Outcome Semantics — Phase 4A.1

> **Table:** `bar_outcomes`
> **Engine:** `src/core/outcome_engine.py`
> **CLI:** `python -m src.outcomes`

## Window Definition

For each bar identified by `(provider, symbol, bar_duration_seconds, timestamp_ms)` and
each `horizon_seconds`, a forward-looking outcome window is defined:

| Field | Computation |
|---|---|
| `close_ref_ms` | `timestamp_ms + bar_duration_seconds * 1000` (the bar's close time) |
| `window_start_ms` | `= close_ref_ms` |
| `window_end_ms` | `window_start_ms + horizon_seconds * 1000` |
| `close_now` | the bar's close price (reference price for all returns) |

## Future Bar Selection

A bar is **in the window** when both conditions hold:

1. `bar.timestamp_ms > anchor.timestamp_ms` (strictly after the anchor bar)
2. `bar.close_time_ms <= window_end_ms` where `close_time_ms = bar.timestamp_ms + bar_duration_seconds * 1000`

Bars within the window are used in strict timestamp order.
No interpolation, no invented bars.

## Metrics

| Metric | Formula |
|---|---|
| `fwd_return` | `(close_at_horizon / close_now) - 1` |
| `max_runup` | `(max_high_in_window / close_now) - 1` |
| `max_drawdown` | `(min_low_in_window / close_now) - 1` |
| `realized_vol` | sample stddev of `log(close[i] / close[i-1])` over `[anchor] + future_bars`, no annualization |
| `close_at_horizon` | close of the **last** bar in the window (by timestamp) |

### Path Helper Fields

| Field | Meaning |
|---|---|
| `max_high_in_window` | absolute high price of the bar producing the max high |
| `min_low_in_window` | absolute low price of the bar producing the min low |
| `max_runup_ts_ms` | close time of the bar producing `max_high_in_window` |
| `max_drawdown_ts_ms` | close time of the bar producing `min_low_in_window` |
| `time_to_max_runup_ms` | `max_runup_ts_ms - window_start_ms` |
| `time_to_max_drawdown_ms` | `max_drawdown_ts_ms - window_start_ms` |

## Quantization

All float metrics are rounded to `quantize_decimals` (config, default 10) **before**
persistence. This guarantees exact floating-point equality across platforms and reruns.

## Statuses

| Status | Condition | Metric behavior |
|---|---|---|
| **OK** | `bars_found > 0` **and** `gap_count <= gap_tolerance_bars` | All metrics computed |
| **INCOMPLETE** | `bars_found == 0` (no future data yet) | All metrics NULL |
| **GAP** | `gap_count > gap_tolerance_bars` | `fwd_return` may be set (from last available close); path metrics (`max_runup`, `max_drawdown`, extrema timestamps) are **NULL** to avoid misleading values across missing spans |

Where:
- `bars_expected = horizon_seconds / bar_duration_seconds`
- `bars_found` = number of future bars actually present in the window
- `gap_count = max(0, bars_expected - bars_found)`

## Status Upgrade Rules (Idempotent)

| Current status | Re-run behavior |
|---|---|
| **INCOMPLETE** | May be upgraded to **OK** or **GAP** when more data arrives (upsert overwrites all metric columns) |
| **OK** | Never changed on rerun with the same `outcome_version` |
| **GAP** | Never changed on rerun with the same `outcome_version` |

This is enforced by the SQL `ON CONFLICT ... DO UPDATE SET ... CASE WHEN status = 'INCOMPLETE' THEN excluded.* ELSE bar_outcomes.* END` pattern.

## Primary Key

```
(provider, symbol, bar_duration_seconds, timestamp_ms, horizon_seconds, outcome_version)
```

Changing `outcome_version` in config causes a full parallel set of outcomes to be
written, leaving the old version's rows untouched for comparison.

## Coverage Fields (Debug)

| Field | Purpose |
|---|---|
| `bars_expected` | How many bars the horizon should contain |
| `bars_found` | How many were actually present |
| `gap_count` | `bars_expected - bars_found` |
| `computed_at_ms` | Wall-clock timestamp of computation (**metadata only** — never used in determinism comparisons) |

## Derived Bar Durations

The config lists `bar_durations_seconds: [60, 300, 900, 3600, 14400, 86400]` and maps
each to a list of horizons. However, only bar durations that **actually exist** in
`market_bars` will produce outcomes. Currently only `bar_duration=60` (1-minute bars)
is persisted by ingestion. The outcome engine correctly skips durations with no data.

When derived bars (5m, 1h, etc.) are added in a future phase, the engine will
automatically begin computing outcomes for them — no code change needed, just
persisting derived bars into `market_bars` with the appropriate `bar_duration`.

## CLI Quick Reference

```bash
# Discover what bar keys exist in the DB
python -m src.outcomes list

# Discover what outcome keys exist
python -m src.outcomes list-outcomes

# Backfill outcomes for a specific key + date range
python -m src.outcomes backfill --provider bybit --symbol BTCUSDT --bar 60 --start 2025-02-01 --end 2025-02-28

# Backfill ALL provider/symbol combos
python -m src.outcomes backfill-all --start 2025-02-01 --end 2025-02-28

# Coverage report (omit filters for global)
python -m src.outcomes coverage
python -m src.outcomes coverage --provider bybit --symbol BTCUSDT
```

## Non-Negotiables

1. **Determinism:** same persisted bars → same outcomes → same DB rows (bit-identical after quantization)
2. **Idempotency:** reruns never duplicate rows, never drift values
3. **No wall-clock in metrics:** `computed_at_ms` is metadata only
4. **No invented data:** gaps produce NULL metrics, never interpolated values
5. **DB is source of truth:** engine reads only from `market_bars`, never from EventBus ordering
