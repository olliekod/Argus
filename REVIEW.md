# Argus Codebase Review

**Reviewer:** Senior Quantitative Engineer
**Date:** 2026-02-03
**Scope:** Full codebase — correctness, performance, simulation integrity, strategy quality
**Severity scale:** CRITICAL (system broken), HIGH (results unreliable), MEDIUM (correctness risk), LOW (improvement opportunity)

---

## Executive Summary

Argus has a well-structured architecture for monitoring crypto volatility and generating put spread recommendations on IBIT/BITO. The connector layer, conditions monitor, economic calendar, and gap risk tracker are competently built. However, the core simulation engine — the 400K paper trader farm — has a **fatal architectural bug**: open positions are never checked for exits or expirations. This means the system currently accumulates open trades indefinitely, never realizes P&L, and cannot produce meaningful performance data. The promotion system, leaderboard, and daily review all depend on data that can never be generated under the current code.

Beyond that critical issue, the review identified 28 findings across trading logic, simulation integrity, performance, and concurrency.

---

## CRITICAL Findings

### 1. Open positions are never closed — the core simulation loop is broken

**Severity:** CRITICAL
**Files:** `src/orchestrator.py:662-705`, `src/trading/paper_trader_farm.py:418-478`

The `_run_research_farm` method in the orchestrator only calls `evaluate_signal` for new entries. It never calls `check_exits()` or `expire_positions()`, which are the only two methods that close trades and realize P&L.

```python
# orchestrator.py:662-705 — the entire research loop
async def _run_research_farm(self):
    while self._running and self.research_enabled:
        # ... builds signal ...
        trades = await self.paper_trader_farm.evaluate_signal(symbol, signal)
        # ^^^ entries only. No exit checking anywhere.
```

**Impact:**
- `PaperTrade.status` is always `"open"` — never `"closed"` or `"expired"`
- `PaperTrader.stats['total_pnl']` is always `0.0` for every trader
- `realized_pnl` is always `NULL` in the database
- The leaderboard (`get_leaderboard`) always shows $0 realized P&L
- The promotion system (`_maybe_promote_configs`) queries for `total_pnl >= 250` — this can never be satisfied
- Daily review P&L figures (`get_trade_activity_summary`) query `WHERE status != 'open'` — returns 0 rows
- The entire system operates as if no trades have ever been completed

**Fix required:** The orchestrator must periodically:
1. Fetch current spread prices for all open positions
2. Call `paper_trader_farm.check_exits(current_prices)`
3. Check for expired positions and call `expire_positions(expiry_date)`

This requires building a price lookup mechanism for open positions, which does not currently exist.

### 2. Correlation limit counter never decrements

**Severity:** CRITICAL
**File:** `src/trading/paper_trader_farm.py:352`

```python
self._positions_by_symbol[symbol] = self._positions_by_symbol.get(symbol, 0) + 1
```

This counter increments on every entry but is never decremented when positions close (partly because positions never close per finding #1). Even if exits were implemented, `check_exits()` and `expire_positions()` don't touch `_positions_by_symbol`. Eventually, `btc_exposure >= max_btc_exposure` will be permanently true, blocking all new entries.

---

## HIGH Findings

### 3. Session filter uses UTC hours instead of Eastern Time

**Severity:** HIGH
**Files:** `src/trading/paper_trader.py:197-213`, `src/trading/paper_trader_farm.py:307-313`

The session filters check `morning` (9:30-11:30), `midday` (11:30-14:00), `afternoon` (14:00-16:00) but operate on UTC hours, not Eastern Time. A "morning" check at hour 10 UTC corresponds to 5:00 AM ET — well before market open. The comment at line 199-200 acknowledges this:

```python
# Convert to Eastern Time for session checks
# In production this would use the timezone-fixed value
# For simplicity here, we assume the provided time reflects market hours
hour = dt.hour + (dt.minute / 60)
```

**Impact:** ~33% of trader configs (those with `session_filter != 'any'`) are filtering on the wrong time zone. Traders labeled "morning" are actually filtering for pre-market hours. The resulting parameter search is testing meaningless time windows.

### 4. Market hours checks use server local time, not Eastern Time

**Severity:** HIGH
**Files:** `src/detectors/ibit_detector.py:210`, `src/connectors/ibit_options_client.py:332-339`

```python
# ibit_detector.py:210
def _check_market_hours(self) -> bool:
    now = datetime.now()  # <-- local time, not Eastern
```

```python
# ibit_options_client.py:332
def get_market_status(self) -> Dict:
    now = datetime.now()  # <-- local time, not Eastern
```

If the server runs in UTC (as is typical for cloud deployments), these checks will report the market as open at the wrong times.

### 5. All 400K traders receive identical trade parameters

**Severity:** HIGH
**File:** `src/trading/paper_trader_farm.py:327-347`

Every trader that passes its entry filter enters the exact same trade:

```python
trade = active_trader.enter_trade(
    symbol=symbol,
    strikes=signal_data.get('strikes', 'N/A'),   # same for all
    expiry=signal_data.get('expiry', 'N/A'),       # same for all
    entry_credit=signal_data.get('credit', 0.40),  # same for all
    contracts=1,                                     # same for all
    ...
)
```

The trader-specific parameters (`profit_target_pct`, `stop_loss_pct`, `trailing_stop_pct`, `position_size_pct`) only apply at exit, which never happens (finding #1). The farm is therefore testing 400K identical entry filters against identical trades. This means:

- There is zero diversity in trade construction
- Varying `dte_target` has no effect (the same expiry is used for all)
- Varying `position_size_pct` has no effect (contracts=1 always)
- The optimization is effectively testing only the entry threshold grid

### 6. Credit spread pricing uses mid-market prices, inflating simulated returns

**Severity:** HIGH
**File:** `src/connectors/ibit_options_client.py:211-215`

```python
short_mid = (short_bid + short_ask) / 2 if short_ask else short_bid
long_mid = (long_bid + long_ask) / 2 if long_ask else long_bid
net_credit = short_mid - long_mid
```

Real execution fills the short leg at the bid and the long leg at the ask. The mid-market credit is always higher than a realistically achievable fill. The `get_spread_fill_estimate` method (line 416-456) exists and correctly models this, but it is never used in the actual trade flow. The code shows this awareness but doesn't apply it.

For IBIT OTM puts, bid-ask spreads can be 10-30% of the mid price. Using mid-market prices overstates credits by 5-15%, which directly inflates simulated win rates and P&L.

### 7. UUID truncation creates collision risk at scale

**Severity:** HIGH
**File:** `src/trading/paper_trader.py:292`

```python
id=str(uuid.uuid4())[:8]
```

Truncating UUID4 to 8 hex characters yields 4 bytes of entropy (2^32 possible values). By the birthday paradox, collision probability reaches 50% at ~77,000 trades. With 400K traders potentially making multiple trades each, collisions are statistically certain. The `paper_trades` table uses `id TEXT PRIMARY KEY`, so collisions will cause `INSERT OR REPLACE` to silently overwrite earlier trades.

---

## MEDIUM Findings

### 8. Gap risk tracker snapshots are never taken

**Severity:** MEDIUM
**File:** `src/orchestrator.py`

The `GapRiskTracker.snapshot_market_close()` method exists but is never called by the orchestrator. Without snapshots, `_last_close_snapshot` is `None` (or only populated from historical DB data), and `calculate_gap()` returns `None`. The gap risk system is effectively non-functional.

### 9. Monte Carlo PoP model uses GBM, not appropriate for crypto/IBIT

**Severity:** MEDIUM
**File:** `src/analysis/gpu_engine.py:126-200`

The Monte Carlo uses Geometric Brownian Motion (constant volatility, log-normal returns). IBIT/BTC exhibits:
- Fat tails (kurtosis >> 3)
- Volatility clustering
- Jump risk (gap events)
- Negative skew under stress

The Heston parameters are defined (lines 62-65) but never used in the simulation. The GBM model underestimates tail risk, leading to overstated PoP values. A 70% PoP under GBM might be 60-65% under a more realistic model.

### 10. IV Rank calculation is a rough proxy, not true IV Rank

**Severity:** MEDIUM
**File:** `src/connectors/ibit_options_client.py:276-323`

True IV Rank = (Current IV - 52wk Low IV) / (52wk High IV - 52wk Low IV). The implementation uses realized volatility * arbitrary multipliers (0.8 and 1.5) as proxies for the IV range, then falls back to hardcoded bounds (0.33 to 0.62). This will produce unreliable values, particularly during regime changes.

### 11. Economic calendar is hardcoded for 2025-2026 only

**Severity:** MEDIUM
**File:** `src/core/economic_calendar.py:86-87`

```python
FOMC_DATES = {2025: FOMC_2025, 2026: FOMC_2026}
CPI_DATES = {2025: CPI_2025, 2026: CPI_2026}
```

After 2026, the code falls back to 2026 dates for all future years, which will be wrong. The events cache (`_events_cache`) also stores only one year's events and never invalidates.

### 12. Database writes serialize in the hot path

**Severity:** MEDIUM
**File:** `src/trading/paper_trader_farm.py:354-356`

Each trade entry calls `_save_trade` which does an individual `INSERT` + implicit `COMMIT`. With potentially thousands of entries per evaluation cycle, this creates severe I/O bottleneck:

```python
for idx in entry_indices:
    # ... create trade ...
    if self.db:
        await self._save_trade(trade)  # Individual INSERT + COMMIT per trade
```

Should batch inserts and commit once at the end of the evaluation.

### 13. Promotion timer resets on process restart

**Severity:** MEDIUM
**File:** `src/orchestrator.py:98, 714`

```python
self._start_time = datetime.now(timezone.utc)  # line 98
days_since_start = (datetime.now(timezone.utc) - self._start_time).days  # line 714
```

The promotion check uses process start time, not the earliest trade timestamp in the database. Restarting the process resets the countdown. If `promote_after_days` is 60, a restart on day 59 requires waiting another 60 days.

### 14. Shutdown sequence sends Telegram after closing the database

**Severity:** MEDIUM
**File:** `src/orchestrator.py:842-846`

```python
await self.db.close()          # line 842
# ...
if self.telegram:
    await self.telegram.send_system_status(...)  # line 846
```

Any callbacks triggered between `db.close()` and complete shutdown that attempt database writes will fail silently or crash.

### 15. No duplicate entry protection within the same evaluation cycle

**Severity:** MEDIUM
**File:** `src/trading/paper_trader_farm.py:208-370`

If `_run_research_farm` fires twice rapidly (e.g., interval is 10 seconds but evaluation takes 1 second), the same trader can enter the same trade twice if the signal is identical. There is no deduplication by (trader_id, symbol, strikes, expiry).

### 16. `active_traders` dict grows unboundedly

**Severity:** MEDIUM
**File:** `src/trading/paper_trader_farm.py:322-324`

Since positions never close, `PaperTrader` instances accumulate in `active_traders` forever. Each instance holds config, open positions list, and stats dict. At 400K potential entries, this becomes a significant memory burden (estimated 200-400MB for full population).

---

## LOW Findings

### 17. Conditions warmth score has very low resolution

The warmth score combines IV (0-3), funding (1-2), momentum (1-2), market bonus (0-1). The theoretical range is 2-8, not 1-10. Many threshold combinations will be functionally identical.

### 18. Python loop for tensor preparation

`_prepare_trader_tensors` (line 179-187) iterates 400K configs in a Python loop to build the tensor. This should use a list comprehension or numpy intermediary for a 5-10x speedup.

### 19. `_btc_iv_history` in IBITDetector is capped at 168 entries

This is 7 days of hourly data. Z-score calculations on such short history are unreliable and will produce high false-positive rates.

### 20. Trailing stop high watermark is stored in `market_conditions` dict

`paper_trader.py:372-374` — The trailing stop stores `high_pnl_pct` inside `trade.market_conditions`, which is serialized to JSON in the database. This means the watermark is lost on process restart (unless trades are reloaded from DB, which they aren't).

### 21. `get_pnl_summary` includes unrealized credit as "open_pnl"

`paper_trader.py:486` — `open_pnl = sum(t.entry_credit * 100 * t.contracts ...)` counts the full entry credit as P&L. This is misleading — it's not P&L, it's maximum potential profit. Unrealized P&L requires current spread prices, which aren't available.

### 22. Rate limiter in DeribitClient has a timing edge case

`deribit_client.py:64` — `(now - self._last_reset).seconds` wraps at 60 seconds (returns only the seconds component, not total seconds). Should use `.total_seconds()` for intervals > 60s.

### 23. SQLite table name injection in cleanup

`database.py:731-732` — `f"DELETE FROM {table} WHERE timestamp < ?"` uses f-string interpolation for table names. While the input comes from internal config, this is a pattern that should use parameterized table references.

### 24. Yahoo Finance scraping is fragile

`yahoo_client.py` and `ibit_options_client.py` depend on Yahoo Finance endpoints that change frequently. The `v8/finance/chart` endpoint has broken multiple times historically. No fallback data source exists.

### 25. `get_btc_price` returns `price_24h_pcnt` as raw value

`orchestrator.py:391` — `ticker.get('price_24h_pcnt', 0) * 100` assumes the field is a decimal (e.g., 0.02 for 2%). But the Bybit WS handler at `bybit_ws.py:208` already multiplies by 100: `'price_change_24h': float(ticker_data.get('price24hPcnt', 0)) * 100`. This creates a double-multiplication when the conditions monitor reads BTC price change via the orchestrator's ticker cache.

### 26. No early assignment modeling for American-style options

IBIT and BITO options are American-style. The PoP calculation and spread pricing assume European-style exercise only. Early assignment risk is particularly relevant near ex-dividend dates and deep ITM scenarios.

---

## Strategy Improvement Recommendations

### A. Fix the simulation lifecycle (prerequisite for everything else)

Before any strategy improvements matter, the simulation must actually close positions. Implement:
1. A periodic exit-checking loop (every 60s during market hours)
2. Automatic expiration handling on expiry dates
3. Current spread price estimation from the options chain

### B. Differentiate trade construction across the farm

Currently all traders get identical (strikes, credit, expiry). Instead:
- Use each trader's `dte_target` to select different expirations
- Apply `position_size_pct` to determine actual contract count
- Vary spread width based on risk tolerance parameters
- This turns the farm into a genuine multi-strategy test, not just an entry-filter grid search

### C. Use realistic fill prices

Replace mid-market pricing with a fill model:
- Short leg: bid price (or bid + small improvement for limits)
- Long leg: ask price (or ask - small improvement)
- Apply the existing `get_spread_fill_estimate` method
- Add a configurable slippage model based on volume/OI

### D. Replace GBM with a jump-diffusion or Heston model

The Heston parameters are already defined but unused. Implementing Heston-based Monte Carlo would:
- Better capture fat tails
- Provide more realistic PoP estimates
- Reduce overfitting to calm-market conditions

### E. Add regime-aware position sizing

The warmth score could drive position sizing rather than binary entry/no-entry:
- Score 8-10: full size
- Score 6-7: half size
- Score 4-5: quarter size
- Below 4: no entry

### F. Implement portfolio-level risk controls

Currently each trader operates independently. Add:
- Maximum aggregate delta exposure
- Maximum aggregate vega exposure
- Correlation-aware position limits (not just symbol count)
- Portfolio margin requirements tracking

### G. Add time-decay harvesting logic

Credit spreads are theta-positive. The system should model and track theta decay explicitly:
- Exit when X% of theta has been captured
- Reduce position if realized theta capture is below model expectations

### H. Volatility regime detection

Instead of fixed IV thresholds, implement dynamic regime detection:
- Rolling percentile rank against 60-90 day IV history
- Regime classification: low/normal/elevated/crisis
- Regime-specific entry and exit parameters

### I. Walk-forward validation

To avoid overfitting the parameter grid:
- Split history into in-sample and out-of-sample periods
- Optimize on in-sample, validate on out-of-sample
- Only promote parameters that work in both periods

### J. Drawdown circuit breaker

Add account-level drawdown protection:
- If aggregate drawdown exceeds X%, halt all new entries for Y hours
- If monthly drawdown exceeds Z%, reduce all position sizes by 50%

---

## Concurrency and Stability Notes

- The `_running` boolean flag is used across multiple async tasks but is not an `asyncio.Event`. This works in single-threaded asyncio but is fragile if threading is introduced.
- The `_maybe_log_price_snapshot` timestamp check is not atomic. Two concurrent ticker callbacks could both pass the staleness check.
- SQLite single-writer contention: all DB writes go through one `aiosqlite` connection. Under heavy load, writes serialize. Consider WAL mode or batching.

---

## Summary of Priority Actions

| Priority | Finding | Impact |
|----------|---------|--------|
| P0 | #1: Exits never checked | Entire simulation produces no results |
| P0 | #2: Position counter never decrements | Entries blocked permanently over time |
| P1 | #3: Session filter wrong timezone | 33% of configs filter incorrectly |
| P1 | #4: Market hours wrong timezone | Signals fire at wrong times |
| P1 | #5: Identical trades for all traders | Optimization value greatly reduced |
| P1 | #6: Mid-market pricing | Simulated returns inflated 5-15% |
| P1 | #7: UUID collision | Trade data silently overwritten |
| P2 | #8-16: Various medium issues | Correctness and reliability risks |
| P3 | #17-26: Low issues | Quality and robustness improvements |

The system cannot produce meaningful simulation results until findings #1 and #2 are resolved. Findings #3-7 should be addressed before any confidence is placed in the simulation output. The strategy improvements (A-J) become relevant only after the simulation lifecycle is working correctly.
