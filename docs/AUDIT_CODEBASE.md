# Argus Codebase Audit Report

**Date:** 2026-02-12
**Scope:** Full codebase audit — architecture, data integrity, determinism, provider semantics, strategy correctness, and critical bug fixes.

---

## Executive Summary

Argus is an options-focused market monitoring and paper-trading platform built around a Pub/Sub event bus, a replay harness for backtesting, and multiple data providers (Alpaca, Tastytrade, Yahoo, DXLink). The architecture is sound at the macro level: clean separation between connectors, detectors, strategies, and an orchestrator coordinating everything.

The audit identified **two critical correctness bugs** (fixed in this commit), **10 ranked risks** across all subsystems, and a **concrete patch plan** for remediation.

### Bugs Fixed

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **Critical** | ATM IV lookup mixes expirations — `_parse_option_symbol` drops YYMMDD, so `get_atm_iv` can return IV from a different expiry | `_parse_option_symbol` now returns 4-tuple with expiry; `get_atm_iv`/`get_greeks_for_strike` accept `expiration_ms` filter; `enrich_snapshot_iv` passes `snapshot.expiration_ms` |
| 2 | **Critical** | `_on_dxlink_greeks_event` is dead code — never called because no DXLink streamer is instantiated; `_greeks_cache` stays empty forever | Added `_start_dxlink_greeks_streamer()`, wired into `run()` alongside Tastytrade polling; added `get_api_quote_token()` to REST client |

---

## 1. Architecture Overview

```
ArgusOrchestrator
├── EventBus (Pub/Sub)
├── Connectors
│   ├── AlpacaDataClient (bars, quotes)
│   ├── AlpacaOptionsConnector (options chains)
│   ├── TastytradeOptionsConnector (options chains REST)
│   ├── TastytradeStreamer (DXLink WebSocket — NOW WIRED)
│   ├── BybitWebSocket (crypto)
│   ├── DeribitClient (crypto options)
│   └── YahooFinanceClient (equities)
├── Core
│   ├── BarBuilder → FeatureBuilder → RegimeDetector
│   ├── PersistenceManager (DB writes)
│   ├── GreeksCache (in-memory IV cache — NOW POPULATED)
│   └── OutcomeEngine (forward returns)
├── Detectors
│   ├── OptionsIVDetector
│   ├── VolatilityDetector
│   └── IBITDetector
├── Strategies
│   ├── VRPCreditSpreadStrategy
│   └── SpreadCandidateGenerator
├── Analysis
│   ├── ReplayHarness (deterministic backtesting)
│   ├── ExecutionModel (fill simulation)
│   └── GreeksEngine (Black-Scholes)
└── Alerts
    └── TelegramBot
```

---

## 2. Critical Bug: Cross-Expiration IV Contamination

### Problem

`_parse_option_symbol` parsed DXLink symbols like `.SPY250321P590` but discarded the YYMMDD expiration date — the regex `\d{6}` was matched but never captured. The returned 3-tuple `(underlying, opt_type, strike)` made it impossible to distinguish options with the same underlying and strike but different expirations.

When `get_atm_iv()` searched the cache for the nearest ATM strike, it would iterate **all** expirations and pick whichever happened to have the closest strike — even if it was from a completely different term. For example:

- Cache holds: `.SPY250321P590` (March, IV=0.22) and `.SPY250418P590` (April, IV=0.30)
- Snapshot for April at underlying=590 calls `get_atm_iv("SPY", 590, ...)`
- **Bug**: could return 0.22 (March IV) for an April snapshot

This systematically corrupts term-specific IV inputs for strategies like VRP credit spreads, which depend on accurate term IV.

### Fix

1. **`_parse_option_symbol`** now returns a 4-tuple `(underlying, opt_type, strike, expiry_yymmdd)`
2. **`_yymmdd_to_epoch_ms`** helper converts YYMMDD strings to midnight-UTC epoch ms
3. **`get_atm_iv`** and **`get_greeks_for_strike`** accept optional `expiration_ms` parameter; when provided, only options matching the same calendar day are considered
4. **`enrich_snapshot_iv`** now passes `snapshot.expiration_ms` to both `get_atm_iv` calls
5. Backward compatibility preserved: callers that don't pass `expiration_ms` get the old (unfiltered) behavior

### Tests Added

- `test_cross_expiration_filtering`: verifies that March and April expirations return correct per-term IV
- `test_no_expiration_filter_matches_all`: verifies backward compatibility
- `test_enrichment_respects_expiration`: verifies enrichment won't apply March IV to an April snapshot

---

## 3. Critical Bug: Dead DXLink Greeks Event Handler

### Problem

`ArgusOrchestrator._on_dxlink_greeks_event` was defined but never referenced anywhere. No DXLink streamer was instantiated in `setup()` or started in `run()`. Consequently:

- `_greeks_cache` remained empty for the entire application lifetime
- `enrich_snapshot_iv()` always fell through to the "no cached Greeks" path
- The advertised IV enrichment from DXLink streaming data never executed in production

### Fix

1. **`TastytradeRestClient.get_api_quote_token()`** added — fetches a DXLink streaming token via `GET /api-quote-tokens`
2. **`_start_dxlink_greeks_streamer()`** added to orchestrator — obtains DXLink credentials, instantiates `TastytradeStreamer` with `on_event=self._on_dxlink_greeks_event` and `event_types=["Greeks"]`, then runs `run_forever()`
3. **`run()`** now creates the streamer task alongside the Tastytrade polling task

The DXLink streamer runs as a long-lived async task. Greeks events flow into `_on_dxlink_greeks_event`, which populates `_greeks_cache`. The polling loop's `enrich_snapshot_iv()` calls then find cached IV and actually enrich snapshots.

---

## 4. Replay Determinism & Lookahead Leaks

### Verdict: CORRECT — no lookahead leaks detected

The replay harness (`src/analysis/replay_harness.py`) implements three strict temporal barriers with correct operator usage:

| Barrier | Location | Condition | Status |
|---------|----------|-----------|--------|
| Outcome | Line ~596 | `window_end_ms > sim_ts_ms` → break | Correct |
| Regime | Line ~608 | `rts > sim_ts_ms` → break | Correct |
| Snapshot | Line ~618 | `snap.recv_ts_ms > sim_ts_ms` → break | Correct |

**Determinism guarantees:**
- No `random.*`, `time.time()`, or `datetime.now()` calls in the main replay loop
- All timestamps are integer milliseconds — no floating-point ordering issues
- Bars are sorted at initialization (`sorted(bars, key=lambda b: b.timestamp_ms)`)
- Cursors (outcome, regime, snapshot) move only forward — no reordering possible
- Visible data is defensively copied before passing to strategies (`dict(...)`, `list(...)`)

**Residual risk — ExecutionModel ledger not reset between runs:**
- `ExecutionModel.reset()` exists (line ~204) but is never called by `ReplayHarness`
- If the same `ExecutionModel` instance is reused across harness runs, the ledger accumulates fills/rejects from all runs
- All existing tests create fresh instances, masking this issue
- **Severity: MEDIUM** — easily fixed by calling `self._exec.reset()` at start of `run()`

**Residual risk — Strategy state not reset:**
- Strategy instances passed to harness retain internal state across runs
- No `reset()` method in the `ReplayStrategy` base class
- **Severity: MEDIUM** — callers must create fresh strategy instances or strategies must implement reset

---

## 5. Provider Semantics & Data Sources

### 5.1 Timestamp Normalization

| Provider | Method | UTC Enforced? | Issue |
|----------|--------|---------------|-------|
| Alpaca bars | `_parse_rfc3339_to_ms()` | Partial | `datetime.timestamp()` uses local TZ if naive; strips "Z" but doesn't enforce UTC |
| Alpaca options | `_now_ms()` / `_poll_time_ms()` | Yes | Correct |
| Tastytrade options | `_now_ms()` / `_date_to_ms()` | Yes | Correct |
| Tastytrade REST | `parse_rfc3339_nano()` | Yes | Enforces UTC at parse |
| Yahoo Finance | `_parse_yahoo_source_ts()` | Fragile | Heuristic `>10B → ms` is brittle; falls back to `0.0` on failure, masking errors |
| Deribit | `_normalize_source_ts()` | Partial | Microsecond threshold at 13 digits (`>10^13`) mishandles edge cases |
| Bybit REST | Direct `int(item[0])` | Undocumented | Assumes API returns ms; no assertion |

### 5.2 WebSocket Reconnection & Data Loss

**Tastytrade DXLink streamer:**
- Events are silently dropped during reconnection — no persistent queue
- Error frames are logged but don't trigger reconnect (line ~375)
- Token refresh callback exceptions don't force reconnect

**Bybit WebSocket:**
- Exponential backoff capped at 300s; total reconnect window ~16 minutes before giving up
- Subscription acknowledgement not verified — if subscription fails silently, reconnect assumes active
- Invalid quotes fall back to zeroed values instead of being skipped

### 5.3 Rate Limiting

| Provider | Implementation | Issue |
|----------|---------------|-------|
| Alpaca bars | 0.5s delay between symbols | Sufficient for 13 symbols; no adaptive limiting |
| Bybit REST | `_throttle()` with `time.monotonic()` | Correct, handles Bybit error code 10016 |
| Deribit | Counter reset per 60s | **Bug**: uses `.seconds` instead of `.total_seconds()` — rate limit resets incorrectly |
| OKX | `1/rate_limit` interval | Correct |
| Yahoo | Hard-coded 1s sleep | No formal rate limiting; may exceed informal limits |

### 5.4 Options Chain Compatibility

Alpaca and Tastytrade options connectors produce compatible schemas (`timestamp_ms`, `recv_ts_ms`, `source_ts_ms`, `contract_id` via SHA-256), but **Tastytrade snapshots have all IV/greeks fields as None** (only DXLink streaming provides greeks). This means:
- `atm_iv` is populated for Alpaca but NULL for Tastytrade in the DB
- Downstream queries that require `atm_iv IS NOT NULL` will silently exclude Tastytrade snapshots

---

## 6. Strategy & Execution Model

### 6.1 VRP Credit Spread Strategy

**Correct:**
- VRP = IV - RV threshold gating
- IV source hierarchy: Tastytrade ATM IV → Brenner-Subrahmanyam approximation → Alpaca IV (if enabled)
- Regime filtering: avoids VOL_SPIKE and TREND_DOWN

**Issues:**
- No RV validation before VRP subtraction (`last_rv` could be None/NaN/negative)
- Brenner-Subrahmanyam approximation assumes ATM options, unreliable for deep OTM/ITM
- Hardcoded 14-DTE assumption in IV derivation

### 6.2 Greeks Engine (Black-Scholes)

**Mathematically correct.** All core formulas verified:
- d1/d2 ✓, Delta (call N(d1), put N(d1)-1) ✓, Gamma ✓, Theta (annualized, /365) ✓, Vega (/100 for per-1%) ✓
- IV solver uses Brent's method with bounds [0.001, 5.0] ✓
- Edge cases: T≤0 returns intrinsic, σ≤0 returns 0, illiquid quotes blocked ✓

**Limitations (documented, not bugs):**
- European approximation only — no early exercise premium for American equity options
- No dividend yield (q) term — shifts delta/gamma for dividend-paying equities
- Fixed 4.5% risk-free rate (auto-refresh available)

### 6.3 Spread Candidate Generator

**Correct:**
- Strike ordering: short put > long put ✓
- Credit = short_bid - long_ask (conservative) ✓
- Max loss = width - credit ✓
- Width validation against configured set [1.0, 2.0, 2.5, 5.0] ✓
- Bid-ask quality filter (25% of mid) ✓

**Issues:**
- No volatility skew/smile adjustment — uses single IV for both strikes
- Expected value formula (`credit × prob_otm`) assumes binary outcome, ignores partial losses
- Delta as probability (`1 - |delta|`) is inaccurate for deep OTM under real distributions

### 6.4 Feature Builder

**Correct:**
- Log returns: `log(close / prev_close)` ✓
- Rolling window: `deque(maxlen=60)` → automatic FIFO ✓
- Realized vol: Bessel's correction, annualization factor √525960 ✓
- No off-by-one errors in window management

### 6.5 Regime Detector

**Correct:**
- Hysteresis prevents oscillation near thresholds ✓
- Minimum dwell time prevents flip-flopping ✓
- Gap detection with optional state reset and confidence decay ✓
- Confidence scaling: multiplicative penalties (not warm ×0.5, repaired ×0.9, gap ×0.8) ✓

### 6.6 Outcome Engine

**Correct:**
- Forward return: `(close_at_horizon / close_now) - 1` ✓
- Max runup/drawdown from entry close ✓
- Realized vol with Bessel's correction ✓
- Status classification (OK/GAP/INCOMPLETE) ✓
- Deterministic: `computed_at_ms = window_end_ms`, no wall-clock calls ✓

---

## 7. Database, Persistence & Event Bus

### 7.1 SQL Injection

**Low risk.** All data INSERT/UPDATE uses parameterized queries. Dynamic SQL exists for table names in PRAGMA/ALTER TABLE statements, but table names are hardcoded (not user-supplied). WHERE clause assembly in `get_regimes()` uses `f"SELECT * FROM regimes WHERE {where}"` but params are parameterized.

### 7.2 Data Durability

| Data Type | Retry Logic | Silent Loss? |
|-----------|-------------|-------------|
| Bars | 3× retry, buffer return on failure | No — bars are never silently dropped |
| Options snapshots | Fire-and-forget (`future.add_done_callback`) | **Yes** — exceptions swallowed |
| Signals | No retry, dropped on first failure | **Yes** |
| Metrics/telemetry | No retry, dropped on first failure | **Yes** |

- WAL mode with `PRAGMA synchronous=NORMAL` — crash after commit can lose data (industry standard is FULL for critical data)
- Disk spool with `pause_on_spool_full=True` is safe; legacy unbounded-memory fallback is dangerous

### 7.3 Event Bus

**Publishing is O(1) and non-blocking** — uses `deque(maxlen=...)` with silent drop-on-overflow. Worker threads call handlers sequentially.

**Risks:**
- `_on_bar` handler holds `_bar_lock` while doing `os.fsync()` (disk I/O under lock blocks entire topic worker)
- `_do_flush` blocks handler for up to 30s waiting on `future.result(timeout=30.0)`
- Signal drops under back-pressure are silent

### 7.4 Orchestrator Shutdown

- Shutdown ordering is logical: flush → bus stop → close connections → DB close
- `persistence.shutdown()` calls `thread.join(timeout=5.0)` synchronously, blocking the async event loop for up to 15s
- **Tasks created with `asyncio.create_task()` are never added to `self._tasks` list** — `stop()` cancels an empty list, leaving zombie tasks

### 7.5 Secrets Handling

- Uses `yaml.safe_load()` — no YAML injection ✓
- Validates against placeholder values (PASTE_, YOUR_) ✓
- **File permissions not set** — secrets YAML has default 644 (should be 600)
- Secrets stored in object attributes — visible to debuggers and exception dumps

---

## 8. Files Changed (Bug Fixes)

| File | Change |
|------|--------|
| `src/core/greeks_cache.py` | `_parse_option_symbol` returns 4-tuple; added `_yymmdd_to_epoch_ms`/`_ONE_DAY_MS`; `get_atm_iv`/`get_greeks_for_strike` accept `expiration_ms`; `enrich_snapshot_iv` passes expiration |
| `src/connectors/tastytrade_rest.py` | Added `get_api_quote_token()` for DXLink streaming token |
| `src/orchestrator.py` | Imported `TastytradeStreamer`; added `_dxlink_streamer` attr; added `_start_dxlink_greeks_streamer()`; wired streamer in `run()` |
| `tests/test_greeks_cache_and_iv_enrichment.py` | Updated `_parse_option_symbol` assertions to 4-tuples; fixed `_make_snapshot` expiration; added cross-expiration and enrichment tests |

---

## 9. Top 10 Critical Risks

Ranked by impact × likelihood. Items 1-2 are already fixed.

| Rank | Risk | Severity | Component | Status |
|------|------|----------|-----------|--------|
| **1** | Cross-expiration IV contamination — wrong-term IV fed to VRP strategy | **CRITICAL** | `greeks_cache.py` | **FIXED** |
| **2** | Dead DXLink streamer — `_greeks_cache` empty forever, IV enrichment never runs | **CRITICAL** | `orchestrator.py` | **FIXED** |
| **3** | Alpaca RFC3339 parser doesn't enforce UTC — `datetime.timestamp()` uses local TZ on naive datetimes | **HIGH** | `alpaca_client.py:59-60` | Open |
| **4** | Deribit rate limiter uses `.seconds` instead of `.total_seconds()` — resets every clock-minute, not every 60s elapsed | **HIGH** | `deribit_client.py:79` | Open |
| **5** | Options snapshot writes are fire-and-forget — failures silently lost | **HIGH** | `persistence.py:~755` | Open |
| **6** | Orchestrator tasks never tracked — `self._tasks` is empty, `stop()` cancels nothing, zombie tasks survive shutdown | **HIGH** | `orchestrator.py:~2852` | Open |
| **7** | Disk I/O (`os.fsync`) under `_bar_lock` in event bus handler — blocks entire bar topic worker | **MEDIUM** | `persistence.py:~377` | Open |
| **8** | DXLink error frames logged but don't trigger reconnect — streamer continues emitting nothing after permanent error | **MEDIUM** | `tastytrade_streamer.py:~375` | Open |
| **9** | ExecutionModel ledger not reset between replay runs — accumulates fills across harness instances | **MEDIUM** | `replay_harness.py:~530` | Open |
| **10** | Secrets file written with default permissions (644) — readable by group/other | **MEDIUM** | `config.py:~104` | Open |

---

## 10. Concrete Patch Plan

### P1 — Critical / This Sprint

#### 10.1 Fix Alpaca UTC timestamp parsing
**File:** `src/connectors/alpaca_client.py:59-60`
**Change:** After stripping "Z" and parsing with `fromisoformat()`, ensure the datetime is UTC-aware before calling `.timestamp()`:
```python
dt = datetime.fromisoformat(ts_str)
if dt.tzinfo is None:
    dt = dt.replace(tzinfo=timezone.utc)
return int(dt.timestamp() * 1000)
```
**Test:** Unit test with a naive datetime string (no TZ suffix) to confirm UTC assumption.

#### 10.2 Fix Deribit rate limiter
**File:** `src/connectors/deribit_client.py:79,84`
**Change:** Replace `.seconds` with `.total_seconds()`:
```python
if (now - self._last_reset).total_seconds() >= 60:
    ...
wait_time = 60 - (now - self._last_reset).total_seconds()
```
**Test:** Unit test that rate limiter correctly resets after 60 actual seconds.

#### 10.3 Add retry logic to options snapshot writes
**File:** `src/core/persistence.py:~755`
**Change:** Replace fire-and-forget with retry pattern matching bar flush logic (3× retry, backoff, log on final failure).
**Test:** Mock DB failure and verify retry + eventual success.

#### 10.4 Track async tasks for cancellation
**File:** `src/orchestrator.py`
**Change:** Every `asyncio.create_task()` call should append to `self._tasks`. In `stop()`, cancel all tracked tasks and `await asyncio.gather(*self._tasks, return_exceptions=True)`.
**Test:** Integration test that verifies no tasks are running after `stop()`.

### P2 — High / Next Sprint

#### 10.5 Move disk I/O out of `_bar_lock`
**File:** `src/core/persistence.py:~364-417`
**Change:** Serialize bar to JSON string under lock, but do the file write + fsync outside the lock:
```python
with self._bar_lock:
    line = json.dumps(asdict(bar), separators=(",", ":")) + "\n"
# Outside lock:
with open(self._spool_path, "a") as f:
    f.write(line)
    f.flush()
    os.fsync(f.fileno())
```
**Test:** Concurrent spool writes don't interleave (append is atomic on Linux for < PIPE_BUF).

#### 10.6 DXLink error frame handling
**File:** `src/connectors/tastytrade_streamer.py:~375`
**Change:** On error frame, increment error counter. If persistent errors (>3 in 60s), trigger reconnect instead of silently continuing.
**Test:** Feed mock error frames and verify reconnect is triggered.

#### 10.7 ExecutionModel auto-reset
**File:** `src/analysis/replay_harness.py:~559`
**Change:** Call `self._exec.reset()` at the start of `run()`.
**Test:** Reuse same `ExecutionModel` across two harness runs; verify second run's ledger is clean.

#### 10.8 Secrets file permissions
**File:** `src/core/config.py:~104`
**Change:** After writing secrets file, set `path.chmod(0o600)`.
**Test:** Verify file mode is 0600 after `save_secrets()`.

### P3 — Medium / Backlog

#### 10.9 VRP strategy: validate RV before subtraction
**File:** `src/strategies/vrp_credit_spread.py:~240`
**Change:** Guard `last_rv` against None, NaN, and negative values before computing VRP.

#### 10.10 Token refresh wiring
**File:** `src/orchestrator.py`
**Change:** Wire `token_refresh_cb` on `TastytradeStreamer` to call `get_api_quote_token()`. On callback exception, force streamer reconnect.

#### 10.11 Cache eviction timer
**File:** `src/orchestrator.py`
**Change:** Add periodic `asyncio.create_task` that calls `_greeks_cache.evict_stale()` every `max_age_ms`.

#### 10.12 Yahoo Finance timestamp robustness
**File:** `src/connectors/yahoo_client.py:~259`
**Change:** Reject quotes with `source_ts == 0.0` instead of silently publishing them. Log at WARNING level.

#### 10.13 Bybit WS: skip invalid quotes instead of zeroing
**File:** `src/connectors/bybit_ws.py:~313`
**Change:** Return early (skip the quote) instead of falling back to zeroed values.

#### 10.14 Tastytrade snapshot NULL IV documentation
**Change:** Document in `data_sources` policy that Tastytrade REST snapshots have `atm_iv = NULL` by design; DXLink streaming is required for IV. Add a health check that warns if DXLink is not connected and Tastytrade is the primary snapshot source.

---

## 11. Audit Methodology

1. **Architecture review** — read orchestrator, event bus, all connectors
2. **Replay determinism** — traced all timestamp comparisons, checked for RNG/wall-clock, verified sort stability
3. **Provider semantics** — verified UTC normalization, reconnection logic, rate limiting, data structure compatibility across all 7 connectors
4. **Strategy correctness** — verified VRP math, Black-Scholes formulas, feature builder rolling windows, regime hysteresis, spread generator strike ordering
5. **Data integrity** — checked SQL injection, schema drift, write durability, transaction isolation
6. **Event bus safety** — checked for blocking handlers, deadlock risks, back-pressure behavior
7. **Orchestrator lifecycle** — verified startup ordering, shutdown grace, task tracking
8. **Secrets handling** — checked file permissions, logging exposure, YAML safety

All findings are based on static analysis of source code as of the audit date.
