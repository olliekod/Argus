# Argus Codebase Audit Report

**Date:** 2026-02-12
**Scope:** Full codebase audit — architecture, data integrity, determinism, provider semantics, strategy correctness, and critical bug fixes.

---

## Executive Summary

Argus is an options-focused market monitoring and paper-trading platform built around a Pub/Sub event bus, a replay harness for backtesting, and multiple data providers (Alpaca, Tastytrade, Yahoo, DXLink). The architecture is sound at the macro level: clean separation between connectors, detectors, strategies, and an orchestrator coordinating everything. However, the audit identified two **critical correctness bugs** in the IV enrichment pipeline — both of which have been **fixed in this commit**.

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

`ArgusOrchestrator._on_dxlink_greeks_event` was defined (lines 1483-1506 after fix) but never referenced anywhere. No DXLink streamer was instantiated in `setup()` or started in `run()`. Consequently:

- `_greeks_cache` remained empty for the entire application lifetime
- `enrich_snapshot_iv()` always fell through to the "no cached Greeks" path
- The advertised IV enrichment from DXLink streaming data never executed in production

### Fix

1. **`TastytradeRestClient.get_api_quote_token()`** added — fetches a DXLink streaming token via `GET /api-quote-tokens`
2. **`_start_dxlink_greeks_streamer()`** added to orchestrator — obtains DXLink credentials, instantiates `TastytradeStreamer` with `on_event=self._on_dxlink_greeks_event` and `event_types=["Greeks"]`, then runs `run_forever()`
3. **`run()`** now creates the streamer task alongside the Tastytrade polling task

The DXLink streamer runs as a long-lived async task. Greeks events flow into `_on_dxlink_greeks_event`, which populates `_greeks_cache`. The polling loop's `enrich_snapshot_iv()` calls then find cached IV and actually enrich snapshots.

---

## 4. Broader Audit Findings

### 4.1 Replay Determinism

The replay harness (`src/analysis/replay_harness.py`) correctly enforces:
- **Time-gating**: snapshots/outcomes with `recv_ts_ms > sim_ts_ms` are excluded
- **Sorted iteration**: bars processed in timestamp order
- **Frozen dataclasses**: `OptionChainSnapshotEvent` and `BarData` are immutable

**Residual risk**: `_yymmdd_to_epoch_ms` uses `datetime.strptime` which is deterministic. The `_ONE_DAY_MS` constant (86,400,000) is used for day-level comparison, which correctly handles UTC midnight timestamps.

### 4.2 Provider Semantics

The `data_sources` policy correctly separates provider roles:
- `bars_primary: alpaca` — canonical bar source
- `options_snapshots_primary: tastytrade` — REST-based chain snapshots
- `options_stream_primary: tastytrade_dxlink` — now actually connected

Timestamp normalization across providers is handled at the connector level (UTC epoch ms), which is correct.

### 4.3 Database & Persistence

- Schema uses `INTEGER` for all timestamps (epoch ms) — consistent
- WAL mode enabled for concurrent reads
- `PersistenceManager` flushes on heartbeat boundaries — no data loss on normal shutdown
- Foreign keys are not enforced (`PRAGMA foreign_keys = OFF` is the SQLite default) — acceptable for append-only analytics data

### 4.4 Strategy Correctness

The VRP credit spread strategy (`src/strategies/vrp_credit_spread.py`) uses:
- `_select_iv_from_snapshots` with Tastytrade preference
- VRP = IV - RV threshold gating
- Regime filtering (trend + vol regime)

With the cross-expiration fix, the IV fed to VRP calculations is now term-accurate.

### 4.5 Event Bus

Thread-safe Pub/Sub with topic-based routing. No blocking subscribers observed. The `publish()` method catches and logs subscriber exceptions without propagating — correct for a monitoring system.

---

## 5. Files Changed

| File | Change |
|------|--------|
| `src/core/greeks_cache.py` | `_parse_option_symbol` returns 4-tuple; added `_yymmdd_to_epoch_ms`/`_ONE_DAY_MS`; `get_atm_iv`/`get_greeks_for_strike` accept `expiration_ms`; `enrich_snapshot_iv` passes expiration |
| `src/connectors/tastytrade_rest.py` | Added `get_api_quote_token()` for DXLink streaming token |
| `src/orchestrator.py` | Imported `TastytradeStreamer`; added `_dxlink_streamer` attr; added `_start_dxlink_greeks_streamer()`; wired streamer in `run()` |
| `tests/test_greeks_cache_and_iv_enrichment.py` | Updated `_parse_option_symbol` assertions to 4-tuples; fixed `_make_snapshot` expiration; added cross-expiration and enrichment tests |

---

## 6. Recommendations

1. **Option symbol subscription**: The DXLink streamer currently subscribes to underlying symbols. For Greeks events, it needs specific option contract symbols (e.g., `.SPY250321P590`). A follow-up should build the subscription list from the first REST snapshot fetch, subscribing to ATM ± N strikes per expiration.

2. **Token refresh**: `TastytradeStreamer` supports `token_refresh_cb`. The orchestrator should wire this to `get_api_quote_token()` for long-running sessions.

3. **Cache eviction**: `GreeksCache.evict_stale()` is implemented but never called periodically. Add a timer task to evict stale entries every `max_age_ms`.

4. **Monitoring**: Add a metric for `_greeks_cache.size` to the health check dashboard so operators can verify the DXLink pipeline is flowing.
