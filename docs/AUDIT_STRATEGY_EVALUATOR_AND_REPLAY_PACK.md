# Audit: Strategy Evaluator & Replay Pack (Post-PR)

**Date:** 2025-02-11  
**Scope:** Code pushed by Claude implementing the strategy evaluation framework and replay pack snapshot support, per the original PR summary and prompt.

---

## Executive Summary

The implementation **matches the PR summary** and prompt deliverables. One **critical bug** was found and fixed: `ExperimentRunner` passed raw pack snapshot **dicts** to `ReplayHarness`, which expects **`MarketDataSnapshot`** objects (and uses `s.recv_ts_ms`), causing a runtime failure when running experiments with packs that contain snapshots. After the fix, all 62 relevant tests pass.

---

## 1. PART 1 — Replay Pack Snapshot Support ✅

| Requirement | Status | Notes |
|-------------|--------|--------|
| Load `option_chain_snapshots` from DB for symbol + date range | ✅ | `_fetch_snapshots()` in `replay_pack.py` calls `db.get_option_chain_snapshots()` |
| Include timestamp_ms, recv_ts_ms, provider, underlying_price, atm_iv, quotes_json | ✅ | All present in snapshot dicts; `atm_iv` nullable |
| Chronological ordering | ✅ | Sorted by `recv_ts_ms` after fetch |
| Include snapshots in replay pack JSON | ✅ | Pack has `"snapshots": snapshots` and metadata `snapshot_count` |
| ReplayHarness receives bars + outcomes + regimes + snapshots | ✅ | `ExperimentRunner.run()` and `run_walk_forward()` pass all four (after fix below) |
| Snapshot gating by recv_ts_ms | ✅ | Harness uses `sim_ts_ms >= snapshot.recv_ts_ms` in loop |
| Graceful degradation for symbols without options | ✅ | Empty snapshot list; no errors |
| Tests: snapshots included, ordering, harness loads correctly | ✅ | `tests/test_replay_pack_snapshots.py` |

**Bug fixed:** `ExperimentRunner` was passing `pack.get("snapshots", [])` (list of dicts) directly to `ReplayHarness`. The harness sorts and uses `s.recv_ts_ms` (attribute access), so it expects `MarketDataSnapshot` instances. **Fix:** Added `_pack_snapshots_to_objects()` to convert pack snapshot dicts to `MarketDataSnapshot` and use the converted list in both `run()` and `run_walk_forward()`.

---

## 2. PART 2 — Universe Support ✅

| Requirement | Status | Notes |
|-------------|--------|--------|
| CLI supports --symbol or --universe | ✅ | Mutually exclusive group in `_build_parser()` |
| Universe mode loads symbols from liquid_etf_universe | ✅ | `get_liquid_etf_universe()` used in `create_universe_packs()` |
| Example: python -m src.tools.replay_pack --universe --start ... --end ... | ✅ | Supported |
| Write packs per symbol to data/packs | ✅ | Default output dir `data/packs`; single-symbol default `data/packs/<symbol>_<start>_<end>.json` |
| Pack file naming | ✅ | `<symbol>_<start_date>_<end_date>.json` (more precise than single date) |
| Do NOT force options for symbols without options data | ✅ | Empty snapshots list per symbol when no data |

---

## 3. PART 3 — Strategy Evaluator ✅

| Requirement | Status | Notes |
|-------------|--------|--------|
| Load experiment JSON outputs | ✅ | `StrategyEvaluator.load_experiments()` |
| Metrics: pnl, sharpe proxy, max drawdown, expectancy, profit factor, trade counts, reject ratios, regime-conditioned | ✅ | `extract_metrics()` + `compute_regime_scores()` |
| Deterministic / edge-case handling: no trades, no losses, zero variance, NaN safety | ✅ | `_safe_float()`, defaults, tests in TestEdgeCases |
| Composite scoring: return, sharpe, drawdown penalty, reject penalty, robustness penalty, regime dependency penalty | ✅ | `compute_composite_score()` with DEFAULT_WEIGHTS |
| Robustness analysis (parameter fragility across sweeps) | ✅ | `compute_robustness_penalty()` using CV of PnL by strategy class |
| Walk-forward stability penalty | ✅ | `compute_walk_forward_penalty()` (sign consistency across same config) |
| Output logs/strategy_rankings_<date>.json | ✅ | `save_rankings()` with default path |
| Manifest references in output | ✅ | Each ranking has `manifest_ref` and `source_file` |

---

## 4. PART 4 — CLI ✅

| Requirement | Status | Notes |
|-------------|--------|--------|
| scripts/evaluate_strategies.py | ✅ | Present |
| Usage: --input logs/experiments | ✅ | Default input dir |
| Output ranking JSON + console summary | ✅ | `save_rankings()` + `print_summary()`; `--output` and `--quiet` supported |

---

## 5. PART 5 — Tests ✅

| Requirement | Status | Notes |
|-------------|--------|--------|
| tests/test_strategy_evaluator.py | ✅ | 57 tests: _safe_float, extract_metrics, all penalties, regime scores, composite, deterministic ranking, edge cases, full pipeline |
| tests/test_replay_pack_snapshots.py | ✅ | Snapshot inclusion, ordering, harness loading, recv_ts_ms gating, round-trip |
| Deterministic ranking, metric edge cases, robustness penalty, regime scoring | ✅ | Covered by test_strategy_evaluator.py |

---

## 6. PART 6 — Documentation ✅

| Requirement | Status | Notes |
|-------------|--------|--------|
| docs/strategy_evaluation.md | ✅ | Metrics, composite scoring, penalty thresholds, deployability interpretation, red flags, checklist, output format, replay pack integration |

---

## 7. Implementation Details Verified

- **Min-max normalization:** Return and Sharpe normalized across experiments in the set. ✅  
- **Robustness:** CV ≤ 0.3 → no penalty, CV ≥ 2.0 → full penalty, zero mean → 0.5. ✅  
- **Walk-forward:** Same strategy class + same params; sign consistency &lt; 80% penalized. ✅  
- **Regime breakdown:** Per-regime PnL from portfolio summary; concentration ratio for penalty. ✅  
- **Database:** `get_option_chain_snapshots(symbol, start_ms, end_ms)` exists and returns rows with `timestamp_ms`, `recv_ts_ms`, etc. ✅  

---

## 8. Optional / Minor Gaps (Non-Blocking)

- **Replay pack CLI:** `bar_duration` is not exposed as a CLI argument (default 60 is hardcoded in `create_replay_pack`). The prompt did not require it; can be added later if needed.
- **Universe pack naming:** Prompt said `data/packs/<symbol>_<date>.json`; implementation uses `<symbol>_<start>_<end>.json`, which is clearer for date ranges.

---

## 9. Changes Made During Audit

1. **ExperimentRunner snapshot conversion**  
   - Added `_pack_snapshots_to_objects(snapshot_dicts)` to convert replay pack snapshot dicts to `MarketDataSnapshot`.  
   - `run()` now passes `snapshots_objs` (not raw dicts) to `ReplayHarness`.  
   - `run_walk_forward()` now loads snapshots from packs, converts them, and passes `snapshots_objs` to `ReplayHarness` for consistency.

---

## 10. Test Run

```
pytest tests/test_strategy_evaluator.py tests/test_replay_pack_snapshots.py tests/test_experiment_runner.py -v --tb=short
# 62 passed
```

---

## Conclusion

The PR deliverables are **complete** and aligned with the summary and prompt. The only critical issue was the snapshot type mismatch in `ExperimentRunner`, which is fixed. The codebase is in good shape for strategy evaluation and replay pack usage with option snapshots.
