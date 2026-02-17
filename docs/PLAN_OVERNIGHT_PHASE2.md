# Overnight Session Strategy — Phase 2 Plan

**Goal:** Complete the “Data Enhancement” phase so the overnight strategy and regime engine fully use global ETF proxies and Alpha Vantage data, with clear verification and optional extensions.

**Source:** [OVERNIGHT_SESSION_STRATEGY_PLAN.md](OVERNIGHT_SESSION_STRATEGY_PLAN.md) §3 (Phase 2), §5 (Global Risk Flow); [MASTER_PLAN.md](../MASTER_PLAN.md) §8.4.

---

## 1. Phase 2 Scope (Recap)

| Sub-phase | Description |
|-----------|-------------|
| **2a** | Global ETF proxies (10 symbols) — Asia + Europe + EM — in config and data path. |
| **2b** | Alpha Vantage daily backfill (10 ETFs + 4 FX), global risk flow computation, regime integration. |

**Outcome:** Regime detection and overnight strategy have a **global risk flow** signal (Asia/Europe/FX composite); replay packs inject it deterministically; strategy can gate on it (`gate_on_risk_flow`, `min_global_risk_flow`).

---

## 2. Current Status (What’s Already Done)

These are already implemented and tested; no Phase 2 work required for them.

| Item | Location | Verification |
|------|----------|--------------|
| **Alpha Vantage config** | `config/config.yaml` → `exchanges.alphavantage` | 10 `daily_symbols` (EWJ, FXI, EWT, EWY, INDA, EWG, EWU, FEZ, EWL, EEM), 4 `fx_pairs` |
| **Alpha Vantage client** | `src/connectors/alphavantage_client.py` | TIME_SERIES_DAILY, FX_DAILY; rate limit; tests in `tests/test_alphavantage_client.py` |
| **Alpha Vantage collector** | `src/connectors/alphavantage_collector.py` | Daily batch 09:00 ET; 14 symbols → `market_bars` (source=alphavantage, bar_duration=86400); `tests/test_alphavantage_collector.py` |
| **Global risk flow computation** | `src/core/global_risk_flow.py` | 0.4×Asia + 0.4×Europe + 0.2×FX; `get_bars_daily_for_risk_flow`; `tests/test_global_risk_flow.py` |
| **Global risk flow updater** | `src/core/global_risk_flow_updater.py` | DB-only; publishes `ExternalMetricEvent`; `tests/test_global_risk_flow_updater.py` |
| **Regime integration** | `src/core/regime_detector.py` | Subscribes to external metrics; merges `global_risk_flow` into `metrics_json` |
| **Replay pack injection** | `src/tools/replay_pack.py` | Injects `global_risk_flow` into each regime from `get_bars_daily_for_risk_flow`; deterministic |
| **Overnight strategy gating** | `src/strategies/overnight_session.py` | `gate_on_risk_flow`, `min_global_risk_flow`; reads from `visible_regimes` |
| **Orchestrator** | `src/orchestrator.py` | Instantiates collector + updater; `_run_external_metrics_loop`; `get_global_risk_flow()` |
| **DB API** | `src/core/database.py` | `get_bars_daily_for_risk_flow(source, symbols, end_ms, lookback_days)` |

---

## 3. Remaining Work (Phase 2 Checklist)

### 3.1 Config and data (optional / documentation)

| # | Task | Priority | Notes |
|---|------|----------|--------|
| 2.1 | **Add 10 global ETFs to Alpaca/Yahoo symbols (optional)** | Low | Only if you want **intraday bars** (and outcomes) for EWJ, FXI, etc. for strategies that trade them. Today: global ETFs are used only for **daily** risk-flow; overnight strategy trades SPY/QQQ/IBIT/etc. from existing bars. |
| 2.2 | **Document symbol sets** | Medium | In `OVERNIGHT_SESSION_STRATEGY_PLAN.md` or this doc: “Overnight equities: pack.symbols = SPY, QQQ, IBIT, …; global ETFs (EWJ, …) are regime-only unless added to Alpaca symbols.” |

### 3.2 Verification and tests

| # | Task | Priority | Description |
|---|------|----------|-------------|
| 2.3 | **E2E: Pack has global_risk_flow in regimes** | High | Script or test: build a replay pack for a date range where `market_bars` has Alpha Vantage daily bars; assert at least one regime in the pack has `metrics_json` containing `global_risk_flow`. |
| 2.4 | **E2E: Overnight strategy respects gate** | High | Test: run OvernightSessionStrategy on a pack with injected `global_risk_flow`; with `gate_on_risk_flow=true` and `min_global_risk_flow=0.01`, when all regimes have risk_flow < 0.01, no LONG intents (or document current behavior). |
| 2.5 | **Research loop once with gate_on_risk_flow=true** | Medium | Run `strategy_research_loop.py --once` with overnight params including `gate_on_risk_flow: true`; confirm no errors and rankings/allocations reflect overnight runs. |

### 3.3 Defaults and sweep

| # | Task | Priority | Description |
|---|------|----------|-------------|
| 2.6 | **Sweep / default for gating** | Low | Consider enabling `gate_on_risk_flow: true` in `config/overnight_sweep.yaml` or in default params so experiments routinely test risk-flow gating (after 2.3–2.5 pass). |

### 3.4 Documentation

| # | Task | Priority | Description |
|---|------|----------|-------------|
| 2.7 | **Mark Phase 2 complete in OVERNIGHT_SESSION_STRATEGY_PLAN.md** | Medium | Add a “Phase 2 status” subsection: implemented (collector, updater, regime merge, replay injection, gating); remaining: verification (2.3–2.5) and optional config (2.1, 2.6). |
| 2.8 | **MASTER_PLAN.md §8.4** | Low | After verification, add one line: “Overnight Phase 2: global risk flow + Alpha Vantage data path verified; E2E tests in place.” |

---

## 4. Implementation Order

1. **Verification (recommended first)**  
   - 2.3: Add test or script that builds a pack and asserts `global_risk_flow` in regimes (may require fixture DB or seeded `market_bars`).  
   - 2.4: Add test that overnight strategy with `gate_on_risk_flow=true` and high `min_global_risk_flow` skips entries when regimes have low risk flow.  
   - 2.5: Manual run of research loop with gating on.

2. **Documentation**  
   - 2.2: Document symbol sets (equities vs regime-only).  
   - 2.7: Update OVERNIGHT_SESSION_STRATEGY_PLAN.md Phase 2 status.  
   - 2.8: Optional MASTER_PLAN tweak.

3. **Optional**  
   - 2.1: Add global ETFs to Alpaca/Yahoo if you want to trade them.  
   - 2.6: Enable `gate_on_risk_flow` in sweep/defaults.

---

## 5. Out of Scope for Phase 2

- **Phase 3:** CME futures (ES/NQ) via DXLink — separate plan.  
- **Phase 4:** Additional Alpha Vantage endpoints or intraday FX — not required for overnight Phase 2.  
- **Trading global ETFs:** Adding EWJ, FXI, etc. to Alpaca and packing them is an optional extension, not a Phase 2 requirement.

---

## 6. Success Criteria

Phase 2 is **complete** when:

1. Alpha Vantage daily data (10 ETFs + 4 FX) is collected and stored in `market_bars`.  
2. Global risk flow is computed and available to the regime detector (live) and to replay packs (injected).  
3. Overnight strategy can gate on `global_risk_flow` and tests confirm behavior.  
4. Documentation reflects what’s implemented and what’s optional (symbol sets, gating defaults).

(1) and (2) are already true; (3) and (4) are the remaining checklist items above.

---

## 7. References

- [OVERNIGHT_SESSION_STRATEGY_PLAN.md](OVERNIGHT_SESSION_STRATEGY_PLAN.md) — Phase 2 description, global risk flow §5, API §9  
- [MASTER_PLAN.md](../MASTER_PLAN.md) §2 (Global risk flow), §8.4 (Next steps)  
- [src/core/global_risk_flow.py](../src/core/global_risk_flow.py), [src/core/global_risk_flow_updater.py](../src/core/global_risk_flow_updater.py)  
- [src/tools/replay_pack.py](../src/tools/replay_pack.py) — injection step  
- [tests/test_global_risk_flow_updater.py](../tests/test_global_risk_flow_updater.py) — replay pack injection determinism test
