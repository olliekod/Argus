# Do Telegram/Dashboard Metrics Affect Strategy Decisions?

This document clarifies which of the indicators you see (BTC IV, Regime, Global Risk Flow, Conditions) **actually feed into** Argus when testing strategies and when deciding which strategy to test.

---

## Short answer

| Indicator | Affects strategy behavior during replay/backtest? | Affects which strategies get tested? | Affects evaluation (scores/kills)? |
|-----------|---------------------------------------------------|--------------------------------------|------------------------------------|
| **Regime** (Vol / Trend) | **Yes** — VRP, Overnight, Dow gate on it | No | **Yes** — regime breakdown, regime-stress, penalties |
| **Global Risk Flow** | **Yes** — Overnight only (when gating enabled) | No | No (not in evaluator) |
| **BTC IV** | **No** — replay uses option IV from snapshots | No | No |
| **Conditions (warmth 1–10)** | **No** — not passed to replay strategies | No | No |

**Which strategies get tested** is determined only by **config** (the list in `research_loop.yaml`). Nothing in the briefing (BTC IV, regime, risk flow, conditions) is used to **select** which strategy to run. Regime and Global Risk Flow **do** affect **how** those strategies behave during each run.

---

## 1. During replay (strategy testing)

Replay runs bars, outcomes, **regimes**, and (for VRP) option snapshots through each strategy. The strategy sees `visible_regimes` and (for VRP) `visible_snapshots`.

### Regime (vol_regime, trend_regime)

- **VRPCreditSpreadStrategy**  
  Reads SPY from `visible_regimes`. It **will not open** if:
  - `vol_regime == "VOL_SPIKE"` or  
  - `trend_regime == "TREND_DOWN"`  
  So regime **directly changes** whether VRP trades in replay.

- **OvernightSessionStrategy**  
  Can read `global_risk_flow` from `visible_regimes` when `gate_on_risk_flow=True`. If risk flow &lt; `min_global_risk_flow`, it **skips entry**. So regime/risk flow **directly change** overnight behavior when gating is on.

- **DowRegimeTimingGateStrategy**  
  Uses symbol regime (vol/trend) for timing.

So: **Regime (and, for Overnight, Global Risk Flow) do contribute to the decisions Argus makes when testing strategies** — they gate entries.

### Global Risk Flow

- Injected into replay pack **regimes** as `global_risk_flow` in `metrics_json`.
- Only **OvernightSessionStrategy** uses it, and only when `gate_on_risk_flow=True` in params.
- So it **contributes** to overnight replay decisions when that gate is enabled.

### BTC IV and Conditions (warmth score)

- **Not** passed into the replay harness or strategy `on_bar` / `generate_intents`.
- VRP uses **option atm_iv** (and realized vol) from **option snapshots** in the pack, not BTC IV from Deribit.
- So **BTC IV and the Conditions score do not affect strategy behavior during replay/backtest.**

---

## 2. Deciding which strategy to test

The research loop (`strategy_research_loop.py`) runs **every strategy listed** in `config/research_loop.yaml` (and its sweep files). It does **not** use:

- Current BTC IV  
- Current regime  
- Global Risk Flow  
- Conditions score  

to **choose** which strategies to run. So **none of the briefing metrics affect which strategy gets tested**; that’s purely config-driven.

---

## 3. After the run: evaluation

After each experiment, the **StrategyEvaluator** uses **regime_breakdown** (PnL and bar counts per regime bucket from the run) to:

- Apply **regime_dependency_penalty** (e.g. if >80% of PnL is from one regime).
- Compute **regime_sensitivity_score** (how balanced performance is across regimes).
- Optionally run **regime-stress** (subset runs by regime).

So **regime does affect evaluation**: it can lower scores and trigger kill reasons. It does **not** decide which strategies are run; it decides how we **score and filter** the results.

---

## 4. Where BTC IV and Conditions *are* used

- **Paper trader farm** (when enabled): each paper trader has a `warmth_min`; the **conditions score** (which uses BTC IV, funding, momentum) must be ≥ that to allow a live/paper entry. So BTC IV **indirectly** affects **paper/live** entries via the conditions score.
- **IBIT detector** (live): uses conditions in its signal checklist.
- **Telegram / dashboard**: all of these (BTC IV, Rank, Regime, Global Risk Flow, Conditions) are for **display and context** only, except where they are explicitly wired into strategies (regime, risk flow) or paper (conditions) as above.

---

## Summary

- **Regime** and **Global Risk Flow** (for Overnight with gating on) **do** contribute to the decisions Argus makes **during** strategy testing (replay).
- **BTC IV** and **Conditions** do **not** affect replay strategy logic; they affect **paper/live** gating and **display**.
- **None** of these metrics are used to **decide which strategy to test**; that’s config-only. **Regime** then contributes to **evaluation** (scores and kill decisions) after the run.
