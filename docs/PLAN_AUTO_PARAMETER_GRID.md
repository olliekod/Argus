# Plan: Automatic Parameter Grid — Completed

**Status:** Implemented and verified.

Sweep YAML values can be specified as numeric range specs in addition to explicit lists; the research loop expands ranges after `yaml.safe_load` so `ExperimentRunner.run_parameter_grid` still receives `Dict[str, List[Any]]`.

---

## Implemented

| Item | Location | Verification |
|------|----------|--------------|
| **expand_sweep_grid** | `src/analysis/sweep_grid.py` | `expand_sweep_grid(raw \| None) -> dict[str, list]`; lists as-is, scalars → `[v]`, range dicts → expanded list via Decimal; optional `round`; step or num_steps; ValueError on bad input. |
| **Integration** | `scripts/strategy_research_loop.py` | `grid = expand_sweep_grid(yaml.safe_load(f))` before `runner.run_parameter_grid(...)`. |
| **VRP sweep file** | `config/vrp_sweep.yaml` | Range for `min_vrp`, lists for `max_vol_regime` / `avoid_trend`. |
| **Docs** | `docs/strategy_research_loop.md` | Parameter sweeps section documents explicit list, range spec (`min`/`max`/`step` or `num_steps`), scalar wrap, and `config/vrp_sweep.yaml`. |
| **Tests** | `tests/test_sweep_grid.py` | Mixed inputs, step vs num_steps, rounding stability, scalar wrap, invalid-spec ValueError. |
| **Alpaca = bars/outcomes only** | `src/core/data_sources.py`, `src/strategies/vrp_credit_spread.py` | `_ALLOWED_OPTIONS_SNAPSHOT_PROVIDERS` = `{"tastytrade", "public"}`; VRP never uses Alpaca for IV; no `allow_alpaca_iv`. |

---

## Verification (double-check)

- **sweep_grid.py:** Range spec requires `min`/`max` and either `step` or `num_steps`; uses `Decimal`; inclusive endpoint when within 1e-12; rounding from `round` or inferred from step or default 4.
- **strategy_research_loop.py:** Imports `expand_sweep_grid`; when `spec.sweep` is set and file exists, opens file → `grid = expand_sweep_grid(yaml.safe_load(f))` → `runner.run_parameter_grid(strat_cls, base_config, grid)`.
- **config/vrp_sweep.yaml:** Exists; `min_vrp` range 0.02–0.08 step 0.01; `max_vol_regime`, `avoid_trend` lists; no `allow_alpaca_iv`.
- **config/research_loop.yaml:** `sweep: "config/vrp_sweep.yaml"` for VRP strategy.
- **data_sources.py:** Alpaca excluded from allowed options snapshot providers; comment states Alpaca is bars/outcomes only.
- **vrp_credit_spread.py:** No `allow_alpaca_iv`; `_select_iv_from_snapshots` has no Alpaca fallback; replay fallback excludes Alpaca (`non_alpaca`).
- **Tests:** `pytest tests/test_sweep_grid.py tests/test_data_sources.py tests/test_research_loop.py tests/test_experiment_runner.py tests/test_iv_consensus_engine.py` → 77 passed.

---

## What’s next

- **NEXT_STEPS_IMPLEMENTATION_PLAN.md** — Research–allocation loop, P1/P2 audit items: all implemented and verified.
- **OVERNIGHT_SESSION_STRATEGY_PLAN.md** — Next strategy work: Phase 1 (OvernightSessionStrategy using existing bars/outcomes/session regime), then Phase 2 (global ETF proxies, Alpha Vantage).
- **Ongoing:** Use the research engine to prove edge (VRP, overnight experiments); document findings; feed into strategy priority and allocation design (per NEXT_STEPS §5).
