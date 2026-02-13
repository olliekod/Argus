# Next Steps Implementation Plan

**Source:** MASTER_PLAN.md §8.4, §8.1, §8.3  
**Created:** 2026-02-13  
**Status:** Draft

This document translates the Master Plan’s recommended next steps into a concrete implementation plan. It covers: (1) closing the research–allocation loop, (2) P1 audit items, (3) P2 quick wins, and (4) future phases.

---

## 1. Close the Research–Allocation Loop

**Goal:** Connect the Strategy Research Loop to StrategyRegistry and AllocationEngine so that the pipeline becomes: *test → filter → allocate → persist* for consumption by a future paper/live path.

### 1.1 Current State

- **Strategy Evaluator** (`strategy_evaluator.py`): Produces rankings with composite scores, DSR, kill reasons, metrics.
- **Strategy Registry** (`strategy_registry.py`): `load_from_rankings(rankings, min_dsr, min_composite_score)` — populates from evaluator output.
- **Allocation Engine** (`allocation_engine.py`): Consumes `Forecast` objects, outputs target weights/contracts.
- **Sizing** (`sizing.py`): `Forecast(strategy_id, instrument, mu, sigma, edge_score, cost, confidence, meta)`.
- **Research Loop** (`strategy_research_loop.py`): Runs outcomes → packs → experiments → evaluate_and_persist. Stops at rankings and candidate set; does **not** call registry or allocation.

### 1.2 Implementation Tasks

| # | Task | Location | Description |
|---|------|----------|-------------|
| 1.1 | Add allocation step after evaluate_and_persist | `strategy_research_loop.py` | New function `run_allocation(config, evaluator)` or extend `evaluate_and_persist` to optionally run allocation. |
| 1.2 | Load config for allocation | `research_loop_config.py` | Use existing `evaluation.allocation` (AllocationOpts). Add `allocations_output_path`, `equity`, `min_dsr`, `min_composite_score` if missing. |
| 1.3 | Build Forecasts from rankings | `strategy_research_loop.py` | Map each non-killed candidate to `Forecast`: `mu` from total_return or expectancy, `sigma` from rolling vol or backtest vol proxy, `edge_score` from composite_score, `instrument` from pack symbols or strategy default. |
| 1.4 | Load registry from rankings | New helper | `registry.load_from_rankings(evaluator.rankings, min_dsr=config.min_dsr, min_composite_score=config.min_composite_score)`. |
| 1.5 | Run AllocationEngine | `strategy_research_loop.py` | `engine = AllocationEngine(config=AllocationConfig(...), equity=config.equity)`; `allocations = engine.allocate(forecasts)`. |
| 1.6 | Persist allocations | `strategy_research_loop.py` | Write JSON: `{generated_at, equity, config, allocations: [{strategy_id, instrument, weight, dollar_risk, contracts}, ...]}`. Path from `evaluation.allocations_output_path`. |
| 1.7 | Add `max_loss_per_contract` for options | Config / strategy metadata | For VRP/options strategies: allow `max_loss_per_contract` in strategy metadata or config so AllocationEngine can compute contract counts. |

### 1.3 Config Additions

In `EvaluationOpts` / `research_loop.example.yaml`:

```yaml
evaluation:
  # ... existing ...
  allocation:  # already exists
    kelly_fraction: 0.25
    per_play_cap: 0.07
    vol_target_annual: 0.10
  allocations_output_path: "logs/allocations.json"
  equity: 10000.0
  min_dsr: 0.95
  min_composite_score: 0.0
```

### 1.4 Forecast Mapping Logic

From each ranking record:

- `strategy_id` → `Forecast.strategy_id`
- `instrument` → Derive from pack symbols (e.g. first symbol) or strategy params; default `"UNKNOWN"`
- `mu` → `total_return_pct / 100` or `expectancy` (annualized proxy)
- `sigma` → Rolling vol from backtest or `max_drawdown_pct / 2` as proxy if unavailable
- `edge_score` → `composite_score`
- `cost` → Estimate from execution metrics (optional; 0 if unknown)
- `confidence` → From regime_sensitivity_score or 0.5 default

### 1.5 Acceptance Criteria

- [ ] Running `strategy_research_loop.py --once` with `evaluation.allocation` and `allocations_output_path` produces `allocations.json`.
- [ ] Allocations include strategy_id, instrument, weight, dollar_risk, contracts where applicable.
- [ ] No allocation step runs if `evaluation.allocation` is null or `allocations_output_path` is null (backward compatible).

---

## 2. P1 Audit Items

### 2.1 10.2 Deribit Rate Limiter (`.total_seconds()`)

**File:** `src/connectors/deribit_client.py`  
**Status:** Verify — codebase shows `total_seconds()` at lines 79, 84. Audit may be outdated.

| # | Task | Description |
|---|------|-------------|
| 2.1a | Audit deribit_client.py | Grep for `.seconds` on timedelta; replace with `.total_seconds()` if any remain. |
| 2.1b | Add unit test | `tests/test_deribit_client.py`: Test that rate limiter correctly waits when limit exceeded (mock time). |

### 2.2 10.4 Orchestrator Task Tracking

**File:** `src/orchestrator.py`  
**Goal:** Ensure every `asyncio.create_task()` used for long-running work is appended to `self._tasks` and that `stop()` cancels and awaits them.

| # | Task | Description |
|---|------|-------------|
| 2.2a | Inventory create_task call sites | List all `create_task` in `orchestrator.py`. Exclude signal_handler’s `create_task(argus.stop())` — that one triggers shutdown, not a tracked task. |
| 2.2b | Verify run() tasks | Confirm all tasks created in `run()` are appended to `self._tasks`. |
| 2.2c | Verify stop() behavior | Ensure `stop()` cancels and awaits `self._tasks` (e.g. `asyncio.gather(*self._tasks, return_exceptions=True)` or equivalent). |
| 2.2d | Document findings | Add a short comment or docstring in orchestrator describing the task-tracking invariant. |

---

## 3. P2 Quick Wins

### 3.1 10.7 ExecutionModel Reset in Replay

**File:** `src/analysis/replay_harness.py`  
**Task:** At the start of `ReplayHarness.run()`, call `self._exec.reset()` (or equivalent) if the execution model exposes a reset method, so the ledger does not accumulate if the harness is ever reused.

**Note:** ExperimentRunner creates a fresh harness and execution model per run, so accumulation is unlikely. This is a defensive fix. ExecutionModel already has `reset()` — we only need to call it.

| # | Task | Description |
|---|------|-------------|
| 3.1a | Call reset at start of run | `replay_harness.py`: At top of `run()`, call `self.execution_model.reset()`. |
| 3.1b | Add test | Verify that two consecutive `harness.run()` calls produce independent results (no cross-run state). |

### 3.2 10.8 Secrets File Permissions

**File:** `src/core/config.py`  
**Task:** After writing the secrets file, set permissions to `0o600` so only the owner can read.

| # | Task | Description |
|---|------|-------------|
| 3.2a | Add chmod after write | `path.chmod(0o600)` after writing secrets. |
| 3.2b | Add test | `tests/test_config.py`: Verify that a written secrets file has mode `0o600`. |

---

## 4. Future Phases (Not in This Sprint)

Per Master Plan §8.4:

1. **Portfolio risk engine (Phase 5 full)** — Exposure limits, correlation awareness, drawdown containment.
2. **Strategy lifecycle** — Rolling performance, degradation detection, quarantine/kill in production.
3. **Live vs backtest drift monitor** — Compare live fills vs simulated; alert on slippage drift.
4. **StrategyLeague enhancements** — Capital competition, smoothed weight updates, degradation detector, eligibility gate.

These will be scoped in separate implementation plans when prioritized.

---

## 5. Recommended Execution Order

1. **Sprint 1 (this plan):**
   - 1.1–1.7: Close research–allocation loop
   - 2.1a–2.1b: Deribit rate limiter (verify/fix)
   - 2.2a–2.2d: Orchestrator task tracking verification

2. **Sprint 2:**
   - 3.1a–3.1b: ExecutionModel reset
   - 3.2a–3.2b: Secrets file permissions
   - E2E check: Confirm IV truth map, replay experiments produce non-zero VRP trades where expected

3. **Ongoing:**
   - Use research engine to prove edge (VRP, Overnight Session experiments)
   - Document findings; feed into strategy priority and allocation design

---

## 6. References

- [MASTER_PLAN.md](../MASTER_PLAN.md) §8.4 Next steps and recommended route
- [strategy_research_loop.md](strategy_research_loop.md) — Research loop usage
- [strategy_evaluation.md](strategy_evaluation.md) — Evaluator metrics and scoring
- [AUDIT_CODEBASE.md](AUDIT_CODEBASE.md) — Full audit (if present in repo)
