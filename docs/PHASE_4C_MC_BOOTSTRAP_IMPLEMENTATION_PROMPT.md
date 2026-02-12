# Phase 4C: MC/Bootstrap on Realized Trades — Implementation Prompt for Codex

**Use this document as the prompt/spec when implementing the remaining Phase 4C work.** The Master Plan (MASTER_PLAN.md §1 and §8.2b) is the authority; this file turns it into a concrete implementation plan.

---

## Copy-paste prompt for Codex (start here)

```
Implement the "MC/bootstrap on realized trades" part of Phase 4C as specified in docs/PHASE_4C_MC_BOOTSTRAP_IMPLEMENTATION_PROMPT.md.

Summary: Replay already exists and produces a trade list. We need to (1) expose the ordered sequence of realized trade PnLs from each replay run; (2) add a new module that resamples that list (iid or block bootstrap) to generate N equity paths and computes median return, 5th percentile drawdown, ruin probability, fraction of paths positive; (3) add kill rules so strategies that survive only a small fraction of paths or have high ruin risk are auto-killed; (4) integrate with ExperimentRunner and StrategyEvaluator so MC summary is stored in the experiment artifact and evaluator adds MC-based kill reasons. Prefer block bootstrap; tests for determinism and kill vs no-kill. Do not change MASTER_PLAN.md. Full spec is in the doc above.
```

---

## 1. Context: What Already Exists

Codex already implemented:

- **Regime sensitivity scoring** in `strategy_evaluator.py` (PnL-per-bar dispersion across regime buckets; integrated into composite score).
- **Kill thresholds and kill list** in `StrategyEvaluator`: configurable thresholds, per-run kill reasons, output `kill_thresholds` and `killed` list (e.g. `robustness_penalty`, `walk_forward_penalty`, `regime_dependency_penalty`, `composite_score_min`).
- **Walk-forward persistence**: `ExperimentRunner.run_walk_forward()` persists each test window with deterministic run IDs and `manifest.walk_forward`; helpers to filter outcomes/regimes/snapshots by window.
- **Regime-stress** in `src/analysis/regime_stress.py`: `map_bars_to_regime_keys()`, `run_regime_subset_stress()` — one replay per regime bucket; returns `subsets`, `fraction_profitable`, `stress_score`.
- **CLI**: `run_experiment.py --regime-stress`, `evaluate_strategies.py --kill-thresholds` and `--output-killed`.

**What is not yet implemented:** Monte Carlo / bootstrap over the **realized trade list** from a single replay run. Regime-stress runs many replays (one per regime); MC/bootstrap takes **one** replay’s trade PnL sequence and resamples it to get many alternative equity paths, then applies kill rules from the path distribution.

---

## 2. Goal

Implement **MC/bootstrap on realized trades** so that:

1. For each strategy/parameter set we can: **run replay once → get the ordered list of realized trade PnLs → resample/reorder that list to generate N alternative equity paths.**
2. From the N paths we compute: **median terminal return, 5th percentile (worst) drawdown, ruin probability** (and optionally: 5th percentile terminal return, fraction of paths that are “acceptable”).
3. We **kill** the strategy/param set if: median return is bad, worst 5–10% of paths are unacceptable, or there is meaningful ruin risk. Exact thresholds can be configurable.
4. **Bootstrap (block sampling)** is the preferred mode: sample **blocks** of consecutive trades (e.g. stationary bootstrap) so that volatility/regime clustering is preserved; plain iid resampling is acceptable as a simpler first step but block bootstrap should be implemented and used by default.

**Authoritative definition:** See MASTER_PLAN.md §1 (Terminology) and §8.2b (“MC / bootstrap on realized trades”). This is **not** option pricing or Heston; it is Monte Carlo over **realized outcomes only** — reordering/resampling the trade list to test sequence dependence and path risk.

---

## 3. Pipeline (Required Flow)

```
Replay (existing) → trade list (ordered PnL per closed trade)
                         ↓
              MC/Bootstrap module
                - Input: list of trade PnLs, starting_cash, n_paths, method="bootstrap"|"iid"
                - Resample: either iid draw with replacement, or block bootstrap (e.g. stationary bootstrap)
                - Build N equity paths (cumulative PnL from start)
                - Per path: terminal_return, max_drawdown, hit_ruin (e.g. equity < fraction of start)
                         ↓
              Aggregate: median return, 5th pct return, 5th pct max_drawdown, P(ruin)
                         ↓
              Kill rule: e.g. kill if median_return < threshold OR 5th pct drawdown > threshold OR P(ruin) > threshold
                         ↓
              Integrate with StrategyEvaluator: new kill reason e.g. "mc_bootstrap_penalty" or "path_stability"
```

---

## 4. Implementation Tasks

### 4.1 Expose trade PnL sequence from replay

- The replay harness’s `VirtualPortfolio` has `_closed_positions`; each position has `realized_pnl`. The **ordered** sequence of trade PnLs (by exit time) is the input to MC/bootstrap.
- **Action:** Ensure this sequence is available to callers. Prefer adding to the artifact that replay already produces (e.g. in `portfolio_summary` or `ReplayResult`) a field such as `trade_pnls: List[float]` — ordered list of realized PnL per closed trade. If you prefer not to change the summary schema, alternatively expose a small helper that, given a `ReplayResult` or the harness after `run()`, returns `List[float]` (e.g. from `harness.portfolio.closed_positions` in order). The MC/bootstrap module must receive **only** the list of trade PnLs and starting capital; no need to re-run replay.

### 4.2 New module: path resampling (MC/bootstrap)

- **Location:** e.g. `src/analysis/mc_bootstrap.py` (or under `regime_stress` if you prefer a single “stress” package).
- **Functions/API (suggested):**
  - `run_mc_paths(trade_pnls: List[float], starting_cash: float, n_paths: int, method: str = "bootstrap", block_size: Optional[int] = None, random_seed: Optional[int] = None) -> Dict[str, Any]`
    - If `method == "iid"`: sample trades with replacement to form each path (same length as original).
    - If `method == "bootstrap"`: use block bootstrap (e.g. stationary bootstrap or fixed-length blocks); preserve approximate sequence structure. Document block choice (e.g. expected block length from data or parameter).
    - Build `n_paths` equity paths: for each path, cumulative sum of sampled trade PnLs; optionally prepend starting_cash so path is [starting_cash, starting_cash + pnl_1, ...].
    - Return dict with at least: `paths` (list of equity curves or terminal returns only to save memory), `median_return`, `p5_return` (5th percentile terminal return), `p5_max_drawdown`, `p95_max_drawdown`, `ruin_probability` (fraction of paths where equity fell below e.g. 20% of starting_cash or 0), `fraction_positive` (fraction of paths with terminal return > 0).
  - Optional: `compute_ruin_probability(paths: List[List[float]], starting_cash: float, ruin_level: float = 0.2) -> float` (fraction of paths where equity ever went <= ruin_level * starting_cash).
- **Determinism:** Use `random_seed` so that same inputs + seed give same paths (for tests and reproducibility).

### 4.3 Kill rules and thresholds

- Add configurable thresholds, e.g.:
  - `mc_median_return_min`: kill if median terminal return (or median total PnL) is below this.
  - `mc_p5_drawdown_max`: kill if 5th percentile max drawdown (worst paths) is above this (e.g. 40% of starting capital).
  - `mc_ruin_prob_max`: kill if estimated ruin probability > this (e.g. 0.05).
  - `mc_fraction_positive_min`: kill if fraction of paths with positive terminal return is below this (e.g. 0.20 → “survives only 20% of paths”).
- Implement a small function or method that, given the output of `run_mc_paths`, returns “killed: bool” and “reason: str” (e.g. `mc_fraction_positive`, `mc_ruin_prob`, `mc_median_return`, `mc_p5_drawdown`).

### 4.4 Integration with experiment runner and evaluator

- **Option A — Post-replay in experiment runner:** After each `run()` (or after walk-forward windows), call the MC/bootstrap module on the trade list for that run; compute path stats; if kill rule triggers, write a flag or reason into the experiment artifact (e.g. `manifest.mc_bootstrap` or `result.mc_bootstrap` with `killed`, `reason`, `metrics`). The evaluator then treats this as a kill reason when present.
- **Option B — Inside evaluator:** Evaluator loads experiment JSONs. If an artifact includes a `trade_pnls` (or equivalent) field, run MC/bootstrap on it and add to kill reasons. This requires either storing `trade_pnls` in the artifact (could be large) or re-running replay to get trade list (wasteful). Prefer **Option A** so replay runs once and MC runs once per run; store only the **MC summary** (median return, p5 drawdown, ruin prob, fraction_positive) in the artifact, not the full path set.
- **Recommendation:** Implement Option A: in `ExperimentRunner.run()` (and optionally in `run_walk_forward()` for each window), after `harness.run()`, get the trade PnL list from the result (or harness), call `run_mc_paths`, then apply kill rules; attach a compact `mc_bootstrap` block to the saved manifest/result. StrategyEvaluator then checks for `result.mc_bootstrap.killed` (or similar) and adds to `kill_reasons` with reason and threshold. Ensure the evaluator’s `kill_thresholds` can include the new MC keys (e.g. `mc_fraction_positive_min`, `mc_ruin_prob_max`).

### 4.5 CLI and config

- **run_experiment.py:** Add an optional flag, e.g. `--mc-bootstrap` (and optionally `--mc-paths 1000`, `--mc-method bootstrap`), to enable MC/bootstrap after each run. When enabled, run MC/bootstrap on the replay’s trade list and persist the summary (and kill decision) in the experiment artifact.
- **evaluate_strategies.py:** Already supports `--kill-thresholds`; ensure the evaluator can apply MC kill reasons from the artifact; if thresholds are passed via YAML/JSON, allow keys like `mc_fraction_positive_min`, `mc_ruin_prob_max` so they can be tuned without code changes.

### 4.6 Tests

- **Unit tests** (e.g. `tests/test_mc_bootstrap.py` or under existing Phase 4C test file):
  - Given a fixed list of trade PnLs and a seed, `run_mc_paths` returns deterministic outputs (same metrics for same inputs + seed).
  - With a clearly “bad” trade list (e.g. one big loss that dominates), ruin probability or fraction_positive is bad and kill rule fires.
  - With a “good” list (many small positive trades), median return and fraction_positive are acceptable and kill rule does not fire.
  - Block bootstrap produces different path distribution than iid when given a list with obvious clustering (e.g. losses in a block); sanity check that block bootstrap runs and returns valid metrics.
- **Integration:** One test that runs a small replay (or mocks a ReplayResult with `trade_pnls`), runs MC/bootstrap, and checks that the kill reason appears in the evaluator’s `killed` list when thresholds are set to strict values.

---

## 5. Out of Scope for This Task

- **Regime-stress** is already implemented; do not duplicate it. MC/bootstrap is about **resampling the trade list from one replay**, not about running multiple replays per regime.
- **Option pricing / Heston:** Do not use or reference the GPU/Heston Monte Carlo. This is only over realized trade PnLs.
- **Deflated Sharpe, Reality Check, slippage sensitivity:** Mentioned in Phase 4C but separate; not required in this implementation prompt.

---

## 6. Acceptance Criteria

- Replay (or its result) exposes the ordered sequence of realized trade PnLs for a run.
- A new module can take that list + starting_cash + n_paths + method and produce: median return, 5th percentile return/drawdown, ruin probability, fraction of paths positive.
- Block bootstrap (preferred) is implemented and used by default; iid resampling is allowed as fallback.
- Kill rules based on these metrics are applied and surface in the evaluator’s kill list (e.g. `mc_fraction_positive`, `mc_ruin_prob`, etc.) with configurable thresholds.
- Experiment runner (and optionally walk-forward) can optionally run MC/bootstrap and store a compact summary in the experiment artifact.
- Unit tests cover determinism, “bad” list → kill, “good” list → no kill, and block vs iid behavior; integration test ties replay → MC → evaluator kill.
- MASTER_PLAN.md is not modified by Codex unless the user asks; this prompt is the implementation spec.

---

## 7. One-Paragraph Summary for Codex

Implement **Monte Carlo / bootstrap on the realized trade list** from replay: add a way to get the ordered list of trade PnLs from each replay run; implement a module that resamples that list (iid or, preferably, block bootstrap) to generate N equity paths; from the paths compute median return, 5th percentile drawdown, ruin probability, and fraction of paths with positive return; add kill rules so that strategies that survive only a small fraction of paths or have high ruin risk are automatically killed; integrate with ExperimentRunner and StrategyEvaluator so MC summary is stored in the experiment artifact and evaluator adds MC-based reasons to the kill list. Tests must verify determinism and kill vs no-kill for good/bad trade lists. See MASTER_PLAN.md §1 and §8.2b for the authoritative definition (MC over realized outcomes only, not option pricing).
