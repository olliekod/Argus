# Argus: Commands Reference & Path to Self-Contained Continuous Program

**Date:** 2026-02-12  
**Purpose:** Single reference for (1) every command you need, (2) current vs end-goal state, (3) next step and a big Claude Opus task.

---

## 1. Codebase scan summary — what’s present and working

| Area | Status | Notes |
|------|--------|--------|
| **Phases 0–4B** | Done | Event bus, bars, regimes, options pipeline, outcome engine, replay harness, experiment runner, execution model. DXLink Greeks + cross-expiration IV fixes applied. |
| **Phase 4C (partial)** | Done | Regime sensitivity scoring, parameter stability auto-kill, **MC/bootstrap on realized trades** (trade_pnls → resample paths → kill rules), regime-stress, walk-forward. Integrated in `ExperimentRunner` and `StrategyEvaluator`. |
| **Public options** | Done | Second IV source; health script; Tastytrade-as-structure path. |
| **Replay pack** | Working | `python -m src.tools.replay_pack` builds packs with bars, outcomes, regimes, snapshots. |
| **Outcome engine** | Working | `python -m src.outcomes` for backfill, coverage, gaps, uptime. |
| **Experiments** | Working | `run_experiment.py` single run + sweep; `--regime-stress`, `--mc-bootstrap`; results in `logs/experiments`. |
| **Evaluation** | Working | `evaluate_strategies.py` loads experiment JSONs, composite scores, kill list, rankings. |
| **StrategyLeague / allocation** | Not implemented | Planned (Phase 8); no registry, capital competition, or promotion/demotion yet. |
| **Strategy lifecycle / kill engine** | Partial | Kill in **research** (evaluator kill list, MC/regime-stress). No **running** lifecycle: no rolling performance, quarantine, or auto-death in production. |
| **Portfolio risk / sizing** | Not implemented | Phase 5; no fractional Kelly, per-play cap, or portfolio risk engine. |
| **Single continuous loop** | Not implemented | No one process that defines strategies → tweaks params → allocates → tests → funds some → retests → kills failures → repeats. |

So: **research and robustness (through 4C) are in place; allocation, sizing, lifecycle, and a single continuous program are not.**

---

## 2. Spot-check errors

- **Three tests fail at collection** (import errors):
  - `tests/test_liquid_etf_universe.py` — imports `_select_sampled_contracts`, `select_spot` from `scripts.tastytrade_health_audit`
  - `tests/test_option_quote_snapshots_schema.py` — imports `_ensure_snapshot_table`, `_prune_snapshots_sql` from same
  - `tests/test_tastytrade_sampling.py` — same as liquid_etf_universe  
  **Cause:** Those helpers no longer exist in `tastytrade_health_audit.py` (refactor). Either reintroduce/alias them in that script or change tests to use current API / skip.

- **README references `python scripts/init_database.py`** but there is no `scripts/init_database.py` in the repo. Either add a small init script or update README to the actual DB init path (e.g. first run of main/orchestrator or another script).

- **Full test run:** Excluding the three broken modules, many tests pass; run took >2 minutes (suite is large). No other critical failures observed in the partial run.

---

## 3. Every command you need (going forward)

### Data & outcomes

```powershell
# Outcomes: backfill for a symbol/date (required before building replay packs)
python -m src.outcomes backfill --provider alpaca --symbol SPY --bar 60 --start YYYY-MM-DD --end YYYY-MM-DD

# Outcomes: backfill all configured providers/symbols
python -m src.outcomes backfill-all --start YYYY-MM-DD --end YYYY-MM-DD

# Outcomes: coverage and diagnostics
python -m src.outcomes list
python -m src.outcomes list-outcomes
python -m src.outcomes coverage
python -m src.outcomes health
python -m src.outcomes gaps --provider alpaca --symbol SPY
python -m src.outcomes uptime --start YYYY-MM-DD --end YYYY-MM-DD
```

### Replay packs (for experiments)

```powershell
# Single symbol
python -m src.tools.replay_pack --symbol SPY --start YYYY-MM-DD --end YYYY-MM-DD --out data/packs/ --provider alpaca

# Full universe
python -m src.tools.replay_pack --universe --start YYYY-MM-DD --end YYYY-MM-DD --out data/packs/ --provider alpaca
```

### Experiments (research engine)

```powershell
# Single run
python scripts/run_experiment.py --strategy VRPCreditSpreadStrategy --pack data/packs/SPY_2026-02-11_2026-02-11.json --output logs/experiments

# With regime-stress and MC/bootstrap (recommended for robustness)
python scripts/run_experiment.py --strategy VRPCreditSpreadStrategy --pack data/packs/SPY_2026-02-11_2026-02-11.json --output logs/experiments --regime-stress --mc-bootstrap --mc-paths 1000

# Parameter sweep (YAML grid)
python scripts/run_experiment.py --strategy VRPCreditSpreadStrategy --pack data/packs/SPY_2026-02-11_2026-02-11.json --sweep config/sweep.yaml --output logs/experiments --mc-bootstrap
```

### Evaluation (rank & kill)

```powershell
# Rank experiments, apply kill thresholds, write rankings
python scripts/evaluate_strategies.py --input logs/experiments --output logs/strategy_rankings.json

# With kill thresholds (YAML or JSON) and output killed list
python scripts/evaluate_strategies.py --input logs/experiments --kill-thresholds config/kill_thresholds.yaml --output-killed logs/killed.json
```

### Options / IV health (when you have live or replay data)

```powershell
python scripts/options_providers_health.py --db data/argus.db --hours 24
python scripts/compare_options_snapshot_providers.py --symbol SPY --start YYYY-MM-DD --end YYYY-MM-DD
```

### System & verification

```powershell
python scripts/verify_system.py
python scripts/verify_system.py --deep
python scripts/e2e_verify.py --symbol SPY --days 5
```

### Legacy / alternate backtest (BITO proxy, not replay-pack)

```powershell
python scripts/run_backtest.py
python scripts/optimize.py
python scripts/apply_params.py
```

### Paper & monitoring (when used)

```powershell
python scripts/paper_performance.py
python scripts/paper_trading_status.py
python scripts/strategy_health.py
python scripts/select_best_trader.py --days 30 --top-n 10
```

### One-shot data pipeline (example: one day SPY)

```powershell
python -m src.outcomes backfill --provider alpaca --symbol SPY --bar 60 --start 2026-02-11 --end 2026-02-11
python -m src.tools.replay_pack --symbol SPY --start 2026-02-11 --end 2026-02-11 --out data/packs/ --provider alpaca
python scripts/run_experiment.py --strategy VRPCreditSpreadStrategy --pack data/packs/SPY_2026-02-11_2026-02-11.json --regime-stress --mc-bootstrap --output logs/experiments
python scripts/evaluate_strategies.py --input logs/experiments
```

**Note:** The plan is to **automate** this. Right now there is no single script or service that runs this pipeline on a schedule or loop; that’s the gap below.

---

## 4. End goal vs current state

**End goal (your words):**  
Argus has strategy sets defined; it tweaks parameters, allocations, capital, risk, sizing; strategies are continually tested; some get funding; they are constantly tested; if they fail they’re killed and the cycle repeats — a **self-contained continuous program**.

**Current state:**

- **Defined strategies:** Yes (e.g. VRPCreditSpreadStrategy, DowRegimeTiming); config/sweep YAML for params.
- **Tweaking parameters:** Yes, via `--sweep` and experiment runner grid.
- **Allocations / capital / risk / sizing:** No. No StrategyLeague, no portfolio risk engine, no fractional Kelly or per-play cap.
- **Continually tested:** Only if you run `run_experiment` + `evaluate_strategies` on a schedule yourself (cron/task scheduler). No in-process loop.
- **Some get funding:** No funding step — no paper/live allocation from “winners.”
- **Kill on failure:** Yes in **research** (evaluator kill list from MC, regime-stress, composite score). No **runtime** kill engine (no rolling PnL, degradation detector, or quarantine in a live loop).
- **Self-contained continuous program:** No. You have a research pipeline (pack → run → evaluate) and separate scripts; no single daemon or scheduler that ties them together.

So the **next step** is to add the **orchestration layer** that turns “research pipeline + evaluation” into a **repeating loop** and then, in order, add allocation/sizing and runtime lifecycle (funding + kill).

---

## 5. Recommended next step (to get there)

**Immediate next step:**  
Implement a **Strategy Research Loop** (or “experiment loop”) that:

1. **Input:** A small config that defines (a) which strategies (and param grids) to consider, (b) which replay packs (or pack generator: symbol + date range) to use, (c) kill thresholds and MC/regime-stress flags.
2. **Loop (e.g. daily or on new data):**
   - Ensure outcomes exist for the pack dates (call outcome backfill or skip if already done).
   - Build replay packs (single symbol or universe as configured).
   - Run experiments (single + sweep) with `--regime-stress` and `--mc-bootstrap`.
   - Run `evaluate_strategies` on the experiment output directory.
   - Persist rankings and killed list; optionally emit a “candidate set” (strategies that passed and are eligible for allocation once that exists).
3. **Output:** Structured results (rankings, killed, run IDs) so a future **StrategyLeague** can consume them.

This gives you a **single, runnable program** (script or small service) that “continually tests” strategies on new or updated data. It does not yet allocate capital or “fund” strategies; that comes after with StrategyLeague + portfolio risk.

**After that (in order):**

- Add **strategy allocation / StrategyLeague** (registry, eligibility gate, smoothed weights, degradation detector) so “candidate set” from the loop gets capital budgets.
- Add **portfolio risk engine** (per-play cap, exposure limits, sizing).
- Add **runtime strategy lifecycle** (rolling metrics, quarantine, kill when edge disappears in live/paper).
- Then paper trading and live execution.

---

## 6. Big next step for Claude Opus 4.6 (heavy/complex task)

Use Claude Opus for the **design and implementation of the Strategy Research Loop** as the first version of the “self-contained continuous program”:

**Task title:** *Design and implement the Argus Strategy Research Loop: a single config-driven process that builds replay packs, runs experiments with regime-stress and MC/bootstrap, evaluates strategies, and persists rankings and killed set on a schedule or on-demand.*

**Suggested scope for Opus:**

1. **Config schema**  
   - Strategies and param grids (or pointers to existing sweep YAML).  
   - Pack spec: symbol list or universe, date range or “last N days,” provider.  
   - Flags: regime-stress, mc-bootstrap, paths, kill thresholds path.  
   - Output dirs and frequency (e.g. daily, or “on new data”).

2. **Loop logic**  
   - Check/create outcomes for pack dates (reuse `src.outcomes` backfill or skip if coverage exists).  
   - Build packs via `src.tools.replay_pack` (CLI or programmatic).  
   - Run experiments (single + sweep) via `ExperimentRunner` (or subprocess to `run_experiment.py`) with the right flags.  
   - Run evaluation via `StrategyEvaluator` (or subprocess to `evaluate_strategies.py`).  
   - Write results (rankings, killed, run IDs) to a known location/format.

3. **Entry point**  
   - One entry point: e.g. `python scripts/strategy_research_loop.py --config config/research_loop.yaml [--once]`  
   - `--once`: run one cycle and exit (for cron).  
   - Without `--once`: optional daemon mode (e.g. run every 24h or when new data is available), if you want it.

4. **Docs and invariants**  
   - Short doc: how to configure and run the loop; how it fits into the future StrategyLeague.  
   - Respect Master Plan invariants: no lookahead, same inputs → same outputs; use existing replay and evaluator, don’t change their contracts.

5. **Tests**  
   - At least one integration test: run one cycle with a tiny pack and a small grid; assert that output dir contains rankings and that killed list format is correct.

This gives you the **first version of the continuous program**: one process that, on a schedule or on-demand, keeps testing strategies and producing a live “candidate set” and “killed set,” ready for a future allocation and lifecycle layer.

---

## 7. Fixing the spot-check errors (optional)

- **Tests:** Either (a) add back to `scripts/tastytrade_health_audit.py` the helpers expected by the three tests (`_select_sampled_contracts`, `select_spot`, `_ensure_snapshot_table`, `_prune_snapshots_sql`) as thin wrappers around current logic, or (b) update the three tests to use the current API and remove the obsolete imports. Option (b) is cleaner if the script’s contract has intentionally changed.
- **init_database:** Add `scripts/init_database.py` that creates DB schema / runs any one-time setup your app expects, or update README to the real initialization method.

---

**Summary:**  
You have a solid research stack (through Phase 4C) and a clear set of commands; the main gap is **no single continuous loop** and no allocation/lifecycle. The next step is the **Strategy Research Loop**; the **big Claude Opus task** is to design and implement that loop so Argus becomes a self-contained program that continually tests strategies and produces candidates and killed sets for future allocation and funding.

**Implementation plan:** See [docs/STRATEGY_RESEARCH_LOOP_IMPLEMENTATION_PLAN.md](STRATEGY_RESEARCH_LOOP_IMPLEMENTATION_PLAN.md) for the full spec: config schema, step-by-step loop logic, task checklist, acceptance criteria, and tests.
