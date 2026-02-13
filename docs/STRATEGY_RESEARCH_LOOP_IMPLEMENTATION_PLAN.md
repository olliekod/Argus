# Strategy Research Loop — Implementation Plan

This document is the **implementation plan** for the Argus Strategy Research Loop: a single config-driven process that builds replay packs, runs experiments (with regime-stress and MC/bootstrap), evaluates strategies, and persists rankings and killed set. It is the first version of the "self-contained continuous program" that continually tests strategies and produces a candidate set for future allocation.

**Authority:** MASTER_PLAN.md §8.4 (recommended next steps), COMMANDS_AND_NEXT_STEPS.md §5–6.

---

## 1. Goals and scope

### Goals

1. **Single entry point:** One command (e.g. `python scripts/strategy_research_loop.py --config config/research_loop.yaml [--once]`) runs the full research cycle.
2. **Config-driven:** Strategies, param grids, pack spec (symbols/dates), and robustness flags come from a YAML config file; no hardcoded symbols or paths.
3. **Deterministic and reproducible:** Same config + same data → same rankings and killed list. Uses existing ExperimentRunner and StrategyEvaluator; no change to their contracts.
4. **Output for downstream:** Writes rankings JSON, killed list JSON, and an optional **candidate set** (strategies that passed all kill rules) so a future StrategyLeague can consume them.

### In scope

- Config schema and validation.
- Outcomes backfill (ensure bars have outcomes for pack date range).
- Replay pack creation (single-symbol or universe).
- Experiment runs (single + parameter sweep) with regime-stress and MC/bootstrap as configured.
- Strategy evaluation and kill-threshold application.
- Persisting rankings, killed list, and candidate set to a known directory.
- CLI with `--config` and `--once`; optional daemon mode (e.g. run every N hours or when new data exists).

### Out of scope (future)

- StrategyLeague / capital allocation.
- Portfolio risk engine or sizing.
- Runtime strategy lifecycle (rolling PnL, quarantine, kill in production).
- Paper or live execution.

---

## 2. Config schema

**File:** `config/research_loop.yaml` (or user-specified path).

```yaml
# ── Pack specification ─────────────────────────────────────────────────────
# How to build replay packs for this cycle.
pack:
  # One of: "single" (use symbols list) or "universe" (liquid ETF universe).
  mode: "single"   # or "universe"
  # For mode=single: list of symbols to pack (e.g. ["SPY", "IBIT"]).
  symbols: ["SPY"]
  # Date range: either fixed or "last_n_days" (computed at run time).
  # Option A: fixed range
  start_date: "2026-02-01"
  end_date: "2026-02-11"
  # Option B: use last_n_days (optional; overrides start/end if set)
  # last_n_days: 7
  # Bars provider (default from data_sources.bars_primary if omitted).
  bars_provider: null   # e.g. "alpaca" or null
  # Options snapshot provider override (default from config).
  options_snapshot_provider: null
  # Output directory for built packs.
  packs_output_dir: "data/packs"
  # DB path for replay_pack and outcomes.
  db_path: "data/argus.db"

# ── Outcomes ───────────────────────────────────────────────────────────────
# Ensure outcomes exist before building packs.
outcomes:
  # If true, run outcomes backfill for pack date range before building packs.
  ensure_before_pack: true
  # Bar duration in seconds (must match bars in DB).
  bar_duration: 60
  # For mode=single: backfill only for pack.symbols.
  # For mode=universe: use backfill-all for the date range (recommended).

# ── Strategies and experiments ──────────────────────────────────────────────
strategies:
  # List of strategy definitions. Each has strategy_class and optional params/sweep.
  - strategy_class: "VRPCreditSpreadStrategy"
    # Base params (merged with sweep if present).
    params: {}
    # Optional: path to YAML param grid (same format as run_experiment.py --sweep).
    sweep: null   # e.g. "config/vrp_sweep.yaml"
  # - strategy_class: "DowRegimeTimingStrategy"
  #   params: {}
  #   sweep: "config/dow_sweep.yaml"

# ── Experiment runner options ───────────────────────────────────────────────
experiment:
  output_dir: "logs/experiments"
  starting_cash: 10000.0
  regime_stress: true
  mc_bootstrap: true
  mc_paths: 1000
  mc_method: "bootstrap"
  mc_block_size: null
  mc_seed: 42
  mc_ruin_level: 0.2
  mc_kill_thresholds: null   # path to YAML or null for defaults

# ── Evaluation ──────────────────────────────────────────────────────────────
evaluation:
  # Same as experiment.output_dir by default (where evaluator reads from).
  input_dir: null   # null => use experiment.output_dir
  kill_thresholds: null   # path to YAML or null for defaults
  rankings_output_dir: "logs"
  # Filename pattern: strategy_rankings_<date>.json (date = run date UTC).
  killed_output_path: null   # e.g. "logs/killed.json" or null to auto-derive
  candidate_set_output_path: null   # e.g. "logs/candidates.json" or null

# ── Loop (daemon) ───────────────────────────────────────────────────────────
# Only used when running without --once.
loop:
  interval_hours: 24
  # Optional: only run if there is new bar data in the last N hours (skip if stale).
  require_recent_bars_hours: null
```

**Validation rules:**

- `pack.mode` is `single` or `universe`. If `single`, `pack.symbols` must be a non-empty list.
- At least one of `pack.start_date`/`pack.end_date` or `pack.last_n_days` must be valid. If `last_n_days` is set, it overrides start/end (compute from today UTC).
- `strategies` must have at least one entry; `strategy_class` must be loadable (same as `run_experiment.py`).
- All paths (sweep, kill_thresholds, output dirs) are resolved relative to project root or as absolute.

---

## 3. Module layout and dependencies

| Component | Location | Role |
|-----------|----------|------|
| **Config loader** | `scripts/strategy_research_loop.py` or `src/analysis/research_loop_config.py` | Load and validate YAML; resolve `last_n_days` to start/end. |
| **Loop orchestrator** | `scripts/strategy_research_loop.py` | Main entry; runs steps in order: outcomes → packs → experiments → evaluation → persist. |
| **Outcomes step** | Subprocess or helper | Call `python -m src.outcomes backfill` (per symbol) or `backfill-all` for date range. |
| **Packs step** | `src.tools.replay_pack` | Call `create_replay_pack()` or `create_universe_packs()` (async) with config. |
| **Experiments step** | `src.analysis.experiment_runner` | Instantiate `ExperimentRunner`, load strategy class, build `ExperimentConfig`, run single or `run_parameter_grid()`. |
| **Evaluation step** | `src.analysis.strategy_evaluator` | Instantiate `StrategyEvaluator(input_dir=..., kill_thresholds=...)`, `load_experiments()`, `evaluate()`, `save_rankings()`, write killed and candidates. |

**Recommendation:** Keep the loop script in `scripts/strategy_research_loop.py` and put only config dataclass/validation in `src/analysis/research_loop_config.py` so the script remains the single entry point and the rest of the codebase stays unchanged.

---

## 4. Step-by-step loop logic

### 4.1 Resolve date range

- If `pack.last_n_days` is set: `end_date = today_utc`, `start_date = end_date - last_n_days`.
- Else use `pack.start_date` and `pack.end_date` from config.
- All downstream steps use this `(start_date, end_date)`.

### 4.2 Ensure outcomes

- If `outcomes.ensure_before_pack` is false, skip.
- If `pack.mode == "universe"`: run subprocess  
  `python -m src.outcomes backfill-all --start <start_date> --end <end_date>`  
  (and pass `--config` if needed). Wait for exit code 0.
- If `pack.mode == "single"`: for each symbol in `pack.symbols`, run  
  `python -m src.outcomes backfill --provider <bars_provider> --symbol <symbol> --bar <bar_duration> --start <start_date> --end <end_date>`.  
  Use `pack.bars_provider` or read from `data_sources.bars_primary` (config).
- On failure: log and either abort the cycle or continue (configurable; recommend abort).

### 4.3 Build replay packs

- Use `src.tools.replay_pack`:
  - **Universe:** `asyncio.run(create_universe_packs(start_date=..., end_date=..., output_dir=pack.packs_output_dir, provider=..., db_path=...))`.
  - **Single:** For each symbol, `asyncio.run(create_replay_pack(symbol=..., start_date=..., end_date=..., output_path=..., provider=..., db_path=...))`.
- Output paths: either a directory of `{symbol}_{start}_{end}.json` or a single path per symbol. The next step needs a **list of pack file paths** to pass to `ExperimentConfig.replay_pack_paths`.

### 4.4 Run experiments

- For each strategy in `strategies`:
  - Load strategy class (same as `run_experiment.py`: `load_strategy_class(strategy_class)` from known modules).
  - Resolve params: merge `params` with optional sweep grid. If `sweep` path is given, load YAML and use `ExperimentRunner.run_parameter_grid(strategy_cls, base_config, grid)`; else `runner.run(config)` once.
  - `ExperimentConfig`: `replay_pack_paths` = list of pack paths from 4.3; `strategy_params` = merged params; all experiment.* flags (regime_stress, mc_bootstrap, mc_paths, etc.). `output_dir` = `experiment.output_dir`.
- **Regime-stress:** If `experiment.regime_stress` is true, after each run (or after sweep) the plan can call `run_regime_subset_stress(...)` with the same bars/outcomes/regimes/snapshots as in the pack and log or attach the result. The current `run_experiment.py` does regime-stress only for single run (not per sweep run). For the loop, either: (A) run regime-stress once per strategy using the full pack data, or (B) run it per experiment run. Recommendation: (A) once per strategy after all param combinations for that strategy, to keep runtime lower; document in plan.
- Experiments write artifacts into `experiment.output_dir`. No need to change artifact format.

### 4.5 Evaluate and persist

- `input_dir` = `evaluation.input_dir` or `experiment.output_dir`.
- Instantiate `StrategyEvaluator(input_dir=input_dir, kill_thresholds=kill_thresholds)` (load kill_thresholds from path if given).
- `evaluator.load_experiments()`; `evaluator.evaluate()`.
- `evaluator.save_rankings(output_path=...)` to `evaluation.rankings_output_dir` with default filename `strategy_rankings_<date>.json`.
- If `evaluation.killed_output_path` is set, write `evaluator.killed` to that path (same format as `evaluate_strategies.py --output-killed`).
- **Candidate set:** Build a list of records from `evaluator._rankings` where `killed` is false; write to `evaluation.candidate_set_output_path` if set. Format: e.g. `{"candidates": [{"run_id", "strategy_id", "strategy_class", "strategy_params", "composite_score", ...}], "generated_at": "ISO8601"}`.

### 4.6 Daemon mode (optional)

- If not `--once`: after one cycle, sleep `loop.interval_hours` then repeat from 4.1 (re-resolve dates if using `last_n_days`). Optional: before running, check `require_recent_bars_hours` (e.g. query DB for max bar timestamp; if older than N hours, skip cycle and sleep).

---

## 5. Implementation tasks (checklist)

### 5.1 Config and validation

- [ ] **Task 5.1.1** Add `config/research_loop.yaml` (or `research_loop.example.yaml`) with the schema above and sensible defaults.
- [ ] **Task 5.1.2** Implement config loader: read YAML, resolve `last_n_days` to `start_date`/`end_date`, validate mode/symbols/strategies. Prefer a dataclass or typed dict; raise clear errors on invalid config.
- [ ] **Task 5.1.3** Resolve paths: all relative paths are relative to project root (e.g. `Path(__file__).resolve().parent.parent` for script in `scripts/`).

### 5.2 Outcomes step

- [ ] **Task 5.2.1** Implement `run_outcomes_backfill(config, start_date, end_date)`:
  - If universe: subprocess `python -m src.outcomes backfill-all --start ... --end ...` with project root on PYTHONPATH.
  - If single: for each symbol, subprocess `python -m src.outcomes backfill --provider ... --symbol ... --bar 60 --start ... --end ...`.
- [ ] **Task 5.2.2** Use `pack.bars_provider` or load from `config/config.yaml` `data_sources.bars_primary` when provider is null.
- [ ] **Task 5.2.3** On non-zero exit, log and exit the loop with non-zero status (or make configurable).

### 5.3 Packs step

- [ ] **Task 5.3.1** Implement `build_packs(config, start_date, end_date) -> List[str]` returning list of pack file paths.
  - Universe: call `create_universe_packs(...)`, then glob or list `packs_output_dir` for `*_start_end.json` (or use return value if the API returns paths).
  - Single: for each symbol, call `create_replay_pack(...)` with `output_path = packs_output_dir / f"{symbol}_{start_date}_{end_date}.json"`.
- [ ] **Task 5.3.2** Use `asyncio.run()` from the sync loop script; pass `db_path`, `bars_provider`, `options_snapshot_provider` from config.
- [ ] **Task 5.3.3** Return the list of absolute paths so the experiment step can pass them to `ExperimentConfig.replay_pack_paths`.

### 5.4 Experiments step

- [ ] **Task 5.4.1** For each strategy in config: load class via same logic as `run_experiment.py` (e.g. `load_strategy_class(name)` from `src.strategies.*`).
- [ ] **Task 5.4.2** Build `ExperimentConfig`: replay_pack_paths, strategy_params, starting_cash, output_dir, mc_bootstrap_enabled, mc_paths, mc_method, mc_block_size, mc_random_seed, mc_ruin_level, mc_kill_thresholds (load from path if given).
- [ ] **Task 5.4.3** If strategy has `sweep` path: load YAML grid, call `runner.run_parameter_grid(strategy_cls, base_config, grid)`. Else call `runner.run(config)`.
- [ ] **Task 5.4.4** If `experiment.regime_stress` is true: after all runs for a strategy, run `run_regime_subset_stress(...)` once with the full pack data (bars/outcomes/regimes/snapshots loaded from the first pack) and log or append to a separate regime_stress artifact (optional; can be Phase 2).

### 5.5 Evaluation step

- [ ] **Task 5.5.1** Instantiate `StrategyEvaluator(input_dir=experiment.output_dir, kill_thresholds=...)`. Load kill_thresholds from YAML path if set.
- [ ] **Task 5.5.2** Call `load_experiments()`, `evaluate()`, `save_rankings(output_path=...)`.
- [ ] **Task 5.5.3** Write killed list to `evaluation.killed_output_path` if set (same structure as `evaluate_strategies.py --output-killed`).
- [ ] **Task 5.5.4** Build candidate set: filter `evaluator._rankings` for `not rec.get("killed")`, serialize to JSON with `generated_at`; write to `evaluation.candidate_set_output_path` if set.

### 5.6 CLI and entry point

- [ ] **Task 5.6.1** Create `scripts/strategy_research_loop.py`:
  - `argparse`: `--config` (required or default `config/research_loop.yaml`), `--once` (store_true; default false for daemon), optional `--dry-run` (resolve config and log steps without running).
  - Load config, validate, then run one cycle: resolve dates → outcomes → packs → experiments → evaluation → persist.
  - If `--once`: exit 0 after one cycle. If not `--once`: loop with `loop.interval_hours` (and optional `require_recent_bars_hours` check).
- [ ] **Task 5.6.2** Ensure script is runnable from repo root with `python scripts/strategy_research_loop.py` (sys.path or PYTHONPATH so `src` and `config` are found).

### 5.7 Documentation and invariants

- [ ] **Task 5.7.1** Add a short section to `docs/COMMANDS_AND_NEXT_STEPS.md` or a new `docs/strategy_research_loop.md`: how to create `research_loop.yaml`, how to run `--once` and daemon, where outputs go, how the candidate set feeds future StrategyLeague.
- [ ] **Task 5.7.2** Document invariants: no lookahead (replay and evaluator unchanged); same config + same data → same outputs; use existing runner and evaluator contracts.

### 5.8 Tests

- [ ] **Task 5.8.1** Unit test: config loader with valid YAML; with `last_n_days`; with invalid mode/symbols (expect validation error).
- [ ] **Task 5.8.2** Integration test: run one cycle with a **tiny** pack (one symbol, one day) and a **minimal** strategy (e.g. one param set, no sweep). Use a fixture pack or build one in test. Assert: `experiment.output_dir` contains at least one experiment JSON; `evaluator.load_experiments()` finds it; after evaluate, `save_rankings` writes a file; killed list and candidate set files (if configured) have the expected structure.
- [ ] **Task 5.8.3** Optional: test that `--dry-run` exits 0 and does not create experiment or ranking files.

---

## 6. Acceptance criteria

- [ ] One command runs the full cycle: `python scripts/strategy_research_loop.py --config config/research_loop.yaml --once`.
- [ ] Config supports single-symbol and universe pack modes, fixed date range or `last_n_days`, and multiple strategies with optional sweep.
- [ ] Outcomes backfill runs before pack build when `outcomes.ensure_before_pack` is true.
- [ ] Replay packs are built via `src.tools.replay_pack`; experiment runner receives the list of pack paths and writes artifacts to the configured output dir.
- [ ] StrategyEvaluator runs on the experiment output dir and produces rankings; killed list and candidate set are written when paths are configured.
- [ ] Without `--once`, the process runs cycles every `loop.interval_hours` (and optionally skips when data is stale).
- [ ] At least one integration test runs one full cycle and checks output files and structure.
- [ ] MASTER_PLAN.md and existing contracts (ExperimentRunner, StrategyEvaluator, replay pack) are unchanged except for additive config and script.

---

## 7. Optional extensions (Phase 2)

- **Regime-stress per strategy:** Run regime-stress for each strategy after sweep and attach summary to a dedicated artifact or log.
- **Slippage sensitivity:** Run experiments with cost multipliers (e.g. +25%, +50%) and add to evaluator or loop config.
- **Notify on completion:** Telegram or webhook when a cycle finishes (summary: N experiments, M killed, K candidates).
- **Stale-data check:** Implement `require_recent_bars_hours` by querying DB for latest bar timestamp and skipping cycle if older than N hours.

---

## 8. One-paragraph summary for implementers

Implement a **Strategy Research Loop** script that reads a YAML config (pack mode/symbols/dates, strategies with optional sweeps, experiment and evaluation options), then in order: (1) optionally backfills outcomes for the pack date range via subprocess to `python -m src.outcomes`; (2) builds replay packs via `src.tools.replay_pack.create_replay_pack` or `create_universe_packs`; (3) runs experiments for each strategy using `ExperimentRunner` (single or parameter grid) with regime-stress and MC/bootstrap as configured; (4) runs `StrategyEvaluator` on the experiment output dir and saves rankings, killed list, and optional candidate set. Entry point: `scripts/strategy_research_loop.py --config ... [--once]`. Add config validation, one integration test, and short docs. Do not change ExperimentRunner or StrategyEvaluator contracts; keep the loop as an orchestrator only.
