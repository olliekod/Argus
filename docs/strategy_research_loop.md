# Strategy Research Loop

The Strategy Research Loop is a config-driven orchestrator that runs the full
strategy research cycle in a single command:

1. **Outcomes backfill** — ensure forward-return outcomes exist for the date range.
2. **Replay pack build** — create packs from the database (single-symbol or universe).
3. **Experiments** — run strategies through `ExperimentRunner` (single or parameter sweep) with optional MC/bootstrap and regime-stress.
4. **Evaluation** — rank strategies via `StrategyEvaluator`, persist rankings, killed list, and candidate set.
5. **Allocate** — when enabled, run `StrategyRegistry` + `AllocationEngine` and persist `allocations.json`.

## Quick start

```bash
# One-shot cycle
python scripts/strategy_research_loop.py \
    --config config/research_loop.yaml --once

# Dry run (validate config, log steps, no execution)
python scripts/strategy_research_loop.py \
    --config config/research_loop.yaml --dry-run

# Daemon mode (loop every N hours)
python scripts/strategy_research_loop.py \
    --config config/research_loop.yaml
```

## Config file

Copy `config/research_loop.example.yaml` to `config/research_loop.yaml` and
edit for your environment. Key sections:

| Section | Purpose |
|---------|---------|
| `pack` | Pack mode (`single`/`universe`), symbols, date range or `last_n_days`, DB path |
| `outcomes` | Whether to backfill outcomes before packing, bar duration |
| `strategies` | List of strategy classes with optional params and sweep grid |
| `experiment` | Output dir, starting cash, MC/bootstrap settings, regime-stress toggle |
| `evaluation` | Input dir, kill thresholds, output paths for rankings/killed/candidates/allocations |
| `loop` | Daemon interval, optional stale-data check |

### Date range

Set either fixed dates or `last_n_days`:

```yaml
pack:
  start_date: "2026-02-01"
  end_date: "2026-02-11"
  # OR
  last_n_days: 7  # overrides start/end; computed from today UTC
```


### Allocation config keys

When `evaluation.allocation` and `evaluation.allocations_output_path` are both set, Step 5 (allocation) runs after evaluation.

| Key | Purpose |
|-----|---------|
| `evaluation.allocation` | Allocation engine settings (`kelly_fraction`, `per_play_cap`, optional `vol_target_annual`, optional `max_loss_per_contract`). |
| `evaluation.allocations_output_path` | JSON output path for allocation targets (for example `logs/allocations.json`). |
| `evaluation.equity` | Equity base used to convert weights into dollar risk/contracts. |
| `evaluation.min_dsr` | Candidate filter floor applied before allocation (`StrategyRegistry.load_from_rankings`). |
| `evaluation.min_composite_score` | Additional candidate floor before allocation. |


`evaluation.allocation.max_loss_per_contract` supports both of the following forms:

- Single float default applied to all strategies that do not provide an override.
- Dict map of `strategy_id -> float` for strategy-specific risk-per-contract assumptions.

Override order used by `run_allocation()` (highest priority first):
1. Ranking row field `max_loss_per_contract` (if present).
2. Ranking `strategy_params.max_loss_per_contract` (if present).
3. Config `evaluation.allocation.max_loss_per_contract` (dict entry for strategy_id, then float default).

If none are provided, allocation behavior is unchanged (contract counts remain null while weights/dollar risk are still produced).

### Parameter sweeps

Point a strategy's `sweep` field at a YAML grid (same format as
`run_experiment.py --sweep`):

```yaml
strategies:
  - strategy_class: "VRPCreditSpreadStrategy"
    params: {}
    sweep: "config/vrp_sweep.yaml"
```

Sweep YAML values can now be expressed as:
- **Explicit list** (existing behavior):
  - `max_vol_regime: ["VOL_LOW", "VOL_NORMAL"]`
- **Range spec** (new behavior):
  - `min_vrp: {min: 0.02, max: 0.08, step: 0.01}`
  - `entry_threshold: {min: 0.1, max: 0.4, num_steps: 3}`
- **Scalar convenience form** (auto-wrapped to a single-value list):
  - `max_vol_regime: "VOL_LOW"`

Range spec options:
- `min` and `max` are required.
- Use either `step` (positive only) or `num_steps` (integer >= 1).
- `num_steps: N` yields `N+1` points including both endpoints.
- Optional `round: <int>` rounds each generated point.
- If `round` is omitted, decimals are inferred from `step`; otherwise default is 4.

#### VRP coarse sweep example

The VRP coarse sweep file is `config/vrp_sweep.yaml`.

```yaml
min_vrp:
  min: 0.02
  max: 0.08
  step: 0.01
max_vol_regime: ["VOL_LOW", "VOL_NORMAL"]
avoid_trend: ["TREND_DOWN"]
```

#### VRP refinement sweep example

```yaml
min_vrp:
  min: 0.045
  max: 0.065
  step: 0.005
  round: 3
max_vol_regime: ["VOL_LOW"]
avoid_trend: ["TREND_DOWN"]
max_vol_regime: ["VOL_LOW"]
```

## Outputs

| File | Description |
|------|-------------|
| `logs/experiments/*.json` | Experiment artifacts (unchanged format) |
| `logs/strategy_rankings_<date>.json` | Ranked strategies with composite scores |
| `logs/killed.json` | Strategies that failed kill thresholds (if configured) |
| `logs/candidates.json` | Strategies that passed all kill rules (if configured) |
| `logs/experiments/<Strategy>_regime_stress.json` | Regime-stress results (if enabled) |
| `logs/allocations.json` | Allocation output with position sizing fields for downstream consumers |

The candidate set is designed for consumption by a future StrategyLeague or
capital allocation layer.

## Invariants

- **No lookahead**: replay harness and evaluator contracts are unchanged.
- **Determinism**: same config + same data → same rankings and killed list.
- **Additive only**: no changes to `ExperimentRunner`, `StrategyEvaluator`,
  `ReplayHarness`, or replay pack APIs.

## Architecture

```
scripts/strategy_research_loop.py    ← CLI entry point & orchestrator
src/analysis/research_loop_config.py ← Config loader & validation
config/research_loop.example.yaml    ← Config template
tests/test_research_loop.py          ← Unit + integration tests
```

The loop script calls existing modules in order:
- `python -m src.outcomes backfill` (subprocess)
- `src.tools.replay_pack.create_replay_pack()` / `create_universe_packs()`
- `src.analysis.experiment_runner.ExperimentRunner.run()` / `run_parameter_grid()`
- `src.analysis.regime_stress.run_regime_subset_stress()`
- `src.analysis.strategy_evaluator.StrategyEvaluator`


### `logs/allocations.json` schema

```json
{
  "generated_at": "...",
  "equity": 10000.0,
  "config": {"kelly_fraction": 0.25, "per_play_cap": 0.07, "vol_target_annual": 0.1},
  "allocations": [
    {"strategy_id": "...", "instrument": "SPY", "weight": 0.07, "dollar_risk": 700.0, "contracts": 2}
  ]
}
```

## Consuming allocations

A future paper/live execution path should:

1. Read `allocations.json` after each completed research cycle (poll or watch file changes).
2. Map `(strategy_id, instrument)` into executable strategy instances/instruments in the runtime registry.
3. Apply `weight`, `dollar_risk`, and `contracts` as sizing targets when placing paper/live orders.

This keeps research/evaluation deterministic while providing a stable handoff contract for execution.
