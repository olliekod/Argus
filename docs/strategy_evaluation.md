# Strategy Evaluation

## Overview

The strategy evaluator ranks experiment results produced by the `ExperimentRunner`. It loads JSON artifacts, computes standardized metrics, applies composite scoring with penalty factors, and outputs a deterministic ranking.

## Quick Start

```bash
# Evaluate all experiments in the default directory
python scripts/evaluate_strategies.py --input logs/experiments

# Custom output path
python scripts/evaluate_strategies.py --input logs/experiments --output logs/my_rankings.json

# Suppress console table
python scripts/evaluate_strategies.py --input logs/experiments --quiet
```

Output: `logs/strategy_rankings_<date>.json`

## Metrics

Each experiment is scored on the following metrics extracted from the replay harness portfolio summary:

| Metric | Description |
|--------|-------------|
| `total_pnl` | Net realized PnL |
| `total_return_pct` | Return as percentage of starting capital |
| `sharpe` | Annualized Sharpe proxy (minute-bar returns) |
| `max_drawdown` | Peak-to-trough drawdown in absolute terms |
| `max_drawdown_pct` | Drawdown as percentage of starting capital |
| `expectancy` | Average PnL per trade |
| `profit_factor` | Gross profit / gross loss |
| `win_rate` | Percentage of winning trades |
| `total_trades` | Number of closed trades |
| `fill_rate` | Fills / (fills + rejects) |

## Composite Scoring

The composite score is a weighted sum of normalized metrics and penalties:

```
score = w_return * norm_return
      + w_sharpe * norm_sharpe
      + w_dd     * drawdown_penalty       (negative weight)
      + w_rej    * reject_penalty          (negative weight)
      + w_rob    * robustness_penalty      (negative weight)
      + w_regime * regime_dep_penalty      (negative weight)
      + w_wf     * walk_forward_penalty    (negative weight)
```

### Default Weights

| Component | Weight | Direction |
|-----------|--------|-----------|
| Return (normalized) | +0.25 | Higher is better |
| Sharpe (normalized) | +0.30 | Higher is better |
| Drawdown penalty | -0.15 | Lower penalty is better |
| Reject penalty | -0.10 | Lower penalty is better |
| Robustness penalty | -0.10 | Lower penalty is better |
| Regime dependency penalty | -0.05 | Lower penalty is better |
| Walk-forward penalty | -0.05 | Lower penalty is better |

### Normalization

Return and Sharpe are normalized to [0, 1] across all experiments in the evaluation set using min-max scaling. This makes scores comparable regardless of absolute scale.

## Penalties

### Drawdown Penalty

| Condition | Penalty |
|-----------|---------|
| max_drawdown_pct <= 5% | 0.0 |
| max_drawdown_pct >= 50% | 1.0 |
| Between | Linear interpolation |

### Reject Penalty

| Condition | Penalty |
|-----------|---------|
| fill_rate >= 0.8 | 0.0 |
| fill_rate == 0.0 | 1.0 |
| Between | Linear interpolation |

### Robustness Penalty

Measures parameter fragility across sweep results. Groups experiments by strategy class and computes the coefficient of variation (CV) of PnL:

| Condition | Penalty |
|-----------|---------|
| CV <= 0.3 | 0.0 |
| CV >= 2.0 | 1.0 |
| Zero mean PnL | 0.5 |
| Single run | 0.0 (not enough data) |

A high CV means performance varies wildly with small parameter changes — a sign of overfitting.

### Regime Dependency Penalty

Penalizes strategies whose profits are concentrated in a single regime:

| Condition | Penalty |
|-----------|---------|
| Max concentration > 90% | 0.8 |
| Max concentration > 80% | 0.5 |
| Otherwise | 0.0 |

### Walk-Forward Stability Penalty

Measures sign consistency across walk-forward windows:

| Condition | Penalty |
|-----------|---------|
| >= 80% same sign | 0.0 |
| <= 50% consistency | 0.8 |
| Between | Linear interpolation |

## Regime-Conditioned Performance

Each strategy is scored per regime bucket (e.g., `regime:SPY`, `session:RTH`). The output includes:

- PnL in each regime
- Bars spent in each regime
- PnL per bar (efficiency)

This helps identify whether a strategy only works in specific market conditions.

## Deployability Interpretation

Use the composite score and its components to assess deployability:

| Score Range | Interpretation |
|-------------|---------------|
| > 0.40 | Strong candidate — review regime scores and robustness before deploying |
| 0.25 - 0.40 | Moderate — may need parameter tuning or regime filtering |
| 0.10 - 0.25 | Weak — high penalties or low absolute performance |
| < 0.10 | Not deployable — significant issues with drawdown, rejects, or fragility |

### Red Flags

- **Robustness penalty > 0.5**: Strategy is parameter-fragile. Small changes destroy edge.
- **Regime dependency > 0.5**: Profits come from a single market state. Will fail when regime changes.
- **Reject penalty > 0.3**: Execution model rejects too many trades. Likely illiquid or wide spreads.
- **Drawdown penalty > 0.7**: Unacceptable risk. Even with positive PnL, capital preservation is at risk.
- **Walk-forward penalty > 0.5**: In-sample overfitting. Performance doesn't generalize across time windows.

### Deployment Checklist

1. Composite score > 0.25
2. Robustness penalty < 0.3
3. Drawdown penalty < 0.5
4. Positive PnL across at least 2 regimes
5. Fill rate > 0.7
6. Walk-forward penalty < 0.3
7. Manual review of regime breakdown for concentration risks

## Output Format

The rankings JSON has this structure:

```json
{
  "generated_at": "2024-01-15T12:00:00+00:00",
  "experiment_count": 10,
  "weights": { ... },
  "rankings": [
    {
      "rank": 1,
      "strategy_id": "OVERNIGHT_MOMENTUM_V2",
      "run_id": "abc12345",
      "composite_score": 0.4521,
      "scoring": {
        "composite_score": 0.4521,
        "components": {
          "return_norm": 0.85,
          "sharpe_norm": 0.72,
          "drawdown_penalty": 0.15,
          "reject_penalty": 0.0,
          "robustness_penalty": 0.1,
          "regime_dependency_penalty": 0.0,
          "walk_forward_penalty": 0.0
        }
      },
      "metrics": { ... },
      "regime_scores": { ... },
      "manifest_ref": { ... }
    }
  ]
}
```

## Replay Pack Integration

The evaluator reads from experiment JSON files produced by `ExperimentRunner`, which in turn consumes replay packs. Replay packs now include option chain snapshots:

```bash
# Pack a single symbol
python -m src.tools.replay_pack --symbol SPY --start 2024-01-01 --end 2024-01-31

# Pack all liquid ETF universe symbols
python -m src.tools.replay_pack --universe --start 2024-01-01 --end 2024-01-31
```

The snapshot data flows through as:
1. `replay_pack.py` fetches from `option_chain_snapshots` DB table
2. Stored in pack JSON with `recv_ts_ms` for gating
3. `ExperimentRunner` loads and passes to `ReplayHarness`
4. Harness gates snapshots by `recv_ts_ms` (data availability barrier)
5. Strategy sees snapshots only when they would have been available in real time
