# Argus Research & Experimentation Pipeline

The Research Pipeline allows you to evaluate strategies consistently and objectively using deterministic "Replay Packs".

## Core Components

### 1. Replay Packs
A Replay Pack is a JSON slice of database records (`market_bars`, `bar_outcomes`, `regimes`) exported for a specific time window. 
Use `src/tools/replay_pack.py` to create them:
```bash
python src/tools/replay_pack.py --symbol SPY --start 2024-01-01 --end 2024-01-07 --out data/spy_jan.json
```

### 2. Experiment Runner
The `ExperimentRunner` orchestrates the execution of strategies over one or more packs. It produces a standardized JSON report for every run, including:
- **Expectancy & Profit Factor**
- **Annualized Sharpe Proxy**
- **Regime-Conditioned performance**
- **Average Holding Time**

### 3. CLI Usage
Run a single experiment:
```bash
python scripts/run_experiment.py --strategy VRPCreditSpreadStrategy --pack data/spy_jan.json --params '{"min_vrp": 0.05}'
```

Run a parameter sweep:
Create a `sweep.yaml`:
```yaml
min_vrp: [0.03, 0.05, 0.07]
max_vol_regime: ["VOL_NORMAL", "VOL_HIGH"]
```
Then run:
```bash
python scripts/run_experiment.py --strategy VRPCreditSpreadStrategy --pack data/spy_jan.json --sweep sweep.yaml
```

## Advanced Training

### Walk-Forward Analysis
The `ExperimentRunner` supports split-based evaluation. It slides a training and evaluation window across the dataset to detect over-fitting and regime-dependent performance degradation.

### Standardized Outputs
All experiments are saved to `logs/experiments/<StrategyName>_<Tag>_<Hash>.json`. These artifacts are deterministic and can be compared across research iterations.

### Metrics Definitions
- **Expectancy**: Average PnL expected per trade.
- **Sharpe Proxy**: Annualized using minute-bar variance. Treat as an estimate for relative comparison.
- **Regime Win Rate**: Helps identify if a strategy only works in certain market conditions (e.g., TREND_UP).
