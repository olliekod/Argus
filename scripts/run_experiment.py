"""
Run Experiment CLI
==================

Entry point for running standardized strategy experiments.

Usage:
  python scripts/run_experiment.py --strategy VRPCreditSpreadStrategy --pack data/spy_pack.json --params '{"min_vrp": 0.06}'
"""

import argparse
import json
import logging
import yaml
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Type

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# So strategy diagnostics (e.g. why 0 trades) are visible
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.analysis.experiment_runner import ExperimentRunner, ExperimentConfig
from src.analysis.replay_harness import ReplayStrategy

def load_strategy_class(name: str) -> Type[ReplayStrategy]:
    """Dynamically load strategy class from src.strategies."""
    # Common locations
    modules = [
        "src.strategies.vrp_credit_spread",
        "src.strategies.dow_regime_timing",
    ]
    
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, name):
                return getattr(mod, name)
        except ImportError:
            continue
            
    raise ImportError(f"Could not find strategy class {name} in known modules.")

def main():
    parser = argparse.ArgumentParser(description="Argus Experiment Runner CLI")
    parser.add_argument("--strategy", required=True, help="Strategy class name (e.g. VRPCreditSpreadStrategy)")
    parser.add_argument("--pack", required=True, action="append", help="Path to one or more Replay Packs (JSON)")
    parser.add_argument("--params", help="JSON string of strategy parameters", default="{}")
    parser.add_argument("--config", help="Path to YAML config file (overrides CLI params)")
    parser.add_argument("--sweep", help="Path to YAML param grid for parameter sweep")
    parser.add_argument("--output", default="logs/experiments", help="Output directory for results")
    parser.add_argument("--cash", type=float, default=10000.0, help="Starting cash")

    args = parser.parse_args()

    # 1. Load params
    params = json.loads(args.params)
    if args.config:
        with open(args.config, "r") as f:
            params.update(yaml.safe_load(f))

    # 2. Resolve Strategy
    try:
        strat_cls = load_strategy_class(args.strategy)
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 3. Initialize Runner
    runner = ExperimentRunner(output_dir=args.output)

    # 4. Handle Sweep vs Single Run
    if args.sweep:
        with open(args.sweep, "r") as f:
            grid = yaml.safe_load(f)
        
        base_config = ExperimentConfig(
            strategy_class=strat_cls,
            strategy_params=params,
            replay_pack_paths=args.pack,
            starting_cash=args.cash
        )
        
        results = runner.run_parameter_grid(strat_cls, base_config, grid)
        print(f"\n--- Sweep Results ({len(results)}) ---")
        for r in sorted(results, key=lambda x: x.portfolio_summary['total_realized_pnl'], reverse=True):
            p = r.portfolio_summary
            print(f"PnL: {p['total_realized_pnl']:>8} | Sharpe: {p['sharpe_annualized_proxy']:>5} | Params: {r.strategy_state.get('params', 'N/A')}")
            
    else:
        config = ExperimentConfig(
            strategy_class=strat_cls,
            strategy_params=params,
            replay_pack_paths=args.pack,
            starting_cash=args.cash
        )
        
        result = runner.run(config)
        
        print("\n" + "="*40)
        print(f" EXPERIMENT COMPLETE: {result.strategy_id}")
        print("="*40)
        p = result.portfolio_summary
        print(f"Total PnL:     {p['total_realized_pnl']:>10}")
        print(f"Return %:      {p['total_return_pct']:>10}%")
        print(f"Sharpe Proxy:  {p['sharpe_annualized_proxy']:>10}")
        print(f"Win Rate:      {p['win_rate']:>10}%")
        print(f"Trades:        {p['total_trades']:>10}")
        print(f"Profit Factor: {p['profit_factor']:>10}")
        print(f"Max DD:        {p['max_drawdown']:>10}")
        print("-" * 40)
        print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
