"""
Experiment Runner
=================

Orchestrates strategy evaluation over Replay Packs.
Provides deterministic walk-forward splitting and parameter sweeps.
"""

import json
import logging
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Type, Optional, Generator

from src.analysis.replay_harness import (
    MarketDataSnapshot,
    ReplayHarness,
    ReplayConfig,
    ReplayStrategy,
    ReplayResult,
)
from src.analysis.execution_model import ExecutionModel, ExecutionConfig
from src.core.outcome_engine import BarData

logger = logging.getLogger("argus.experiment_runner")

@dataclass
class ExperimentConfig:
    """Settings for a single experiment run."""
    strategy_class: Type[ReplayStrategy]
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    replay_pack_paths: List[str] = field(default_factory=list)
    starting_cash: float = 10000.0
    execution_config: Optional[ExecutionConfig] = None
    output_dir: str = "logs/experiments"
    tag: str = "default"

class ExperimentRunner:
    """Standardized handler for strategy research experiments."""

    def __init__(self, output_dir: str = "logs/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_pack(self, path: str) -> Dict[str, Any]:
        """Load a Replay Pack JSON file."""
        with open(path, "r") as f:
            pack = json.load(f)
        return pack

    @staticmethod
    def _pack_snapshots_to_objects(snapshot_dicts: List[Dict[str, Any]]) -> List[MarketDataSnapshot]:
        """Convert replay pack snapshot dicts to MarketDataSnapshot for ReplayHarness."""
        out: List[MarketDataSnapshot] = []
        for s in snapshot_dicts or []:
            recv_ts = s.get("recv_ts_ms")
            if recv_ts is None:
                recv_ts = s.get("timestamp_ms", 0)
            out.append(
                MarketDataSnapshot(
                    symbol=s.get("symbol", "SPY"),
                    recv_ts_ms=recv_ts,
                    underlying_price=float(s.get("underlying_price", 0.0)),
                    atm_iv=s.get("atm_iv") if s.get("atm_iv") is not None else None,
                    source=s.get("provider", ""),
                )
            )
        return out

    def run(self, config: ExperimentConfig) -> ReplayResult:
        """Run a single experiment configuration."""
        # 1. Prepare data from packs
        all_bars: List[BarData] = []
        all_outcomes: List[Dict[str, Any]] = []
        all_regimes: List[Dict[str, Any]] = []
        all_snapshots: List[Any] = []
        
        for pack_path in config.replay_pack_paths:
            pack = self.load_pack(pack_path)
            # Reconstruct BarData objects
            for b in pack.get("bars", []):
                bar = BarData(
                    timestamp_ms=b["timestamp_ms"],
                    open=b["open"], high=b["high"], low=b["low"], 
                    close=b["close"], volume=b.get("volume", 0)
                )
                # Apply symbol if present
                if "symbol" in b:
                    object.__setattr__(bar, 'symbol', b["symbol"])
                all_bars.append(bar)
            
            all_outcomes.extend(pack.get("outcomes", []))
            all_regimes.extend(pack.get("regimes", []))
            all_snapshots.extend(pack.get("snapshots", []))

        snapshots_objs = self._pack_snapshots_to_objects(all_snapshots)

        # Warn if pack is missing data many strategies need (e.g. VRP needs outcomes + snapshots)
        if not all_outcomes:
            print("WARNING: Pack has 0 outcomes. Strategies that need realized_vol (e.g. VRPCreditSpreadStrategy) will never signal. Run: python -m src.outcomes backfill --provider <source> --symbol <sym> --bar 60 --start YYYY-MM-DD --end YYYY-MM-DD then re-pack.")
        if not snapshots_objs:
            print("WARNING: Pack has 0 option snapshots. Strategies that need IV/options will never signal.")

        # 2. Instantiate Strategy
        strategy = config.strategy_class(config.strategy_params)
        
        # 3. Setup Replay
        exec_model = ExecutionModel(config.execution_config)
        replay_cfg = ReplayConfig(starting_cash=config.starting_cash)
        
        harness = ReplayHarness(
            bars=all_bars,
            outcomes=all_outcomes,
            strategy=strategy,
            execution_model=exec_model,
            regimes=all_regimes,
            snapshots=snapshots_objs,
            config=replay_cfg
        )
        
        # 4. Execute
        result = harness.run()
        
        # 5. Save Artifact
        self._save_result(config, result)
        
        return result

    def run_parameter_grid(self, strategy_cls: Type[ReplayStrategy], 
                           base_config: ExperimentConfig, 
                           param_grid: Dict[str, List[Any]]) -> List[ReplayResult]:
        """Run a parameter sweep/grid search."""
        import itertools
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        results = []
        print(f"Starting sweep: {len(combinations)} combinations.")
        for i, params in enumerate(combinations):
            print(f"Sweep {i+1}/{len(combinations)}: {params}")
            config = ExperimentConfig(
                strategy_class=strategy_cls,
                strategy_params=params,
                replay_pack_paths=base_config.replay_pack_paths,
                starting_cash=base_config.starting_cash,
                execution_config=base_config.execution_config,
                tag=f"sweep_{i}"
            )
            results.append(self.run(config))
        return results

    def _save_result(self, config: ExperimentConfig, result: ReplayResult):
        """Persist result to JSON artifact with rich manifest."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Environment Metadata
        git_commit = "UNKNOWN"
        try:
            import subprocess
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except: pass

        # 2. Input Integrity (Pack Hashes)
        pack_info = []
        for p in config.replay_pack_paths:
            try:
                content = open(p, "rb").read()
                pack_info.append({
                    "path": str(p),
                    "hash": hashlib.sha256(content).hexdigest()[:12]
                })
            except:
                pack_info.append({"path": str(p), "hash": "ERROR"})

        # 3. Deterministic Run ID
        # Strategy + Sorted Params + Sorted Pack Paths
        input_data = f"{result.strategy_id}_{json.dumps(config.strategy_params, sort_keys=True)}_{sorted(config.replay_pack_paths)}"
        run_id = hashlib.md5(input_data.encode()).hexdigest()[:8]
        filename = f"{result.strategy_id}_{config.tag}_{run_id}.json"
        
        output_file = self.output_dir / filename
        
        artifact = {
            "manifest": {
                "run_id": run_id,
                "strategy_class": config.strategy_class.__name__,
                "strategy_params": config.strategy_params,
                "execution_config": config.execution_config.__dict__ if config.execution_config else "DEFAULT",
                "replay_packs": pack_info,
                "environment": {
                    "git_commit": git_commit,
                    "python_version": f"{datetime.now().year}.{datetime.now().month}", # Or sys.version
                    "timestamp": timestamp
                }
            },
            "result": result.summary()
        }
        
        with open(output_file, "w") as f:
            json.dump(artifact, f, indent=2)
        
        logger.info(f"Experiment result saved to {output_file}")

    def split_walk_forward(self, bars: List[BarData], 
                          train_days: int, test_days: int, 
                          step_days: Optional[int] = None) -> Generator[tuple[List[BarData], List[BarData]], None, None]:
        """Generator that yields (train_bars, test_bars) windows based on unique trading days."""
        if not bars: return
        
        # 1. Group bars by date (YYYY-MM-DD)
        sorted_bars = sorted(bars, key=lambda b: b.timestamp_ms)
        date_groups: Dict[str, List[BarData]] = {}
        from datetime import timezone
        for b in sorted_bars:
            date_str = datetime.fromtimestamp(b.timestamp_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
            if date_str not in date_groups:
                date_groups[date_str] = []
            date_groups[date_str].append(b)
        
        unique_dates = sorted(date_groups.keys())
        step_days = step_days or test_days
        
        # 2. Slide window across dates
        current_idx = 0
        while current_idx + train_days + test_days <= len(unique_dates):
            train_dates = unique_dates[current_idx : current_idx + train_days]
            test_dates = unique_dates[current_idx + train_days : current_idx + train_days + test_days]
            
            train_bars = []
            for d in train_dates: train_bars.extend(date_groups[d])
            
            test_bars = []
            for d in test_dates: test_bars.extend(date_groups[d])
            
            if train_bars and test_bars:
                yield (train_bars, test_bars)
            
            current_idx += step_days

    def run_walk_forward(self, config: ExperimentConfig, 
                         train_days: int, test_days: int) -> List[Dict[str, Any]]:
        """Run rolling walk-forward evaluation."""
        # 1. Load all data
        all_bars: List[BarData] = []
        all_outcomes: List[Dict[str, Any]] = []
        all_regimes: List[Dict[str, Any]] = []
        all_snapshots: List[Dict[str, Any]] = []

        for pack_path in config.replay_pack_paths:
            pack = self.load_pack(pack_path)
            for b in pack.get("bars", []):
                bar = BarData(timestamp_ms=b["timestamp_ms"], open=b["open"], high=b["high"], low=b["low"], close=b["close"])
                if "symbol" in b: object.__setattr__(bar, 'symbol', b["symbol"])
                all_bars.append(bar)
            all_outcomes.extend(pack.get("outcomes", []))
            all_regimes.extend(pack.get("regimes", []))
            all_snapshots.extend(pack.get("snapshots", []))

        snapshots_objs = self._pack_snapshots_to_objects(all_snapshots)

        results = []
        for i, (train_bars, test_bars) in enumerate(self.split_walk_forward(all_bars, train_days, test_days)):
            print(f"Window {i+1}: Train {len(train_bars)} bars, Test {len(test_bars)} bars")
            # In a real system, we'd "train" (optimize) on train_bars
            # For now, we just run the strategy on test_bars to see how it performs in different periods

            # Setup harness for test window (harness gates outcomes/regimes/snapshots by time)
            strategy = config.strategy_class(config.strategy_params)
            exec_model = ExecutionModel(config.execution_config)
            harness = ReplayHarness(
                bars=test_bars,
                outcomes=all_outcomes,
                strategy=strategy,
                execution_model=exec_model,
                regimes=all_regimes,
                snapshots=snapshots_objs,
                config=ReplayConfig(starting_cash=config.starting_cash)
            )
            
            res = harness.run()
            results.append({
                "window": i,
                "pnl": res.portfolio_summary["total_realized_pnl"],
                "sharpe": res.portfolio_summary["sharpe_proxy"],
                "win_rate": res.portfolio_summary["win_rate"]
            })
            
        return results
