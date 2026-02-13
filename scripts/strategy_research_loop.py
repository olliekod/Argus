#!/usr/bin/env python3
"""
Strategy Research Loop
=======================

Single entry point that runs the full strategy research cycle:

1. Resolve date range from config.
2. Optionally backfill outcomes for the date range.
3. Build replay packs (single-symbol or universe).
4. Run experiments for each strategy (single or parameter sweep)
   with optional MC/bootstrap and regime-stress.
5. Evaluate experiments, persist rankings, killed list, and candidate set.

Usage::

    # One-shot cycle
    python scripts/strategy_research_loop.py --config config/research_loop.yaml --once

    # Daemon mode (runs every loop.interval_hours)
    python scripts/strategy_research_loop.py --config config/research_loop.yaml

    # Dry-run (validate config, log steps, no execution)
    python scripts/strategy_research_loop.py --config config/research_loop.yaml --dry-run

Authority: MASTER_PLAN.md §8.4, COMMANDS_AND_NEXT_STEPS.md §5-6.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# Ensure repo root is on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml

from src.analysis.research_loop_config import (
    ConfigValidationError,
    ResearchLoopConfig,
    load_research_loop_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("argus.research_loop")


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy class loader (same logic as run_experiment.py)
# ═══════════════════════════════════════════════════════════════════════════

_STRATEGY_MODULES = [
    "src.strategies.vrp_credit_spread",
    "src.strategies.dow_regime_timing",
]


def load_strategy_class(name: str):
    """Dynamically load a ReplayStrategy class by name."""
    for mod_name in _STRATEGY_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, name):
                return getattr(mod, name)
        except ImportError:
            continue
    raise ImportError(f"Could not find strategy class '{name}' in known modules.")


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1: Outcomes backfill
# ═══════════════════════════════════════════════════════════════════════════


def run_outcomes_backfill(config: ResearchLoopConfig) -> None:
    """Backfill outcomes for the configured date range.

    Calls ``python -m src.outcomes backfill`` (single) or
    ``python -m src.outcomes backfill-all`` (universe) as a subprocess.
    """
    if not config.outcomes.ensure_before_pack:
        logger.info("Outcomes backfill skipped (ensure_before_pack=false).")
        return

    start = config.pack.start_date
    end = config.pack.end_date

    if config.pack.mode == "universe":
        cmd = [
            sys.executable, "-m", "src.outcomes",
            "backfill-all",
            "--start", start,
            "--end", end,
        ]
        logger.info("Running outcomes backfill-all: %s to %s", start, end)
        _run_subprocess(cmd)
    else:
        # Determine bars provider
        bars_provider = config.pack.bars_provider
        if bars_provider is None:
            bars_provider = _read_bars_primary()

        for symbol in config.pack.symbols:
            cmd = [
                sys.executable, "-m", "src.outcomes",
                "backfill",
                "--provider", bars_provider,
                "--symbol", symbol,
                "--bar", str(config.outcomes.bar_duration),
                "--start", start,
                "--end", end,
            ]
            logger.info(
                "Running outcomes backfill: %s/%s %s to %s",
                bars_provider, symbol, start, end,
            )
            _run_subprocess(cmd)


def _read_bars_primary() -> str:
    """Read data_sources.bars_primary from config/config.yaml."""
    cfg_path = _REPO / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("data_sources", {}).get("bars_primary", "alpaca")
    return "alpaca"


def _run_subprocess(cmd: List[str]) -> None:
    """Run a subprocess command and raise on non-zero exit."""
    logger.debug("Subprocess: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (exit {result.returncode}): {' '.join(cmd)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2: Build replay packs
# ═══════════════════════════════════════════════════════════════════════════


def build_packs(config: ResearchLoopConfig) -> List[str]:
    """Build replay packs and return a list of pack file paths.

    Uses ``src.tools.replay_pack`` (async API) wrapped in ``asyncio.run()``.
    """
    from src.tools.replay_pack import create_replay_pack, create_universe_packs

    start = config.pack.start_date
    end = config.pack.end_date
    output_dir = config.pack.packs_output_dir
    db_path = config.pack.db_path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if config.pack.mode == "universe":
        logger.info("Building universe packs: %s to %s", start, end)
        paths = asyncio.run(create_universe_packs(
            start_date=start,
            end_date=end,
            output_dir=output_dir,
            provider=config.pack.bars_provider,
            db_path=db_path,
            snapshot_provider=config.pack.options_snapshot_provider,
        ))
        logger.info("Built %d universe packs.", len(paths))
        return [str(p) for p in paths]
    else:
        paths: List[str] = []
        for symbol in config.pack.symbols:
            out_path = str(
                Path(output_dir) / f"{symbol}_{start}_{end}.json"
            )
            logger.info("Building pack: %s -> %s", symbol, out_path)
            asyncio.run(create_replay_pack(
                symbol=symbol,
                start_date=start,
                end_date=end,
                output_path=out_path,
                provider=config.pack.bars_provider,
                db_path=db_path,
                snapshot_provider=config.pack.options_snapshot_provider,
            ))
            paths.append(out_path)
        logger.info("Built %d packs.", len(paths))
        return paths


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3: Run experiments
# ═══════════════════════════════════════════════════════════════════════════


def run_experiments(config: ResearchLoopConfig, pack_paths: List[str]) -> None:
    """Run experiments for each strategy in the config.

    Uses ``ExperimentRunner`` from ``src.analysis.experiment_runner``.
    Supports single runs and parameter sweeps.  Optionally runs
    regime-stress after all runs for a strategy.
    """
    from src.analysis.experiment_runner import ExperimentRunner, ExperimentConfig

    runner = ExperimentRunner(output_dir=config.experiment.output_dir)

    # Load MC kill thresholds from file if specified
    mc_kill_thresholds: Dict[str, float] = {}
    if config.experiment.mc_kill_thresholds:
        mc_kill_path = Path(config.experiment.mc_kill_thresholds)
        if mc_kill_path.exists():
            with open(mc_kill_path) as f:
                mc_kill_thresholds = dict(yaml.safe_load(f) or {})

    for spec in config.strategies:
        logger.info("Running experiments for strategy: %s", spec.strategy_class)

        strat_cls = load_strategy_class(spec.strategy_class)

        base_config = ExperimentConfig(
            strategy_class=strat_cls,
            strategy_params=dict(spec.params),
            replay_pack_paths=list(pack_paths),
            starting_cash=config.experiment.starting_cash,
            output_dir=config.experiment.output_dir,
            mc_bootstrap_enabled=config.experiment.mc_bootstrap,
            mc_paths=config.experiment.mc_paths,
            mc_method=config.experiment.mc_method,
            mc_block_size=config.experiment.mc_block_size,
            mc_random_seed=config.experiment.mc_seed,
            mc_ruin_level=config.experiment.mc_ruin_level,
            mc_kill_thresholds=mc_kill_thresholds,
        )

        if spec.sweep:
            sweep_path = Path(spec.sweep)
            if not sweep_path.exists():
                logger.warning(
                    "Sweep file not found: %s — running single experiment.",
                    spec.sweep,
                )
                result = runner.run(base_config)
                logger.info(
                    "Single run complete: %s PnL=%.2f Sharpe=%.2f",
                    result.strategy_id,
                    result.portfolio_summary["total_realized_pnl"],
                    result.portfolio_summary["sharpe_annualized_proxy"],
                )
            else:
                with open(sweep_path) as f:
                    grid = yaml.safe_load(f)
                logger.info(
                    "Running parameter sweep from %s", spec.sweep,
                )
                results = runner.run_parameter_grid(strat_cls, base_config, grid)
                logger.info(
                    "Sweep complete: %d runs for %s",
                    len(results), spec.strategy_class,
                )
        else:
            result = runner.run(base_config)
            logger.info(
                "Single run complete: %s PnL=%.2f Sharpe=%.2f",
                result.strategy_id,
                result.portfolio_summary["total_realized_pnl"],
                result.portfolio_summary["sharpe_annualized_proxy"],
            )

        # Regime-stress (once per strategy after all runs)
        if config.experiment.regime_stress:
            _run_regime_stress(runner, pack_paths, strat_cls, spec.params, config)


def _run_regime_stress(
    runner,
    pack_paths: List[str],
    strat_cls,
    strategy_params: Dict[str, Any],
    config: ResearchLoopConfig,
) -> None:
    """Run regime subset stress test for a strategy."""
    try:
        from src.analysis.regime_stress import run_regime_subset_stress
        from src.core.outcome_engine import BarData

        bars: List[BarData] = []
        outcomes: List[Dict[str, Any]] = []
        regimes: List[Dict[str, Any]] = []
        snapshots: List[Dict[str, Any]] = []

        for pack_path in pack_paths:
            pack = runner.load_pack(pack_path)
            for b in pack.get("bars", []):
                bar = BarData(
                    timestamp_ms=b["timestamp_ms"],
                    open=b["open"],
                    high=b["high"],
                    low=b["low"],
                    close=b["close"],
                    volume=b.get("volume", 0),
                )
                if "symbol" in b:
                    object.__setattr__(bar, "symbol", b["symbol"])
                bars.append(bar)
            outcomes.extend(pack.get("outcomes", []))
            regimes.extend(pack.get("regimes", []))
            snapshots.extend(pack.get("snapshots", []))

        if not bars:
            logger.warning("No bars loaded — skipping regime stress.")
            return

        stress = run_regime_subset_stress(
            bars=bars,
            outcomes=outcomes,
            regimes=regimes,
            snapshots=snapshots,
            strategy_class=strat_cls,
            strategy_params=dict(strategy_params),
            starting_cash=config.experiment.starting_cash,
        )

        stress_path = (
            Path(config.experiment.output_dir)
            / f"{strat_cls.__name__}_regime_stress.json"
        )
        stress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stress_path, "w") as f:
            json.dump(stress, f, indent=2)

        logger.info(
            "Regime stress for %s: score=%.3f saved to %s",
            strat_cls.__name__,
            stress.get("stress_score", 0.0),
            stress_path,
        )
    except Exception as exc:
        logger.warning("Regime stress failed for %s: %s", strat_cls.__name__, exc)


# ═══════════════════════════════════════════════════════════════════════════
#  Step 4: Evaluate and persist
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_and_persist(config: ResearchLoopConfig) -> str:
    """Run strategy evaluation and persist rankings, killed list, candidates.

    Returns the rankings output path.
    """
    from src.analysis.strategy_evaluator import StrategyEvaluator

    input_dir = config.evaluation.input_dir

    # Load kill thresholds from file if specified
    kill_thresholds = None
    if config.evaluation.kill_thresholds:
        kt_path = Path(config.evaluation.kill_thresholds)
        if kt_path.exists():
            with open(kt_path) as f:
                kill_thresholds = yaml.safe_load(f)

    evaluator = StrategyEvaluator(
        input_dir=input_dir,
        output_dir=config.evaluation.rankings_output_dir,
        kill_thresholds=kill_thresholds,
    )

    count = evaluator.load_experiments()
    if count == 0:
        logger.warning("No experiment files found in %s", input_dir)
        return ""

    evaluator.evaluate()

    # Save rankings
    rankings_path = evaluator.save_rankings()
    logger.info("Rankings saved to %s (%d experiments)", rankings_path, count)

    # Print summary
    evaluator.print_summary()

    # Save killed list
    if config.evaluation.killed_output_path:
        killed_out = {
            "killed_count": len(evaluator.killed),
            "killed": evaluator.killed,
        }
        killed_path = Path(config.evaluation.killed_output_path)
        killed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(killed_path, "w") as f:
            json.dump(killed_out, f, indent=2)
        logger.info("Killed list written to %s", killed_path)

    # Build and save candidate set
    if config.evaluation.candidate_set_output_path:
        candidates = [
            {
                "run_id": rec.get("run_id", ""),
                "strategy_id": rec.get("strategy_id", ""),
                "strategy_class": rec.get("strategy_class", ""),
                "strategy_params": rec.get("strategy_params", {}),
                "composite_score": rec.get("composite_score", 0.0),
                "metrics": rec.get("metrics", {}),
            }
            for rec in evaluator.rankings
            if not rec.get("killed", False)
        ]
        candidate_set = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        cand_path = Path(config.evaluation.candidate_set_output_path)
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cand_path, "w") as f:
            json.dump(candidate_set, f, indent=2)
        logger.info(
            "Candidate set written to %s (%d candidates)",
            cand_path, len(candidates),
        )

    return rankings_path


# ═══════════════════════════════════════════════════════════════════════════
#  Cycle orchestrator
# ═══════════════════════════════════════════════════════════════════════════


def run_cycle(config: ResearchLoopConfig, dry_run: bool = False) -> None:
    """Execute one full research cycle.

    Steps: resolve dates -> outcomes -> packs -> experiments -> evaluate.
    """
    logger.info(
        "=== Research cycle starting: %s to %s, %d strategies ===",
        config.pack.start_date,
        config.pack.end_date,
        len(config.strategies),
    )

    if dry_run:
        logger.info("[DRY RUN] Config validated. Steps that would run:")
        if config.outcomes.ensure_before_pack:
            logger.info("  1. Outcomes backfill (%s, %s to %s)",
                        config.pack.mode, config.pack.start_date,
                        config.pack.end_date)
        else:
            logger.info("  1. Outcomes backfill: SKIPPED")
        logger.info("  2. Build packs: mode=%s symbols=%s",
                     config.pack.mode, config.pack.symbols)
        for s in config.strategies:
            logger.info("  3. Experiment: %s sweep=%s",
                         s.strategy_class, s.sweep or "none")
        logger.info("  4. Evaluate: input=%s", config.evaluation.input_dir)
        logger.info("[DRY RUN] No actions taken.")
        return

    # Step 1: Outcomes backfill
    logger.info("--- Step 1/4: Outcomes backfill ---")
    run_outcomes_backfill(config)

    # Step 2: Build packs
    logger.info("--- Step 2/4: Build replay packs ---")
    pack_paths = build_packs(config)
    if not pack_paths:
        logger.error("No packs built — aborting cycle.")
        return

    # Step 3: Run experiments
    logger.info("--- Step 3/4: Run experiments ---")
    run_experiments(config, pack_paths)

    # Step 4: Evaluate and persist
    logger.info("--- Step 4/4: Evaluate and persist ---")
    rankings_path = evaluate_and_persist(config)

    logger.info("=== Research cycle complete. Rankings: %s ===", rankings_path)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Argus Strategy Research Loop — run the full "
                    "research cycle: outcomes -> packs -> experiments -> evaluation.",
    )
    parser.add_argument(
        "--config",
        default="config/research_loop.yaml",
        help="Path to research loop YAML config "
             "(default: config/research_loop.yaml)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Run a single cycle and exit (default: daemon mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config and log steps without executing anything",
    )

    args = parser.parse_args()

    # Load and validate config
    try:
        config = load_research_loop_config(args.config)
    except (FileNotFoundError, ConfigValidationError) as exc:
        logger.error("Config error: %s", exc)
        sys.exit(1)

    if args.once or args.dry_run:
        try:
            run_cycle(config, dry_run=args.dry_run)
        except Exception:
            logger.exception("Research cycle failed.")
            sys.exit(1)
    else:
        # Daemon mode
        logger.info(
            "Starting daemon mode (interval=%.1fh).",
            config.loop.interval_hours,
        )
        while True:
            try:
                # Re-load config each cycle to pick up changes
                config = load_research_loop_config(args.config)
                run_cycle(config)
            except (FileNotFoundError, ConfigValidationError) as exc:
                logger.error("Config error (will retry next cycle): %s", exc)
            except Exception:
                logger.exception("Research cycle failed (will retry next cycle).")

            sleep_seconds = config.loop.interval_hours * 3600
            logger.info(
                "Sleeping %.1f hours until next cycle...",
                config.loop.interval_hours,
            )
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
