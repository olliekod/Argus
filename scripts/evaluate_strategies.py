#!/usr/bin/env python3
"""
Strategy Evaluation CLI
========================

Loads experiment JSON outputs, computes composite rankings, and writes
a ranked strategy report.

Usage::

    python scripts/evaluate_strategies.py --input logs/experiments
    python scripts/evaluate_strategies.py --input logs/experiments --output logs/rankings.json
    python scripts/evaluate_strategies.py --input logs/experiments --quiet

The ranking JSON is written to ``logs/strategy_rankings_<date>.json``
by default.
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so ``src`` imports work
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.strategy_evaluator import StrategyEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and rank strategy experiment results.",
    )
    parser.add_argument(
        "--input",
        default="logs/experiments",
        help="Directory containing experiment JSON files (default: logs/experiments)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for rankings JSON. Default: logs/strategy_rankings_<date>.json",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress console summary table",
    )
    args = parser.parse_args()

    evaluator = StrategyEvaluator(input_dir=args.input)
    count = evaluator.load_experiments()

    if count == 0:
        print(f"No experiment files found in {args.input}")
        sys.exit(1)

    evaluator.evaluate()
    out_path = evaluator.save_rankings(output_path=args.output)

    if not args.quiet:
        evaluator.print_summary()

    print(f"\nRankings written to: {out_path}")
    print(f"Evaluated {count} experiments.")


if __name__ == "__main__":
    main()
