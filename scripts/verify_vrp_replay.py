"""Repeatable Sprint 2 E2E replay verifier for VRP strategy.

Builds a replay pack, runs VRPCreditSpreadStrategy once, and prints core counts:
- bars_count
- outcomes_count
- snapshots_count
- trade_count
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to import path for script execution
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.experiment_runner import ExperimentConfig, ExperimentRunner
from src.strategies.vrp_credit_spread import VRPCreditSpreadStrategy
from src.tools.replay_pack import create_replay_pack


def _default_pack_path(symbol: str, start: str, end: str, provider: str) -> Path:
    safe_provider = (provider or "default").replace("/", "_")
    return Path("data/packs") / f"{symbol}_{start}_{end}_{safe_provider}.json"


def _reasons_for_zero_trades(pack: dict, expected_provider: str) -> list[str]:
    reasons: list[str] = []
    bars = pack.get("bars", [])
    outcomes = pack.get("outcomes", [])
    snapshots = pack.get("snapshots", [])

    if not bars:
        reasons.append("missing bars for the selected provider/date range")
    if not outcomes:
        reasons.append("missing outcomes (RV unavailable); run outcomes backfill for the same provider as bars")
    if not snapshots:
        reasons.append("missing option snapshots (IV unavailable)")

    iv_ready = 0
    for s in snapshots:
        atm_iv = s.get("atm_iv")
        try:
            if atm_iv is not None and float(atm_iv) > 0:
                iv_ready += 1
        except (TypeError, ValueError):
            continue
    if snapshots and iv_ready == 0:
        reasons.append("snapshots exist but none have atm_iv>0 (indicative feed or provider mismatch likely)")

    meta_provider = str(pack.get("metadata", {}).get("provider", ""))
    if meta_provider and expected_provider and meta_provider != expected_provider:
        reasons.append(f"provider mismatch: pack provider={meta_provider}, expected={expected_provider}")

    if not reasons:
        reasons.append("VRP threshold/regime gating too strict for this window (inspect strategy diagnostics)")

    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Build replay pack + run VRP replay smoke check.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--provider", required=True, help="Bars/outcomes provider")
    parser.add_argument("--pack_out", default=None, help="Optional output JSON path for replay pack")
    parser.add_argument("--db", default="data/argus.db", help="Path to argus.db")
    args = parser.parse_args()

    pack_path = Path(args.pack_out) if args.pack_out else _default_pack_path(args.symbol, args.start, args.end, args.provider)
    pack = asyncio.run(
        create_replay_pack(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            output_path=str(pack_path),
            provider=args.provider,
            db_path=args.db,
        )
    )

    bars_count = len(pack.get("bars", []))
    outcomes_count = len(pack.get("outcomes", []))
    snapshots_count = len(pack.get("snapshots", []))

    runner = ExperimentRunner(output_dir="logs/experiments")
    result = runner.run(
        ExperimentConfig(
            strategy_class=VRPCreditSpreadStrategy,
            strategy_params={"min_vrp": 0.05},
            replay_pack_paths=[str(pack_path)],
            starting_cash=10_000.0,
        )
    )
    trade_count = int(result.portfolio_summary.get("total_trades", 0))

    print(f"bars_count={bars_count}")
    print(f"outcomes_count={outcomes_count}")
    print(f"snapshots_count={snapshots_count}")
    print(f"trade_count={trade_count}")

    if trade_count == 0:
        print("WARNING: trade_count is 0. Likely reasons:")
        for reason in _reasons_for_zero_trades(pack, args.provider):
            print(f"- {reason}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
