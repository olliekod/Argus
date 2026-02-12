"""
Daily Replay Pack Tool
=======================

Slices market_bars, bar_outcomes, regimes, and option chain snapshots
from the database for a specific symbol and time window, and saves them
to a JSON file for deterministic offline replay.

Supports:
- Single-symbol mode: ``--symbol SPY``
- Universe mode: ``--universe`` (loads all liquid ETF symbols)

Provider defaults are read from the ``data_sources`` policy in
``config/config.yaml``.  No ``--provider`` flag is required for
normal usage.  Advanced overrides (``--bars-provider``,
``--options-snapshot-provider``) are available but rarely needed.

Option chain snapshots are included when available.  Symbols without
options data simply produce packs with an empty ``snapshots`` list.
"""

import asyncio
import json
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.database import Database
from src.core.data_sources import get_data_source_policy, DataSourcePolicy
from src.core.liquid_etf_universe import get_liquid_etf_universe


async def _fetch_snapshots(
    db: Database,
    symbol: str,
    start_ms: int,
    end_ms: int,
    provider_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load option chain snapshots for a symbol and date range.

    By default includes all providers (Alpaca + Tastytrade) for cross-validation.
    Pass provider_filter to restrict to one provider (e.g. "alpaca" or "tastytrade").

    Returns a list of snapshot dicts ordered chronologically by
    ``recv_ts_ms`` (falling back to ``timestamp_ms`` for legacy rows).

    Each dict includes:
    - timestamp_ms, recv_ts_ms, provider, underlying_price
    - atm_iv (if available)
    - quotes_json payload
    """
    raw = await db.get_option_chain_snapshots(
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
    )

    snapshots: List[Dict[str, Any]] = []
    for row in raw:
        if provider_filter is not None and row.get("provider") != provider_filter:
            continue
        recv_ts = row.get("recv_ts_ms")
        if recv_ts is None:
            recv_ts = row.get("timestamp_ms", 0)

        snapshots.append({
            "timestamp_ms": row.get("timestamp_ms", 0),
            "recv_ts_ms": recv_ts,
            "provider": row.get("provider", ""),
            "underlying_price": row.get("underlying_price", 0.0),
            "atm_iv": row.get("atm_iv"),
            "quotes_json": row.get("quotes_json", ""),
            "symbol": row.get("symbol", symbol),
            "n_strikes": row.get("n_strikes", 0),
        })

    # Sort by recv_ts_ms for strict chronological ordering
    snapshots.sort(key=lambda s: s["recv_ts_ms"])
    return snapshots


async def _fetch_snapshots_multi(
    db: Database,
    symbol: str,
    start_ms: int,
    end_ms: int,
    providers: List[str],
) -> List[Dict[str, Any]]:
    """Fetch snapshots for multiple providers and merge chronologically."""
    all_snaps: List[Dict[str, Any]] = []
    for prov in providers:
        snaps = await _fetch_snapshots(db, symbol, start_ms, end_ms, provider_filter=prov)
        all_snaps.extend(snaps)
    all_snaps.sort(key=lambda s: s["recv_ts_ms"])
    return all_snaps


async def create_replay_pack(
    symbol: str,
    start_date: str,  # YYYY-MM-DD
    end_date: str,    # YYYY-MM-DD
    output_path: str,
    provider: Optional[str] = None,
    bar_duration: int = 60,
    db_path: str = "data/argus.db",
    snapshot_provider: Optional[str] = None,
    include_secondary_options: bool = False,
    policy: Optional[DataSourcePolicy] = None,
) -> Dict[str, Any]:
    """Create a replay pack for a single symbol.

    Provider defaults come from the data-source policy unless
    explicitly overridden via *provider* or *snapshot_provider*.

    Returns the pack dict (also written to *output_path*).
    """
    if policy is None:
        policy = get_data_source_policy()

    # Resolve effective providers from policy + overrides
    bars_provider = provider if provider is not None else policy.bars_provider
    effective_snapshot_provider = (
        snapshot_provider if snapshot_provider is not None
        else policy.options_snapshot_provider
    )

    db = Database(db_path)
    await db.connect()

    try:
        # 1. Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000) + (24 * 3600 * 1000) - 1

        print(f"Packing data for {symbol} (bars={bars_provider}, snapshots={effective_snapshot_provider}) from {start_date} to {end_date}...")

        # 2. Fetch Bars
        bars_raw = await db.get_bars_for_outcome_computation(
            source=bars_provider,
            symbol=symbol,
            bar_duration=bar_duration,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        print(f"  Fetched {len(bars_raw)} bars.")

        # 3. Fetch Outcomes (always from bars_primary)
        outcomes = await db.get_bar_outcomes(
            provider=bars_provider,
            symbol=symbol,
            bar_duration_seconds=bar_duration,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        print(f"  Fetched {len(outcomes)} outcomes.")

        # 4. Fetch Regimes (Symbol and Market)
        market = "EQUITIES"
        symbol_regimes = await db.get_regimes(scope=symbol, start_ms=start_ms, end_ms=end_ms)
        market_regimes = await db.get_regimes(scope=market, start_ms=start_ms, end_ms=end_ms)
        all_regimes = symbol_regimes + market_regimes
        print(f"  Fetched {len(all_regimes)} regimes ({len(symbol_regimes)} symbol, {len(market_regimes)} market).")

        # 5. Fetch Option Chain Snapshots
        if include_secondary_options:
            snap_providers = policy.snapshot_providers(include_secondary=True)
            # Override primary if user specified a custom snapshot provider
            if snapshot_provider is not None:
                snap_providers = [snapshot_provider] + [
                    p for p in policy.options_snapshots_secondary
                    if p != snapshot_provider
                ]
            snapshots = await _fetch_snapshots_multi(
                db, symbol, start_ms, end_ms, snap_providers
            )
            print(f"  Fetched {len(snapshots)} option chain snapshots (providers: {snap_providers}).")
        else:
            snapshots = await _fetch_snapshots(
                db, symbol, start_ms, end_ms, provider_filter=effective_snapshot_provider
            )
            print(f"  Fetched {len(snapshots)} option chain snapshots (provider: {effective_snapshot_provider}).")

        # 6. Build Pack
        secondary_included = include_secondary_options
        pack: Dict[str, Any] = {
            "metadata": {
                "symbol": symbol,
                "provider": bars_provider,
                "bars_provider": bars_provider,
                "options_snapshot_provider": effective_snapshot_provider,
                "secondary_options_included": secondary_included,
                "bar_duration": bar_duration,
                "start_date": start_date,
                "end_date": end_date,
                "packed_at": datetime.utcnow().isoformat(),
                "bar_count": len(bars_raw),
                "outcome_count": len(outcomes),
                "regime_count": len(all_regimes),
                "snapshot_count": len(snapshots),
            },
            "bars": bars_raw,
            "outcomes": outcomes,
            "regimes": all_regimes,
            "snapshots": snapshots,
        }

        # 7. Write
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(pack, f, indent=2)

        print(f"  Pack saved to {output_path}")
        return pack

    finally:
        await db.close()


async def create_universe_packs(
    start_date: str,
    end_date: str,
    output_dir: str = "data/packs",
    provider: Optional[str] = None,
    bar_duration: int = 60,
    db_path: str = "data/argus.db",
    symbols: Optional[List[str]] = None,
    snapshot_provider: Optional[str] = None,
    include_secondary_options: bool = False,
    policy: Optional[DataSourcePolicy] = None,
) -> List[str]:
    """Create replay packs for every symbol in the liquid ETF universe.

    Provider defaults come from the data-source policy.

    Returns a list of output file paths that were written.
    """
    if symbols is None:
        symbols = get_liquid_etf_universe()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    for sym in symbols:
        out_path = os.path.join(output_dir, f"{sym}_{start_date}_{end_date}.json")
        try:
            await create_replay_pack(
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
                output_path=out_path,
                provider=provider,
                bar_duration=bar_duration,
                db_path=db_path,
                snapshot_provider=snapshot_provider,
                include_secondary_options=include_secondary_options,
                policy=policy,
            )
            written.append(out_path)
        except Exception as exc:
            print(f"  WARNING: Failed to pack {sym}: {exc}")

    print(f"\nUniverse packing complete: {len(written)}/{len(symbols)} symbols.")
    return written


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create Replay Pack(s) from the Argus database.  "
                    "Defaults follow the data_sources policy in config/config.yaml.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", help="Single symbol to pack (e.g. SPY)")
    group.add_argument(
        "--universe",
        action="store_true",
        default=False,
        help="Pack all symbols in the liquid ETF universe",
    )

    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (single-symbol mode) or directory (universe mode). "
             "Defaults to data/packs/",
    )

    # ── Provider overrides (advanced) ─────────────────────────────────
    parser.add_argument(
        "--provider",
        default=None,
        help="(Legacy) Alias for --bars-provider.  Still accepted for backward compatibility.",
    )
    parser.add_argument(
        "--bars-provider",
        default=None,
        help="Override bars/outcomes provider (default: data_sources.bars_primary from config).",
    )
    parser.add_argument(
        "--options-snapshot-provider",
        default=None,
        help="Override primary options snapshot provider (default: data_sources.options_snapshots_primary from config).",
    )
    parser.add_argument(
        "--snapshot-provider",
        default=None,
        help="(Legacy) Alias for --options-snapshot-provider.",
    )
    parser.add_argument(
        "--include-secondary-options",
        action="store_true",
        default=False,
        help="Also include secondary options snapshots (e.g. Alpaca structural cross-check).",
    )
    parser.add_argument("--db", default="data/argus.db", help="Path to argus.db")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve provider overrides: new flags take precedence over legacy
    bars_prov = args.bars_provider or args.provider  # None = use policy default
    snap_prov = args.options_snapshot_provider or args.snapshot_provider  # None = use policy default

    if args.universe:
        out_dir = args.out or "data/packs"
        asyncio.run(create_universe_packs(
            start_date=args.start,
            end_date=args.end,
            output_dir=out_dir,
            provider=bars_prov,
            db_path=args.db,
            snapshot_provider=snap_prov,
            include_secondary_options=args.include_secondary_options,
        ))
    else:
        out_path = args.out or f"data/packs/{args.symbol}_{args.start}_{args.end}.json"
        asyncio.run(create_replay_pack(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            output_path=out_path,
            provider=bars_prov,
            db_path=args.db,
            snapshot_provider=snap_prov,
            include_secondary_options=args.include_secondary_options,
        ))


if __name__ == "__main__":
    main()
