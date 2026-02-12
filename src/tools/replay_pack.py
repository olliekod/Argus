"""
Daily Replay Pack Tool
=======================

Slices market_bars, bar_outcomes, regimes, and option chain snapshots
from the database for a specific symbol and time window, and saves them
to a JSON file for deterministic offline replay.

Supports:
- Single-symbol mode: ``--symbol SPY``
- Universe mode: ``--universe`` (loads all liquid ETF symbols)

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
from src.core.liquid_etf_universe import get_liquid_etf_universe


def _atm_iv_from_quotes_json(quotes_json: str, underlying_price: float) -> Optional[float]:
    """Fill ATM put IV from quotes_json when snapshot has no atm_iv.

    Provider IV is always preferred; derived IV is used only when provider
    did not supply it. Order: (1) top-level atm_iv (from connector/provider),
    (2) ATM put's iv field (provider on the quote), (3) derived from bid/ask
    via GreeksEngine (Black-Scholes) when neither is present.
    """
    if not quotes_json or not underlying_price or underlying_price <= 0:
        return None
    try:
        data = json.loads(quotes_json)
        if not isinstance(data, dict):
            return None
        # Top-level atm_iv from serialized OptionChainSnapshotEvent
        top = data.get("atm_iv")
        if top is not None and top != "" and float(top) > 0:
            return float(top)
        puts = data.get("puts") or []
        if not puts:
            return None
        timestamp_ms = int(data.get("timestamp_ms") or 0)
        expiration_ms = int(data.get("expiration_ms") or 0)
        T_years = 0.0
        if timestamp_ms and expiration_ms and expiration_ms > timestamp_ms:
            T_years = (expiration_ms - timestamp_ms) / (1000.0 * 365.25 * 24 * 3600)
        # Find ATM put (strike closest to underlying)
        best_put: Optional[Dict[str, Any]] = None
        best_dist = float("inf")
        for q in puts:
            if not isinstance(q, dict):
                continue
            strike = q.get("strike")
            if strike is None:
                continue
            try:
                s = float(strike)
                dist = abs(s - underlying_price)
                if dist < best_dist:
                    best_dist = dist
                    best_put = q
            except (TypeError, ValueError):
                continue
        if not best_put:
            return None
        # Prefer provider iv on the ATM put
        iv = best_put.get("iv")
        if iv is not None and iv != "" and float(iv) > 0:
            return float(iv)
        # Build list of puts with usable price, sorted by distance to ATM
        others_with_price = []
        for q in puts:
            if not isinstance(q, dict) or q is best_put:
                continue
            strike = q.get("strike")
            if strike is None:
                continue
            try:
                s = float(strike)
            except (TypeError, ValueError):
                continue
            if (q.get("iv") and float(q.get("iv") or 0) > 0) or float(q.get("mid") or 0) > 0 or float(q.get("last") or 0) > 0 or (float(q.get("bid") or 0) > 0 or float(q.get("ask") or 0) > 0):
                others_with_price.append((abs(s - underlying_price), q))
        others_with_price.sort(key=lambda x: x[0])
        candidates: List[Dict[str, Any]] = [best_put] + [q for _, q in others_with_price]
        for put in candidates:
            iv = put.get("iv")
            if iv is not None and iv != "" and float(iv) > 0:
                return float(iv)
            K = float(put.get("strike", 0))
            if K <= 0 or T_years <= 0:
                continue
            bid, ask = put.get("bid"), put.get("ask")
            mid_from_bid_ask = None
            if bid is not None and ask is not None and (float(bid or 0) > 0 or float(ask or 0) > 0):
                mid_from_bid_ask = (float(bid) + float(ask)) / 2.0
            mid_or_last = mid_from_bid_ask
            if (mid_or_last is None or mid_or_last <= 0) and put.get("mid"):
                try:
                    mid_or_last = float(put["mid"])
                except (TypeError, ValueError):
                    pass
            if (mid_or_last is None or mid_or_last <= 0) and put.get("last"):
                try:
                    mid_or_last = float(put["last"])
                except (TypeError, ValueError):
                    pass
            if mid_or_last and mid_or_last > 0:
                try:
                    from src.analysis.greeks_engine import GreeksEngine
                    engine = GreeksEngine()
                    kwargs = {}
                    if bid is not None and ask is not None and float(bid or 0) > 0:
                        kwargs["bid"] = float(bid)
                        kwargs["ask"] = float(ask)
                    iv_val, _ = engine.implied_volatility(
                        mid_or_last, underlying_price, K, T_years, "put", **kwargs
                    )
                    if iv_val and iv_val > 0:
                        return iv_val
                except Exception:
                    continue
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def _bar_timestamp_to_ms(ts: Any) -> int:
    """Convert bar timestamp (ISO str or number) to milliseconds (UTC)."""
    if ts is None:
        return 0
    if isinstance(ts, (int, float)):
        return int(ts * 1000) if ts < 1e12 else int(ts)
    try:
        s = str(ts)
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        try:
            return int(float(ts) * 1000)
        except Exception:
            return 0


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
        atm_iv = row.get("atm_iv")
        if atm_iv is None or (isinstance(atm_iv, (int, float)) and atm_iv <= 0):
            underlying = float(row.get("underlying_price") or 0)
            atm_iv = _atm_iv_from_quotes_json(row.get("quotes_json", "") or "", underlying)

        snapshots.append({
            "timestamp_ms": row.get("timestamp_ms", 0),
            "recv_ts_ms": recv_ts,
            "provider": row.get("provider", ""),
            "underlying_price": row.get("underlying_price", 0.0),
            "atm_iv": atm_iv,
            "quotes_json": row.get("quotes_json", ""),
            "symbol": row.get("symbol", symbol),
            "n_strikes": row.get("n_strikes", 0),
        })

    # Sort by recv_ts_ms for strict chronological ordering
    snapshots.sort(key=lambda s: s["recv_ts_ms"])
    return snapshots


async def create_replay_pack(
    symbol: str,
    start_date: str,  # YYYY-MM-DD
    end_date: str,    # YYYY-MM-DD
    output_path: str,
    provider: str = "tastytrade",
    bar_duration: int = 60,
    db_path: str = "data/argus.db",
    snapshot_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a replay pack for a single symbol.

    Returns the pack dict (also written to *output_path*).
    """
    db = Database(db_path)
    await db.connect()

    try:
        # 1. Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000) + (24 * 3600 * 1000) - 1

        print(f"Packing data for {symbol} ({provider}) from {start_date} to {end_date}...")

        # 2. Fetch Bars
        bars_raw = await db.get_bars_for_outcome_computation(
            source=provider,
            symbol=symbol,
            bar_duration=bar_duration,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        print(f"  Fetched {len(bars_raw)} bars.")

        # 3. Fetch Outcomes
        outcomes = await db.get_bar_outcomes(
            provider=provider,
            symbol=symbol,
            bar_duration_seconds=bar_duration,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        print(f"  Fetched {len(outcomes)} outcomes.")

        # If no bars or outcomes, hint which provider/source has data in the DB
        if len(bars_raw) == 0 or len(outcomes) == 0:
            bar_inv = await db.get_bar_inventory()
            outcome_inv = await db.get_outcome_inventory()
            bar_sources = [r for r in bar_inv if r.get("symbol") == symbol]
            outcome_providers = [r for r in outcome_inv if r.get("symbol") == symbol]
            if bar_sources and len(bars_raw) == 0:
                sources_str = ", ".join(f"{r['source']} ({r.get('bar_count', 0)} bars)" for r in bar_sources)
                print(f"  Hint: No bars for provider={provider!r}. Bars in DB for {symbol}: {sources_str}. Try --provider <source>.")
            if outcome_providers and len(outcomes) == 0:
                prov_str = ", ".join(f"{r['provider']}" for r in outcome_providers)
                print(f"  Hint: No outcomes for provider={provider!r}. Outcomes in DB for {symbol}: provider(s) {prov_str}. Try --provider <provider>.")

        # 4. Fetch Regimes (Symbol and Market)
        market = "EQUITIES"
        symbol_regimes = await db.get_regimes(scope=symbol, start_ms=start_ms, end_ms=end_ms)
        market_regimes = await db.get_regimes(scope=market, start_ms=start_ms, end_ms=end_ms)
        all_regimes = symbol_regimes + market_regimes
        print(f"  Fetched {len(all_regimes)} regimes ({len(symbol_regimes)} symbol, {len(market_regimes)} market).")

        # 5. Fetch Option Chain Snapshots (all providers by default; use snapshot_provider to filter)
        snapshots = await _fetch_snapshots(
            db, symbol, start_ms, end_ms, provider_filter=snapshot_provider
        )
        # Cap recv_ts_ms so snapshots are releaseable: last sim_ts_ms in replay is last bar close
        max_bar_ts_ms = max(
            (_bar_timestamp_to_ms(b.get("timestamp")) for b in bars_raw),
            default=0,
        )
        max_sim_ms = max_bar_ts_ms + bar_duration * 1000
        for s in snapshots:
            if s["recv_ts_ms"] > max_sim_ms:
                s["recv_ts_ms"] = max_sim_ms
        n_with_iv = sum(1 for s in snapshots if s.get("atm_iv") is not None and float(s.get("atm_iv") or 0) > 0)
        print(f"  Fetched {len(snapshots)} option chain snapshots ({n_with_iv} with atm_iv).")
        if n_with_iv == 0 and snapshots:
            s0 = snapshots[0]
            prov = s0.get("provider", "?")
            try:
                data = json.loads(s0.get("quotes_json") or "{}")
                puts = data.get("puts") or []
                n_with_iv_field = sum(1 for p in puts if isinstance(p, dict) and (float(p.get("iv") or 0) > 0))
                n_with_mid_last = sum(1 for p in puts if isinstance(p, dict) and (float(p.get("mid") or 0) > 0 or float(p.get("last") or 0) > 0))
                n_with_bid_ask = sum(1 for p in puts if isinstance(p, dict) and (float(p.get("bid") or 0) > 0 or float(p.get("ask") or 0) > 0))
                print(f"  Hint (no atm_iv): first snapshot provider={prov!r}, puts: {len(puts)} total, {n_with_iv_field} with iv>0, {n_with_mid_last} with mid/last>0, {n_with_bid_ask} with bid/ask>0.")
                if n_with_iv_field == 0 and n_with_mid_last == 0 and n_with_bid_ask == 0:
                    print("  This looks like indicative-only data (no greeks, no quotes). For IV in replay, use option snapshots from a feed that provides greeks (e.g. Tastytrade) or ensure Tastytrade options polling runs so packs include snapshots with atm_iv.")
            except Exception as e:
                print(f"  Hint (no atm_iv): could not inspect quotes_json: {e}")

        # 6. Build Pack
        pack: Dict[str, Any] = {
            "metadata": {
                "symbol": symbol,
                "provider": provider,
                "bar_duration": bar_duration,
                "start_date": start_date,
                "end_date": end_date,
                "packed_at": datetime.now(timezone.utc).isoformat(),
                "bar_count": len(bars_raw),
                "outcome_count": len(outcomes),
                "regime_count": len(all_regimes),
                "snapshot_count": len(snapshots),
            },
            "bars": [
                {**b, "timestamp_ms": _bar_timestamp_to_ms(b.get("timestamp"))}
                for b in bars_raw
            ],
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
    provider: str = "tastytrade",
    bar_duration: int = 60,
    db_path: str = "data/argus.db",
    symbols: Optional[List[str]] = None,
    snapshot_provider: Optional[str] = None,
) -> List[str]:
    """Create replay packs for every symbol in the liquid ETF universe.

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
        description="Create Replay Pack(s) from the Argus database.",
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
    parser.add_argument("--provider", default="tastytrade", help="Provider for bars/outcomes")
    parser.add_argument(
        "--snapshot-provider",
        default=None,
        help="Restrict option chain snapshots to this provider (e.g. alpaca, tastytrade). Default: include all providers.",
    )
    parser.add_argument("--db", default="data/argus.db", help="Path to argus.db")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.universe:
        out_dir = args.out or "data/packs"
        asyncio.run(create_universe_packs(
            start_date=args.start,
            end_date=args.end,
            output_dir=out_dir,
            provider=args.provider,
            db_path=args.db,
            snapshot_provider=getattr(args, "snapshot_provider", None),
        ))
    else:
        default_file = f"data/packs/{args.symbol}_{args.start}_{args.end}.json"
        if args.out is None:
            out_path = default_file
        else:
            p = Path(args.out)
            if p.suffix != ".json" or p.exists() and p.is_dir():
                # Treat as output directory: write SYMBOL_start_end.json inside it
                out_path = str(Path(args.out).resolve() / f"{args.symbol}_{args.start}_{args.end}.json")
            else:
                out_path = args.out
        asyncio.run(create_replay_pack(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            output_path=out_path,
            provider=args.provider,
            db_path=args.db,
            snapshot_provider=getattr(args, "snapshot_provider", None),
        ))


if __name__ == "__main__":
    main()
