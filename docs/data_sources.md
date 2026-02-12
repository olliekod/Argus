# Argus Data Source Policy

This document describes exactly where each type of market data comes
from, what each provider is and is not used for, and how the policy
is enforced across replay packs, experiments, and strategies.

## Quick Reference

| Data Type | Primary Provider | Notes |
|-----------|-----------------|-------|
| 1-min OHLCV Bars | **Alpaca** | Market Data API v2 |
| Bar Outcomes (forward returns) | Derived from **Alpaca** bars | OutcomeEngine consumes bars_primary |
| Regimes | Computed from **Alpaca** bars | Optionally quote-aware if quote data exists |
| Options Chain Snapshots (IV/surface) | **Tastytrade** REST or **Public.com** REST | Choose one as primary; keep the other as secondary/fallback |
| Options Real-Time Quotes + Greeks | **Tastytrade** DXLink | Live mode only |
| Bars (backfill/sanity) | Yahoo *(secondary)* | Optional cross-validation |

## Configuration

The canonical policy lives in `config/config.yaml` under the
`data_sources` key:

```yaml
data_sources:
  bars_primary: alpaca
  outcomes_from: bars_primary
  options_snapshots_primary: tastytrade  # or public
  options_snapshots_secondary:
    - public  # or tastytrade (the non-primary provider)
  options_stream_primary: tastytrade_dxlink
  bars_secondary:
    - yahoo
```

The helper module `src/core/data_sources.py` exposes
`get_data_source_policy()` which returns a frozen
`DataSourcePolicy` dataclass.  All downstream tools read from this
single source of truth.

## Where Bars Come From

- **Provider:** Alpaca (`bars_primary: alpaca`)
- **Timeframe:** 1-minute OHLCV
- **Symbols:** Configured in `exchanges.alpaca.symbols`
- **API:** `https://data.alpaca.markets/v2`
- **Secondary:** Yahoo Finance is available for backfill / sanity
  checking (`bars_secondary: [yahoo]`).  Yahoo bars are never used
  as the primary source for outcomes or replay packs.

## Where Outcomes Come From

Outcomes are **always derived from bars_primary** (Alpaca bars).
The `OutcomeEngine` computes forward returns, max runup/drawdown,
and realized volatility over configurable horizons using the same
bars that populate replay packs.

There is no separate "outcomes provider" — the `outcomes_from`
config key is always `bars_primary`.

## Where Option Snapshots Come From

### Primary: Tastytrade REST

- **Config key:** `options_snapshots_primary: tastytrade`
- **What it provides:** Option chain quotes (bid/ask per strike),
  expiry dates, `atm_iv` (at-the-money implied volatility), and
  any surface fields the Tastytrade API returns.
- **This is the authoritative IV source** for all strategies.


### Alternate Primary: Public.com REST

- **Config key:** `options_snapshots_primary: public`
- **What it provides:** Batch IV/Greeks (`impliedVolatility`, delta, gamma, theta, vega, rho)
  via Public.com REST by OSI symbol.
- **Integration mode:** Current v1 uses hybrid structure sourcing
  (Alpaca chain structure + Public Greeks) and emits provider=`public` snapshots.
- **Replay impact:** Replay packs and harness include these snapshots automatically
  because they are read from `option_chain_snapshots` without provider-specific schema.

### Secondary: the non-primary options provider

- **Config key:** `options_snapshots_secondary: [public]` when primary is `tastytrade`,
  or `options_snapshots_secondary: [tastytrade]` when primary is `public`.
- **What it provides:** Validation and substitution when the primary provider has gaps.
- **Policy:** Collect both when enabled; replay pack build can include both streams
  or prefer primary with secondary gap-fill fallback.

> Alpaca is bars/outcomes only in this setup. It is **not** an options data source.

## Where Greeks / IV Come From

### Replay Mode (backtesting)

1. **Primary options snapshots** (`tastytrade` or `public`) — `atm_iv` from snapshot.
2. **Derived IV** — If `atm_iv` is null, derive from options quotes in `quotes_json` when possible.
3. **Secondary provider snapshots** — used for validation and optional gap-fill fallback in replay-pack creation.

### Live Mode

1. **DXLink streaming** (`options_stream_primary:
   tastytrade_dxlink`) — Real-time option quotes and Greeks
   streamed via the Tastytrade DXLink WebSocket.
2. **Tastytrade REST snapshots** — Polled periodically as a
   fallback if the stream is down.

## Replay Pack Composition

When building a replay pack (`src/tools/replay_pack.py`), the
defaults follow the data-source policy:

| Pack Component | Source | Policy Key |
|---------------|--------|------------|
| Bars | `bars_primary` (Alpaca) | `data_sources.bars_primary` |
| Outcomes | Derived from `bars_primary` | `data_sources.outcomes_from` |
| Regimes | Computed from bars | — |
| Option Snapshots | `options_snapshots_primary` (Tastytrade) | `data_sources.options_snapshots_primary` |

No `--provider` flag is needed for normal usage.  Advanced
overrides:

```
--bars-provider       Override bars/outcomes provider
--options-snapshot-provider  Override snapshot provider
--include-secondary-options  Also fetch secondary provider snapshots
--provider            Legacy alias for --bars-provider
--snapshot-provider   Legacy alias for --options-snapshot-provider
```

Pack metadata records which providers were used:

```json
{
  "metadata": {
    "bars_provider": "alpaca",
    "options_snapshot_provider": "tastytrade",
    "secondary_options_included": false
  }
}
```

## Experiment Manifests

Experiment artifacts (`logs/experiments/*.json`) record
data-source provenance:

```json
{
  "manifest": {
    "data_sources": {
      "from_packs": {
        "bars_providers": ["alpaca"],
        "options_snapshot_providers": ["tastytrade"],
        "secondary_options_included": false
      },
      "config_policy": {
        "bars_primary": "alpaca",
        "options_snapshots_primary": "tastytrade",
        "options_stream_primary": "tastytrade_dxlink"
      }
    }
  }
}
```

## End-to-End Verification

Run the full verification pipeline:

```bash
python scripts/e2e_verify.py --symbol SPY --days 5
```

This confirms:
1. Bars/outcomes coverage from `bars_primary`
2. Options snapshot coverage + `atm_iv` fill rate from
   `options_snapshots_primary`
3. DXLink module availability
4. Replay pack builds with policy defaults
5. Smoke experiment (VRPCreditSpreadStrategy) runs
6. Strategy evaluator produces rankings

Artifacts: `logs/e2e/<date>/summary.json` and `summary.md`.


## Public Connector Notes

- Public REST calls are throttled by a shared client-side limiter (`public.rate_limit_rps`, default `10`) across account lookup and greeks requests.
- Public options connector needs a structure source for expirations/contracts. It can use:
  - `exchanges.alpaca.options` (async structure path), or
  - `tastytrade.snapshot_sampling` (sync structure path wrapped for async use).
- If Public returns 400/422 on greeks, verify `osiSymbols` query serialization against Public docs (repeated key vs comma-separated).

## Provider Comparison Workflow

Use the comparison tool to choose the options primary provider:

```bash
python scripts/compare_options_snapshot_providers.py --symbol SPY --start 2026-01-01 --end 2026-01-03
```

The report compares Tastytrade vs Public overlap (receipt freshness and `atm_iv` agreement) and prints a recommendation.
Afterward, set:

- `data_sources.options_snapshots_primary` = preferred provider
- `data_sources.options_snapshots_secondary` = other provider

## Replay Pack Primary-with-Fallback

Replay pack builder supports primary-preferred merge with secondary gap-fill:

```bash
python -m src.tools.replay_pack --symbol SPY --start 2026-01-01 --end 2026-01-03 \
  --options-snapshot-fallback --options-snapshot-gap-minutes 3
```

This mode keeps primary snapshots by default and inserts secondary snapshots only when primary has a multi-minute gap.
Metadata records whether fallback was enabled/used and how many snapshots were filled.
