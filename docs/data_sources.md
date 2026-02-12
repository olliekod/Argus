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
| Options Chain Snapshots (IV/surface) | **Tastytrade** REST | Authoritative IV source |
| Options Real-Time Quotes + Greeks | **Tastytrade** DXLink | Live mode only |
| Options Snapshots (structural) | Alpaca *(secondary)* | Cross-check only; **no IV/greeks** |
| Bars (backfill/sanity) | Yahoo *(secondary)* | Optional cross-validation |

## Configuration

The canonical policy lives in `config/config.yaml` under the
`data_sources` key:

```yaml
data_sources:
  bars_primary: alpaca
  outcomes_from: bars_primary
  options_snapshots_primary: tastytrade
  options_snapshots_secondary:
    - alpaca
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

### Secondary: Alpaca (structural cross-check)

- **Config key:** `options_snapshots_secondary: [alpaca]`
- **What it provides:** Strike/expiry structure, underlying price,
  contract counts.
- **What it does NOT provide:** Reliable IV or Greeks.  Alpaca
  option snapshots do not include `atm_iv` or Greeks fields.
- **Usage:** Structural coverage sanity checks only.  Strategies
  **must not** depend on IV/greeks from Alpaca unless the
  `allow_alpaca_iv` flag is explicitly set.

## Where Greeks / IV Come From

### Replay Mode (backtesting)

1. **Tastytrade snapshots** — `atm_iv` field from snapshot.
2. **Derived IV** — If `atm_iv` is null, a Brenner-Subrahmanyam
   approximation is computed from the closest-to-ATM put's bid/ask
   in the Tastytrade `quotes_json` payload.
3. **Alpaca IV** — Only used if strategy explicitly sets
   `allow_alpaca_iv: true`.  This is a last resort and is
   discouraged.

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
--include-secondary-options  Also fetch Alpaca snapshots
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
