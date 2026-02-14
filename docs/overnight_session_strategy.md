# Overnight Session Momentum Strategy

## Overview

The **OVERNIGHT_SESSION_V1** strategy captures momentum signals during session
transitions — the windows where overnight information is priced in and
institutional order flow shifts.

## Entry Logic

### Equities (SPY, QQQ, IBIT, etc.)

| Window | Description |
|--------|-------------|
| RTH Tail | Last `entry_window_minutes` min before 16:00 ET close |
| PRE Market | First `entry_window_minutes` min of pre-market (04:00–09:30 ET) |

### Crypto (BTC, ETH, SOL, etc.)

Entries trigger on **session transitions**: ASIA→EU, EU→US (the previous session
must differ from the current session).

### Signal Gating

An entry is only generated when the best available forward return (from
`visible_outcomes`) at the configured `horizon_seconds` exceeds
`fwd_return_threshold`.  **V1 is long-only** — all entries are BUY.

## Exit Logic

Explicit `CLOSE` intent emitted when `(sim_ts - entry_ts) >= horizon_seconds`.
No reliance on harness TTL.

## Parameters

| Key | Type | Default | Sweep Range |
|-----|------|---------|-------------|
| `fwd_return_threshold` | float | 0.005 | 0.002 – 0.020 (step 0.003) |
| `entry_window_minutes` | int | 30 | [15, 30, 45] |
| `horizon_seconds` | int | 14400 | [3600, 14400, 28800] |
| `gate_on_risk_flow` | bool | False | [false, true] |
| `min_global_risk_flow` | float | -0.005 | — |

## Global Risk Flow (Optional Gate)

When `gate_on_risk_flow = true`, entries are blocked if the
`global_risk_flow` metric is below `min_global_risk_flow`.

### GlobalRiskFlow Definition

```
GlobalRiskFlow = 0.4 * AsiaReturn + 0.4 * EuropeReturn + 0.2 * FXRiskSignal
```

| Component | Symbols | Weight |
|-----------|---------|--------|
| AsiaReturn | EWJ, FXI, EWT, EWY, INDA | 0.4 |
| EuropeReturn | EWG, EWU, FEZ, EWL | 0.4 |
| FXRiskSignal | USD/JPY daily return | 0.2 |

Weights redistribute proportionally when components are missing.

### Data Source

Daily bars from **Alpha Vantage** (free tier: 5 calls/min, 25/day).
Persisted to `market_bars` with `source="alphavantage"`, `bar_duration=86400`.

**Lookahead prevention:** Daily bars are timestamped at 00:00 UTC.
Only bars with `timestamp < sim_time` are used (strict less-than).

### Backfill

```bash
# Compact (100 days)
python scripts/alphavantage_daily_backfill.py

# Full history
python scripts/alphavantage_daily_backfill.py --full

# Dry-run
python scripts/alphavantage_daily_backfill.py --dry-run
```

## Files

| File | Purpose |
|------|---------|
| `src/strategies/overnight_session.py` | Strategy implementation |
| `src/core/global_risk_flow.py` | GlobalRiskFlow computation |
| `src/connectors/alphavantage_client.py` | Alpha Vantage REST client |
| `scripts/alphavantage_daily_backfill.py` | Daily bar backfill CLI |
| `config/overnight_sweep.yaml` | Parameter sweep grid |
| `tests/test_overnight_session.py` | Strategy unit tests (20) |
| `tests/test_global_risk_flow.py` | Risk flow unit tests (15) |

## Setup

1. Add your Alpha Vantage API key to `config/secrets.yaml`:
   ```yaml
   alphavantage:
     api_key: "YOUR_KEY_HERE"
   ```

2. Enable in `config/config.yaml`:
   ```yaml
   alphavantage:
     enabled: true
   ```

3. Run backfill:
   ```bash
   python scripts/alphavantage_daily_backfill.py
   ```

## Phase 3 — CME Futures (Future Work)

No existing futures ingestion infrastructure.  E-mini S&P 500 (ES) and
Nikkei 225 futures would provide pre-open signals with better coverage
than ETF proxies, but require a new data connector (CME or broker-provided).
