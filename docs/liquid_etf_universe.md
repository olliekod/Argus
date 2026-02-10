# Liquid ETF Universe

Argus now tracks the following liquid ETF universe across bars and options audits:

`SPY, QQQ, IWM, DIA, TLT, GLD, XLF, XLK, XLE, SMH`

## Coverage

- **Bars (1-minute)**: Alpaca + Yahoo are configured for this universe.
- **Options audits**: `scripts/tastytrade_health_audit.py` supports all universe symbols for nested chain normalization and DXLink quotes probes.

## Snapshot persistence policy

We persist **sampled** option quote snapshots only (not full-chain ticks):

- nearest 2 non-expired expiries
- ±5 strikes around spot proxy
- calls + puts
- deterministic ordering and capped sample size (<= 80 contracts/underlying)

Retention is configured in `config/config.yaml` under:

```yaml
tastytrade:
  snapshot_sampling:
    retention_days: 14
```

(Argus does not delete these rows automatically unless a separate retention job is run.)

## Commands

```bash
python scripts/verify_system.py
python scripts/tastytrade_health_audit.py --symbol SPY --quotes --duration 15
python scripts/tastytrade_health_audit.py --universe --quotes --duration 10
python scripts/provider_benchmark.py --duration 10
```
