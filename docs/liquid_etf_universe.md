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

Guardrails:
- `--max-snapshots-per-underlying-per-run` defaults to `200` and blocks oversized writes unless `--force` is used.
- Snapshot writes are append-only and keep `provider` + `recv_ts` for provenance.

## Commands

```bash
python scripts/verify_system.py
python scripts/verify_system.py --deep
python scripts/tastytrade_health_audit.py --symbol SPY --quotes --duration 15 --json-out logs/spy.json
python scripts/tastytrade_health_audit.py --universe --quotes --duration 10 --json-out logs/universe.json
python scripts/provider_benchmark.py --duration 10 --json-out logs/bench.json
python scripts/prune_option_snapshots.py --days 14
```


Greeks scaffolding: pass `--greeks` (and optional `--require-greeks`) to request DXLink Greeks events. If unavailable, output includes a TODO fallback note for derived greeks via IV solve + Black–Scholes.
