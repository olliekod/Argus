# Replay Pack, IV, and “Zero Trades” – Summary

Summary of work on replay packs, option snapshot IV (implied volatility), and fixing experiments that showed 0 trades / 0 PnL.

---

## 1. Replay pack design and commands

### What a replay pack contains
- **Bars:** 1m OHLCV from one provider (e.g. Alpaca)
- **Outcomes:** Forward-return / runup-drawdown metrics per bar (same provider as bars)
- **Regimes:** Symbol + market regime classifications
- **Option chain snapshots:** All providers (Alpaca + Tastytrade) by default, each with `provider`, `recv_ts_ms`, `underlying_price`, `atm_iv`, `quotes_json`

### Single-pack approach (recommended)
- **Bars and outcomes:** One provider only (e.g. Alpaca) for a consistent backbone.
- **Snapshots:** Include **both** Alpaca and Tastytrade in the same pack so you get maximum coverage and can gate by `recv_ts_ms` and compare providers.
- One pack per symbol/date; no need to merge two packs at experiment time.

### Commands (SPY example; date = 2026-02-11)

```powershell
# 1. Backfill outcomes (so the pack has realized_vol for the strategy)
python -m src.outcomes backfill --provider alpaca --symbol SPY --bar 60 --start 2026-02-11 --end 2026-02-11

# 2. Build the pack (bars + outcomes + regimes + option snapshots)
python -m src.tools.replay_pack --symbol SPY --start 2026-02-11 --end 2026-02-11 --out data/packs/ --provider alpaca

# 3. Run experiment
python scripts/run_experiment.py --strategy VRPCreditSpreadStrategy --pack data/packs/SPY_2026-02-11_2026-02-11.json --output logs/experiments
```

- Use `--provider alpaca` (or whatever source your **bars** use) so bars and outcomes match.
- Do **not** pass `--snapshot-provider` if you want both Alpaca and Tastytrade snapshots in the pack.

---

## 2. Why the experiment showed “all zeroes”

### Strategy needs IV and RV
`VRPCreditSpreadStrategy` only generates intents when it has:
- **IV** from option snapshots (`atm_iv`) → `last_iv`
- **Realized vol** from bar outcomes (`realized_vol`) → `last_rv`

If either is missing, `generate_intents()` returns `[]` → 0 trades, 0 PnL.

### Issues we fixed (in order)

| Issue | Cause | Fix |
|-------|--------|-----|
| **KeyError: 'timestamp_ms'** | Pack stored bars with `timestamp` (DB format); experiment runner expected `timestamp_ms`. | In replay pack, normalize each bar to include `timestamp_ms` (from `_bar_timestamp_to_ms(timestamp)`). |
| **0 outcomes** | Outcomes had not been computed for the bar provider (e.g. Alpaca). | Run `python -m src.outcomes backfill --provider alpaca --symbol SPY --bar 60 --start ... --end ...` then re-pack. |
| **0 trades despite outcomes** | Strategy never saw IV: “no IV yet” warning. | See “IV and snapshots” below. |
| **Snapshots never released** | Harness only releases a snapshot when `sim_ts_ms >= recv_ts_ms`. Snapshots were stored with `recv_ts_ms` **after** the last bar close (e.g. poll 40 min after close). | When building the pack, cap each snapshot’s `recv_ts_ms` to **last bar close** (`max_sim_ms = last_bar_timestamp_ms + bar_duration_ms`), so all snapshots become visible by end of replay. |
| **Wrong cap** | We initially capped `recv_ts_ms` to `end_ms` (midnight). Last bar closes at market close (~4 PM), so `sim_ts_ms` never reached the cap. | Cap to `max_sim_ms` (last bar close), not `end_ms`. |
| **0 with atm_iv** | DB snapshot rows had `atm_iv` null; derivation from `quotes_json` failed because Alpaca snapshots were **indicative-only** (no greeks, no bid/ask/mid/last on puts). | See “IV: provider vs derived” and “Alpaca indicative” below. |

---

## 3. IV: provider vs derived

- **Provider IV is always preferred.** Derived IV is used only when the provider did not supply it.
- In the pack, when filling `atm_iv` from `quotes_json`, the order is:
  1. Top-level `atm_iv` (from connector when the snapshot was stored)
  2. ATM put’s `iv` field (provider on the quote)
  3. **Derived** from bid/ask or mid/last via **GreeksEngine** (Black–Scholes) when neither (1) nor (2) is present.

The same idea applies in `src/analysis/greeks_engine.py`: use `provider_iv` first; only then derive from mid (and optionally enforce illiquid guard with bid/ask).

---

## 4. Filling `atm_iv` when the snapshot has none

In `replay_pack.py`, when a snapshot row has no (or zero) `atm_iv`, we try to fill it from `quotes_json`:

1. **Top-level `atm_iv`** in the parsed JSON → use if > 0.
2. **ATM put** = put with strike closest to `underlying_price`.
3. **Provider IV** on that put → use if > 0.
4. **Derived IV:**  
   - Prefer **bid/ask** mid; if missing or zero, use put’s **mid** or **last**.  
   - If any put has a usable price (iv, mid, last, or bid/ask), we consider all such puts (sorted by distance to ATM) and derive IV via `GreeksEngine.implied_volatility(...)`.

If the JSON has no usable prices on any put (all iv/bid/ask/mid/last zero), we cannot fill `atm_iv`.

---

## 5. Alpaca indicative and “0 with atm_iv”

- Your Alpaca option snapshots for that day had **no greeks and no quotes**: `atm_iv` null, and every put had `iv=None`, `bid=0`, `ask=0`, and no usable `mid`/`last`.
- That is typical of an **indicative** feed (contract structure but no live greeks/quotes).
- So for that data we **cannot** compute or derive `atm_iv`; the pack correctly reports “0 with atm_iv”.

**What to do for IV in replay:**
- Use option snapshots from a feed that **does** provide greeks (and/or last/mid), e.g. **Tastytrade**.
- Run Tastytrade options polling for your underlyings so that future (and, if available, past) packs include Tastytrade snapshots with `atm_iv`.
- Pack **without** `--snapshot-provider` so the pack includes both Alpaca and Tastytrade; then any snapshot that has `atm_iv` (e.g. from Tastytrade) will give the strategy IV.

---

## 6. Diagnostics added

- **Experiment runner:** If the pack has 0 outcomes or 0 snapshots, it prints a short warning so you know why strategies like VRP might produce 0 trades.
- **VRPCreditSpreadStrategy:** One-time logs when it can’t signal: “no IV yet”, “no RV yet”, “VRP <= min_vrp”, or “regime filter (vol=… trend=…)”.
- **Replay pack:** After fetching snapshots it prints:
  - `Fetched N option chain snapshots (M with atm_iv).`
  - If M = 0, a **hint** summarizing the first snapshot: provider, puts count, how many puts have iv>0, mid/last>0, bid/ask>0, and a note that indicative-only data cannot supply IV.

---

## 7. Files touched (reference)

- **`src/tools/replay_pack.py`**  
  Bar `timestamp_ms` normalization; `_atm_iv_from_quotes_json` (provider first, then derive from mid/last/bid/ask); cap `recv_ts_ms` to `max_sim_ms`; diagnostic “N with atm_iv” and hint when 0.
- **`src/analysis/experiment_runner.py`**  
  Warning when pack has 0 outcomes or 0 snapshots.
- **`src/strategies/vrp_credit_spread.py`**  
  One-time diagnostic logs for missing IV/RV, VRP threshold, and regime filter.
- **`scripts/run_experiment.py`**  
  `logging.basicConfig(level=logging.INFO)` so strategy diagnostics are visible.

---

## 8. Quick checklist for “replay pack + IV + no zeroes”

1. **Bars:** Stored under the same `source` you pass as `--provider` (e.g. Alpaca).
2. **Outcomes:** Backfill for that provider/symbol/date so the pack has `realized_vol` and `window_end_ms`.
3. **Pack:** Use `--provider <bar_source>`, no `--snapshot-provider`, so snapshots from all providers (e.g. Alpaca + Tastytrade) are included.
4. **Snapshots:** Either have `atm_iv` in the DB or have `quotes_json` with at least one put that has iv or mid/last/bid/ask so we can derive it; for indicative-only data, use another feed (e.g. Tastytrade) that provides greeks.
5. **Re-run:** After backfill or pack changes, re-run the pack then the experiment so the strategy sees outcomes and (when available) IV.
