# Bitcoin / IBIT in Regime and Risk

Argus now factors **crypto-equity correlation** into the **risk regime** so that when Bitcoin (and thus IBIT) sells off or spikes, the overall regime can tilt risk-off or risk-on. Heavy hitters and risk appetite are still tied to crypto; this makes that explicit in the regime that gates VRP and Overnight.

---

## What changed

### 1. IBIT in the risk basket (RegimeDetector)

- **Default risk basket** is now `["SPY", "IBIT", "TLT", "GLD"]`.
- **Voting:**
  - **SPY and IBIT** (risk-on proxies): TREND_UP + calm vol → risk-on vote (+1); TREND_DOWN or VOL_SPIKE → risk-off vote (-1).
  - **TLT and GLD** (defensive): TREND_UP (flight to safety) → risk-off vote (-1); TREND_DOWN → risk-on vote (+1).
- **Aggregate:** Sum of votes → RISK_ON / RISK_OFF / NEUTRAL. So when IBIT (and typically Bitcoin) drops or spikes vol, the regime can shift to risk-off even if SPY is mixed, and when both SPY and IBIT are trending up with calm vol, regime is more clearly risk-on.

**Requirement:** IBIT must have **bars** (e.g. from Alpaca). If `exchanges.alpaca.symbols` includes `IBIT`, the bar builder publishes IBIT bars and the regime detector will use them. No extra config needed for the default basket.

### 2. Config overrides (optional)

In your **regime thresholds** (e.g. `config/thresholds.yaml` or wherever the orchestrator loads regime config), you can set:

- **`risk_basket`** — e.g. `["SPY", "IBIT", "TLT", "GLD"]` (default) or add/remove symbols.
- **`risk_basket_defensive`** — symbols that vote “flight to safety” when trending up; default `["TLT", "GLD"]`.
- **`risk_basket_min_confidence`** — minimum symbol confidence to count a vote (default 0.5).

The **orchestrator** currently constructs `RegimeDetector` with no thresholds override, so the detector uses its internal defaults (including IBIT in the basket). If you later load regime thresholds from config, you can set `risk_basket` there to include or drop IBIT.

---

## Why IBIT (and not raw BTC) in the basket?

- **IBIT** is an equity bar that the existing pipeline already has; the regime detector is bar-driven, so adding IBIT reuses the same vol/trend logic as SPY.
- **Bitcoin spot** (e.g. from Bybit) is a different feed and scale; to use it in the **same** regime you’d either need to turn it into a synthetic “bar” or add it as an **external metric** (e.g. BTC 1d return) and merge it into risk. That’s a possible follow-up (e.g. add BTC return to global risk flow or as its own metric). For now, IBIT is a direct way to get “when Bitcoin (via IBIT) moves, regime reflects it.”

---

## Optional: BTC in global risk flow (future)

If you want **raw Bitcoin** movement (e.g. from Bybit/Deribit) to also affect the single “global risk flow” number that Overnight (and any other consumer) uses, you could:

- Compute **BTC 1d (or 4h) return** from your crypto feed.
- Add a term to the global risk flow formula, e.g. `+ w_btc * btc_return`, or publish `btc_return_1d` as an external metric so it appears in regime `metrics_json`.

That would be a separate change (e.g. in `GlobalRiskFlowUpdater` or a small BTC-return publisher). The current change (IBIT in the risk basket) already ties regime to crypto-equity movement via IBIT.

---

## Summary

- **Regime risk state** now uses **SPY + IBIT + TLT + GLD** by default. When Bitcoin (and IBIT) takes a hit or spikes, that contributes to risk-off; when IBIT and SPY are both trending up with calm vol, regime can be risk-on.
- **VRP** and **Overnight** (and any strategy that reads regime) therefore see this crypto-equity link in the risk regime without any new feeds; IBIT bars are already in the pipeline if Alpaca symbols include IBIT.
