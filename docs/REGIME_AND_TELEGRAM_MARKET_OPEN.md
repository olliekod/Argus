# Regime Detection and Telegram Market Open Message

This document explains how **regime** is detected in Argus, how it is exposed in the **Telegram market open** notification, and how it relates to next steps.

---

## 1. Are we detecting the regime?

**Yes.** Regime is detected in two layers:

### 1.1 Symbol-level regime (per bar)

- **Component:** `RegimeDetector` (`src/core/regime_detector.py`).
- **Input:** Bar events from the event bus (e.g. SPY, TLT, GLD 1‑min bars).
- **Output:** For each symbol, every bar produces a **symbol regime** with:
  - **Volatility regime:** `VOL_LOW` | `VOL_NORMAL` | `VOL_HIGH` | `VOL_SPIKE` (from vol z-score vs recent history).
  - **Trend regime:** `TREND_UP` | `TREND_DOWN` | `RANGE` (from EMA slope and trend strength, with optional hysteresis).
  - **Liquidity regime:** `LIQ_HIGH` | `LIQ_NORMAL` | `LIQ_LOW` | `LIQ_DRIED` (from spread % and volume percentile).
- **Publishing:** Symbol regimes are published to `regimes.symbol` and persisted to the DB.

### 1.2 Market-level regime (EQUITIES)

- **Component:** Same `RegimeDetector` aggregates a **risk basket** (default: SPY, TLT, GLD) into a single **market regime** for EQUITIES.
- **Logic:**  
  - **SPY:** TREND_UP + (VOL_LOW or VOL_NORMAL) → risk-on; TREND_DOWN or VOL_SPIKE → risk-off; else neutral.  
  - **TLT/GLD:** TREND_UP → risk-off vote; TREND_DOWN → risk-on vote.  
  - Votes are summed; result is **risk_regime:** `RISK_ON` | `RISK_OFF` | `NEUTRAL` (or `UNKNOWN` if SPY not available / low confidence).
- **Session:** Session regime (PRE | RTH | POST | CLOSED for equities) is derived from bar timestamp only (no wall clock).
- **Publishing:** Market regime events are published to `regimes.market` and include `metrics_json` (SPY vol/trend, etc.) and, when configured, **global_risk_flow** (from the external metrics pipeline).

### 1.3 Global Risk Flow (separate metric)

- **Component:** `GlobalRiskFlowUpdater` + `compute_global_risk_flow()`.
- **Input:** Daily bars from Alpha Vantage (Asia/Europe ETFs + FX) in `market_bars`.
- **Output:** Single number: `0.4×AsiaReturn + 0.4×EuropeReturn + 0.2×FXRiskSignal`. Positive ≈ risk-on, negative ≈ risk-off.
- **Integration:** Injected into regime pipeline via `ExternalMetricEvent` → `regime_detector.set_external_metric("global_risk_flow", value)` → appears in `metrics_json` on subsequent market regime events. Also exposed to **ConditionsMonitor** for the conditions score and for **Telegram** (risk flow line).

So: **we are detecting regime** (symbol vol/trend/liquidity and market risk/session), and **global risk flow** is an additional regime-related input.

---

## 2. Is this part of our next steps?

- **Regime detection itself** is already implemented and running (Phases 0–2, MASTER_PLAN).
- **Showing regime in the Telegram market open message** was not explicitly in the Phase 2 plan; it is a **product/UX enhancement** that is now implemented:
  - The market open notification includes a **Regime** line: Risk (RISK_ON/RISK_OFF/NEUTRAL), SPY Vol, and SPY Trend when available.
  - **Global Risk Flow** is now correctly wired in conditions (bug fix: `get_current_conditions()` no longer returns before setting `risk_flow` from the updater), so the briefing shows the correct risk-flow label (risk-on / risk-off / neutral).

So: **detecting regime is current behavior**; **surfacing it in the market open Telegram message** was the missing piece and is now done.

---

## 3. How we determine regime (detail)

| Layer | Source | How it’s determined |
|-------|--------|----------------------|
| **Vol regime** | Bar indicators (ATR, vol history) | Vol z-score vs recent bars; thresholds → VOL_LOW / VOL_NORMAL / VOL_HIGH / VOL_SPIKE. |
| **Trend regime** | Bar indicators (EMA fast/slow, slope, strength) | EMA slope + trend strength vs thresholds; optional hysteresis and min dwell bars. |
| **Liquidity regime** | Quote/spread and volume | Spread % and volume percentile → LIQ_* (used for symbol regime; market risk mainly uses SPY vol/trend). |
| **Risk regime (market)** | SPY + TLT + GLD symbol regimes | Voting rule above; SPY drives most of the signal; TLT/GLD can push to risk-on/risk-off. |
| **Session regime** | Bar timestamp only | Deterministic: PRE / RTH / POST / CLOSED (equities) or ASIA / EU / US / OFFPEAK (crypto) from time-of-day in ET. |
| **Global risk flow** | Alpha Vantage daily bars | Asia/Europe/FX daily returns → single composite; DB-only at update time. |

**At market open (9:30 AM ET):** The “current” regime in the Telegram message is the **last emitted** market regime for EQUITIES. That is usually from the **last bar** processed (e.g. last PRE bar or previous day’s last RTH bar). So it is “last known” state, not a special “open” computation. Once the first RTH bar is processed, the next regime event will reflect RTH.

---

## 4. What the Telegram market open message contains (after changes)

- **Conditions:** Score (1–10), warmth label, BTC IV (and rank if available).
- **Global Risk Flow:** risk-on / risk-off / neutral (from numeric value; now correctly wired via conditions).
- **Regime:** Risk (RISK_ON/RISK_OFF/NEUTRAL) and, when available, SPY Vol and SPY Trend.
- **Instrument prices:** e.g. IBIT/BITO if available.
- **Farm status:** Configs, active traders, open positions.

---

## 5. References

- `src/core/regime_detector.py` — symbol and market regime logic.
- `src/core/regimes.py` — regime event schemas and thresholds.
- `src/core/market_regime_detector.py` — optional market-level risk from basket (separate from RegimeDetector’s own risk aggregation).
- `src/core/conditions_monitor.py` — conditions score and `get_current_conditions()` (includes risk_flow).
- `src/orchestrator.py` — `_send_market_open_notification()`, `_format_current_regime_for_telegram()`.
