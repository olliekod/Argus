# Overnight Session Strategy — Implementation Plan

**Source:** MASTER_PLAN.md §4 (Primary Strategies #1), ChatGPT feed recommendations  
**Created:** 2026-02-13  
**Updated:** 2026-02-13 — Alpha Vantage as international data source

This document defines a phased implementation plan for the **Overnight Session Momentum / Seasonality** strategy: what to build, what data to use, and how it integrates with Argus. It synthesizes the Master Plan’s direction with practical feed recommendations for cross-session signals.

**International market data:** Argus uses **Alpha Vantage API** for global ETF proxies (10 symbols) and FX (4 pairs). **Slow signals only:** daily bars, regime indicators, overnight return modeling, session correlation — **not** for per-second or execution feeds. Cache aggressively; 25 requests/day budget (10 + 4 = 14 used, ~11 buffer). Config format:

```yaml
alphavantage:
  api_key: "XXXXXXXXXXXX2342234"
```

---

## 1. Strategy Overview

### 1.1 Goal

Capture predictable returns around session transitions and overnight periods:

- **US equities:** Close → overnight gap → next open (CLOSED → RTH transition)
- **Crypto:** Asia → Europe → US handovers (ASIA → EU → US transitions)
- **ETF proxies:** IBIT overnight vs BTC; SPY overnight drift; liquid ETF universe (SPY, QQQ, DIA, IWM, GLD, TLT, XLE, XLF, XLK, SMH, NVDA)
- **Global influence:** Asia/Europe session returns as regime inputs (when feeds exist)

### 1.2 Why This Strategy First

- Master Plan lists it as **#1 priority**
- Uses **bars + outcomes + session regime** — no IV/options dependency
- Works with existing Argus data (Alpaca, Yahoo, Bybit)
- “Simple, robust, hard to overfit”
- No options snapshots required (unlike VRP)

---

## 2. Data Requirements — What You Actually Need

### 2.1 Core vs Auxiliary

For most cross-session strategies, you do **not** need every exchange’s order book. You need:

| Category | Purpose | Examples |
|----------|---------|----------|
| **Global ETF proxies** | Asia/Europe sentiment via US-listed ETFs | EWJ, FXI, EWG, EWU, FEZ, EWL, EEM, EWT, EWY, INDA |
| **FX pairs** | Risk-on/off, session momentum; often cleaner than equities | EUR/USD, USD/JPY, GBP/USD, AUD/USD |
| **Index futures** | 24h global sentiment (Phase 3, if DXLink supports) | ES, NQ (CME Globex) |

These cover Asia growth & tech, China risk, Europe industrial & financial flow, EM risk appetite, without redundant markets.

### 2.2 Current Argus Coverage

| Feed | Coverage | Overnight relevance |
|------|----------|---------------------|
| **Alpaca** | US equities/ETFs only; no Asia/Europe exchange data | ✅ Liquid ETF universe (SPY, QQQ, IBIT, DIA, IWM, GLD, TLT, XLE, XLF, XLK, SMH, NVDA) + 10 global ETFs |
| **Yahoo Finance** | US ETFs, quotes, chart API | ✅ Same ETFs |
| **Alpha Vantage** | **International market data** — global equities, FX, indices | ✅ Asia/Europe indices, FX pairs; daily OHLCV time series |
| **Tastytrade / DXLink** | US derivatives, possibly Globex futures | ⚠️ Check if ES/NQ available via DXLink |
| **Public** | US retail focus | No global depth |
| **Bybit** | Crypto perpetuals | ✅ BTC/USDT 24/7 — Asia/EU/US session data |

**Alpha Vantage supplies:** 10 ETF symbols (EWJ, FXI, EWT, EWY, INDA, EWG, EWU, FEZ, EWL, EEM) + 4 FX pairs; daily only; 25 req/day budget. See §9 for API details.

---

## 3. Phased Implementation

### Phase 1 — Strategy Logic (No New Feeds)

**Goal:** Ship a working `OvernightSessionStrategy` using existing bars and outcomes.

**Data:** Alpaca/Yahoo bars for liquid ETF universe (SPY, QQQ, IBIT, DIA, IWM, GLD, TLT, XLE, XLF, XLK, SMH, NVDA) + 10 global ETFs; Bybit for BTC/USDT. Outcomes from OutcomeEngine. Session regime from `get_session_regime()`.

**Scope:**

1. **ReplayStrategy implementation** (`src/strategies/overnight_session.py`)
   - Implement `ReplayStrategy` (`on_bar`, `generate_intents`, `strategy_id`)
   - Use `session_regime` to detect transitions (e.g. last bar of RTH, first bar of PRE)
   - Use `visible_outcomes` for forward returns (e.g. 4h, 8h, overnight horizon)
   - Emit `TradeIntent` for LONG/SHORT based on session + outcome logic

2. **Entry/exit rules (minimal V1)**
   - **Equities:** Enter at last N minutes of RTH or first N of PRE; exit at next RTH open or fixed horizon
   - **Crypto:** Enter at session transitions (ASIA→EU, EU→US); use outcome horizon to close
   - **Horizons:** Use existing outcome horizons (e.g. 14400 = 4h, 28800 = 8h, 86400 = 1d)

3. **Config and wiring**
   - Add module to `_STRATEGY_MODULES` in `strategy_research_loop.py`
   - Add config block in `research_loop.yaml`

**Deliverables:**

- `src/strategies/overnight_session.py`
- Unit tests for session-transition detection
- Integration in research loop; verify runs on existing packs

**Dependencies:** None beyond current bars + outcomes + session.

---

### Phase 2 — Data Enhancement (Low Effort)

**Goal:** Add global proxies and regime signals without new connectors.

**Status: COMPLETE (2026-02-17).** All core implementation and verification done:

- **Alpha Vantage collector** — 10 ETFs + 4 FX daily → `market_bars` (source=alphavantage). Budget: 14 calls/day within 25 free-tier limit.
- **Global risk flow computation** — `src/core/global_risk_flow.py`: 0.4×Asia + 0.4×Europe + 0.2×FX; weight redistribution; strict less-than lookahead prevention.
- **DB-only updater** — `src/core/global_risk_flow_updater.py`: reads from DB (no API calls at update time); publishes `ExternalMetricEvent`; 5-min cache.
- **Regime integration** — Regime detector subscribes to external metrics; merges `global_risk_flow` into `metrics_json`.
- **Replay pack injection** — `src/tools/replay_pack.py` injects `global_risk_flow` into regime `metrics_json` deterministically from AV daily bars; `sort_keys=True` for reproducibility.
- **Overnight strategy gating** — `gate_on_risk_flow` / `min_global_risk_flow` in `OvernightSessionStrategy`; reads from `visible_regimes["EQUITIES"]["metrics_json"]`.
- **E2E verification** — `tests/test_phase2_e2e.py` (30 tests): replay pack injection, strategy gating suppression, replay harness integration, deterministic behavior, edge cases.
- **Research loop wiring** — `OvernightSessionStrategy` in `config/research_loop.yaml`; `config/overnight_sweep.yaml` includes `gate_on_risk_flow: [false, true]`.

**Symbol sets:**
- **Overnight traded assets:** SPY, QQQ, IBIT, DIA, IWM, GLD, TLT, XLE, XLF, XLK, SMH, NVDA (liquid ETF universe) + BTC/USDT (crypto).
- **Regime-only global ETFs (Alpha Vantage daily):** EWJ, FXI, EWT, EWY, INDA, EWG, EWU, FEZ, EWL (9 equity ETFs) + FX:USDJPY, FX:EURUSD, FX:GBPUSD, FX:AUDUSD (4 FX pairs). These are **not traded** — used only for global risk flow computation and regime features.

**Remaining optional items:**
- Add global ETFs to Alpaca/Yahoo configs if intraday bars are needed for these symbols (currently daily only via AV).
- Enable `gate_on_risk_flow: true` as default in sweep configs after research confirms value.
- EEM (Emerging Markets) is in the plan docs but not in the risk-flow computation symbols (by design — 9 ETFs suffice).

#### 2a. Global ETF Proxies (10 Symbols — Asia + Europe)

Add these symbols to Alpaca/Yahoo and Alpha Vantage daily backfill:

| Symbol | Region | Purpose |
|--------|--------|---------|
| **Asia** | | |
| EWJ | Japan | Nikkei proxy; core Asia signal |
| FXI | China | China large caps; risk-on/off driver |
| EWT | Taiwan | Semiconductor/global tech signal |
| EWY | South Korea | Export & tech cycle signal |
| INDA | India | Growing global macro influence |
| **Europe** | | |
| EWG | Germany | Eurozone industrial core |
| EWU | United Kingdom | Financial & global exposure |
| FEZ | Eurozone | Broad large caps |
| EWL | Switzerland | Defensive Europe exposure |
| **EM risk** | | |
| EEM | Emerging markets | Broad EM risk gauge |

**Captures:** Asia growth & tech, China risk, Europe industrial & financial flow, EM risk appetite — no redundant markets.

**Tasks:**

- Extend `config/config.yaml` `exchanges.alpaca.symbols` and `exchanges.yahoo.symbols` with these 10
- Add to Alpha Vantage daily backfill (TIME_SERIES_DAILY); persist to `market_bars` with `source=alphavantage`
- Ensure outcomes backfill includes these symbols
- **Cache aggressively:** fetch once per day; regime indicators, overnight return modeling, session correlation read from DB

#### 2b. Alpha Vantage — FX Pairs (4 Pairs) + Daily Backfill

**Alpha Vantage** provides global equity and FX data. Use for **slow signals only:** daily bars, regime indicators, overnight return modeling, session correlation. **Not** for per-second or execution feeds.

**Recommended 4 FX pairs** (FX is often a cleaner session signal than equities):

| Pair | Purpose |
|------|---------|
| EUR/USD | Europe vs US risk regime |
| USD/JPY | Global risk-on/off proxy; predicts equity futures drift |
| GBP/USD | London session momentum |
| AUD/USD | Asia/commodity risk proxy; predicts equity futures drift |

**Data source:**

| Data type | Endpoint | Response format |
|-----------|----------|-----------------|
| **10 ETF symbols** | `TIME_SERIES_DAILY` | JSON: `Time Series (Daily)` → date → OHLCV |
| **4 FX pairs** | `FX_DAILY` | JSON: `Time Series FX (Daily)` → date → OHLC (no volume) |

**API budget (25 requests/day):**

| Source | Calls/day | Buffer |
|--------|-----------|--------|
| 10 symbols | 10 | |
| 4 FX pairs | 4 | |
| **Total** | **14** | ~11 for retries, diagnostics, occasional extra symbols |

**Tasks:**

- Add `src/connectors/alphavantage_client.py` — REST client for TIME_SERIES_DAILY, FX_DAILY
- Config: `alphavantage.api_key` (from secrets)
- Backfill script: fetch 10 symbols + 4 FX pairs **once per day**; persist to `market_bars` with `source=alphavantage`
- **Cache aggressively:** all downstream consumers (regime detector, overnight return model, session correlation) read from DB
- Compute **global risk flow** (Asia return, Europe return, FX risk signal) as regime features; feed into regime engine. See §5.

**Deliverables:**

- Alpha Vantage connector + daily backfill script; regime feature extraction (incl. global risk flow)

---

### Phase 3 — CME Futures (If DXLink Supports)

**Goal:** Use ES/NQ overnight drift as a regime or signal input.

**Assumption:** DXLink may already stream Globex futures. CME trades nearly 24h.

**Key contracts:**

- ES (S&P 500)
- NQ (Nasdaq)
- RTY (Russell)
- 6E (EUR/USD futures)
- 6J (JPY/USD futures)

**Tasks:**

1. **Confirm DXLink futures support**
   - Check Tastytrade/DXLink docs for ES, NQ, 6E, 6J
   - If available: add futures symbols to DXLink config; bars flow into same `market_bars` (or futures table)

2. **Ingestion path**
   - If DXLink streams futures: create bar builder path for futures (similar to options snapshot path)
   - Persist to DB; outcomes backfill for futures symbols

3. **Strategy integration**
   - Use ES overnight return as regime input (e.g. “ES overnight positive → bullish bias for SPY open”)
   - Or: trade ES/NQ directly if execution path exists

**Dependencies:** DXLink futures availability; new bar-ingestion path if not already present.

---

### Phase 4 — Alpha Vantage Daily FX Refresh (Optional Extension)

**Goal:** Ensure FX regime features stay fresh via **daily** FX_DAILY refresh.

Alpha Vantage FX endpoints:

- **FX_DAILY** — Daily OHLC time series; use for regime features (Asia/Europe session, risk-on/off)
- **CURRENCY_EXCHANGE_RATE** — Real-time; **not used** (per-second/execution feeds out of scope)

**Pairs:** EUR/USD, USD/JPY, GBP/USD, AUD/USD (4 calls/day).

**Tasks:**

- Schedule daily FX_DAILY fetch (e.g. after Asia close); persist to `market_bars` with `source=alphavantage`
- Regime feature: “USDJPY risk-on/off predicts equity strength”; strategy uses FX regime as gate
- All reads from DB; no intraday polling

**Dependencies:** Alpha Vantage API key (same as Phase 2b); included in 14-call daily budget.

---

## 4. Strategy Logic — Detailed Design (Phase 1)

### 4.1 Session Transitions

| Market | Transition | Typical behavior |
|--------|------------|------------------|
| Equities | RTH close → CLOSED | Overnight gap risk |
| Equities | CLOSED → PRE | Pre-market drift |
| Equities | PRE → RTH | Open gap, high vol |
| Crypto | ASIA → EU | Europe handover |
| Crypto | EU → US | US handover |
| Crypto | US → OFFPEAK | Low liquidity |

### 4.2 Entry Rules (V1)

**Equities (liquid ETF universe + 10 global ETFs):**

- **Entry window:** Last 15–30 minutes of RTH, or first 30 minutes of PRE
- **Signal:** Visible outcome `fwd_return` (e.g. 4h or 8h) above threshold → LONG; below → SHORT (or flat)
- **Regime gate:** Skip if `visible_regimes` indicate risk-off / high vol

**Crypto (BTC/USDT):**

- **Entry window:** First N minutes of EU or US session (session transition)
- **Signal:** Prior session return (ASIA or EU) as momentum; outcome horizon for exit

### 4.3 Exit Rules (V1)

- **Time-based:** Close at fixed horizon (e.g. next RTH open for equities; 4h for crypto)
- **Outcome-based:** Use `visible_outcomes` to close when max_runup or max_drawdown hit target

### 4.4 Outcome Horizons

Existing outcome config (`horizons_seconds_by_bar`) typically includes:

- 60, 300, 900, 3600, 14400, 86400

For overnight:

- **14400 (4h):** Intra-overnight
- **28800 (8h):** Full overnight (if configured)
- **86400 (1d):** Next-day close

Add 28800 if not present for overnight experiments.

### 4.5 ReplayStrategy Interface

```python
class OvernightSessionStrategy(ReplayStrategy):
    strategy_id = "OVERNIGHT_SESSION_V1"

    def on_bar(self, bar, sim_ts_ms, session_regime, visible_outcomes,
               visible_regimes=None, visible_snapshots=None):
        # Track session transitions, latest bar, outcomes
        ...

    def generate_intents(self, sim_ts_ms) -> List[TradeIntent]:
        # Emit LONG/SHORT at entry windows when signal + regime allow
        ...
```

### 4.6 Market Configuration

Replay pack / harness use `market: "EQUITIES"` or `"CRYPTO"` per pack. Overnight strategy should support both:

- **Equities pack:** PRE, RTH, POST, CLOSED from `get_session_regime("EQUITIES", ts)`
- **Crypto pack:** ASIA, EU, US, OFFPEAK from `get_session_regime("CRYPTO", ts)`

Consider separate config entries: one for equities (liquid ETF universe + 10 global ETFs), one for crypto (BTC/USDT).

---

## 5. Global Risk Flow — Regime Feature Layer

**Global risk flow** is not separate from regime detection. It is a **feature layer** that feeds into the regime engine. Argus already has:

```
ingestion → indicators → regime → strategy → allocation
```

Add:

```
global session data → global features (incl. global risk flow) → regime engine
```

Everything else stays the same.

### 5.1 What Regime Detection Needs

The regime engine answers:

- Is market risk-on or risk-off?
- Is volatility rising or falling?
- Is capital rotating globally?
- Are markets trending or mean reverting?

SPY alone does not show where flows originate. US futures often move because Asia sold overnight, Europe rallied at open, FX moved risk-off, or EM markets dumped. Regime detection works better with global drivers.

### 5.2 What “Global Risk Flow” Is

A **compressed summary** of global session moves, used as an input to regime detection.

**Step 1 — Measure sessions**

- **Asia return:** AsiaClose − AsiaOpen (from Asia ETF proxies, e.g. EWJ, FXI)
- **Europe return:** EuropeClose − EuropeOpen (from Europe ETF proxies, e.g. EWG, EWU)
- **Overnight ES futures move:** (if DXLink available) US futures move before open
- **FX risk move:** USD/JPY change overnight (risk-on/off proxy)

**Step 2 — Combine into one signal**

Instead of feeding 10 ETFs + 4 FX pairs separately, compress:

```
GlobalRiskFlow = 0.4 × AsiaReturn + 0.4 × EuropeReturn + 0.2 × FXRiskSignal
```

(Weights are illustrative; tune or use PCA/composite.)

**Step 3 — Feed into regime engine**

Regime inputs become:

- US volatility
- SPY trend
- **Global risk flow**
- Overnight range

Regime detection improves because it sees **where pressure comes from**, not just price movement.

### 5.3 Why Compress?

Feeding 14 correlated signals causes overfitting, noise, conflicting signals, and unstable regime classification. Compressing into a few factors (PCA, global risk indices, composite momentum) improves stability.

### 5.4 Example

Asia sold overnight, Europe flat, USD/JPY falling (risk-off) → **GlobalRiskFlow &lt; 0** → US open likely weak, spreads widen, vol higher → regime engine switches to defensive strategies, lower sizing, avoid long delta.

### 5.5 Why This Helps VRP

VRP trades benefit when vol is overpriced after risk-off and underpriced in calm markets. Global session flow predicts vol expansion, US open risk, overnight gap behavior. Global risk flow helps VRP timing.

### 5.6 Implementation

- **Phase 2b:** Compute Asia return, Europe return, FX risk signal from cached bars; combine into `GlobalRiskFlow`.
- **Regime detector:** Add `global_risk_flow` as an input feature alongside vol, trend, etc.
- **Strategy gating:** Overnight and VRP strategies use regime (which now includes global risk flow) to gate entries.

---

## 6. Feed Addition Order

Implement in this order:

1. **Alpha Vantage connector** — Add `alphavantage_client.py`; TIME_SERIES_DAILY (10 ETF symbols), FX_DAILY (4 FX pairs). Config: `alphavantage.api_key`.
2. **Add global ETF proxies** — Config: 10 symbols (EWJ, FXI, EWT, EWY, INDA, EWG, EWU, FEZ, EWL, EEM) to Alpaca/Yahoo.
3. **Alpha Vantage daily backfill** — 10 symbols + 4 FX pairs; persist to `market_bars`; cache aggressively; ~14 calls/day, ~11 buffer.
4. **Confirm CME futures via DXLink** — If available, highest leverage for 24h sentiment (Phase 3).

Phase 1 does not wait on any of these.

---

## 7. Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Data Ingestion (existing + Phase 2)               │
├─────────────────────────────────────────────────────────────────────────┤
│  Alpaca/Yahoo: liquid ETF universe + 10 global ETFs (EWJ, FXI, ...)   │
│  Bybit: BTC/USDT                                                        │
│  Alpha Vantage: 10 symbols (EWJ–EEM) + 4 FX (EUR/USD, USD/JPY, ...)    │
│                 Daily only; ~14 calls/day; aggressive cache             │
│  (Phase 3) DXLink: ES, NQ (if available)                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OutcomeEngine │ RegimeDetector (+ global risk flow) │ Sessions          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Replay Pack (bars, outcomes, regimes, [snapshots for VRP])             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OvernightSessionStrategy (ReplayStrategy)                              │
│  - session_regime transitions                                           │
│  - visible_outcomes (fwd_return, horizons)                              │
│  - visible_regimes (optional gate)                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Strategy Research Loop → Evaluator → Registry → Allocation             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Checklist (Phase 1)

| # | Task | Location | Description |
|---|------|----------|-------------|
| 1.1 | Create strategy module | `src/strategies/overnight_session.py` | `OvernightSessionStrategy(ReplayStrategy)` |
| 1.2 | Session transition detection | `overnight_session.py` | Detect RTH→POST→CLOSED, CLOSED→PRE→RTH; ASIA→EU→US |
| 1.3 | Entry logic | `overnight_session.py` | Entry windows + outcome threshold + regime gate |
| 1.4 | Exit logic | `overnight_session.py` | Time-based or outcome-based close |
| 1.5 | Outcome horizon config | `outcomes` config | Ensure 28800 (8h) if needed for overnight |
| 1.6 | Register strategy | `strategy_research_loop.py` | Add to `_STRATEGY_MODULES` |
| 1.7 | Research loop config | `research_loop.yaml` | Add `OvernightSessionStrategy` under `strategies` |
| 1.8 | Unit tests | `tests/test_overnight_session.py` | Session transitions, entry/exit logic |
| 1.9 | Verify E2E | Manual | Run `--once`; confirm non-zero trades where expected |

---

## 9. Alpha Vantage API Reference

Argus uses Alpha Vantage for **slow signals only:** daily bars, regime indicators, overnight return modeling, session correlation. **Not** for per-second or execution feeds. Full docs: <https://www.alphavantage.co/documentation/>

### 9.1 Config

```yaml
alphavantage:
  api_key: "XXXXXXXXXXXX2342234"
```

API key goes in `secrets.yaml` (or equivalent); never commit. Load via `config.get_secret("alphavantage", "api_key")`.

### 9.2 Recommended Symbol Set (10 ETF + 4 FX)

| Type | Symbols | Calls/day |
|------|---------|-----------|
| **Asia + Europe ETFs** | EWJ, FXI, EWT, EWY, INDA, EWG, EWU, FEZ, EWL, EEM | 10 |
| **FX pairs** | EUR/USD, USD/JPY, GBP/USD, AUD/USD | 4 |
| **Total** | | **14** |

**Budget:** 25 requests/day. Leaves ~11 for retries, diagnostics, occasional extra symbols. **Cache aggressively:** fetch once per day; all consumers read from DB.

### 9.3 Endpoints Relevant to Overnight Strategy

| Endpoint | Function | Params | Use |
|----------|----------|--------|-----|
| **ETF daily** | `TIME_SERIES_DAILY` | symbol (EWJ, FXI, etc.), outputsize=compact/full | Asia/Europe session returns, regime features |
| **FX daily** | `FX_DAILY` | from_currency, to_currency, outputsize=compact/full | Regime: USDJPY risk-on/off, EURUSD flows, session momentum |
| **Realtime FX** | `CURRENCY_EXCHANGE_RATE` | **Not used** | Out of scope (no per-second feeds) |
| **Intraday** | `TIME_SERIES_INTRADAY` | Premium; **Not used** | Out of scope |

### 9.4 Symbol Conventions (Global Equity — Optional)

For direct exchange symbols (e.g. TSCO.LON, MBG.DEX), use Search Endpoint (`SYMBOL_SEARCH`). Primary overnight set uses US-listed ETFs (EWJ, EWG, etc.) via TIME_SERIES_DAILY.

### 9.5 Response Parsing

- **Equity:** Top-level key `Time Series (Daily)`; date strings as keys; values `{"1. open": str, "2. high": str, "3. low": str, "4. close": str, "5. volume": str}`. Parse to float; convert date to epoch ms for `market_bars`.
- **FX daily:** Top-level key `Time Series FX (Daily)`; same structure but no volume.
- **Error:** Response may contain `"Error Message"` or `"Note"` (rate limit) — check before parsing.

### 9.6 Rate Limits and Caching

- **25 requests/day** (free tier). 10 symbols + 4 FX = 14; ~11 buffer for retries, diagnostics.
- **Cache aggressively:** fetch once per day (e.g. after Asia close); persist to DB; regime detector, overnight model, session correlation read from cache.
- Use `outputsize=compact` (100 days) when sufficient; `full` for initial backfill only.

---

## 10. References

- [MASTER_PLAN.md](../MASTER_PLAN.md) §4 — Strategy priorities
- [RESEARCH_ENGINE_AND_CLOSED_LOOP.md](RESEARCH_ENGINE_AND_CLOSED_LOOP.md) — Research loop usage
- [src/core/sessions.py](../src/core/sessions.py) — Session boundaries (EQUITIES, CRYPTO)
- [src/strategies/vrp_credit_spread.py](../src/strategies/vrp_credit_spread.py) — Reference ReplayStrategy
- [docs/outcome_semantics.md](outcome_semantics.md) — Outcome windows and horizons
- [Alpha Vantage API Documentation](https://www.alphavantage.co/documentation/) — International equity, FX, and data formats
