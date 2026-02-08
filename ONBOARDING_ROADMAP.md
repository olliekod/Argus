# Argus Onboarding Roadmap

**Purpose:** Walk you through the entire Argus codebase so you understand the **WHY** and **HOW** of every major piece. Written for someone with little math background; we explain concepts in plain language first.

**Quick navigation:**  
[Part 0: What Argus Is](#part-0-what-argus-is-one-paragraph) · [Part 1: Architecture](#part-1-high-level-architecture-the-plumbing) · [Part 2: Core Concepts](#part-2-core-concepts-minimal-jargon) · [Part 3: Data Flow](#part-3-data-flow-step-by-step-why-and-how) · [Part 4: Config & Safety](#part-4-configuration-and-safety) · [Part 5: Soak/Tape](#part-5-soak-tape-and-guards) · [Part 6: Learning Order](#part-6-learning-order-your-roadmap) · [Part 7: Glossary](#part-7-glossary-quick-reference) · [Part 8: Why Each Piece](#part-8-why-each-piece-exists-summary)

---

## Part 0: What Argus Is (One Paragraph)

**Argus** is a 24/7 market monitor that:

1. **Pulls live data** from exchanges (crypto: Bybit; options/IV: Deribit; ETFs: Yahoo, Alpaca).
2. **Turns ticks into 1-minute bars** and then into **regimes** (e.g. “trending up”, “high volatility”).
3. **Runs “detectors”** that look for specific setups (e.g. “BTC volatility spiked and IBIT dropped → good time to sell put spreads”).
4. **Runs a huge “paper trader farm”** — hundreds of thousands of virtual traders, each with different parameters (IV threshold, profit target, etc.), to see which parameter combos would have made money.
5. **Sends you Telegram alerts** with concrete trade ideas (e.g. IBIT put spread with strikes and size) for you to execute **manually** (e.g. on Robinhood).

So: **observe → detect → paper-test many strategies → recommend the best ones for you to trade by hand.** No automated execution of real money (unless you explicitly switch to “live” mode later).

---

## Part 1: High-Level Architecture (The “Plumbing”)

Everything flows through a **central event bus** (pub/sub). Think of it as a post office: components **publish** messages to named **topics**, and other components **subscribe** to those topics and react when a message arrives. Nobody talks directly to each other; they all go through the bus.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA SOURCES (Connectors)                                                  │
│  Bybit WS, Deribit, Yahoo, Alpaca, (Polymarket optional)                    │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ publish QuoteEvent / MetricEvent
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  EVENT BUS (src/core/bus.py)                                                 │
│  Topics: market.quotes | market.bars | market.metrics | regimes.* |        │
│          signals.* | system.heartbeat | system.minute_tick | ...             │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ BarBuilder    │     │ FeatureBuilder  │     │ RegimeDetector  │
│ quotes → bars │     │ bars → returns, │     │ bars → trend/    │
│               │     │ vol, jump score │     │ vol/session     │
└───────┬───────┘     └────────┬────────┘     └────────┬────────┘
        │                      │                       │
        │ bars                 │ metrics               │ regime events
        ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PERSISTENCE (writes bars, signals, metrics to SQLite)                       │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        │ bars + regimes (and metrics) feed detectors
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DETECTORS (IBIT, BITO, Options IV, Volatility) + STRATEGIES (gates)        │
│  → SignalEvent published to signals.detections (or signals.raw/ranked)       │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Telegram      │     │ Paper Trader    │     │ Database        │
│ (alerts)      │     │ Farm (97K+      │     │ (signals, bars,  │
│               │     │ configs)        │     │ positions)      │
└───────────────┘     └─────────────────┘     └─────────────────┘
```

**Why a bus?** So that adding a new data source or a new consumer doesn’t require changing everyone else. One component publishes; many can subscribe. The bus also gives you **back-pressure**: if a topic queue gets full, the oldest messages can be dropped (for high-volume market data) so the system doesn’t stall.

**Entry point:** `main.py` → `asyncio.run(main())` → `src/orchestrator.py` → `ArgusOrchestrator`. The orchestrator loads config, creates the bus, creates all connectors and subscribers, then starts the bus and runs until you hit Ctrl+C.

---

## Part 2: Core Concepts (Minimal Jargon)

### 2.1 Events (What Flows on the Bus)

- **QuoteEvent** — One “tick”: a single price update (bid, ask, last, mid) for a symbol from one exchange. Has `source` (e.g. `bybit`, `yahoo`) and timestamps.
- **BarEvent** — One 1-minute OHLCV bar (open, high, low, close, volume) for a symbol, aligned to UTC minute boundaries. Built from many QuoteEvents.
- **MetricEvent** — Non-price data: funding rate, open interest, **IV (implied volatility)**, etc.
- **SignalEvent** — A trading idea from a detector: e.g. “sell IBIT put spread now” with priority (e.g. Tier 1 = immediate).
- **HeartbeatEvent / MinuteTickEvent** — Time-based triggers for flushing bars and doing periodic work.

**Where they’re defined:** `src/core/events.py`. Topic names live there too (e.g. `TOPIC_MARKET_QUOTES`, `TOPIC_MARKET_BARS`).

### 2.2 Why 1-Minute Bars?

Raw ticks are too noisy and too many. Bars give you a stable “candle” per minute so you can compute things like “how much did price move this minute?” and “how jumpy was it recently?” (volatility). Everything downstream (features, regimes, strategies) is built on bars so that behavior is **deterministic**: same bars in → same indicators and signals out (no randomness, no wall-clock dependence in the core pipeline).

### 2.3 Regimes (Simple Idea)

A **regime** is a **label for “what kind of market is this right now?”** — e.g.:

- **Trend:** RANGE vs TREND_UP vs TREND_DOWN (from EMAs and price relative to them).
- **Volatility:** VOL_LOW / VOL_NORMAL / VOL_HIGH / VOL_SPIKE (from recent move size vs history).
- **Session:** PRE / RTH / POST / CLOSED for equities; ASIA / EU / US for crypto.

**Why it matters:** Your strategy might say “only sell put spreads when trend is not down and volatility is high.” The regime detector puts those labels on the market so the rest of the system can gate or score signals.

### 2.4 Indicators (The Little Math You’ll See)

- **EMA (Exponential Moving Average)** — A smoothed average of price that gives more weight to recent prices. Used to define “trend” (price above/below EMA).
- **RSI (Relative Strength Index)** — A 0–100 number: high = “overbought,” low = “oversold.” Used in some regime/strategy logic.
- **ATR (Average True Range)** — Average size of price swings. Used to measure volatility and “how big is a normal move?”
- **Realized volatility** — How much price actually moved recently (e.g. standard deviation of returns). Compared to history to get “high” vs “low” vol.

You don’t need to derive formulas; just know: **EMA = trend, RSI = overbought/oversold, ATR/vol = how jumpy**. All of these are implemented in `src/core/indicators.py` in a **deterministic** way (batch and incremental, with tests for parity).

### 2.5 Options Terms (For IBIT/BITO Alerts)

- **IV (Implied Volatility)** — Market’s expectation of how much the underlying will move; from options prices. High IV = options are expensive (good for **selling** premium).
- **Put spread** — Sell a put, buy a lower put; you collect premium and cap downside. IBIT detector suggests these when conditions are right.
- **PoP (Probability of Profit)** — Chance the trade is profitable at expiry (from option pricing). Used for position sizing.
- **DTE (Days to Expiry)** — How many days until the option expires. Config has things like `target_expiry_days`, `time_exit_dte`.

---

## Part 3: Data Flow Step by Step (WHY and HOW)

### Step 1: Connectors Publish Quotes and Metrics

**Where:** `src/connectors/` (e.g. `bybit_ws.py`, `deribit_client.py`, `yahoo_client.py`, `alpaca_client.py`).

**What they do:** Connect to exchanges or data providers. When new data arrives (ticker, funding, IV), they build a `QuoteEvent` or `MetricEvent` and call `event_bus.publish(topic, event)`.

**Orchestrator’s role:** It creates each connector and passes a callback (e.g. `_on_bybit_ticker`). That callback normalizes the payload and publishes to the bus. So the flow is: **exchange → callback → EventBus**.

**Why multiple connectors?** Bybit = crypto perpetuals; Deribit = BTC/ETH options (IV); Yahoo/Alpaca = IBIT/BITO equity and options data. Each feed is normalized to the same event types so the rest of the system doesn’t care where the data came from.

### Step 2: Bar Builder Turns Quotes into Bars

**Where:** `src/core/bar_builder.py`.

**What it does:** Subscribes to `market.quotes`. For each symbol it keeps an **accumulator** for the current minute. When a new quote arrives it updates high/low/close/volume; when the minute changes (or a tick for the next minute arrives), it **closes** the bar and publishes a `BarEvent` to `market.bars`.

**Rules you’ll see in code:** Use exchange timestamp only; reject quotes without valid `source_ts`; align bars to UTC minute start; fix OHLCV if they violate invariants (e.g. high ≥ open, close). Volume is handled as **deltas** of cumulative exchange volume so we don’t double-count.

**Why:** Downstream components need a single, consistent bar stream. BarBuilder is the only place that turns ticks into bars, so all regime and feature logic is based on the same bars.

### Step 3: Feature Builder Adds Returns and Volatility Metrics

**Where:** `src/core/feature_builder.py`.

**What it does:** Subscribes to `market.bars`. For each bar it computes things like 1-bar **log return**, **rolling realized volatility**, and a **jump score** (big move vs recent vol). It publishes these as `MetricEvent` to `market.metrics`.

**Why:** Detectors and strategies often need “how much did price move?” and “how volatile is it?” in a standard form. FeatureBuilder centralizes that so everyone gets the same numbers.

### Step 4: Regime Detector Labels the Market

**Where:** `src/core/regime_detector.py`; regime types in `src/core/regimes.py`.

**What it does:** Subscribes to `market.bars`. Keeps **per-symbol state** (EMA, RSI, ATR, rolling vol) and updates it bar-by-bar. From that it decides: trend regime (RANGE / TREND_UP / TREND_DOWN), volatility regime (LOW / NORMAL / HIGH / SPIKE), and session regime (RTH, PRE, etc.). It publishes `SymbolRegimeEvent` and `MarketRegimeEvent` to `regimes.symbol` and `regimes.market`.

**Why:** Strategies and the paper farm need to know “what kind of market is this?” so they can allow or block trades (e.g. “only in RTH,” “only when vol is high”). Doing it once in RegimeDetector keeps behavior consistent and deterministic.

### Step 5: Persistence Writes to the Database

**Where:** `src/core/persistence.py`.

**What it does:** Subscribes to `market.bars`, `signals.*`, `market.metrics`, `system.heartbeat`. It batches bars and writes them to SQLite. Bars are **high priority** (never dropped on overflow; retried); signals medium; metrics/heartbeats lower (can be dropped under load). Flush is triggered by a 1-second timer, heartbeats, and shutdown.

**Why:** You need a durable record of bars and signals for backtesting, analysis, and debugging. Persistence isolates “how we write” from “how we compute,” and priority rules ensure the most important data (bars) is never lost.

### Step 6: Detectors and Strategies Produce Signals

**Where:**  
- Detectors: `src/detectors/` (`ibit_detector.py`, `options_iv_detector.py`, `volatility_detector.py`, etc.).  
- Strategies: `src/strategies/` (`dow_regime_timing.py`, `router.py`).

**What they do:**

- **Detectors** use bars, metrics, and sometimes options data to decide “is this an opportunity?” For example, **IBITDetector** checks: BTC IV above threshold, IBIT down by X%, IV rank sufficient → build a **TradeRecommendation** (strikes, size, expiry) and emit a **SignalEvent** (or trigger paper trades and Telegram).
- **Options IV detector** mainly **logs** IV for IBIT/BITO context; it doesn’t send Deribit trade alerts (you don’t trade there).
- **Volatility detector** labels expansion/compression; used as **context** for other strategies.
- **Strategies** like **DowRegimeTimingGate** emit **filter** signals (e.g. “allow sell put spread for IBIT in RTH”) that other logic can use. The **SignalRouter** collects raw signals, scores them (data quality, regime alignment), and publishes **ranked** signals.

**Why detectors vs strategies:** Detectors are the “opportunity finders” tied to concrete products (IBIT put spreads, IV spikes). Strategies are more abstract “gates” or scoring layers. Both feed into the same idea: **only act when conditions match**.

### Step 7: Paper Trader Farm (Why So Many “Traders”?)

**Where:** `src/trading/paper_trader_farm.py`, `src/trading/trader_config_generator.py`, `src/trading/paper_trader.py`.

**What it does:** The farm holds a **list of configs** (97K–400K+), each a different combination of parameters: warmth_min, pop_min, dte_target, gap_tolerance_pct, profit_target_pct, stop_loss_pct, session_filter, budget_tier, IV range, strategy type (bull put, bear call, iron condor, straddle sell). Only **IBIT** and **BITO** are tradeable underlyings; crypto symbols are data only.

When conditions are right (e.g. bar + regime + options data), the farm **evaluates** which configs would enter a trade. It doesn’t instantiate 400K Python objects; it uses **tensors** and batch logic (and optional GPU) to compute “would this config enter?” for many configs at once. Configs that enter get an actual **PaperTrader** instance to manage the position; exits are evaluated on bars/heartbeats.

**Why so many configs?** You don’t know in advance which threshold combo (e.g. “IV > 60%, PoP > 70%, 30 DTE”) will perform best. By running a huge set in parallel on the same live data, you **discover** which parameter sets make money. Later you can “follow” the best trader(s) manually (research mode, promotion rules in config).

### Step 8: Alerts and Dashboard

**Where:** `src/alerts/telegram_bot.py`, `src/dashboard/web.py`.

**What they do:** Telegram subscribes to signals (or is called by detectors/orchestrator) and sends you formatted messages (e.g. IBIT put spread with strikes and size). The dashboard serves a local web UI that shows status, PnL summary, farm status, recent logs, and can run simple commands (e.g. export tape). Both are optional and configured in `config/config.yaml`.

---

## Part 4: Configuration and Safety

- **config/config.yaml** — System-wide: exchanges, assets, alerts, database path, circuit breakers, research mode, soak guards, dashboard port, etc.
- **config/thresholds.yaml** — Per-detector thresholds (IV %, drop %, cooldown hours, etc.). README says: **90-day rule** after adopting a strategy — don’t change these for 90 days to avoid curve-fitting to noise.
- **config/strategy_params.json** — Optimized paper/live params (e.g. from `scripts/optimize.py`); used by trade logic and paper traders.

**Modes:** `ARGUS_MODE=collector` (default) = observation and paper trading only; no live execution. `ARGUS_MODE=live` would allow execution (guarded by `_guard_collector_mode` in orchestrator).

**Circuit breakers:** In config: max daily loss %, max consecutive losses, cooldown. These auto-pause trading when hit.

---

## Part 5: Soak, Tape, and Guards

For long-running stability:

- **SoakGuardian** (`src/soak/guards.py`) — Watches for quote drops, heartbeat staleness, persist queue depth, bar liveness, disk space, etc. Can send alerts if something is wrong.
- **TapeRecorder** (`src/soak/tape.py`) — Optionally records a rolling buffer of events (e.g. quotes) for replay and deterministic backtesting.
- **ResourceMonitor** — Tracks resource usage.

These are “operational” pieces so you can run Argus 24/7 and catch regressions or resource issues.

---

## Part 6: Learning Order (Your Roadmap)

Follow this order so each piece builds on the previous one.

1. **README.md** — High-level goal and setup.
2. **main.py** → **src/orchestrator.py** (first ~450 lines) — How the app starts, how the bus and DB are created, and the order of `setup()` (config → db → bar_builder, persistence, feature_builder, regime_detector → connectors → telegram → detectors → query layer → bus start → dashboard).
3. **src/core/events.py** — All event types and topic names. Reference while reading other modules.
4. **src/core/bus.py** — How publish/subscribe and worker threads work. No need to memorize; just “events go to queues, workers drain and call handlers.”
5. **src/core/bar_builder.py** — How quotes become one bar per symbol per minute. Focus on `_BarAccumulator` and when a bar is closed and published.
6. **src/core/indicators.py** — EMA, RSI, ATR in simple form (batch + incremental). Skim; you’ll see them again in regime_detector.
7. **src/core/regimes.py** — Regime enums and data structures (SymbolRegimeEvent, MarketRegimeEvent, data quality flags).
8. **src/core/regime_detector.py** — How bars become trend/vol/session regimes. Skim the update logic; notice it’s bar-driven and deterministic.
9. **src/core/feature_builder.py** — Short: bars → returns, realized vol, jump score → MetricEvent.
10. **src/core/persistence.py** — Priority queues and what gets written to DB (bars first, then signals, then metrics/heartbeats).
11. **config/config.yaml** and **config/thresholds.yaml** — So you know where behavior is tuned.
12. **src/detectors/base_detector.py** — Common detector interface.
13. **src/detectors/ibit_detector.py** — Full flow: BTC IV + IBIT drop + IV rank → TradeCalculator → recommendation and/or paper trade and Telegram.
14. **src/analysis/trade_calculator.py** — How a concrete IBIT put spread recommendation is built (strikes, size, Greeks, PoP).
15. **src/trading/trader_config_generator.py** — How the parameter grid is built (PARAM_RANGES, strategy types, regime compatibility).
16. **src/trading/paper_trader_farm.py** — Initialize configs, evaluate entries in batch, manage active PaperTrader instances and exits.
17. **src/strategies/dow_regime_timing.py** — How gate signals (e.g. “allow IBIT put spread in RTH”) are produced from bar + regime.
18. **src/strategies/router.py** — How raw signals are scored and ranked.
19. **src/alerts/telegram_bot.py** — How messages are sent.
20. **src/soak/guards.py** and **src/soak/tape.py** — Optional; for long-run and replay.

After that, dive into any script in `scripts/` (e.g. `optimize.py`, `run_backtest.py`, `paper_trading_status.py`) when you need to understand how a specific workflow is driven.

---

## Part 7: Glossary (Quick Reference)

| Term | Meaning |
|------|--------|
| **Bar** | One candle: open, high, low, close, volume for one minute (UTC-aligned). |
| **Bus** | Central pub/sub event system; components publish/subscribe by topic. |
| **Collector mode** | Observation + paper only; no live trading. |
| **Detector** | Module that watches market data and emits trading signals (e.g. IBIT put spread). |
| **DTE** | Days to expiry (option). |
| **EMA** | Exponential moving average (smoothed price trend). |
| **IV** | Implied volatility (from options prices). |
| **PoP** | Probability of profit (from option model). |
| **Put spread** | Sell higher put, buy lower put; collect premium, cap risk. |
| **Regime** | Label for market state: trend (up/down/range), vol (low/normal/high/spike), session (RTH, pre, post). |
| **RSI** | Relative strength index (overbought/oversold). |
| **Signal** | A trading recommendation (e.g. SignalEvent) from a detector or strategy. |
| **Source** | Which connector produced the data (bybit, deribit, yahoo, alpaca). |
| **Strategy** | Logic that gates or scores trades (e.g. “only in RTH,” “rank by regime alignment”). |
| **Tick** | Single price update; many ticks aggregate into one bar. |
| **Topic** | Named channel on the bus (e.g. `market.quotes`, `market.bars`). |

---

## Part 8: Why Each Piece Exists (Summary)

| Piece | WHY it exists |
|-------|----------------|
| **Event bus** | Decouple data sources from consumers; one publish, many subscribe; back-pressure. |
| **BarBuilder** | Single place to turn ticks into bars so all downstream logic sees the same candles. |
| **FeatureBuilder** | Central place for returns/vol/jump so detectors don’t duplicate math. |
| **RegimeDetector** | Single place to label “what kind of market” so strategies and farm can filter consistently. |
| **Persistence** | Durable store for bars and signals; priority so bars are never dropped. |
| **Detectors** | Turn “market conditions” into concrete trade ideas (IBIT put spread, etc.). |
| **Strategies (gates)** | Add rules like “only in RTH” or “only when trend is not down.” |
| **Paper Trader Farm** | Test hundreds of thousands of parameter sets on live data to find what works. |
| **Telegram** | Deliver actionable alerts to you for manual execution. |
| **Soak/guards/tape** | Run 24/7 safely and replay/debug with deterministic tape. |

By the end of this roadmap you should be able to open any of these files and understand **why** it’s there and **how** it fits into the big picture. For deeper “how,” follow the learning order and read the code next to this doc.
