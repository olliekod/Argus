# News Sentiment — Pattern Detection Plan

**Goal:** Add a news-sentiment signal to Argus so the system can **detect patterns** that combine price/regime data with qualitative context (headlines, macro narrative). Sentiment is treated as an input to pattern detection: regime confirmation, divergence, and event-driven context.

**Status:** Plan only — not yet implemented.

---

## 1. Why Sentiment as Pattern Detection

Argus already detects:

- **Regime** (trend, volatility, session) from bars and indicators.
- **Global risk flow** from daily ETF/FX returns.
- **Economic blackouts** (FOMC, CPI, Jobs) as hard dates.

It does **not** yet use:

- Whether the **narrative** (news) aligns or diverges with price/regime.
- **Sentiment extremes** that often precede reversals or confirm breakouts.
- **Event context** around macro releases (beyond blackout windows).

Using news sentiment as a **pattern input** allows:

| Pattern | Description | Use in Argus |
|--------|--------------|--------------|
| **Regime confirmation** | Risk-on regime + positive news sentiment → stronger conviction. | Increase weight or allow entries when regime and sentiment align. |
| **Divergence** | Price/regime bullish but sentiment very negative (or vice versa). | Flag for caution or reduce size; possible mean-reversion setup. |
| **Sentiment extremes** | Extreme fear/greed in headlines. | Gate or scale: e.g. avoid adding risk in extreme fear until confirmation. |
| **Event context** | Sentiment shift around CPI/FOMC. | Enrich regime around events; optional blackout extension. |

So the “agent” is not necessarily heavy ML: it can be a **scored signal** (from API, keywords, or ML) that Argus **weighs** together with existing regime/risk-flow in its decision logic.

---

## 2. Architecture (Reuse Existing Pipeline)

News sentiment will plug into the same pipeline as `global_risk_flow`:

```
[News sources] → [Sentiment collector] → ExternalMetricEvent("news_sentiment", value)
       → RegimeDetector (set_external_metric) → metrics_json on MarketRegimeEvent
       → Strategies read visible_regimes["EQUITIES"]["metrics_json"] → gate/weight
```

- **Event:** `ExternalMetricEvent(key="news_sentiment", value=..., timestamp_ms=...)`
- **Value shape:** JSON-serialisable. Suggested: numeric score (e.g. -1 to +1 or 0–100) plus optional fields (e.g. `label`, `source`, `n_headlines`) for logging and future pattern logic.
- **Replay:** Replay packs can inject `news_sentiment` into `metrics_json` per regime (same as `global_risk_flow`) so backtests are deterministic when historical sentiment is available.

No change to RegimeDetector schema: it already merges any `key` from `ExternalMetricEvent` into `_risk_metrics` → `metrics_json`.

---

## 3. Implementation Phases

### Phase 1 — Integration point and stub (low effort)

| # | Task | Description |
|---|------|-------------|
| 1.1 | **NewsSentimentUpdater** | New module (e.g. `src/core/news_sentiment_updater.py`) that holds the contract: `async def update() -> Optional[float]` (or dict). On success, publish `ExternalMetricEvent(key="news_sentiment", value=...)`. |
| 1.2 | **Stub implementation** | First version returns a **constant** (e.g. `0.0` or `None`) or a small dict `{"score": 0.0, "label": "stub"}` so the rest of the pipeline can be wired without a real news source. |
| 1.3 | **Orchestrator** | In the same loop as `_run_external_metrics_loop`, or in a separate loop, call the news sentiment updater periodically (e.g. same interval as global risk flow, or 1h). Config key e.g. `news_sentiment.enabled`, `news_sentiment.interval_seconds`. |
| 1.4 | **Replay pack injection** | In `replay_pack.py`, when injecting `global_risk_flow` into `metrics_json`, support optional `news_sentiment` from a small helper (e.g. `get_news_sentiment_for_replay(sim_ts_ms)`) that returns stub or historical value if available. |

**Outcome:** Strategies can read `news_sentiment` from `metrics_json`; replay can inject it; no real news yet.

---

### Phase 2 — Real sentiment (choose one path)

Pick **one** of (a)–(c) for the first non-stub implementation.

| Path | Description | Effort | Dependencies |
|------|-------------|--------|--------------|
| **(a) Third-party API** | Use Alpha Vantage News Sentiment, Benzinga, or similar. Collector calls API, maps response to a single score (and optional label), publishes it. | Low | API key; rate limits. |
| **(b) Headlines + keyword/lexicon** | Fetch headlines (RSS, NewsAPI, etc.), score with a word list or simple lexicon (e.g. Loughran–McDonald or custom), aggregate (e.g. average, or decay by age). | Low–Medium | News source; no ML. |
| **(c) NLP / light ML** | Run a pre-trained model (e.g. FinBERT) on headline text; aggregate per time window; publish score. | Medium | `transformers` + model; optional GPU. |

**Tasks:**

| # | Task | Description |
|---|------|-------------|
| 2.1 | **Config** | Add `news_sentiment` section: `enabled`, `interval_seconds`, and source-specific keys (API URL/key, or feed URLs, or model name). |
| 2.2 | **Collector implementation** | Implement the chosen path inside the updater: fetch → score → `ExternalMetricEvent("news_sentiment", value)`. |
| 2.3 | **Value contract** | Standardise `value`: e.g. `{"score": float, "label": str}` or a single `float`. Document in this plan and in the updater docstring. |
| 2.4 | **Logging and alerts** | Log sentiment updates; optionally send a digest to Telegram (like Reddit sentiment) for monitoring. |

**Outcome:** Live (or replay) regime events carry a real `news_sentiment` in `metrics_json`.

---

### Phase 3 — Strategy use (pattern detection)

Use the signal in at least one strategy so that “pattern” = regime + risk flow + sentiment.

| # | Task | Description |
|---|------|-------------|
| 3.1 | **Extract helper** | In strategies that need it (e.g. overnight), add `_extract_news_sentiment(visible_regimes) -> Optional[float]` (or dict), analogous to `_extract_global_risk_flow`. |
| 3.2 | **Gating** | Add optional params, e.g. `gate_on_news_sentiment: bool`, `min_news_sentiment: float`. When gate is on, suppress entry if sentiment is below threshold (e.g. extreme fear). |
| 3.3 | **Confirmation / divergence (optional)** | Use sentiment to confirm regime (e.g. risk-on + positive sentiment) or flag divergence (e.g. risk-on + very negative sentiment). Start with logging or a simple “reduce size” rule; refine with backtests. |

**Outcome:** Strategies can gate or weight on news sentiment; pattern logic (confirmation/divergence) is in place and testable.

---

### Phase 4 — Optional: LLM or richer ML

Only if Phase 2–3 show promise:

- **LLM “agent”:** Batch of headlines → LLM → structured output (score + short summary). Use for richer “weighing” or narrative summary; validate cost and latency.
- **Backtest with historical sentiment:** If historical sentiment data exists (e.g. from API or scraped), run replay with injected `news_sentiment` and measure impact on PnL and drawdown.

---

## 4. Config Sketch

```yaml
# config/config.yaml (example)

news_sentiment:
  enabled: false
  interval_seconds: 3600
  # Path (a): API
  # api_key: "${NEWS_SENTIMENT_API_KEY}"
  # api_url: "https://..."
  # Path (b): Headlines + keyword
  # feeds: ["https://...", "https://..."]
  # lexicon: "loughran_mcdonald"  # or path to custom
  # Path (c): NLP
  # model_name: "ProsusAI/finbert"
  # max_headlines: 50
```

---

## 5. Replay and Backtests

- **Determinism:** For a given `sim_ts_ms`, replay must resolve the same `news_sentiment` when using historical sentiment (or stub). So either:
  - Replay pack precomputes and stores `news_sentiment` per regime row, and injection uses it; or
  - A replay-time helper returns a stub (e.g. 0) or looks up from a small time-series (e.g. CSV by date).
- **Sweeps:** Once gating params exist (`gate_on_news_sentiment`, `min_news_sentiment`), add them to strategy sweep configs so sensitivity can be tested.

---

## 6. Success Criteria

1. **Integration:** `news_sentiment` appears in `metrics_json` (live and replay when injected).
2. **Pattern use:** At least one strategy (e.g. overnight) can gate or weight on `news_sentiment`.
3. **Observability:** Sentiment updates are logged and optionally visible in Telegram.
4. **Backtest:** Replay runs with injected sentiment and gating; no regression; optional sweep over sentiment thresholds.

---

## 7. References

- **External metrics:** `src/core/events.py` (`ExternalMetricEvent`), `src/core/regime_detector.py` (`set_external_metric`, `_on_external_metric`).
- **Global risk flow (pattern):** `src/core/global_risk_flow_updater.py`, `src/orchestrator.py` (`_run_external_metrics_loop`), `src/strategies/overnight_session.py` (`_extract_global_risk_flow`, `gate_on_risk_flow`).
- **Replay injection:** `src/tools/replay_pack.py` (injecting into `metrics_json`).
- **Existing sentiment (display only):** `src/core/sentiment_collector.py` (Fear & Greed), `src/core/reddit_monitor.py` (Reddit keyword sentiment).
