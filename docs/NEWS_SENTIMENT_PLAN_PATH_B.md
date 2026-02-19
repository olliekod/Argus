# News Sentiment — Path (b): Headlines + Lexicon

**Goal:** Replace the Alpha Vantage NEWS_SENTIMENT implementation with a **headlines + keyword/lexicon** approach that uses **no Alpha Vantage API calls** (preserving quota for equity/FX). Optionally enhance with a lightweight sklearn model (e.g. RandomForest) if labeled data becomes available.

**Status:** Implemented (Path B complete).

## Quick Wins Status

- ✅ Added overnight sweep toggle for `gate_on_news_sentiment`.
- ✅ Added Telegram market-open news sentiment digest line.
- ✅ Added Telegram `/sentiment` command (news + Fear & Greed + Reddit).
- ✅ Marked Path B plan complete.

---

## 1. Rationale

- **Alpha Vantage quota:** Equity daily bars + FX already consume API calls. Adding NEWS_SENTIMENT would exhaust free-tier limits (typically 25/day).
- **Path (b):** Fetch headlines from free sources (RSS, NewsAPI free tier), score with a finance-specific lexicon (Loughran-McDonald), aggregate. Zero Alpha Vantage cost.
- **Optional ML:** Lexicon counts (positive, negative, uncertainty, etc.) can be used as features for a small sklearn model (LogisticRegression or RandomForest) if labeled headlines are available later. Phase 1 uses lexicon-only; ML is a future enhancement.

---

## 2. What We Keep vs Replace

| Component | Action |
|-----------|--------|
| `NewsSentimentUpdater` class | **Keep** — same interface (`async update() -> Optional[Dict]`), same `ExternalMetricEvent` and value contract |
| Value contract `{score, label, n_headlines}` | **Keep** — strategies and replay expect this |
| Orchestrator wiring | **Keep** — no changes |
| Replay injection | **Keep** — no changes |
| Strategy gating / divergence | **Keep** — no changes |
| **Fetcher** (Alpha Vantage API) | **Replace** → RSS/NewsAPI headline fetcher |
| **Scorer** (Alpha Vantage scores) | **Replace** → Lexicon-based scorer (Loughran-McDonald) |
| Config `api_key`, `api_url` | **Replace** → `feeds`, `lexicon`, optional `newsapi_key` |

---

## 3. Architecture

```
[RSS / NewsAPI] → fetch headlines
       ↓
[Lexicon Scorer] → Loughran-McDonald (positive/negative word lists)
       ↓
[Aggregate] → per-headline score → mean (optionally decay by age)
       ↓
[NewsSentimentUpdater] → ExternalMetricEvent("news_sentiment", value)
       ↓
RegimeDetector → metrics_json → Strategies
```

**Optional future:** Lexicon features (pos_count, neg_count, uncertainty_count, etc.) → sklearn RandomForest/LogReg (trained on labeled data) → score. Only if labeled headlines exist.

---

## 4. Implementation Plan

### Phase 1 — Lexicon-based (core)

#### 4.1 Headline fetcher

| # | Task | Description |
|---|------|-------------|
| 1.1 | **`HeadlineFetcher`** | New module `src/core/headline_fetcher.py` (or inline in updater). Fetches headlines from configurable sources. |
| 1.2 | **RSS feeds** | Support RSS URLs (e.g. Reuters Business, Bloomberg, Yahoo Finance). `feedparser` is standard and free. No API key. |
| 1.3 | **NewsAPI (optional)** | If `newsapi_key` in config, fetch from NewsAPI free tier (100 req/day). Query `q=stock market OR economy` or similar. Fallback to RSS if key missing. |
| 1.4 | **Contract** | `async fetch_headlines(limit: int = 50) -> List[Dict]` with `{"title": str, "summary": str, "published": datetime|str, "source": str}`. |

**Suggested RSS feeds (free, no key):**
- Yahoo Finance: `https://finance.yahoo.com/news/rssindex`
- Reuters Business: `https://www.reutersagency.com/feed/`
- MarketWatch: `https://feeds.content.dowjones.io/public/rss/mw_topstories`
- Or use `feedparser` with any finance RSS URL from config.

#### 4.2 Lexicon scorer

| # | Task | Description |
|---|------|-------------|
| 2.1 | **Loughran-McDonald** | Bundle a minimal lexicon (positive/negative word lists) or load from CSV. Source: [SRAF Notre Dame](https://sraf.nd.edu/loughranmcdonald-master-dictionary/). For research use; commercial requires permission. Alternative: use `finbert` or `vaderSentiment` if already in stack, or a small curated list. |
| 2.2 | **Scoring formula** | Per headline: `score = (pos_count - neg_count) / (pos_count + neg_count + 1)` or similar. Map to [-1, 1]. Handle empty/zero cases → 0. |
| 2.3 | **Aggregation** | Mean of per-headline scores. Optional: weight by recency (exponential decay) so older headlines count less. |

**Lexicon options:**
- **Bundled CSV:** Ship `data/loughran_negative.txt` and `data/loughran_positive.txt` (or CSV with Negative/Positive columns) in repo. Load at startup.
- **External file:** Config `lexicon_path` pointing to a local file.
- **Fallback:** Minimal in-code lists (~50 words each) if we want zero external files; less accurate but works.

#### 4.3 NewsSentimentUpdater refactor

| # | Task | Description |
|---|------|-------------|
| 3.1 | **Remove Alpha Vantage** | Delete `_extract_feed_score`, API params, `api_key` handling. |
| 3.2 | **Add fetcher + scorer** | In `update()`: call `HeadlineFetcher.fetch_headlines()` (or inline), then `LexiconScorer.score_headlines(headlines)`. |
| 3.3 | **Keep stub** | When `enabled=false` or fetch fails, emit stub `{score: 0, label: "stub", n_headlines: 0}`. |
| 3.4 | **Throttle** | Keep reasonable throttle (e.g. 60s min interval) to avoid hammering RSS/NewsAPI. |

#### 4.4 Config

```yaml
news_sentiment:
  enabled: false
  interval_seconds: 3600
  # Path (b): Headlines + lexicon (no Alpha Vantage)
  feeds:
    - "https://finance.yahoo.com/news/rssindex"
    - "https://feeds.content.dowjones.io/public/rss/mw_topstories"
  lexicon: "loughran_mcdonald"   # or path to custom CSV
  lexicon_path: null             # optional: override bundled path
  max_headlines: 50
  # Optional: NewsAPI for more sources (100 free req/day)
  newsapi_key: null              # or "${NEWSAPI_KEY}"
```

#### 4.5 Dependencies

- **feedparser** — RSS parsing. Add to `requirements.txt` if not present.
- **aiohttp** — already used. For async HTTP to RSS/NewsAPI.
- **Optional:** `requests` or `aiohttp` for NewsAPI if used.

---

### Phase 2 — Optional: Lexicon features + sklearn model

Only if labeled headlines are available (manual labels or external dataset).

| # | Task | Description |
|---|------|-------------|
| 2.1 | **Feature extraction** | Per headline: `pos_count`, `neg_count`, `uncertainty_count`, `word_count`, etc. from Loughran-McDonald. |
| 2.2 | **Training** | Train `RandomForestClassifier` or `LogisticRegression` on (features, label). Label = bullish/neutral/bearish from historical data. |
| 2.3 | **Inference** | At runtime: extract features → model.predict_proba() → map to score in [-1, 1]. |
| 2.4 | **Config** | `use_model: true`, `model_path: "data/news_sentiment_model.pkl"`. Fallback to lexicon-only if model missing. |

**Note:** This phase requires labeled data. Without it, stay with Phase 1 (lexicon-only).

---

## 5. File Changes Summary

| File | Changes |
|------|---------|
| `src/core/news_sentiment_updater.py` | Replace Alpha Vantage with headline fetcher + lexicon scorer. Keep `update()`, `_stub_payload()`, `_label_from_score()`, `close()`. |
| `src/core/headline_fetcher.py` | **New** — RSS + optional NewsAPI fetching. |
| `src/core/lexicon_scorer.py` | **New** — Load Loughran-McDonald (or custom), score text. |
| `data/loughran_positive.txt` | **New** (optional) — Bundled positive words, one per line. |
| `data/loughran_negative.txt` | **New** (optional) — Bundled negative words. |
| `config/config.yaml` | Replace `api_key`/`api_url` with `feeds`, `lexicon`, `max_headlines`, `newsapi_key`. |
| `requirements.txt` | Add `feedparser` if missing. |
| Tests | Update `test_news_sentiment_updater` to mock headline fetcher; remove Alpha Vantage mocks. |

---

## 6. Value Contract (unchanged)

```python
{
    "score": float,      # [-1.0, +1.0]
    "label": str,        # "bullish" | "neutral" | "bearish" | "stub"
    "n_headlines": int,
}
```

Strategies and replay expect this. No change.

---

## 7. Replay

- `get_news_sentiment_for_replay()` continues to return stub. No change.
- Future: if historical headline/sentiment data is stored, replay could inject real values.

---

## 8. Success Criteria

1. **Zero Alpha Vantage calls** for news sentiment.
2. **Headlines** fetched from RSS (and optionally NewsAPI).
3. **Lexicon scoring** produces score in [-1, 1], same value contract.
4. **Tests** pass with mocked fetcher; integration test with real RSS (optional, may be flaky).
5. **Config** has no `api_key` for news sentiment; `feeds` and `lexicon` drive behavior.

---

## 9. References

- Loughran-McDonald: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
- feedparser: https://feedparser.readthedocs.io/
- NewsAPI: https://newsapi.org/ (free tier 100 req/day)
- Reddit keyword pattern: `src/core/reddit_monitor.py` (BULLISH_KEYWORDS, BEARISH_KEYWORDS) — similar idea but Loughran-McDonald is finance-tuned.
