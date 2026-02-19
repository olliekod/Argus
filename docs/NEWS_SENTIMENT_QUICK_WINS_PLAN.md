# News Sentiment — Quick Wins Implementation Plan

**Goal:** Complete the four quick wins to round out the Path B implementation.

---

## Overview

| # | Task | Effort | Files |
|---|------|--------|-------|
| 1 | Add `gate_on_news_sentiment` to overnight sweep | Low | `config/overnight_sweep.yaml` |
| 2 | Telegram news sentiment digest (periodic) | Low–Medium | `src/core/news_sentiment_updater.py`, `src/orchestrator.py` |
| 3 | `/sentiment` command (news + Fear & Greed + Reddit) | Medium | `src/alerts/telegram_bot.py`, `src/orchestrator.py` |
| 4 | Mark NEWS_SENTIMENT_PLAN_PATH_B complete | Trivial | `docs/NEWS_SENTIMENT_PLAN_PATH_B.md` |

---

## 1. Add `gate_on_news_sentiment` to Overnight Sweep

**What:** Extend the sweep grid so experiments test news sentiment gating in addition to risk-flow gating.

**File:** `config/overnight_sweep.yaml`

**Changes:**
- Add `gate_on_news_sentiment: values: [false, true]`
- Add `min_news_sentiment: values: [-0.50, 0.0, 0.10]` (or a smaller set if grid explosion is a concern)

**Sweep explosion note:** Current grid has `gate_on_risk_flow: [false, true]`. Adding `gate_on_news_sentiment: [false, true]` and `min_news_sentiment: [-0.50, 0.0]` multiplies combinations. Options:
- **Minimal:** `gate_on_news_sentiment: [false, true]` only; keep `min_news_sentiment` as default from params (-0.50)
- **Full:** Add both; accept larger grid

**Recommended:** Add only `gate_on_news_sentiment: values: [false, true]` so the research loop can compare gated vs not; `min_news_sentiment` stays at default from `research_loop.yaml` params.

---

## 2. Telegram News Sentiment Digest (Periodic)

**What:** Send a periodic news sentiment summary to Telegram. Options:
- **(A)** Append to market open notification (once per day at 9:30 AM ET)
- **(B)** Separate daily digest at a fixed time (e.g. 10 AM ET)
- **(C)** Send when news sentiment updates (every `interval_seconds`, could be noisy)

**Recommended:** **(A)** — Add a sentiment section to the existing market open notification. Low effort, no new scheduled task, once per day.

**Implementation:**

1. **Cache last payload in NewsSentimentUpdater**
   - Add `_last_payload: Optional[Dict[str, Any]] = None`
   - In `update()`, after publishing, set `self._last_payload = payload`
   - Add `def get_last_payload(self) -> Optional[Dict[str, Any]]`

2. **Format helper for Telegram**
   - Add to `news_sentiment_updater.py` or a small util:
   ```python
   def format_news_sentiment_telegram(payload: Dict) -> str:
       """One-line or short block for Telegram."""
       if not payload or payload.get("label") == "stub":
           return "News sentiment: (stub/unavailable)"
       score, label, n = payload.get("score", 0), payload.get("label", "?"), payload.get("n_headlines", 0)
       emoji = "🟢" if label == "bullish" else "🔴" if label == "bearish" else "⚪"
       return f"{emoji} News: {label} ({score:+.2f}) | {n} headlines"
   ```

3. **Orchestrator: append to market open**
   - In `_send_market_open_notification()`, after building `lines`:
   - If `self.news_sentiment_updater` and enabled: call `get_last_payload()`, format, append to `lines`
   - Note: At 9:30 AM, the updater may not have run yet (runs on interval). Options:
     - **(i)** Trigger a one-off `await self.news_sentiment_updater.update()` before building the message (adds ~1 fetch at open)
     - **(ii)** Use cached payload from last run (might be from yesterday evening)
   - **Recommended:** **(i)** — run update at market open so the digest is fresh.

---

## 3. `/sentiment` Command (News + Fear & Greed + Reddit)

**What:** New Telegram command that returns a combined sentiment view:
- **News sentiment** (headlines + lexicon) — from NewsSentimentUpdater
- **Fear & Greed** — from SentimentCollector (alternative.me)
- **Reddit sentiment** — from RedditMonitor (if configured)

**Implementation:**

1. **NewsSentimentUpdater**
   - Ensure `get_last_payload()` exists (from task 2). If no cached payload, call `update()` once to populate.

2. **Orchestrator: callback for `/sentiment`**
   - Add `_get_sentiment: Optional[Callable]` to TelegramBot callbacks.
   - Implement `async def _get_sentiment_summary(self) -> str` in orchestrator:
     - Fetch news (from NewsSentimentUpdater cache or update)
     - Fetch Fear & Greed (instantiate SentimentCollector, call `get_sentiment()`)
     - Fetch Reddit (from self.reddit_monitor.fetch_sentiment() if configured)
     - Format into a single message (sections for each source; "N/A" if unavailable)
   - Wire `set_callbacks(get_sentiment=...)` — add param to `set_callbacks`, and add `_get_sentiment` handler.

3. **TelegramBot**
   - Add `CommandHandler("sentiment", self._cmd_sentiment)`
   - Implement `async def _cmd_sentiment(self, update, context)`:
     - Call `self._get_sentiment()` (or await if async)
     - Reply with the formatted message
   - Add `_get_sentiment: Optional[Callable]` to `set_callbacks`
   - Add `/sentiment` to HELP_TEXT

4. **Format**
   - Section 1: News (score, label, n_headlines)
   - Section 2: Fear & Greed (value, label, trend)
   - Section 3: Reddit (if available: score, top tickers; else "Not configured")

---

## 4. Mark NEWS_SENTIMENT_PLAN_PATH_B Complete

**What:** Update the plan doc to reflect implementation status.

**File:** `docs/NEWS_SENTIMENT_PLAN_PATH_B.md`

**Changes:**
- Add a **Status** section at the top: "Implemented (Path B complete). Quick wins: sweep, Telegram digest, /sentiment."
- Or add a "Quick Wins Status" subsection listing the four items and their completion.

---

## Implementation Order

1. **Task 4** (doc) — 2 min
2. **Task 1** (sweep) — 5 min
3. **Task 2** (digest) — 30–45 min (cache + format + market open hook)
4. **Task 3** (`/sentiment`) — 45–60 min (callback, handler, wiring, SentimentCollector usage)

---

## Dependencies

- **Task 2** and **Task 3** both need `get_last_payload()` on NewsSentimentUpdater — implement that first.
- **Task 3** needs `SentimentCollector` — already exists; orchestrator will instantiate on demand or hold a reference (instantiate in `_setup_conditions_and_daily_review` or similar, or lazily in the callback).

---

## Verification

| Task | How to verify |
|------|---------------|
| 1 | Run `strategy_research_loop.py` with overnight + sweep; confirm sweep includes `gate_on_news_sentiment` rows |
| 2 | Start Argus, wait for market open (or temporarily change market open check for testing); confirm Telegram gets sentiment line in open notification |
| 3 | Send `/sentiment` in Telegram; confirm response shows news, Fear & Greed, and Reddit (or N/A) |
| 4 | Read the updated plan doc |
