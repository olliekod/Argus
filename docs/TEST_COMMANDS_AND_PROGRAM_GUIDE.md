# Argus: Test Commands & Program Guide

This document lists **the commands you need to test that data is pulling through correctly**, using the **modern engine and data-source policy**. Bars come from **bars_primary** (e.g. Alpaca); options snapshots from **options_snapshots_primary** (Tastytrade) and **options_snapshots_secondary** (Public when enabled). **Alpaca does not supply options data** in the current setup.

---

## Part 1: Commands to Test the Flow

Run from the project root with your venv activated (e.g. `.\venv\Scripts\Activate.ps1`). Default DB: `data/argus.db`.

---

### 1. One comprehensive verifier (recommended)

| Command | What it does | What you're testing |
|--------|----------------|---------------------|
| `python scripts/verify_argus.py --quick` | Loads config and data-source policy; checks DB exists and has expected tables. | Core setup and policy (no data checks). |
| `python scripts/verify_argus.py --data --symbol SPY --days 5` | **+** Bars coverage (bars_primary), outcomes coverage, options snapshots (primary + secondary from policy). | That bars, outcomes, and options (Tastytrade + Public when enabled) are present for the symbol/range. |
| `python scripts/verify_argus.py --replay --symbol SPY --days 5` | **+** Builds a replay pack (policy defaults) and runs VRP smoke experiment. | Full pipeline: bars → outcomes → snapshots → pack → strategy run. |
| `python scripts/verify_argus.py --full` | **+** Validates research loop config (dry-run). | That the research loop YAML is valid. |
| `python scripts/verify_argus.py --full --validate` | Same as `--full` but **exits with code 1** if any step failed (for CI/scripts). | Pass/fail gate for automation. |

**Options:** `--db path`, `--symbol SYMBOL`, `--days N`, `--start YYYY-MM-DD`, `--end YYYY-MM-DD`, `--research-config path`.

**When to use:** Run `--quick` after setup; run `--data` or `--replay` to confirm data flow; run `--full --validate` before commits or in CI.

---

### 2. Outcomes (bars + forward returns)

| Command | What it does | What you're testing |
|--------|----------------|---------------------|
| `python -m src.outcomes list` | Bar inventory and coverage %. | That bars exist for your bars_primary provider/symbols. |
| `python -m src.outcomes list-outcomes` | Outcome inventory by status. | That outcomes exist for the same provider/symbols. |
| `python -m src.outcomes coverage [--provider X] [--symbol Y]` | Coverage report for outcomes. | That outcomes are present for the dates/symbols you use in replay. |
| `python -m src.outcomes backfill --provider alpaca --symbol SPY --bar 60 --start YYYY-MM-DD --end YYYY-MM-DD` | Backfill **outcomes** for existing bars (does not fetch new bars). | Populates forward returns so replay and strategies have labels. |

**When to use:** After you have bars (from live ingestion or a bar backfill), run **backfill** so replay packs and the research loop have outcomes. Use **list** / **list-outcomes** / **coverage** to confirm.

---

### 3. Options snapshots (Tastytrade + Public, policy-driven)

| Command | What it does | What you're testing |
|--------|----------------|---------------------|
| `python scripts/options_providers_health.py [--hours 24] [--symbol SPY] [--db path]` | Per-provider report: snapshot counts, % with `atm_iv`, from **policy** primary + secondary. | That Tastytrade (and Public when enabled) are writing snapshots and IV. |
| `python scripts/verify_ingestion_debug.py [--validate]` | Validates fresh snapshots for config symbols, **policy** providers (primary + secondary), minute alignment, recv_ts now-ish. | That options ingestion is healthy (both providers when configured). |

**When to use:** Before relying on VRP or any strategy that needs IV; use **options_providers_health** to see counts and IV %, **verify_ingestion_debug --validate** for a strict pass/fail.

---

### 4. Replay and VRP smoke

| Command | What it does | What you're testing |
|--------|----------------|---------------------|
| `python -m src.tools.replay_pack --symbol SPY --start YYYY-MM-DD --end YYYY-MM-DD [--out data/packs/]` | Builds a replay pack from DB using **policy** (bars_primary + options_snapshots_primary). | That bars, outcomes, regimes, and snapshots slice correctly for a symbol/range. |
| `python scripts/verify_vrp_replay.py --symbol SPY --start YYYY-MM-DD --end YYYY-MM-DD [--provider alpaca] [--db path]` | Builds pack (provider optional; defaults to **bars_primary**), runs VRP once, prints bars/outcomes/snapshots/trades. Exits 1 with reasons if trade_count is 0. | End-to-end VRP path; when omitted, `--provider` uses config. |

**When to use:** **verify_vrp_replay** is the focused “does VRP get data and run?” check. Use **replay_pack** when you need a pack file for manual experiments.

---

### 5. Research loop

| Command | What it does | What you're testing |
|--------|----------------|---------------------|
| `python scripts/strategy_research_loop.py --config config/research_loop.yaml --dry-run` | Validates research loop config; logs steps without executing. | That `research_loop.yaml` is valid. |
| `python scripts/strategy_research_loop.py --config config/research_loop.yaml --once` | One full cycle: recent bars check, outcome backfill (if configured), pack build, experiments, evaluation, optional allocation. | That the full research pipeline runs with your data. |

**When to use:** **--dry-run** after editing the research config; **--once** to test one full cycle.

---

### 6. End-to-end (single script, full pipeline)

| Command | What it does | What you're testing |
|--------|----------------|---------------------|
| `python scripts/e2e_verify.py [--symbol SPY] [--days 5] [--db path]` | Bars → outcomes → options snapshots (primary; secondary in report) → DXLink probe → pack build → VRP smoke → evaluator. Writes to `logs/e2e/<date>/`. | Full pipeline in one command; policy-driven (bars_primary, options primary + secondary in report). |

**When to use:** When you want one command that runs the entire verification chain.

---

### 7. Unit tests

| Command | What it does | What you're testing |
|--------|----------------|---------------------|
| `python -m pytest tests -q` | Full test suite. | Core logic, replay, outcomes, IV consensus, etc. |

---

## Partial IV: solutions so you can still trade

Not every option snapshot has `atm_iv` (e.g. Tastytrade needs DXLink for IV; some minutes have no IV). You are **not** stuck: the stack is set up to use whatever IV is available.

### What’s in place

1. **Use any snapshot that has IV**  
   VRP chooses the **most recent visible snapshot that has atm_iv > 0** (by `recv_ts_ms`). If only some snapshots have IV, it still trades when one of them is visible.

2. **Derived IV from quotes**  
   When `atm_iv` is missing, IV can be derived from the snapshot’s option **bid/ask** (Brenner–Subrahmanyam style). For that to work in replay, **quotes_json** is passed through:
   - `MarketDataSnapshot` has an optional `quotes_json` field.
   - Pack load (experiment runner) and DB load (replay harness) set `quotes_json` so the strategy can derive IV when the provider didn’t supply it.

3. **Any-provider fallback in replay**  
   If no Tastytrade snapshot has IV and derivation fails, VRP falls back to **any visible snapshot with atm_iv > 0** (e.g. Public), so partial IV from another provider can still drive trades.

4. **Robust derivation**  
   Quote fields (`bid`, `ask`, `strike`, `underlying_price`) are coerced to `float` (handles JSON strings). Valid IV range for derivation is relaxed to 0.005–5.0.

### What you can do

- **Run with DXLink** so Tastytrade snapshots get `atm_iv` from the Greeks stream; more bars will have IV.
- **Use Public as secondary** (`public_options.enabled: true`); many Public snapshots have IV, and the strategy can use them when visible.
- **Confirm what’s in the pack** with the IV diagnostic in `verify_vrp_replay` (counts with `atm_iv` and with derivable quotes). If “with atm_iv > 0” is > 0, the strategy can trade on that once those snapshots are visible (recv_ts ≤ bar time).

### Replay: 0 trades but IV is present

If `verify_vrp_replay` shows **with atm_iv > 0** in the pack but **trade_count=0**:

1. **Bar–snapshot alignment**  
   Replay now keeps only bars whose sim time (bar close) can release at least one snapshot. So you no longer burn hundreds of bars with 0 visible snapshots before market open.

2. **IV is being selected**  
   When the strategy finds IV it logs once: `VRP: first IV selected … (visible_snapshots=N)`. If you see that, IV is working; 0 trades then come from **VRP or regime gating** (e.g. `min_vrp` too high, or regime not BULLISH/NEUTRAL).

3. **What to do**  
   - Relax `min_vrp` in strategy params (e.g. `strategy_params={"min_vrp": 0.02}`) or loosen regime filters.  
   - Inspect strategy diagnostics (regime, VRP value at bar time).

### Do I need to restart main.py?

**No**, for replay. Replay reads from the **pack file** (and the DB only when building the pack). Restarting `main.py` affects **live** ingestion (new bars and option snapshots). It does **not** change how an already-built pack is replayed. Restart main.py if you want more or fresher data for **future** packs; it will not fix “replay of today’s pack still shows 0 trades.”

---

## Part 2: How the Program Works Now

- **Bars:** Connectors (e.g. Alpaca, Yahoo) publish quotes; bar builder produces 1m OHLCV bars. **bars_primary** (config) is the source for bars and for outcome computation.
- **Options:** Option chain snapshots come from **options_snapshots_primary** (Tastytrade) and **options_snapshots_secondary** (Public when `public_options.enabled`). IV is from IVConsensusEngine (DXLink + snapshots). **Alpaca does not provide options** in the current config.
- **Outcomes:** OutcomeEngine computes forward returns from bars (same provider as bars_primary) and stores them; replay and strategies use these with no lookahead.
- **Replay:** Packs slice bars, outcomes, regimes, and snapshots from the DB; experiment runner replays through strategies (e.g. VRP) with a conservative execution model.
- **Research loop:** Automates outcomes → packs → experiments → evaluation → optional allocation; can run once (`--once`) or as a daemon.

---

## Part 3: What Is Left (Functionality)

- **Done:** Phases 0–5 prelude (bars, regimes, outcomes, replay, research loop, sizing, allocation), IV consolidation (Tastytrade + Public), policy-driven verification.
- **Next:** Overnight Session Strategy (Phase 1), optional Alpha Vantage + global risk flow; audit backlog; full portfolio risk engine; execution/slippage measurement; later phases (see MASTER_PLAN.md).

---

## Quick checklist: “Is data pulling through?”

1. **`python scripts/verify_argus.py --quick`** — Config and DB.
2. **`python scripts/verify_argus.py --data --symbol SPY --days 5`** — Bars, outcomes, options (Tastytrade + Public when enabled).
3. **`python scripts/verify_argus.py --replay --symbol SPY --days 5`** — Pack build + VRP smoke.
4. **`python scripts/verify_argus.py --full --validate`** — All steps including research loop config; exit 1 if any fail.

Or run **`python scripts/e2e_verify.py --symbol SPY --days 5`** for the full pipeline in one go.

---

*All verification commands use the data-source policy (config/config.yaml data_sources). Bars = bars_primary; options = options_snapshots_primary + options_snapshots_secondary when enabled. No Alpaca options.*
