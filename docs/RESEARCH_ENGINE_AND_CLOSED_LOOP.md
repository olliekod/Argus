# Research Engine and Closed-Loop Usage

This document explains **step by step** how to use the research engine (VRP and, when available, overnight session experiments) and how Argus is designed to run as a **closed loop** that eventually funnels into execution and future phases. You verify things at the beginning; once the pipeline is trusted, it can run autonomously.

---

## 1. What the research engine does (one cycle)

A single **research cycle** is:

1. **Outcomes backfill** — Compute forward-return outcomes for the configured date range from bars already in the DB (same provider as bars).
2. **Replay pack build** — Slice bars, outcomes, regimes, and option snapshots from the DB into one or more replay pack JSON files.
3. **Experiments** — Run each configured strategy (e.g. VRP) over those packs; optional parameter sweep, MC/bootstrap, regime-stress.
4. **Evaluation** — Rank strategies, apply kill rules (DSR, Reality Check, slippage sensitivity, etc.), persist rankings, killed list, and candidate set.
5. **Allocate** — (Optional) Build forecasts from candidates, run AllocationEngine, write `allocations.json` for downstream (e.g. paper/live).

**Data dependency:** The loop **reads** bars, outcomes (after step 1), regimes, and option snapshots from the database. It **does not** ingest bars or snapshots itself. So for a closed loop you need **bars (and optionally snapshots) already being written to the DB** by some other process (e.g. orchestrator, or a separate ingest job). The loop then runs on a schedule and consumes that data.

---

## 2. One-time setup and verification (do this first)

### 2.1 Config file

1. Copy the example config and set your paths and symbols:

   ```bash
   cp config/research_loop.example.yaml config/research_loop.yaml
   ```

2. Edit `config/research_loop.yaml`:
   - **pack.mode** — `"single"` for one or a few symbols, or `"universe"` for the full liquid ETF universe.
   - **pack.symbols** — For `single`, e.g. `["SPY", "QQQ", "IBIT"]` for VRP; add/change as needed.
   - **pack date range** — Either set `start_date` / `end_date` (fixed window) or **`last_n_days: 7`** (rolling window; recommended for autonomy so each run uses the last N days).
   - **pack.db_path** — Same DB that your bar (and optionally snapshot) ingest uses, e.g. `data/argus.db`.
   - **outcomes.ensure_before_pack** — `true` so the loop backfills outcomes before building packs.
   - **outcomes.bar_duration** — Must match bar length in DB (e.g. `60` for 1m).
   - **strategies** — Start with VRP only; add more when implemented (see §4). To **automatically test all parameter variations**, set `sweep` to a YAML grid file; the loop runs every combination (no manual runs). The **GPU Heston model** is for options pricing / Probability of Profit (PoP), not for strategy parameter search — see MASTER_PLAN §1.

     ```yaml
     strategies:
       - strategy_class: "VRPCreditSpreadStrategy"
         params: {}   # base params; merged with each sweep row
         sweep: null  # or "config/vrp_sweep.yaml" to run all grid combinations automatically
     ```

   - **evaluation** — Set output paths if you want rankings/killed/candidates/allocations written:
     - `rankings_output_dir: "logs"`
     - `killed_output_path: "logs/killed.json"` (optional)
     - `candidate_set_output_path: "logs/candidates.json"` (optional)
     - **Allocation (closed loop):** Set `allocations_output_path: "logs/allocations.json"` and keep `evaluation.allocation` with `kelly_fraction`, `per_play_cap`, etc. so Step 5 runs.
   - **loop.interval_hours** — Used in daemon mode (e.g. `24`).
   - **loop.require_recent_bars_hours** — Optional. If set (e.g. `12`), the loop **skips** the cycle when the newest bar in the DB is older than that (avoids running on stale data).

3. Validate config (no execution):

   ```bash
   python scripts/strategy_research_loop.py --config config/research_loop.yaml --dry-run
   ```

   You should see “[DRY RUN] Config validated” and the list of steps that would run.

### 2.2 Ensure the DB has data for the chosen window

The loop needs **bars** (and for VRP, **option snapshots** with usable IV) in the DB for the pack date range.

- **Bars:** Come from your bar ingest (e.g. Alpaca via orchestrator or a separate script). If you don’t have bars yet, run your ingest for the symbols and dates you configured, then confirm bars exist (e.g. check bar inventory or run a small backfill for outcomes and see it succeed).
- **Outcomes:** The loop creates these in Step 1 when `ensure_before_pack: true`; no separate step needed.
- **Option snapshots (for VRP):** Must be in the DB for the same window if you want non-zero VRP trades. Use a feed that provides greeks/IV (e.g. Tastytrade); indicative-only feeds won’t yield IV. See [replay_pack_and_iv_summary.md](replay_pack_and_iv_summary.md).

**Quick data check (optional):** Run the Sprint 2 verifier for one symbol and one day that you know have bars + outcomes + snapshots with IV. It should report non-zero bars/outcomes/snapshots and ideally non-zero trade_count:

```bash
python scripts/verify_vrp_replay.py --symbol SPY --start 2026-02-11 --end 2026-02-11 --provider alpaca --db data/argus.db
```

If trade_count is 0, the script prints diagnostics (missing outcomes, no atm_iv, etc.); fix data or config accordingly.

### 2.3 Run one full cycle (one-shot)

Run a **single** cycle to confirm everything works before relying on the daemon:

```bash
python scripts/strategy_research_loop.py --config config/research_loop.yaml --once
```

- Step 1: Outcomes backfill for your pack range.
- Step 2: Packs built under `pack.packs_output_dir`.
- Step 3: Experiments run; artifacts under `experiment.output_dir` (e.g. `logs/experiments/`).
- Step 4: Rankings (and optionally killed/candidates) written.
- Step 5: If allocation is configured, `logs/allocations.json` is written.

Check logs for errors. Confirm outputs exist: e.g. `logs/strategy_rankings_*.json`, and if allocation is on, `logs/allocations.json`. Optionally open `logs/allocations.json` and confirm structure (strategy_id, instrument, weight, dollar_risk, contracts).

---

### 2.4 Why VRP shows 0 trades and "no IV available"

If you see **"skipping trade — no IV available (provider atm_iv absent and derived IV failed)"** on every replay, the packs have **option snapshots but none with usable `atm_iv`**:

- Snapshots are loaded from the DB; each snapshot can have `atm_iv` set at persist time (from the connector) or filled at pack-build time from `quotes_json` (provider IV on the ATM put, or derived from bid/ask/mid via Black–Scholes).
- If the stored snapshots have **no greeks and no usable quotes** (e.g. indicative-only, or Tastytrade response didn't include `iv` on puts when they were stored), then `atm_iv` stays null and derivation fails → VRP never sees IV → 0 trades.

So the loop and strategy are behaving correctly; the bottleneck is **option snapshot quality**, not bar count.

### 2.5 How much data do you need to run reliably?

**Bars:** You need enough bars to cover the pack window (e.g. `last_n_days` or `start_date`/`end_date`). A few days of 1m bars per symbol is plenty for a first pass.

**Option snapshots (for VRP):** You need **the same time window** as the bars, but the critical thing is **IV quality**, not length:

- **4 hours of options data is enough** to see VRP trades **if** those snapshots have IV: either `atm_iv` stored when the snapshot was written, or `quotes_json` with put `iv` or bid/ask/mid so the pack builder can derive it.
- If your 4 hours of options data are **structure-only** (no greeks, no usable quotes), you will get 0 trades no matter how many snapshots you have.

**Practical steps:** (1) Run `python scripts/options_providers_health.py --symbol SPY --hours 24 --db data/argus.db` to see if any snapshots have atm_iv. (2) Ensure new snapshots are collected with a Tastytrade path that returns greeks so `atm_iv` is stored. (3) If you only have 4 hours of IV-rich data, set the pack window to that range.

Summary: **You don't need more days of data; you need option snapshots that contain IV (or derivable quotes).** A short window with good IV is enough for VRP to produce non-zero trades.

### 2.6 Why "Loaded 0 candidates" and "Allocations: skipped" even with allocations_output_path set

Step 5 (Allocate) only runs when at least one ranking row passes the registry filters. The registry **excludes** any row where the evaluator set `killed: true` (deploy gates: DSR, reality check, MC, slippage sensitivity, etc.). With **0 trades**, many of those runs get kill reasons, so all 5 experiments can be marked killed → **0 candidates** → allocation is skipped and **no allocations file is written**, even if you set `evaluation.allocations_output_path` (e.g. `logs/allocations.json`).

So: **0 trades → often all killed → 0 candidates → allocation skipped.** Once you have IV and VRP produces non-zero trades, some runs may pass the kill filters; then the registry will load candidates and Step 5 will write the allocations file. The config key is **`evaluation.allocations_output_path`** (e.g. `logs/allocations.json`).

---

## 3. Using the research engine by hand (repeated runs)

After setup, you can run the loop whenever you want with the same config:

```bash
# One cycle
python scripts/strategy_research_loop.py --config config/research_loop.yaml --once
```

- Use **`last_n_days`** in config so each run automatically uses the latest N days of data.
- To add more symbols: edit `pack.symbols` and re-run.
- To add parameter sweeps: set `sweep: "config/vrp_sweep.yaml"` (or another grid) for a strategy and re-run.

### 2.2 Automatic variation testing (sweep) vs GPU Heston

- **“Test all variations”** — That’s what the **sweep** is for. You create a YAML file (e.g. `config/vrp_sweep.yaml`) where each key is a parameter and each value is a **list** of values to try. The research loop runs **every combination** automatically (no manual runs). Example:

  ```yaml
  # config/vrp_sweep.yaml
  min_vrp: [0.02, 0.04, 0.06, 0.08]
  max_vol_regime: ["VOL_LOW", "VOL_NORMAL"]
  ```
  Then set `sweep: "config/vrp_sweep.yaml"` for that strategy. One cycle runs 4×2 = 8 experiments, and the evaluator ranks them.

- **GPU Heston model** — Used for **options pricing and Probability of Profit (PoP)** (e.g. “given this spread and IV, what’s the chance of profit?”). It is **not** used for strategy parameter search or for “automatically testing all variations” of strategy params. See MASTER_PLAN §1.

- **Possible future:** Auto-generating a grid from a range (e.g. “min_vrp from 0.01 to 0.10 step 0.01”) is not built in today; you list the values in the sweep YAML. MASTER_PLAN §9.4 notes “automatic parameter robustness scoring, kill unstable parameter sets” as still needed.

**Documenting findings:** Keep a simple log (or doc) of what you ran (symbols, dates, params), what rankings/allocations you got, and what you changed. That’s the “document findings and feed into strategy priority and allocation design” part of the plan.

---

## 4. Overnight session experiments (when that strategy exists)

The Master Plan lists **Overnight Session Momentum / Seasonality** as a top-priority direction. **OvernightSessionStrategy** is implemented in `src/strategies/overnight_session.py`, wired in `strategy_research_loop.py` and `config/research_loop.yaml` (with `config/overnight_sweep.yaml`). **VRPCreditSpreadStrategy** and **DowRegimeTiming** are also wired.

**When an overnight/session strategy is implemented:**

1. Add its module to the strategy loader in `scripts/strategy_research_loop.py` (e.g. `_STRATEGY_MODULES`).
2. Add an entry under `strategies` in `config/research_loop.yaml`:

   ```yaml
   strategies:
     - strategy_class: "VRPCreditSpreadStrategy"
       params: {}
       sweep: null
     - strategy_class: "OvernightSessionStrategy"   # example name
       params: {}
       sweep: null  # or a sweep file
   ```

3. Run the loop as usual (`--once` or daemon). The evaluator will rank all strategies together; allocation will include the new one if it passes filters.

Until then, “overnight session experiments” means: **once that strategy is built and registered, you run it through the same research loop and config**; no separate “overnight engine” is required.

---

## 5. Making it closed-loop and autonomous

Argus is intended to be a **closed loop**: the same pipeline runs on a schedule, consumes fresh data, and produces rankings and allocations that a future execution layer can consume. You verify at the start; after that it can run without hand-holding.

### 5.1 Daemon mode (loop runs every N hours)

Run **without** `--once` so the script loops forever, re-running a full cycle every `loop.interval_hours`:

```bash
python scripts/strategy_research_loop.py --config config/research_loop.yaml
```

- Each iteration: load config (so you can change YAML and have it picked up next cycle) → run one full cycle (outcomes → packs → experiments → evaluate → allocate) → sleep `interval_hours`.
- Use **`last_n_days`** in config so every cycle uses the latest window of data.
- Optionally set **`loop.require_recent_bars_hours`** so that if the DB has no recent bars (e.g. ingest died), the cycle is **skipped** instead of running on stale data.

### 5.2 What must be true for full autonomy

1. **Bars (and optionally option snapshots) are written to the DB continuously or on a schedule** by some other process (orchestrator, cron job, etc.). The research loop does not ingest; it only reads.
2. **Config uses a rolling window** (`last_n_days`) so each cycle evaluates recent history.
3. **Allocation is enabled** (`evaluation.allocations_output_path` and `evaluation.allocation` set) so every cycle produces `allocations.json`.
4. **Stale-data guard** (`require_recent_bars_hours`) is set if you want to skip cycles when data is too old.

Then the loop is “closed” in the sense: **data in → research cycle → rankings + allocations out**, on a fixed schedule.

### 5.3 How this funnels to future phases

- **Execution (paper/live):** A future component (scheduler, orchestrator, or separate service) can **watch or poll** `logs/allocations.json` after each cycle and map `(strategy_id, instrument, weight, dollar_risk, contracts)` to orders. See [strategy_research_loop.md](strategy_research_loop.md) “Consuming allocations”.
- **Portfolio risk / StrategyLeague:** The same rankings and candidate set feed into Phase 5 (full risk engine) and StrategyLeague (capital competition, degradation, etc.) when those are implemented; the loop already produces the inputs they need.
- **Drift / lifecycle:** When live vs backtest drift and strategy lifecycle are built, they will consume the same allocations and experiment artifacts; no change to how you run the research loop.

So: **run the research loop (one-shot or daemon)** → **verify once** → **then let it run autonomously**; downstream phases will consume its outputs when they exist.

---

## 6. Minimal checklist (summary)

| Step | Action |
|------|--------|
| 1 | Copy `config/research_loop.example.yaml` → `config/research_loop.yaml`; set symbols, `last_n_days` (or dates), DB path, outcomes bar_duration, and allocation output path. |
| 2 | `python scripts/strategy_research_loop.py --config config/research_loop.yaml --dry-run` |
| 3 | Ensure DB has bars (and for VRP, option snapshots with IV) for the configured window. |
| 4 | (Optional) `python scripts/verify_vrp_replay.py ...` for one symbol/date to confirm non-zero trades when data is good. |
| 5 | `python scripts/strategy_research_loop.py --config config/research_loop.yaml --once`; check logs and outputs. |
| 6 | For autonomy: run without `--once`, use `last_n_days` and optionally `require_recent_bars_hours`; ensure bar (and snapshot) ingest is running so the DB stays fresh. |

Once this runs reliably, the research engine **is** the closed loop; future phases (execution, risk, lifecycle) will consume its outputs and you won’t need to change how you start or schedule the loop.
