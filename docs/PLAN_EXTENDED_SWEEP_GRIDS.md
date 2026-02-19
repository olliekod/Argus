# Extended Parameter Sweep Grids — Implementation Plan

Extend sweep configs for HighVol, VRP, and Overnight strategies, and clarify where a Hidden Markov Model (HMM) fits in the regime/strategy pipeline.

---

## 1. Overview

| Strategy | Current sweep | Proposed additions | Est. combinations (new) |
|----------|---------------|-------------------|-------------------------|
| **HighVol** | Not in loop | Add strategy + sweep | ~12–24 |
| **VRP** | min_vrp, max_vol_regime, avoid_trend | More regime values | +50–100% |
| **Overnight** | fwd_return, entry_window, horizon, gates | min_global_risk_flow, min_news_sentiment | +2–4× |

Sweep grid format (from `expand_sweep_grid`):
- **Explicit list:** `param: [a, b, c]` — used as-is
- **Range spec:** `param: { min: x, max: y, step: z }` or `{ min, max, num_steps: n }`
- **Scalar:** `param: value` → wrapped as `[value]`

---

## 2. Task Breakdown

### 2.1 HighVol Credit Strategy

**Files:**
- `config/high_vol_sweep.yaml` (new)
- `config/research_loop.yaml`
- `scripts/strategy_research_loop.py` (_STRATEGY_MODULES)

**Steps:**

1. **Create `config/high_vol_sweep.yaml`**
   - `min_iv`: range 0.10–0.20, step 0.02 (or values: [0.10, 0.12, 0.15, 0.18])
   - `allowed_vol_regimes`: list of lists, e.g.:
     - `[["VOL_SPIKE", "VOL_HIGH"]]` (default)
     - `[["VOL_SPIKE"]]` (spike only)
     - `[["VOL_HIGH"]]` (high only)

2. **Add HighVol to `_STRATEGY_MODULES`**
   - Module: `src.strategies.high_vol_credit`
   - Class: `HighVolCreditStrategy`

3. **Add HighVol to `config/research_loop.yaml`**
   - `strategy_class: "HighVolCreditStrategy"`
   - `params: {}` (or base defaults)
   - `sweep: "config/high_vol_sweep.yaml"`

4. **HighVol param naming**
   - Strategy expects `thresholds` dict with `min_iv`, `allowed_vol_regimes`
   - Experiment runner merges sweep rows with `params`; pass as top-level keys that strategy maps to thresholds (or ensure strategy accepts both `params` and `thresholds`).
   - Check `HighVolCreditStrategy.__init__(thresholds)` — if it expects `thresholds`, research loop must pass `params: { ... }` where keys match. Current pattern: `params` from research_loop are passed as first constructor arg. HighVol uses `thresholds` — so `params` becomes `thresholds`.

**Verification:**
- `python scripts/strategy_research_loop.py --config config/research_loop.yaml --once`
- Confirm HighVol experiments run and produce artifacts.

---

### 2.2 VRP Credit Spread Strategy

**File:** `config/vrp_sweep.yaml`

**Current:**
```yaml
min_vrp: { min: 0.02, max: 0.08, step: 0.01 }
max_vol_regime: ["VOL_LOW", "VOL_NORMAL"]
avoid_trend: ["TREND_DOWN"]
```

**Additions:**

1. **max_vol_regime** — expand options (vol regimes: VOL_LOW, VOL_NORMAL, VOL_HIGH, VOL_SPIKE)
   - Current: `["VOL_LOW", "VOL_NORMAL"]`
   - Optional: add `"VOL_HIGH"` to test VRP in elevated vol (may overlap with HighVol’s domain)

2. **avoid_trend** — sweep more trend combinations
   - Current: only `TREND_DOWN` (single value)
   - Option A: keep as single value, add `["TREND_DOWN", "RANGE"]` if strategy can avoid multiple
   - Option B: strategy may use `avoid_trend` as a list; sweep:
     - `[["TREND_DOWN"]]`
     - `[["TREND_DOWN", "TREND_UP"]]` (only trade in RANGE)
   - Check `vrp_credit_spread.py` — currently `trend != self._thresholds["avoid_trend"]` (single string). So we sweep different single values: `avoid_trend: ["TREND_DOWN", "RANGE"]` would mean "avoid RANGE" in one config — unusual. Keep as `["TREND_DOWN"]` or add a new param `allowed_trends` if we want to test "only RANGE" etc.

3. **Optional: min_vrp refinement**
   - Add `num_steps` variant for finer grid (e.g. 5 steps between 0.02–0.08) if runtime allows.

**Recommendation:** Extend `max_vol_regime` list; leave `avoid_trend` as-is unless we add `allowed_trends` to the strategy.

---

### 2.3 Overnight Session Strategy

**File:** `config/overnight_sweep.yaml`

**Current:** fwd_return_threshold, entry_window_minutes, horizon_seconds, gate_on_risk_flow, gate_on_news_sentiment

**Additions:**

1. **min_global_risk_flow** — when `gate_on_risk_flow: true`
   - Values: `[-0.01, -0.005, 0.0, 0.005]`
   - Or range: `{ min: -0.01, max: 0.01, step: 0.005 }`

2. **min_news_sentiment** — when `gate_on_news_sentiment: true`
   - Values: `[-0.5, -0.25, 0.0, 0.25]`
   - Or range: `{ min: -0.5, max: 0.5, step: 0.25 }`

3. **Combinatorics note**
   - If we add 4×4 values for these, combinations multiply. Consider:
     - Coarse first: 2 values each
     - Or conditional: only sweep min_global_risk_flow when gate_on_risk_flow=true (would require research loop support for conditional sweeps — not currently supported; simpler to sweep all and let strategy no-op when gate is false).

**Implementation:** Add keys to `overnight_sweep.yaml`; OvernightSessionStrategy already reads them from `_cfg`.

---

### 2.4 RegimeConditionalStrategy (Router) Sweep

The router selects VRP vs HighVol vs Overnight by regime. Its own params are:
- `vrp_params`, `high_vol_params`, `overnight_params`

Sweeping the router would mean sweeping nested param sets. The research loop currently runs **individual** strategies (VRP, HighVol, Overnight), not the router.

**Options:**
- **A)** Run router as a strategy with sweep over `vrp_params.min_vrp`, `high_vol_params.min_iv`, etc. — larger refactor.
- **B)** Keep running VRP, HighVol, Overnight separately; router is used in verify/replay for regime routing. Sweep each sub-strategy independently.

**Recommendation:** B for now. Router sweep can be a later phase.

---

## 3. Hidden Markov Model (HMM) Placement

From MASTER_PLAN §5.1:

> **Markov / HMM regime switching** — Classify state (low-vol grind, high-vol trend, chop); map states → allowed strategy families. **Use as one vote among features**; require confirmation from vol + microstructure; not sole driver.

### 3.1 Where HMM Fits

| Layer | Role | HMM relevance |
|-------|------|---------------|
| **Regime detection** | Vol, trend, session, liquidity | HMM as an **additional regime signal** (e.g. "HMM state 2 = chop") |
| **Strategy routing** | VRP vs HighVol vs Overnight | Router consumes regime; HMM state would be one input |
| **Position sizing** | Kelly, vol target | HMM state could modulate size (e.g. reduce in chop) |

### 3.2 Recommended Placement

1. **Regime feature layer**
   - Add HMM as a **feature** in `RegimeDetector` or a parallel `HMMRegimeClassifier`.
   - Inputs: bar returns, vol, or other features.
   - Output: discrete state (e.g. 0=low-vol, 1=high-vol, 2=chop).
   - Emit as `hmm_state` or `regime_hmm` into `metrics_json` or regime payload.

2. **Integration**
   - Regime detector merges HMM state with existing vol/trend/session.
   - Strategies and router can gate on `hmm_state` (e.g. only trade when HMM != chop).
   - Sweep: add `hmm_avoid_states: [[2], [1,2]]` to test sensitivity.

3. **Phase**
   - After BOCPD (per MASTER_PLAN: implement BOCPD first).
   - HMM is "one vote among features" — implement once BOCPD is stable.
   - Likely Phase 7 (Strategy expansion) or a "Regime methods expansion" sub-phase.

### 3.3 Dependencies

- Library: `hmmlearn` or similar.
- Features: rolling returns, vol, or derived features from `FeatureBuilder`.
- Determinism: fix random seed for HMM fitting; document in MASTER_PLAN.

---

## 4. Implementation Order

| # | Task | Effort | Blocked by |
|---|------|--------|------------|
| 1 | Create `config/high_vol_sweep.yaml` | Low | — |
| 2 | Add HighVol to `_STRATEGY_MODULES` and `research_loop.yaml` | Low | — |
| 3 | Extend `config/vrp_sweep.yaml` (max_vol_regime) | Low | — |
| 4 | Extend `config/overnight_sweep.yaml` (min_global_risk_flow, min_news_sentiment) | Low | — |
| 5 | Run research loop; document combinatorics and runtime | Low | 1–4 |
| 6 | (Future) HMM regime classifier | Medium | BOCPD, feature layer |

---

## 5. Combinatorics and Runtime

Approximate combination counts (before additions):

- VRP: ~7 × 2 × 1 = 14
- Overnight: ~7 × 3 × 3 × 2 × 2 = 252
- HighVol (new): ~5 × 3 = 15

After Overnight additions (4 × 4 for min_global_risk_flow, min_news_sentiment):
- Overnight: 252 × 16 = 4,032 (too large)

**Mitigation:** Use 2 values each for new params initially:
- `min_global_risk_flow: [-0.005, 0.0]`
- `min_news_sentiment: [-0.5, 0.0]`
- Overnight: 252 × 4 = 1,008

Or restrict other dimensions (e.g. fewer horizon_seconds) to keep total < 500 per strategy.

---

## 6. Verification Checklist

- [ ] `config/high_vol_sweep.yaml` exists and is valid YAML
- [ ] `expand_sweep_grid(yaml.safe_load(open("config/high_vol_sweep.yaml")))` succeeds
- [ ] HighVol strategy runs via research loop
- [ ] VRP sweep produces more combinations (if extended)
- [ ] Overnight sweep includes new params
- [ ] `scripts/verify_vrp_replay.py --strategy high_vol` (if wired) passes
- [ ] MASTER_PLAN §5.1 or §8 updated with HMM placement note

---

## 7. References

- MASTER_PLAN.md §5.1 (Regime methods), §8.4 (Next steps)
- config/vrp_sweep.yaml, config/overnight_sweep.yaml
- src/analysis/sweep_grid.py (expand_sweep_grid)
- src/strategies/high_vol_credit.py, vrp_credit_spread.py, overnight_session.py
