# Phase 4C Deploy Gates & Phase 5 Prelude — Implementation Plan

This document is the **implementation plan** for the next steps after Phase 4C: (A) optional deploy gates (DSR, Reality Check, slippage sensitivity) and (B) Phase 5 prelude (strategy allocation engine with fractional Kelly sizing).

**Authority:** MASTER_PLAN.md §8.2b, §8.4, §9.3.

---

## 1. Overview and priorities

| Track | Description | Priority | Effort |
|-------|-------------|----------|--------|
| **Track A** | Deploy gates — DSR, slippage sensitivity, Reality Check | High (blocks safe deploy) | 1–2 sprints |
| **Track B** | Phase 5 prelude — allocation engine, fractional Kelly sizing | High (required before live) | 1–2 sprints |
| **Track C** | P1 audit fixes | Medium (correctness) | 0.5 sprint |

**Recommended order:** Track A first (deploy gates ensure only robust strategies get capital), then Track B (sizing layer). Track C can run in parallel.

---

## 2. Track A: Deploy gates (fancy math)

### 2.1 Deflated Sharpe Ratio (DSR)

**Source:** Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality."

DSR corrects for:
1. **Selection bias** — best of \(N\) trials inflates observed Sharpe.
2. **Non-normality** — skew and kurtosis of returns.

#### Step 1: Effective number of trials \(N\)

Trials are often correlated (e.g. overlapping param sweeps). Estimate effective \(N\) via clustering:

- **Option 1 (simple):** Use total number of experiments in evaluation (conservative).
- **Option 2 (recommended):** Cluster by return correlation; \(N\) = number of clusters (ONC algorithm or hierarchical).

#### Step 2: Threshold Sharpe \(SR_0\) (False Strategy Theorem)

The expected maximum Sharpe among \(N\) unskilled strategies is:

\[
SR_0 = \sqrt{\mathbf{V}[\widehat{SR}_n]} \cdot \left( (1-\gamma) \Phi^{-1}\!\left[1 - \frac{1}{N}\right] + \gamma \Phi^{-1}\!\left[1 - \frac{1}{Ne}\right] \right)
\]

Where:
- \(\mathbf{V}[\widehat{SR}_n]\) = cross-sectional variance of Sharpe ratios across trials
- \(\gamma \approx 0.5772\) = Euler–Mascheroni constant
- \(e \approx 2.718\) = Euler's number
- \(\Phi^{-1}\) = inverse standard normal CDF

#### Step 3: DSR formula

\[
\text{DSR} = \Phi\left( \frac{(\widehat{SR}^* - SR_0) \cdot \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 SR_0 + \frac{\hat{\gamma}_4 - 1}{4} SR_0^2}} \right)
\]

Where:
- \(\widehat{SR}^*\) = observed (non-annualized) Sharpe of best strategy
- \(SR_0\) = threshold from above
- \(T\) = number of return observations (sample length)
- \(\hat{\gamma}_3\) = skewness of returns
- \(\hat{\gamma}_4\) = kurtosis of returns
- \(\Phi\) = standard normal CDF

**Interpretation:** DSR is a probability (0–1). Deploy only if \(\text{DSR} \geq 0.95\) (or configurable threshold).

#### Implementation sketch

- **Input:** Experiment artifacts from StrategyEvaluator (or candidate set); need return series per run for skew/kurtosis.
- **Module:** `src/analysis/deflated_sharpe.py`
- **Output:** DSR value per strategy; add to evaluation manifest; new kill reason `dsr_below_threshold` if DSR < threshold.

---

### 2.2 Slippage sensitivity sweeps

**Goal:** Ensure edge survives reasonable cost inflation. Run experiments at multiple cost multipliers and fail if edge disappears.

Let \(C_0\) = baseline transaction cost (from ExecutionModel). Run replays at:
- \(C_0\) (baseline)
- \(1.25 C_0\) (+25%)
- \(1.50 C_0\) (+50%)

For each strategy/param set, compute:
- \(\widehat{SR}_0\), \(\widehat{SR}_{0.25}\), \(\widehat{SR}_{0.50}\) = Sharpe at each cost level
- \(\mu_0\), \(\mu_{0.25}\), \(\mu_{0.50}\) = mean return per trade

**Kill rule:**
\[
\text{Kill if } \widehat{SR}_{0.50} < 0 \quad \text{or} \quad \mu_{0.50} \leq 0
\]

(Edge disappears at +50% costs → fragile to execution.)

**Optional:** Linear decay model \(\widehat{SR}(c) = \widehat{SR}_0 - \beta (c - 1)\). Kill if \(\beta\) too high (edge decays fast with costs).

#### Implementation sketch

- **Location:** Extend `ExperimentRunner` or add `run_cost_sensitivity_sweep(config, cost_multipliers=[1.0, 1.25, 1.50])`.
- **ExecutionModel:** Add `cost_multiplier` to `ExecutionConfig`; scale slippage/fees accordingly.
- **Output:** Per-strategy cost-sensitivity block in artifact; evaluator kill reason `slippage_sensitivity`.

---

### 2.3 Reality Check / SPA test

**Source:** White (2000), Hansen (2005). Tests whether best strategy outperforms benchmark after correcting for data-snooping.

**Idea:** Bootstrap the null "no strategy beats benchmark" and compute \(p\)-value for best strategy.

Let:
- \(d_{k,t}\) = excess return of strategy \(k\) over benchmark at time \(t\)
- \(\bar{d}_k = \frac{1}{T}\sum_t d_{k,t}\) = mean excess return
- \(V_k = \widehat{\text{Var}}(\bar{d}_k)\) = variance of mean (HAC if needed)

**Test statistic:**
\[
V_n = \max_{k=1,\ldots,n} \frac{\sqrt{T} \bar{d}_k}{\sqrt{V_k}}
\]

**Bootstrap:** Resample blocks of \((d_{1,t},\ldots,d_{n,t})\); compute \(V_n^*\) each iteration; \(p\)-value = proportion of \(V_n^* \geq V_n\).

**Deploy gate:** Only deploy if \(p < 0.05\) (best strategy significantly beats benchmark after multiple-testing correction).

#### Implementation sketch (simplified)

- **Input:** Return series of all experiments + benchmark (e.g. buy-and-hold SPY).
- **Module:** `src/analysis/reality_check.py` or use `arch.bootstrap.reality_check` (if available).
- **Output:** \(p\)-value; kill reason `reality_check_failed` if \(p \geq 0.05\).

---

## 3. Track B: Phase 5 prelude — allocation & sizing (fancy math)

### 3.1 Forecast normalization (Layer A)

Standard object per strategy/instrument:

\[
\mathcal{F} = (\hat{\mu}, \hat{\sigma}, \text{edge\_score}, \hat{c}, \text{confidence})
\]

- \(\hat{\mu}\) = expected return (from walk-forward or replay)
- \(\hat{\sigma}\) = volatility (rolling realized or from backtest)
- \(\text{edge\_score}\) = composite score from StrategyEvaluator
- \(\hat{c}\) = estimated cost per trade
- \(\text{confidence}\) ∈ [0, 1] (e.g. from regime coverage, sample size)

### 3.2 Fractional Kelly sizing (Layer B)

**Full Kelly (single asset):**
\[
f^* = \frac{\mu - r}{\sigma^2} = \frac{\hat{\mu}}{\hat{\sigma}^2}
\]
(with risk-free rate \(r \approx 0\) for simplicity).

**Fractional Kelly (conservative):**
\[
f = c \cdot f^* = c \cdot \frac{\hat{\mu}}{\hat{\sigma}^2}, \qquad c \in [0.10, 0.50]
\]

**Quarter-Kelly** (\(c = 0.25\)) is the standard for automated deploy.

**Caps:**
\[
w = \text{clip}\left( f, -w_{\max}, w_{\max} \right)
\]

Typical \(w_{\max} \in [0.05, 0.10]\) (5–10% of equity per play).

**Skip if edge < costs:**
\[
\text{Size} = 0 \quad \text{if} \quad \hat{\mu} \leq \hat{c}
\]

### 3.3 Vol-target overlay

\[
w_{\text{vol}} = w \cdot \frac{\sigma_{\text{target}}}{\hat{\sigma}}
\]

Where \(\sigma_{\text{target}}\) = desired portfolio vol (e.g. 10% annualized).

### 3.4 Per-play cap (non-negotiable)

From MASTER_PLAN §8.5:
\[
w_{\text{play}} \leq 0.07 \quad \text{(7\% of equity per position)}
\]

### 3.5 Options spread sizing

\[
\text{contracts} = \left\lfloor \frac{\text{risk\_budget\_usd}}{\text{max\_loss\_per\_contract}} \right\rfloor
\]

\(\text{risk\_budget\_usd}\) comes from portfolio-level sizing (Kelly share × equity × cap).

### 3.6 Estimation error shrinkage (Layer C)

Shrink \(\hat{\mu}\) toward 0 by confidence:
\[
\hat{\mu}_{\text{shrunk}} = \text{confidence} \cdot \hat{\mu}
\]

Use \(\hat{\mu}_{\text{shrunk}}\) in Kelly formula.

---

## 4. Implementation tasks (checklist)

### Track A: Deploy gates

| Task | Description | Module |
|------|-------------|--------|
| A.1 | Implement `compute_deflated_sharpe_ratio(returns, n_trials, ...)` | `src/analysis/deflated_sharpe.py` |
| A.2 | Add clustering for effective \(N\) (optional; start with \(N\) = experiment count) | Same |
| A.3 | Integrate DSR into StrategyEvaluator; new kill reason `dsr_below_threshold` | `strategy_evaluator.py` |
| A.4 | Add `cost_multiplier` to ExecutionConfig; run cost-sensitivity sweep | `experiment_runner.py`, `execution_model.py` |
| A.5 | Add slippage-sensitivity kill reason | `strategy_evaluator.py` |
| A.6 | Implement Reality Check / SPA bootstrap (or integrate `arch`) | `src/analysis/reality_check.py` |
| A.7 | Wire deploy gates into research loop config (e.g. `evaluation.dsr_min`, `evaluation.slippage_sweep`) | `research_loop_config.py` |

### Track B: Phase 5 prelude

| Task | Description | Module |
|------|-------------|--------|
| B.1 | Define `Forecast` dataclass \((\hat{\mu}, \hat{\sigma}, \text{edge\_score}, \hat{c}, \text{confidence})\) | `src/analysis/sizing.py` |
| B.2 | Implement `fractional_kelly_size(forecast, c=0.25, w_max=0.07)` | Same |
| B.3 | Implement vol-target overlay | Same |
| B.4 | Implement options spread sizing (`contracts_from_risk_budget`) | Same |
| B.5 | Strategy registry (minimal): list of (strategy_id, params) from candidate set | `src/analysis/strategy_registry.py` |
| B.6 | Allocation engine: consume forecasts, output target exposures under caps | `src/analysis/allocation_engine.py` |

### Track C: P1 audit fixes

| Task | Description | File |
|------|-------------|------|
| C.1 | Alpaca UTC: ensure naive datetime → assume UTC before `.timestamp()` | `src/connectors/alpaca_client.py` |
| C.2 | Deribit: replace `.seconds` with `.total_seconds()` | `src/connectors/deribit_client.py` |
| C.3 | Options snapshot write retry (3×, backoff) | `src/core/persistence.py` |
| C.4 | Verify orchestrator task tracking; ensure `stop()` cancels all tasks | `src/orchestrator.py` |

---

## 5. Config schema additions

### Deploy gates (add to `evaluation` section)

```yaml
evaluation:
  # ... existing ...
  dsr_min: 0.95              # Require DSR >= this to avoid kill
  dsr_trials_mode: "count"   # "count" or "clustered"
  slippage_sweep: true       # Run +25%, +50% cost sweeps
  slippage_sweep_kill_if_edge_disappears: true
  reality_check_benchmark: "buy_and_hold"  # or null to skip
  reality_check_p_max: 0.05
```

### Allocation (new section for Phase 5)

```yaml
allocation:
  kelly_fraction: 0.25       # Quarter-Kelly
  per_play_cap: 0.07         # 7% max per position
  vol_target_annual: 0.10    # 10% target vol (null = no overlay)
  min_edge_over_cost: 0.0    # Skip if mu <= c
```

---

## 6. Tests

| Test | Description |
|------|-------------|
| `test_deflated_sharpe.py` | DSR formula for known N, V[SR]; deterministic; DSR increases with SR, T; decreases with N |
| `test_dsr_kill_threshold` | Strategy with low DSR gets `dsr_below_threshold` kill reason |
| `test_slippage_sensitivity` | Run sweep; verify kill when edge disappears at +50% |
| `test_fractional_kelly` | Kelly size = c·μ/σ²; caps applied; zero when μ ≤ c |
| `test_per_play_cap` | Allocation never exceeds 7% per strategy |

---

## 7. Acceptance criteria

- **DSR:** Evaluator can compute DSR for candidate strategies; kill if DSR < threshold; configurable.
- **Slippage:** Experiments can run at 1.0×, 1.25×, 1.50× costs; kill if edge disappears.
- **Reality Check:** Optional; \(p\)-value for best vs benchmark; kill if \(p \geq 0.05\).
- **Sizing:** `fractional_kelly_size()` returns correct \(w\); per-play cap enforced.
- **Allocation:** Engine consumes candidate set + forecasts, outputs target exposures; no allocation > 7% per play.

---

## 8. References

- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management*.
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*.
- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of Business & Economic Statistics*.
- MASTER_PLAN.md §9.3 — Sizing stack.
