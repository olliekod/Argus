# Medium-Term Scope (Planning Only)

This document captures medium-term work items identified after Sprint 1. It is intentionally scope-only (no implementation in this sprint).

## 1) Portfolio Risk Engine

Planned capabilities:
- Exposure caps at instrument, sector/theme, and portfolio levels.
- Correlation-aware position sizing to avoid hidden concentration.
- Drawdown containment logic (dynamic de-risking after threshold breaches).
- Covariance shrinkage for more stable cross-strategy covariance estimates.
- Portfolio volatility targeting with adaptive leverage/deleverage bands.

Proposed tests:
- Unit tests for cap enforcement under conflicting constraints.
- Scenario tests where highly correlated strategies are down-weighted vs independent strategies.
- Regression tests that verify drawdown containment triggers and recovers deterministically.
- Numerical stability tests for covariance shrinkage on short/noisy histories.
- End-to-end tests validating target portfolio volatility tracking over rolling windows.

## 2) Strategy Lifecycle

Planned capabilities:
- Rolling health metrics (edge, drawdown, turnover, slippage, fill quality).
- Degradation detection based on trend breaks and confidence decay.
- Quarantine and kill-state logic with explicit reactivation criteria.
- Auditable state machine transitions (active → watchlist → quarantine → killed).

Proposed tests:
- Deterministic state-transition unit tests for all lifecycle paths.
- Replay tests for transient degradation vs persistent degradation.
- Tests for cool-down timers and re-entry gates.
- Regression tests ensuring killed strategies cannot re-enter without explicit criteria.

## 3) Live vs Backtest Drift Monitor

Planned capabilities:
- Side-by-side comparison of live fills vs backtest/simulated fills.
- Slippage drift monitoring with threshold-based alerts.
- Attribution of drift components (spread widening, latency, partial fills, missed fills).

Proposed tests:
- Mock fill stream tests with controlled slippage drift trajectories.
- Alert trigger tests around threshold boundaries and debounce windows.
- Backfill/replay tests validating stable drift metrics across reruns.

## 4) StrategyLeague Enhancements (Reference)

Reference-only enhancement directions:
- League-level capital budgeting and admission criteria.
- Diversity-aware ranking and capital assignment.
- Lifecycle-state integration for strategy eligibility.
- Improved observability for promotion/demotion decisions.
