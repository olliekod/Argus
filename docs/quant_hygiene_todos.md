# Quant hygiene TODOs

## Deflated Sharpe Ratio (DSR)
- **Why**: Classic Sharpe overstates significance when many strategies/parameter sweeps are tried; DSR adjusts for selection bias.
- **Where to add**: Simulation reporting and backtest summary output (e.g., the reporting path used by backtests and the performance summary displayed in CLI/plots).
- **Inputs needed**:
  - Number of trials/strategies evaluated (count of variants, grid searches, walk-forward runs).
  - Backtest length (number of returns/observations).
  - Mean/standard deviation of returns (existing Sharpe inputs).
  - Expected maximum Sharpe under the null (depends on trials and observation length).
- **Action**: Add a reporting helper to compute DSR alongside Sharpe and store both in the run summary artifact.

## Look-ahead contamination checklist
- **Common leaks**:
  - Using the close/high/low of a bar to decide trades within the same bar.
  - Using future realized volatility or labels when generating features.
  - Applying full-day VWAP or end-of-day indicators to intraday decisions.
  - Using future corporate actions (splits/dividends) without point-in-time adjustments.
  - Smoothing/filters that pull in data beyond the decision timestamp.
- **Where to enforce**:
  - Feature builders / indicators: ensure features only access data at or before the decision timestamp.
  - Signal generation: enforce that signals reference completed bars only.
  - Backtest runner: validate that “next bar” fills do not leak same-bar closes.
- **Action**: Add guardrails in the feature pipeline to reject look-ahead fields and add unit tests for time alignment.

## Survivorship bias (equities/options/ETFs)
- **Risk**: Backtests over current constituents ignore delisted names and skew performance.
- **Survivorship-free vs acceptable**:
  - **Survivorship-free**: historical constituents for broad universes (SP500, Russell, sector universes), plus delisting returns.
  - **Often acceptable**: single-name tickers like IBIT/SPY/QQQ/NVDA when the strategy is explicitly single-asset.
- **Data vendor/approach**:
  - Use point-in-time constituent datasets for index universes.
  - For options, prefer vendors that include expired series and delisted underlyings.
- **Action**: Document data provenance per strategy and add a checklist item in strategy reviews before production.
