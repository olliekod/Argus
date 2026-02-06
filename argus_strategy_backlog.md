# ARGUS --- Strategy & Research Backlog

*(Saved for later implementation)*

## Purpose

This document stores **all strategy concepts, research ideas, and
trading hypotheses** to be implemented and tested **after** the core
Argus infrastructure is stable.

Argus' job is to: 1. Collect clean market data. 2. Build reliable
features. 3. Detect regimes. 4. Simulate strategies. 5. Execute or alert
with discipline.

Strategies are layered on top of this foundation.

This document prevents idea loss while keeping development focused.

------------------------------------------------------------------------

# SECTION 1 --- Primary Strategy Candidates

These are realistic, automatable strategies suitable for Argus.

------------------------------------------------------------------------

## Strategy 1 --- First 5-Minute Breakout + Fair Value Gap (FVG)

**Current top candidate.**

### Concept

-   Use first 5-minute candle of NY session.
-   Define range high/low.
-   Wait for breakout.
-   Confirm breakout via Fair Value Gap.
-   Enter continuation.
-   Fixed R:R or structural exit.

### Strengths

-   Mechanically defined.
-   Easy automation.
-   Session anchored.
-   Backtestable.

### Weaknesses

-   Fails in range days.
-   Needs regime filter.

### Implementation Order

1.  Detect FVG.
2.  Test fill probability.
3.  Add classification.
4.  Optimize entries.

------------------------------------------------------------------------

## Strategy 2 --- Session Structure (Asia → London → NY)

Used as **directional bias**, not standalone.

### Concept

-   Asia builds range.
-   London sweeps liquidity.
-   NY expands in sweep direction.

### Strengths

-   Matches institutional liquidity behavior.
-   Works across markets.

### Weaknesses

-   Sweep detection difficult.
-   Not deterministic.

### Usage

Provides: - Regime input - Bias filter - Direction weighting

------------------------------------------------------------------------

## Strategy 3 --- Day-of-Week Statistical Patterns

Statistical tendencies.

### Examples

-   Friday high \< Thursday high → Monday low likely.
-   Weekly expansion patterns.

### Usage

Not a strategy alone. Used as: - Regime modifier - Probability weighting
feature.

------------------------------------------------------------------------

## Strategy 4 --- 8:00--8:15 Futures Candle Structure

Primarily ES futures.

### Concept

Early session candle defines continuation levels.

### Status

Lower priority unless futures trading added.

------------------------------------------------------------------------

# SECTION 2 --- Fair Value Gap Refinements (Advanced)

To implement after baseline FVG testing.

------------------------------------------------------------------------

## FVG Types

### Consolidation Gap

Third candle compresses. High fill probability.

### Breakaway Gap

Third candle expands strongly. Gap often not filled.

### Rejection Gap

Third candle wicks aggressively. Low reliability.

### Future Work

Add classification:

    FVG:
        type = consolidation | breakaway | rejection

Used later for: - Position sizing - Entry filtering - Confirmation
rules.

------------------------------------------------------------------------

# SECTION 3 --- Options Strategy Focus

Core real-money goal.

------------------------------------------------------------------------

## IBIT/BITO Put Credit Spread Strategy

### Concept

Sell put spreads on ETFs during bullish or neutral regimes.

### Argus Role

Argus: - Detects regime. - Sends trade alerts. - Suggests strikes. -
Controls risk rules.

Execution currently manual.

------------------------------------------------------------------------

### Required Inputs

-   Market regime
-   Volatility regime
-   BTC/ETF correlation
-   IV levels
-   Liquidity conditions

------------------------------------------------------------------------

### Future Automation

When API available: - Auto order placement - Auto exits - Position
monitoring

------------------------------------------------------------------------

# SECTION 4 --- Advanced Alpha Modules

To mature system beyond basic signals.

------------------------------------------------------------------------

## IV Surface Tracking

Track: - Strike skew - Expiry structure - Mispricing

Enables: - Relative value options trades.

------------------------------------------------------------------------

## Correlation Engine

Monitor divergence: BTC ↔ IBIT/BITO Crypto ↔ ETFs

Used for: - Regime shifts - Arbitrage opportunities.

------------------------------------------------------------------------

## Macro Event Awareness

Calendar integration: - CPI - FOMC - Jobs data

System auto-pauses trading near events.

------------------------------------------------------------------------

## Liquidity & Slippage Engine

Estimate: - Fill quality - Slippage risk - Trade feasibility.

------------------------------------------------------------------------

## Portfolio Risk Aggregation

Track: - Net delta - Gamma - Theta - Exposure concentration

Critical before scaling capital.

------------------------------------------------------------------------

## Polymarket Integration

Use prediction markets for: - Sentiment divergence - Arbitrage setups -
Reaction signals.

------------------------------------------------------------------------

# SECTION 5 --- Strategy Evaluation Framework

Before deploying capital, every strategy must pass:

### 1. Regime Dependency Test

When does it fail?

### 2. Risk-Adjusted Returns

Sharpe, drawdown, win/loss.

### 3. Slippage Sensitivity

Realistic fills.

### 4. Robustness Test

Works across: - assets - months - volatility regimes.

------------------------------------------------------------------------

# SECTION 6 --- Implementation Order (Future)

After system stability:

1.  Basic FVG detection.
2.  Session regime model.
3.  FVG backtests.
4.  Options regime alerts.
5.  Spread trade assistant.
6.  Tape-based backtesting.
7.  Strategy sweeps.
8.  Automated execution layer.

------------------------------------------------------------------------

# SECTION 7 --- Idea Parking Lot

Add future ideas here.

Examples: - Liquidity voids - VWAP mean reversion - Funding rate
trades - Options gamma exposure models - Sentiment divergence

------------------------------------------------------------------------

# Guiding Principle

Argus should:

    Test ideas first,
    Trade only what survives testing.
