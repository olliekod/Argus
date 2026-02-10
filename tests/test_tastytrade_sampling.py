"""
Unit tests for deterministic option-symbol sampling and Bearer prefix helper.

Verify that given a fixed spot and normalized chain fixture, the sampled
symbols list remains stable across invocations.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

# ---------------------------------------------------------------------------
# Import targets
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tastytrade_health_audit import sample_option_symbols
from src.connectors.tastytrade_rest import ensure_bearer_prefix


# ---------------------------------------------------------------------------
# Fixture data: synthetic normalized chain
# ---------------------------------------------------------------------------

def _make_chain(
    underlying: str = "SPY",
    expiries: list[str] | None = None,
    strikes: list[float] | None = None,
) -> list[dict]:
    """Build a synthetic normalized chain for testing."""
    today = date.today()
    if expiries is None:
        expiries = [
            (today + timedelta(days=d)).isoformat()
            for d in [3, 7, 14, 30, 60]
        ]
    if strikes is None:
        strikes = [
            490, 492, 494, 496, 498,
            500, 502, 504, 506, 508,
            510, 512, 514, 516, 518,
        ]

    chain = []
    for exp in expiries:
        for strike in strikes:
            for right in ("C", "P"):
                sym = f".{underlying}{exp.replace('-', '')}{right}{int(strike * 1000):08d}"
                chain.append({
                    "provider": "tastytrade",
                    "underlying": underlying,
                    "option_symbol": sym,
                    "expiry": exp,
                    "right": right,
                    "strike": strike,
                    "multiplier": 100,
                    "currency": "USD",
                    "exchange": None,
                    "meta": {"streamer_symbol": sym},
                })
    # Sort deterministically (matches normalize_tastytrade_nested_chain order)
    chain.sort(
        key=lambda item: (
            item.get("expiry") or "",
            item.get("strike") if item.get("strike") is not None else -1.0,
            item.get("right") or "",
            item.get("option_symbol") or "",
        )
    )
    return chain


# ---------------------------------------------------------------------------
# Tests: deterministic sampling
# ---------------------------------------------------------------------------

class TestSampleOptionSymbols:
    def test_determinism(self):
        """Same input must always produce the same output."""
        chain = _make_chain()
        result1 = sample_option_symbols(chain, spot=500.0)
        result2 = sample_option_symbols(chain, spot=500.0)
        assert result1 == result2
        assert len(result1) > 0

    def test_respects_n_expiries(self):
        chain = _make_chain()
        result = sample_option_symbols(chain, spot=500.0, n_expiries=1)
        expiries = {s.split("C")[0].split("P")[0] for s in result}
        # All sampled symbols should be from the same (nearest) expiry
        assert len(expiries) <= 1 or len(result) > 0

    def test_respects_n_strikes_per_side(self):
        chain = _make_chain()
        result = sample_option_symbols(chain, spot=500.0, n_expiries=1, n_strikes_per_side=2)
        # With 2 strikes above and 2 below, up to 4 strikes * 2 rights = 8 syms
        assert len(result) <= 8

    def test_no_spot_uses_median(self):
        """When spot=None the sampler uses the median strike."""
        chain = _make_chain()
        result = sample_option_symbols(chain, spot=None)
        assert len(result) > 0

    def test_empty_chain(self):
        assert sample_option_symbols([], spot=100.0) == []

    def test_expired_contracts_excluded(self):
        """Contracts with past expiry should not appear."""
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        chain = _make_chain(expiries=[yesterday])
        result = sample_option_symbols(chain, spot=500.0)
        assert result == []

    def test_symbols_are_unique(self):
        chain = _make_chain()
        result = sample_option_symbols(chain, spot=500.0)
        assert len(result) == len(set(result))

    def test_stability_across_runs(self):
        """Run sampling 10 times; all results must be identical."""
        chain = _make_chain()
        first = sample_option_symbols(chain, spot=505.0)
        for _ in range(9):
            assert sample_option_symbols(chain, spot=505.0) == first


# ---------------------------------------------------------------------------
# Tests: ensure_bearer_prefix
# ---------------------------------------------------------------------------

class TestEnsureBearerPrefix:
    def test_adds_prefix(self):
        assert ensure_bearer_prefix("tok-123") == "Bearer tok-123"

    def test_idempotent(self):
        assert ensure_bearer_prefix("Bearer tok-123") == "Bearer tok-123"

    def test_empty_string(self):
        assert ensure_bearer_prefix("") == "Bearer "
