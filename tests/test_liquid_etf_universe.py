from datetime import datetime, timezone

from src.core.liquid_etf_universe import LIQUID_ETF_UNIVERSE, get_liquid_etf_universe
from scripts.tastytrade_health_audit import _sample_option_symbols


def test_liquid_etf_universe_ordering_alphabetical():
    universe = get_liquid_etf_universe()
    assert universe == sorted(universe)
    assert tuple(universe) == LIQUID_ETF_UNIVERSE


def test_sampling_determinism_for_new_underlying():
    contracts = []
    for expiry in ("2026-01-16", "2026-02-20", "2025-01-10"):
        for strike in range(90, 111):
            for right in ("C", "P"):
                contracts.append(
                    {
                        "underlying": "SPY",
                        "expiry": expiry,
                        "strike": float(strike),
                        "right": right,
                        "option_symbol": f".SPY{expiry.replace('-', '')}{right}{strike:08d}",
                    }
                )

    now = datetime(2025, 12, 1, tzinfo=timezone.utc)
    sample1 = _sample_option_symbols(contracts, now_utc=now)
    sample2 = _sample_option_symbols(list(reversed(contracts)), now_utc=now)

    assert [c["option_symbol"] for c in sample1] == [c["option_symbol"] for c in sample2]
    assert len(sample1) <= 80
    assert all(c["expiry"] in {"2026-01-16", "2026-02-20"} for c in sample1)
