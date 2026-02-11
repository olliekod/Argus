from __future__ import annotations

from unittest.mock import MagicMock

from src.core.events import BarEvent, QuoteEvent, TOPIC_REGIMES_SYMBOL
from src.core.regime_detector import RegimeDetector


BASE_TS = 1_700_000_000


def _bar(symbol: str, close: float, ts_s: int, high_off: float = 0.2, low_off: float = 0.2) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        open=close,
        high=close + high_off,
        low=close - low_off,
        close=close,
        volume=1000.0,
        timestamp=float(ts_s),
        source="test",
        bar_duration=60,
    )


def _quote(symbol: str, bid: float, ask: float, ts_s: int, recv_s: int) -> QuoteEvent:
    return QuoteEvent(
        symbol=symbol,
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2.0,
        last=(bid + ask) / 2.0,
        timestamp=float(ts_s),
        source="test",
        receive_time=float(recv_s),
    )


def _detector(thresholds=None):
    bus = MagicMock()
    bus.subscribe = MagicMock()
    published = []

    def _pub(topic, event):
        published.append((topic, event))

    bus.publish.side_effect = _pub
    d = RegimeDetector(bus=bus, thresholds=thresholds)
    return d, published


def _symbol_events(published):
    return [e for topic, e in published if topic == TOPIC_REGIMES_SYMBOL]


def test_hysteresis_prevents_flip_flop_near_boundary():
    d, _ = _detector(
        {
            "vol_hysteresis_enabled": True,
            "vol_hysteresis_band": 0.2,
            "vol_high_z": 1.0,
            "vol_spike_z": 2.5,
            "vol_low_z": -0.5,
        }
    )
    st = d._get_or_create_state("X", 60)
    st.prev_vol_regime = "VOL_HIGH"
    st.bars_since_vol_change = 99

    regime, _ = d._classify_vol_regime(0.9, st)
    assert regime == "VOL_HIGH"  # held by exit threshold 0.8


def test_min_dwell_bars_blocks_early_transition_when_enabled():
    d, _ = _detector({"min_dwell_bars": 3})
    st = d._get_or_create_state("X", 60)
    st.prev_vol_regime = "VOL_NORMAL"
    st.bars_since_vol_change = 1

    regime, _ = d._classify_vol_regime(1.5, st)
    assert regime == "VOL_NORMAL"


def test_gap_handling_can_reset_and_decay_confidence():
    d, published = _detector(
        {
            "warmup_bars": 5,
            "gap_confidence_decay_threshold_ms": 60_000,
            "gap_confidence_decay_multiplier": 0.5,
            "gap_reset_window_threshold_ms": 3 * 60 * 60 * 1000,
            "gap_warmth_decay_bars": 2,
        }
    )

    for i in range(8):
        d._on_bar(_bar("SPY", 100 + i * 0.1, BASE_TS + (i * 60)))

    # Multi-hour gap
    d._on_bar(_bar("SPY", 101.0, BASE_TS + (8 * 60) + (4 * 60 * 60)))

    st = d._get_or_create_state("SPY", 60)
    assert st.bars_processed <= 1  # reset path invoked

    last_event = _symbol_events(published)[-1]
    assert last_event.confidence < 1.0


def test_quote_liquidity_prefers_visible_snapshot_and_falls_back_when_absent():
    d, published = _detector({"quote_liquidity_enabled": True, "warmup_bars": 1})

    # quote visible at asof
    d._on_quote(_quote("SPY", bid=100.0, ask=100.2, ts_s=BASE_TS, recv_s=BASE_TS + 1))
    d._on_bar(_bar("SPY", 100.1, BASE_TS + 2, high_off=1.0, low_off=1.0))
    first = _symbol_events(published)[-1]
    expected_quote_spread = (100.2 - 100.0) / ((100.2 + 100.0) / 2.0)
    assert first.spread_pct == expected_quote_spread

    # symbol without quote snapshots falls back to bar proxy
    d._on_bar(_bar("QQQ", 200.0, BASE_TS + 3, high_off=0.5, low_off=0.5))
    second = _symbol_events(published)[-1]
    proxy = ((200.0 + 0.5) - (200.0 - 0.5)) / (((200.0 + 0.5) + (200.0 - 0.5)) / 2.0)
    assert second.spread_pct == proxy


def test_determinism_across_runs():
    thresholds = {"quote_liquidity_enabled": True, "warmup_bars": 1}
    d1, p1 = _detector(thresholds)
    d2, p2 = _detector(thresholds)

    for d in (d1, d2):
        d._on_quote(_quote("SPY", 100.0, 100.1, BASE_TS, BASE_TS + 1))
        for i in range(10):
            d._on_bar(_bar("SPY", 100 + (i * 0.2), BASE_TS + 2 + (i * 60)))

    e1 = _symbol_events(p1)
    e2 = _symbol_events(p2)
    assert len(e1) == len(e2)
    for a, b in zip(e1, e2):
        assert (a.vol_regime, a.trend_regime, a.liquidity_regime, a.spread_pct, a.confidence) == (
            b.vol_regime,
            b.trend_regime,
            b.liquidity_regime,
            b.spread_pct,
            b.confidence,
        )
