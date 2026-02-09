from datetime import datetime, timezone

from scripts.alpaca_option_chain_snapshot import (
    compute_spread_bps,
    parse_rfc3339_to_datetime,
    percentile,
    quote_age_seconds,
)


def test_parse_rfc3339_to_datetime_handles_nanoseconds_and_z():
    dt = parse_rfc3339_to_datetime("2024-01-01T12:34:56.123456789Z")
    assert dt.tzinfo is not None
    assert dt == datetime(2024, 1, 1, 12, 34, 56, 123456, tzinfo=timezone.utc)


def test_compute_spread_bps():
    assert compute_spread_bps(10.0, 10.5) == 0.5 / 10.25 * 10000
    assert compute_spread_bps(0, 10.5) is None
    assert compute_spread_bps(10.0, 0) is None
    assert compute_spread_bps(None, 10.0) is None


def test_percentile_helper():
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 50) == 3
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 5


def test_quote_age_seconds():
    now = datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
    ts = "2024-01-01T12:00:00.000000Z"
    assert quote_age_seconds(ts, now) == 1.0
