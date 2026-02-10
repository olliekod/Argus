import sqlite3

from scripts.tastytrade_health_audit import _ensure_snapshot_table


def test_option_quote_snapshots_schema_creation(tmp_path):
    db_path = tmp_path / "snapshots.sqlite"
    _ensure_snapshot_table(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cols = conn.execute("PRAGMA table_info(option_quote_snapshots)").fetchall()
        col_names = [c[1] for c in cols]
    finally:
        conn.close()

    expected = [
        "id",
        "ts_utc",
        "provider",
        "underlying",
        "option_symbol",
        "expiry",
        "strike",
        "right",
        "bid",
        "ask",
        "mid",
        "event_ts",
        "recv_ts",
    ]
    assert col_names == expected
