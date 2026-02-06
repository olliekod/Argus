import asyncio
import tempfile
import threading
import time

from src.core.bus import EventBus
from src.core.events import BarEvent
from src.core.persistence import PersistenceManager


class _DummyDB:
    async def execute_many(self, sql: str, params_list):
        await asyncio.sleep(0)

    async def execute(self, sql: str, params=()):
        await asyncio.sleep(0)

    async def insert_market_metric(self, **kwargs):
        await asyncio.sleep(0)


def _start_loop(loop: asyncio.AbstractEventLoop) -> threading.Thread:
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return thread


def test_persistence_flush_timing_updates_status():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)

    try:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as td:
            pm = PersistenceManager(bus, _DummyDB(), loop, spool_dir=td)
            bar = BarEvent(
                symbol="BTC",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=time.time(),
                source="test",
                bar_duration=60,
                tick_count=1,
            )
            pm._on_bar(bar)
            pm._do_flush()
            status = pm.get_status()

            assert status["timing"]["last_latency_ms"] is not None
            assert status["extras"]["bars_writes_total"] == 1
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


def test_persist_lag_ms_uses_sane_source_ts():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)

    try:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as td:
            pm = PersistenceManager(bus, _DummyDB(), loop, spool_dir=td)
            now = time.time()
            bar = BarEvent(
                symbol="BTC",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=now,
                source="test",
                bar_duration=60,
                tick_count=1,
                source_ts=now - 0.2,
            )
            pm._on_bar(bar)
            pm._do_flush()
            status = pm.get_status()

            lag_ms = status["extras"]["persist_lag_ms"]
            assert lag_ms is not None
            assert lag_ms < 2_000
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


def test_persist_lag_ignores_ms_source_ts():
    loop = asyncio.new_event_loop()
    thread = _start_loop(loop)

    try:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as td:
            pm = PersistenceManager(bus, _DummyDB(), loop, spool_dir=td)
            now = time.time()
            bar = BarEvent(
                symbol="BTC",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                timestamp=now,
                source="test",
                bar_duration=60,
                tick_count=1,
                source_ts=now * 1000.0,
            )
            pm._on_bar(bar)
            pm._do_flush()
            status = pm.get_status()

            assert status["extras"]["persist_lag_ms"] is None
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
