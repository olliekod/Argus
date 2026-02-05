import pytest

from src.trading.collector_guard import CollectorModeViolation
from src.trading.paper_trader import PaperTrader, TraderConfig, StrategyType
from src.trading.paper_trader_farm import PaperTraderFarm


@pytest.mark.asyncio
async def test_paper_trader_farm_raises_in_collector_mode(monkeypatch):
    monkeypatch.setenv("ARGUS_MODE", "collector")
    farm = PaperTraderFarm()
    with pytest.raises(CollectorModeViolation):
        await farm.evaluate_signal("IBIT", {})


def test_paper_trader_enter_trade_raises_in_collector_mode(monkeypatch):
    monkeypatch.setenv("ARGUS_MODE", "collector")
    trader = PaperTrader(
        config=TraderConfig(trader_id="t1", strategy_type=StrategyType.BULL_PUT)
    )
    with pytest.raises(CollectorModeViolation):
        trader.enter_trade(
            symbol="IBIT",
            strikes="100/95",
            expiry="2025-01-17",
            entry_credit=0.5,
            contracts=1,
            market_conditions={},
        )
