"""
VRP Credit Spread Strategy (Reference Strategy #2)
=================================================

Generates credit spread signals when Volatility Risk Premium (VRP) is high,
conditioned on market and symbol regimes.

Signal Logic:
1.  Monitor IV (from Greeks/Snapshots) and RV (Realized Volatility).
2.  Compute VRP = IV - RV.
3.  If VRP > Threshold AND Regime == BULLISH/NEUTRAL AND Volatility is STABLE:
    - Emit intent to SELL Put Spread.
"""

import logging
from typing import Dict, List, Any, Optional
from src.analysis.replay_harness import ReplayStrategy, TradeIntent
from src.core.outcome_engine import BarData, OutcomeResult
from src.core.regimes import VolRegime, TrendRegime

logger = logging.getLogger(__name__)

class VRPCreditSpreadStrategy(ReplayStrategy):
    """Ref strategy for VRP put spread selling."""
    
    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self._thresholds = thresholds or {
            "min_vrp": 0.05,       # 5 points of IV over RV
            "max_vol_regime": "VOL_NORMAL",
            "avoid_trend": "TREND_DOWN",
        }
        self.last_close = 0.0
        self.last_iv = None
        self.last_rv = None
        self._logged_no_iv = False
        self._logged_no_rv = False
        self._logged_vrp_or_regime = False

    @property
    def strategy_id(self) -> str:
        return "VRP_PUT_SPREAD_V1"

    def on_bar(
        self,
        bar: BarData,
        sim_ts_ms: int,
        session_regime: str,
        visible_outcomes: Dict[int, OutcomeResult],
        *,
        visible_regimes: Optional[Dict[str, Dict[str, Any]]] = None,
        visible_snapshots: Optional[List[Any]] = None,
    ) -> None:
        self.last_close = bar.close
        self.visible_regimes = visible_regimes or {}
        
        # Extract IV from latest snapshot
        if visible_snapshots:
            latest_snap = visible_snapshots[-1]
            if hasattr(latest_snap, 'atm_iv'):
                self.last_iv = latest_snap.atm_iv
            elif isinstance(latest_snap, dict):
                self.last_iv = latest_snap.get('atm_iv')

        # Extract realized_vol from latest outcome (if available)
        # Note: outcomes are usually trailing, so we might need to look back
        for ts in sorted(visible_outcomes.keys(), reverse=True):
            o = visible_outcomes[ts]
            if o.realized_vol is not None:
                self.last_rv = o.realized_vol
                break

    def generate_intents(self, sim_ts_ms: int) -> List[TradeIntent]:
        intents = []
        min_vrp = self._thresholds["min_vrp"]

        # 1. Check if we have data
        if self.last_iv is None:
            if not self._logged_no_iv:
                logger.warning("VRPCreditSpreadStrategy: no IV yet (need option snapshot with atm_iv). Snapshots may lack atm_iv or recv_ts_ms may be after sim time.")
                self._logged_no_iv = True
            return []
        if self.last_rv is None:
            if not self._logged_no_rv:
                logger.warning("VRPCreditSpreadStrategy: no RV yet (need visible outcome with realized_vol). Outcomes may lack window_end_ms or realized_vol.")
                self._logged_no_rv = True
            return []

        # 2. Compute VRP
        vrp = self.last_iv - self.last_rv

        # 3. Check Regimes
        symbol_regime = self.visible_regimes.get("SPY", {})
        vol = symbol_regime.get("vol_regime", "UNKNOWN")
        trend = symbol_regime.get("trend_regime", "UNKNOWN")

        # 4. Gating Logic
        if vrp <= min_vrp:
            if not self._logged_vrp_or_regime:
                logger.info("VRPCreditSpreadStrategy: VRP=%.4f <= min_vrp=%.2f (iv=%.4f rv=%.4f); no signal.", vrp, min_vrp, self.last_iv, self.last_rv)
                self._logged_vrp_or_regime = True
            return []
        if vol == "VOL_SPIKE" or trend == "TREND_DOWN":
            if not self._logged_vrp_or_regime:
                logger.info("VRPCreditSpreadStrategy: regime filter (vol=%s trend=%s); no signal.", vol, trend)
                self._logged_vrp_or_regime = True
            return []

        # Signal: Sell Put Spread
        intents.append(TradeIntent(
            symbol="SPY",
            side="SELL",
            quantity=1,
            intent_type="OPEN",
            tag="VRP_EDGE",
            meta={
                "vrp": vrp,
                "iv": self.last_iv,
                "rv": self.last_rv,
                "vol": vol,
                "trend": trend
            }
        ))
        return intents

    def finalize(self) -> Dict[str, Any]:
        return {
            "last_iv": self.last_iv,
            "last_rv": self.last_rv
        }
