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

from typing import Dict, List, Any, Optional
from src.analysis.replay_harness import ReplayStrategy, TradeIntent
from src.core.outcome_engine import BarData, OutcomeResult
from src.core.regimes import VolRegime, TrendRegime

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
        
        # 1. Check if we have data
        if self.last_iv is None or self.last_rv is None:
            return []
            
        # 2. Compute VRP
        vrp = self.last_iv - self.last_rv
        
        # 3. Check Regimes
        symbol_regime = self.visible_regimes.get("SPY", {})
        vol = symbol_regime.get("vol_regime", "UNKNOWN")
        trend = symbol_regime.get("trend_regime", "UNKNOWN")
        
        # 4. Gating Logic
        if vrp > self._thresholds["min_vrp"]:
            if vol != "VOL_SPIKE" and trend != "TREND_DOWN":
                # Signal: Sell Put Spread
                # For Phase 5 Signal-only, we just emit the intent
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
