"""
Liquidation Cascade Detector
============================

Detects large liquidation events for snapback trading.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from .base_detector import BaseDetector


class LiquidationDetector(BaseDetector):
    """
    Detector for post-liquidation snapback opportunities.
    
    Strategy: When large liquidation cascades occur (>$5M),
    price often overshoots then snaps back. Enter opposite 
    direction after cascade peak.
    """
    
    def __init__(self, config: Dict[str, Any], db):
        super().__init__(config, db)
        
        # Thresholds
        self.min_liquidation = config.get('min_liquidation_usd', 5_000_000)
        self.time_window = config.get('time_window_minutes', 5)
        self.entry_delay = config.get('entry_delay_seconds', 30)
        self.snapback_threshold = config.get('snapback_threshold_percent', 0.3)
        
        # Exit params
        self.take_profit = config.get('take_profit_percent', 0.5)
        self.stop_loss = config.get('stop_loss_percent', 1.0)
        self.max_hold = config.get('max_hold_minutes', 30)
        
        # Recent cascade tracking
        self._recent_cascades: Dict[str, Dict] = {}
        
        self.logger.info(
            f"LiquidationDetector initialized: min={self.min_liquidation:,.0f} USD"
        )
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Analyze liquidation data for cascade + snapback opportunities.
        
        Args:
            market_data: Liquidation cascade data from Coinglass
            
        Returns:
            Detection if opportunity found
        """
        if not self.enabled:
            return None
        
        # Check if this is a cascade detection
        if not market_data.get('cascade_detected'):
            return None
        
        symbol = market_data.get('symbol', 'BTC')
        total_liq = market_data.get('total_liquidations_usd', 0)
        long_liq = market_data.get('long_liquidations_usd', 0)
        short_liq = market_data.get('short_liquidations_usd', 0)
        dominant_side = market_data.get('dominant_side', 'unknown')
        
        # Check minimum threshold
        if total_liq < self.min_liquidation:
            return None
        
        # Check if we already detected this cascade recently
        cascade_key = f"{symbol}_{dominant_side}"
        if cascade_key in self._recent_cascades:
            last_cascade = self._recent_cascades[cascade_key]
            elapsed = (
                datetime.now(timezone.utc) - 
                datetime.fromisoformat(last_cascade['timestamp'])
            ).seconds
            
            if elapsed < self.time_window * 60 * 2:  # Within 2x time window
                return None  # Already tracked
        
        # Record cascade
        self._recent_cascades[cascade_key] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_usd': total_liq,
        }
        
        # Determine trade direction (fade the liquidation)
        # If longs got liquidated, price dropped, so we go LONG (expecting bounce)
        is_long = dominant_side == 'long'
        
        # Estimate entry price (we don't have real-time price in this data)
        estimated_edge_bps = self.snapback_threshold * 100
        
        # Alert tier based on size
        alert_tier = 1 if total_liq >= 5_000_000 else 2
        
        detection = self.create_detection(
            opportunity_type='liquidation',
            asset=symbol,
            exchange='multi',
            detection_data={
                'total_liquidations_usd': total_liq,
                'long_liquidations_usd': long_liq,
                'short_liquidations_usd': short_liq,
                'dominant_side': dominant_side,
                'trade_direction': 'LONG' if is_long else 'SHORT',
            },
            estimated_edge_bps=estimated_edge_bps,
            alert_tier=alert_tier,
            notes=f"{'LONG' if is_long else 'SHORT'} - fade {dominant_side} liquidations"
        )
        
        await self.log_detection(detection)
        
        return detection
    
    def calculate_edge(self, detection: Dict) -> float:
        """Calculate net edge after costs."""
        raw_edge = detection.get('raw_edge_bps', 30)  # Default 30 bps expected
        costs = self.slippage_bps + (self.fee_bps * 2)
        return raw_edge - costs
    
    def calculate_stops(self, entry_price: float, is_long: bool) -> Dict[str, float]:
        """Override with liquidation-specific stops."""
        if is_long:
            stop_loss = entry_price * (1 - self.stop_loss / 100)
            take_profit = entry_price * (1 + self.take_profit / 100)
        else:
            stop_loss = entry_price * (1 + self.stop_loss / 100)
            take_profit = entry_price * (1 - self.take_profit / 100)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
        }
