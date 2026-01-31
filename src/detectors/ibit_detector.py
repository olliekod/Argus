"""
IBIT Options Opportunity Detector
=================================

Detects opportunities to sell put spreads on IBIT (BlackRock Bitcoin ETF).
Alerts when conditions are favorable for premium selling.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_detector import BaseDetector
from ..core.utils import calculate_z_score, calculate_mean, calculate_std


class IBITDetector(BaseDetector):
    """
    Detector for IBIT options opportunities.
    
    Strategy: When BTC volatility spikes and IBIT drops significantly,
    sell put spreads to collect elevated premium.
    
    This is a MANUAL trading strategy - alerts user to check Robinhood.
    """
    
    def __init__(self, config: Dict[str, Any], db):
        super().__init__(config, db)
        
        # Thresholds
        self.btc_iv_threshold = config.get('btc_iv_threshold', 70)  # Deribit BTC IV
        self.ibit_drop_threshold = config.get('ibit_drop_threshold', -3)  # % drop in 24h
        self.combined_score_threshold = config.get('combined_score_threshold', 1.5)
        
        # Alert cooldown (don't spam alerts)
        self.cooldown_hours = config.get('cooldown_hours', 4)
        self._last_alert_time: Optional[datetime] = None
        
        # Data cache
        self._btc_iv_history: List[float] = []
        self._ibit_price_history: List[Dict] = []
        self._current_btc_iv: float = 0
        self._current_ibit_data: Optional[Dict] = None
        
        self.logger.info(
            f"IBITDetector initialized: BTC IV >{self.btc_iv_threshold}%, "
            f"IBIT drop >{abs(self.ibit_drop_threshold)}%"
        )
    
    def update_btc_iv(self, iv: float) -> None:
        """Update BTC IV data from Deribit."""
        self._current_btc_iv = iv
        self._btc_iv_history.append(iv)
        
        # Keep 7 days of hourly data
        if len(self._btc_iv_history) > 168:
            self._btc_iv_history = self._btc_iv_history[-168:]
    
    def update_ibit_data(self, data: Dict) -> None:
        """Update IBIT price data from Yahoo Finance."""
        self._current_ibit_data = data
        self._ibit_price_history.append({
            'price': data.get('price', 0),
            'timestamp': data.get('timestamp'),
        })
        
        # Keep 30 days of data
        if len(self._ibit_price_history) > 720:
            self._ibit_price_history = self._ibit_price_history[-720:]
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Analyze for IBIT options opportunities.
        
        Can be called with either BTC IV data or IBIT price data.
        Checks if conditions are met for a put spread opportunity.
        
        Args:
            market_data: Either Deribit IV data or IBIT price data
            
        Returns:
            Detection if opportunity found
        """
        if not self.enabled:
            return None
        
        # Update appropriate data based on source
        if market_data.get('source') == 'deribit':
            iv = market_data.get('atm_iv', 0)
            if iv > 0:
                self.update_btc_iv(iv)
        elif market_data.get('source') == 'yahoo':
            self.update_ibit_data(market_data)
        
        # Need both data points to evaluate
        if not self._current_btc_iv or not self._current_ibit_data:
            return None
        
        # Check cooldown
        if self._last_alert_time:
            elapsed = (datetime.utcnow() - self._last_alert_time).total_seconds() / 3600
            if elapsed < self.cooldown_hours:
                return None
        
        # Evaluate conditions
        btc_iv = self._current_btc_iv
        ibit_change = self._current_ibit_data.get('price_change_pct', 0)
        ibit_price = self._current_ibit_data.get('price', 0)
        market_state = self._current_ibit_data.get('market_state', 'CLOSED')
        
        # Condition 1: BTC IV elevated
        iv_elevated = btc_iv >= self.btc_iv_threshold
        
        # Condition 2: IBIT dropped significantly
        ibit_dropped = ibit_change <= self.ibit_drop_threshold
        
        # Calculate combined score
        # Higher is better for selling puts
        iv_score = btc_iv / self.btc_iv_threshold if self.btc_iv_threshold > 0 else 0
        drop_score = abs(ibit_change) / abs(self.ibit_drop_threshold) if ibit_change < 0 else 0
        combined_score = (iv_score + drop_score) / 2
        
        # Check if we should alert
        if combined_score < self.combined_score_threshold:
            return None
        
        if not (iv_elevated or ibit_dropped):
            return None
        
        # Calculate suggested strikes
        # Typically sell puts 5-10% OTM
        suggested_short_strike = round(ibit_price * 0.92, 0)  # 8% OTM
        suggested_long_strike = round(ibit_price * 0.85, 0)   # 15% OTM (defines max loss)
        spread_width = suggested_short_strike - suggested_long_strike
        
        # Calculate BTC IV z-score if we have history
        iv_z_score = 0
        if len(self._btc_iv_history) > 10:
            iv_z_score = calculate_z_score(btc_iv, self._btc_iv_history)
        
        detection = self.create_detection(
            opportunity_type='ibit_options',
            asset='IBIT',
            exchange='robinhood',
            detection_data={
                'btc_iv': btc_iv,
                'btc_iv_z_score': iv_z_score,
                'ibit_price': ibit_price,
                'ibit_change_24h': ibit_change,
                'combined_score': combined_score,
                'market_state': market_state,
                'suggested_short_strike': suggested_short_strike,
                'suggested_long_strike': suggested_long_strike,
                'spread_width': spread_width,
            },
            current_price=ibit_price,
            estimated_edge_bps=100,  # Manual trade
            alert_tier=1,  # High priority - actionable
            notes=f"IBIT put spread opportunity - BTC IV: {btc_iv:.0f}%, IBIT: {ibit_change:+.1f}%"
        )
        
        # Record alert time
        self._last_alert_time = datetime.utcnow()
        
        await self.log_detection(detection)
        
        return detection
    
    def calculate_edge(self, detection: Dict) -> float:
        """Edge calculation for manual trades - placeholder."""
        return 100
    
    def get_current_conditions(self) -> Dict:
        """Get current market conditions for IBIT opportunity."""
        return {
            'btc_iv': self._current_btc_iv,
            'btc_iv_avg': calculate_mean(self._btc_iv_history) if self._btc_iv_history else 0,
            'ibit_price': self._current_ibit_data.get('price', 0) if self._current_ibit_data else 0,
            'ibit_change_24h': self._current_ibit_data.get('price_change_pct', 0) if self._current_ibit_data else 0,
            'market_open': self._current_ibit_data.get('market_state') == 'REGULAR' if self._current_ibit_data else False,
        }
    
    def is_market_open(self) -> bool:
        """Check if stock market is currently open."""
        if not self._current_ibit_data:
            return False
        return self._current_ibit_data.get('market_state') == 'REGULAR'
