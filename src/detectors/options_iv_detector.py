"""
Options IV Spike Detector
=========================

Detects implied volatility spikes for manual options trading.
"""

from typing import Any, Dict, List, Optional

from .base_detector import BaseDetector
from ..core.utils import calculate_z_score, calculate_mean, calculate_std


class OptionsIVDetector(BaseDetector):
    """
    Detector for options implied volatility spikes.
    
    Strategy: When IV spikes >80% during panic, sell premium
    via put spreads or other strategies.
    
    NOTE: This is for MANUAL trading - alerts user to check Deribit.
    """
    
    def __init__(self, config: Dict[str, Any], db):
        super().__init__(config, db)
        
        # Thresholds
        self.iv_threshold = config.get('threshold_percent', 80)
        self.z_score_threshold = config.get('z_score_threshold', 2.0)
        self.lookback_days = config.get('lookback_days', 30)
        
        # Option selection
        self.min_dte = config.get('min_days_to_expiry', 3)
        self.max_dte = config.get('max_days_to_expiry', 14)
        self.otm_percent = config.get('otm_percent', 10)
        
        # IV history cache
        self._iv_history: Dict[str, List[float]] = {}
        
        self.logger.info(
            f"OptionsIVDetector initialized: IV threshold={self.iv_threshold}%"
        )
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Analyze ATM IV for spike opportunities.
        
        Args:
            market_data: IV data from Deribit
                - currency: BTC or ETH
                - atm_iv: Current ATM implied volatility
                - index_price: Current underlying price
                
        Returns:
            Detection if IV spike found
        """
        if not self.enabled:
            return None
        
        currency = market_data.get('currency', 'BTC')
        atm_iv = market_data.get('atm_iv')
        index_price = market_data.get('index_price')
        
        if atm_iv is None or atm_iv == 0:
            return None
        
        # Update IV history
        await self._update_history(currency, atm_iv)
        
        history = self._iv_history.get(currency, [])
        
        # Need some history for comparison
        if len(history) < 10:
            return None
        
        # Check absolute threshold
        if atm_iv < self.iv_threshold:
            return None
        
        # Calculate z-score
        z_score = calculate_z_score(atm_iv, history)
        
        if z_score < self.z_score_threshold:
            return None
        
        # IV spike detected!
        mean_iv = calculate_mean(history)
        std_iv = calculate_std(history)
        
        # Calculate suggested strikes
        suggested_put_strike = index_price * (1 - self.otm_percent / 100)
        suggested_put_spread_width = index_price * 0.05  # 5% width
        
        detection = self.create_detection(
            opportunity_type='options_iv',
            asset=currency,
            exchange='deribit',
            detection_data={
                'current_iv': atm_iv / 100,  # Store as decimal
                'mean_iv': mean_iv / 100,
                'std_iv': std_iv / 100,
                'z_score': z_score,
                'underlying_price': index_price,
                'suggested_put_strike': round(suggested_put_strike, -3),  # Round to 1000
                'suggested_spread_width': suggested_put_spread_width,
            },
            current_price=index_price,
            estimated_edge_bps=100,  # Arbitrary - manual trade
            alert_tier=1,  # Always tier 1 - immediate action
            notes=f"IV SPIKE: {atm_iv:.0f}% (avg: {mean_iv:.0f}%) - Check Deribit for put spreads"
        )
        
        await self.log_detection(detection)
        
        return detection
    
    async def _update_history(self, currency: str, iv: float) -> None:
        """Update IV history cache."""
        if currency not in self._iv_history:
            # Could load from database here
            self._iv_history[currency] = []
        
        self._iv_history[currency].append(iv)
        
        # Keep reasonable history (assuming ~1 update per minute)
        max_entries = self.lookback_days * 24 * 60
        if len(self._iv_history[currency]) > max_entries:
            self._iv_history[currency] = self._iv_history[currency][-max_entries:]
        
        # Store in database
        await self.db.insert_options_iv(
            asset=currency,
            expiry='ATM',  # Synthetic ATM entry
            strike=0,
            option_type='atm',
            iv=iv / 100,
        )
    
    def calculate_edge(self, detection: Dict) -> float:
        """Edge calculation not applicable for manual trades."""
        return 100  # Placeholder
    
    def get_iv_history(self, currency: str) -> List[float]:
        """Get cached IV history."""
        return self._iv_history.get(currency, [])
    
    def get_current_iv(self, currency: str) -> Optional[float]:
        """Get most recent IV value."""
        history = self._iv_history.get(currency, [])
        return history[-1] if history else None
