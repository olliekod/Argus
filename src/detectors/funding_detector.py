"""
Funding Rate Detector
=====================

Detects extreme funding rates for mean reversion strategy.
"""

from typing import Any, Dict, List, Optional

from .base_detector import BaseDetector
from ..core.utils import calculate_z_score, calculate_mean, calculate_std


class FundingDetector(BaseDetector):
    """
    Detector for funding rate mean reversion opportunities.
    
    Strategy: When funding rates spike to extremes (>2.5 std devs),
    enter opposite direction expecting reversion to mean.
    """
    
    def __init__(self, config: Dict[str, Any], db):
        super().__init__(config, db)
        
        # Thresholds
        self.threshold_percent = config.get('threshold_percent', 0.08)
        self.z_score_threshold = config.get('z_score_threshold', 2.5)
        self.lookback_periods = config.get('lookback_periods', 30)
        
        # Filters
        self.min_volume = config.get('min_volume_usd', 50_000_000)
        self.min_oi = config.get('min_open_interest', 100_000_000)
        
        # Cache for funding history
        self._funding_history: Dict[str, List[float]] = {}
        
        self.logger.info(
            f"FundingDetector initialized: threshold={self.threshold_percent:.4%}, "
            f"z_score={self.z_score_threshold}"
        )
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Analyze funding rate data for opportunities.
        
        Args:
            market_data: Dict with:
                - symbol: Asset symbol
                - funding_rate: Current funding rate
                - volume_24h: 24h volume
                - open_interest: Open interest
                - mark_price: Current mark price
                
        Returns:
            Detection dict if opportunity found
        """
        if not self.enabled:
            return None
        
        symbol = market_data.get('symbol', '')
        funding_rate = market_data.get('funding_rate', 0)
        volume = market_data.get('volume_24h', 0) or market_data.get('turnover_24h', 0)
        oi = market_data.get('open_interest_value', 0) or market_data.get('open_interest', 0)
        price = market_data.get('mark_price', 0) or market_data.get('last_price', 0)
        exchange = market_data.get('exchange', 'bybit')
        
        # Skip if funding rate is zero (hasn't loaded yet)
        if funding_rate == 0:
            return None
        
        # Update history
        await self._update_history(symbol, funding_rate, exchange)
        
        # Get historical rates
        history = self._funding_history.get(symbol, [])
        
        if len(history) < 10:
            # Not enough history yet
            return None
        
        # Apply filters
        if volume < self.min_volume:
            self.logger.debug(f"{symbol}: Volume {volume:,.0f} below threshold")
            return None
        
        if oi < self.min_oi:
            self.logger.debug(f"{symbol}: OI {oi:,.0f} below threshold")
            return None
        
        # Check absolute threshold
        if abs(funding_rate) < self.threshold_percent / 100:
            return None
        
        # Calculate z-score
        z_score = calculate_z_score(funding_rate, history)
        
        # Check z-score threshold
        if abs(z_score) < self.z_score_threshold:
            return None
        
        # We have an opportunity!
        mean_rate = calculate_mean(history)
        std_rate = calculate_std(history)
        
        # Calculate edge
        raw_edge_bps = abs(funding_rate - mean_rate) * 10000
        
        # Direction: SHORT if positive funding (longs pay shorts)
        is_long = funding_rate < 0
        
        # Calculate stops
        stops = self.calculate_stops(price, is_long)
        
        # Create detection
        detection = self.create_detection(
            opportunity_type='funding_rate',
            asset=symbol,
            exchange=exchange,
            detection_data={
                'current_funding': funding_rate,
                'mean_funding': mean_rate,
                'std_funding': std_rate,
                'z_score': z_score,
                'periods_analyzed': len(history),
                'open_interest': oi,
                'volume_24h': volume,
            },
            current_price=price,
            volume_24h=volume,
            estimated_edge_bps=raw_edge_bps,
            entry_price=price,
            stop_loss=stops['stop_loss'],
            take_profit=stops['take_profit'],
            alert_tier=2,
            notes=f"{'LONG' if is_long else 'SHORT'} signal - funding {'negative' if is_long else 'positive'}"
        )
        
        # Log to database
        await self.log_detection(detection)
        
        return detection
    
    async def _update_history(
        self,
        symbol: str,
        funding_rate: float,
        exchange: str
    ) -> None:
        """Update funding rate history cache."""
        if symbol not in self._funding_history:
            # Try to load from database
            history = await self.db.get_funding_history(symbol, self.lookback_periods)
            self._funding_history[symbol] = [h['funding_rate'] for h in history]
        
        # Add current rate
        self._funding_history[symbol].append(funding_rate)
        
        # Trim to lookback period
        if len(self._funding_history[symbol]) > self.lookback_periods:
            self._funding_history[symbol] = self._funding_history[symbol][-self.lookback_periods:]
        
        # Save to database
        await self.db.insert_funding_rate(
            exchange=exchange,
            asset=symbol,
            funding_rate=funding_rate,
        )
    
    def calculate_edge(self, detection: Dict) -> float:
        """Calculate net edge after costs."""
        raw_edge = detection.get('raw_edge_bps', 0)
        # Round trip costs (entry + exit)
        costs = self.slippage_bps + (self.fee_bps * 2)
        return raw_edge - costs
    
    def get_history(self, symbol: str) -> List[float]:
        """Get cached funding history for a symbol."""
        return self._funding_history.get(symbol, [])
