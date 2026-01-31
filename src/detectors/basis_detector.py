"""
Basis Arbitrage Detector
========================

Detects spot-perp price divergence for basis arbitrage.
"""

from typing import Any, Dict, Optional

from .base_detector import BaseDetector


class BasisDetector(BaseDetector):
    """
    Detector for spot-perpetual basis arbitrage opportunities.
    
    Strategy: When spot and perp prices diverge significantly,
    profit from convergence (buy cheap, sell expensive).
    """
    
    def __init__(self, config: Dict[str, Any], db):
        super().__init__(config, db)
        
        # Thresholds
        self.threshold_percent = config.get('threshold_percent', 0.5)
        self.min_edge_bps = config.get('min_edge_after_fees_bps', 10)
        
        # Filters
        self.min_volume_spot = config.get('min_volume_spot', 10_000_000)
        self.min_volume_perp = config.get('min_volume_perp', 50_000_000)
        
        # Price cache
        self._spot_prices: Dict[str, float] = {}
        self._perp_prices: Dict[str, float] = {}
        
        self.logger.info(
            f"BasisDetector initialized: threshold={self.threshold_percent}%"
        )
    
    def update_spot_price(self, symbol: str, price: float) -> None:
        """Update cached spot price."""
        # Normalize symbol (remove :USDT suffix if present)
        base_symbol = symbol.split(':')[0].replace('/', '')
        self._spot_prices[base_symbol] = price
    
    def update_perp_price(self, symbol: str, price: float) -> None:
        """Update cached perp price."""
        base_symbol = symbol.split(':')[0].replace('/', '')
        self._perp_prices[base_symbol] = price
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Analyze for basis arbitrage opportunities.
        
        Can be called with either spot or perp data - will check
        for opportunities when both prices are available.
        
        Args:
            market_data: Price update data
            
        Returns:
            Detection dict if opportunity found
        """
        if not self.enabled:
            return None
        
        symbol = market_data.get('symbol', '')
        price = market_data.get('last_price', 0)
        exchange = market_data.get('exchange', '')
        volume = market_data.get('volume_24h', 0) or market_data.get('quote_volume_24h', 0)
        
        if price == 0:
            return None
        
        # Determine if this is spot or perp data
        is_perp = ':' in symbol or exchange == 'bybit'
        
        # Normalize symbol
        base_symbol = symbol.split(':')[0].replace('/', '')
        
        # Update appropriate price cache
        if is_perp:
            self._perp_prices[base_symbol] = price
        else:
            self._spot_prices[base_symbol] = price
        
        # Check if we have both prices
        spot_price = self._spot_prices.get(base_symbol)
        perp_price = self._perp_prices.get(base_symbol)
        
        if not spot_price or not perp_price:
            return None
        
        # Calculate basis
        basis = (perp_price - spot_price) / spot_price
        basis_bps = basis * 10000
        
        # Check threshold
        if abs(basis_bps) < (self.threshold_percent * 100):
            return None
        
        # Calculate edge
        raw_edge_bps = abs(basis_bps)
        net_edge_bps = self.calculate_edge({'raw_edge_bps': raw_edge_bps})
        
        # Check minimum edge
        if net_edge_bps < self.min_edge_bps:
            self.logger.debug(
                f"{base_symbol}: Basis {basis_bps:.1f} bps but net edge "
                f"{net_edge_bps:.1f} bps below threshold"
            )
            return None
        
        # Determine direction
        # If perp > spot: sell perp, buy spot (then close when converged)
        is_long_spot = perp_price > spot_price
        
        detection = self.create_detection(
            opportunity_type='basis',
            asset=symbol,
            exchange='multi',  # Arbitrage across exchanges
            detection_data={
                'spot_price': spot_price,
                'perp_price': perp_price,
                'basis_bps': basis_bps,
                'spot_exchange': 'binance',
                'perp_exchange': 'bybit',
            },
            current_price=perp_price,
            volume_24h=volume,
            estimated_edge_bps=raw_edge_bps,
            entry_price=perp_price,
            alert_tier=2,
            notes=f"{'Buy spot, sell perp' if is_long_spot else 'Sell spot, buy perp'}"
        )
        
        # Override would_trigger based on net edge
        detection['would_trigger_entry'] = net_edge_bps >= self.min_edge_bps
        detection['net_edge_bps'] = net_edge_bps
        
        if detection['would_trigger_entry']:
            await self.log_detection(detection)
            return detection
        
        return None
    
    def calculate_edge(self, detection: Dict) -> float:
        """Calculate net edge after costs."""
        raw_edge = detection.get('raw_edge_bps', 0)
        # Basis arb has costs on both legs
        costs = (self.slippage_bps * 2) + (self.fee_bps * 4)  # Entry/exit on both
        return raw_edge - costs
    
    def get_current_basis(self, symbol: str) -> Optional[float]:
        """Get current basis for a symbol in bps."""
        base_symbol = symbol.split(':')[0].replace('/', '')
        spot = self._spot_prices.get(base_symbol)
        perp = self._perp_prices.get(base_symbol)
        
        if spot and perp:
            return (perp - spot) / spot * 10000
        return None
