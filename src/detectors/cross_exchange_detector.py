"""
Cross-Exchange Arbitrage Detector
=================================

Detects price discrepancies between exchanges.
"""

from typing import Any, Dict, List, Optional

from .base_detector import BaseDetector


class CrossExchangeDetector(BaseDetector):
    """
    Detector for cross-exchange price arbitrage.
    
    Strategy: Monitor same asset across multiple exchanges,
    detect when price differs enough to profit after costs.
    
    Note: Execution requires fast (<10s) trades on both exchanges.
    """
    
    def __init__(self, config: Dict[str, Any], db):
        super().__init__(config, db)
        
        # Thresholds
        self.threshold_percent = config.get('threshold_percent', 0.15)
        self.min_edge_bps = config.get('min_edge_after_fees_bps', 5)
        self.exchanges = config.get('exchanges', ['bybit', 'binance', 'okx'])
        
        # Filters
        self.min_liquidity = config.get('min_liquidity_depth_usd', 100_000)
        
        # Price cache by exchange
        self._prices: Dict[str, Dict[str, float]] = {}
        
        for exchange in self.exchanges:
            self._prices[exchange] = {}
        
        self.logger.info(
            f"CrossExchangeDetector initialized: threshold={self.threshold_percent}%, "
            f"exchanges={self.exchanges}"
        )
    
    def update_price(self, exchange: str, symbol: str, price: float) -> None:
        """Update price for an exchange."""
        if exchange not in self._prices:
            self._prices[exchange] = {}
        
        base_symbol = symbol.split(':')[0].replace('/', '')
        self._prices[exchange][base_symbol] = price
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Analyze for cross-exchange arbitrage.
        
        Args:
            market_data: Price update
            
        Returns:
            Detection if opportunity found
        """
        if not self.enabled:
            return None
        
        symbol = market_data.get('symbol', '')
        price = market_data.get('last_price', 0)
        exchange = market_data.get('exchange', '')
        
        if price == 0 or exchange not in self.exchanges:
            return None
        
        # Update price cache
        base_symbol = symbol.split(':')[0].replace('/', '')
        self.update_price(exchange, base_symbol, price)
        
        # Find all prices for this symbol
        prices_by_exchange = {}
        for exch in self.exchanges:
            if base_symbol in self._prices.get(exch, {}):
                prices_by_exchange[exch] = self._prices[exch][base_symbol]
        
        # Need at least 2 exchanges
        if len(prices_by_exchange) < 2:
            return None
        
        # Find max spread
        exchanges = list(prices_by_exchange.keys())
        prices = list(prices_by_exchange.values())
        
        max_price = max(prices)
        min_price = min(prices)
        high_exchange = exchanges[prices.index(max_price)]
        low_exchange = exchanges[prices.index(min_price)]
        
        spread_bps = (max_price - min_price) / min_price * 10000
        
        # Check threshold
        if spread_bps < (self.threshold_percent * 100):
            return None
        
        # Calculate edge
        net_edge = self.calculate_edge({'raw_edge_bps': spread_bps})
        
        if net_edge < self.min_edge_bps:
            return None
        
        detection = self.create_detection(
            opportunity_type='cross_exchange',
            asset=symbol,
            exchange='multi',
            detection_data={
                'buy_exchange': low_exchange,
                'buy_price': min_price,
                'sell_exchange': high_exchange,
                'sell_price': max_price,
                'spread_bps': spread_bps,
                'all_prices': prices_by_exchange,
            },
            current_price=price,
            estimated_edge_bps=spread_bps,
            alert_tier=3,  # Usually too small to act on
            notes=f"Buy {low_exchange} @ {min_price:.2f}, Sell {high_exchange} @ {max_price:.2f}"
        )
        
        detection['net_edge_bps'] = net_edge
        detection['would_trigger_entry'] = net_edge >= self.min_edge_bps
        
        # Only log significant opportunities
        if net_edge >= self.min_edge_bps * 2:
            await self.log_detection(detection)
        
        return detection if net_edge >= self.min_edge_bps else None
    
    def calculate_edge(self, detection: Dict) -> float:
        """Calculate net edge after costs."""
        raw_edge = detection.get('raw_edge_bps', 0)
        # Cross-exchange has: 2x slippage, 2x fees, plus potential transfer costs
        costs = (self.slippage_bps * 2) + (self.fee_bps * 2) + 5  # +5 bps transfer estimate
        return raw_edge - costs
    
    def get_all_spreads(self) -> Dict[str, Dict]:
        """Get current spreads for all tracked symbols."""
        spreads = {}
        
        # Get all symbols
        all_symbols = set()
        for exch_prices in self._prices.values():
            all_symbols.update(exch_prices.keys())
        
        for symbol in all_symbols:
            prices = {}
            for exch, exch_prices in self._prices.items():
                if symbol in exch_prices:
                    prices[exch] = exch_prices[symbol]
            
            if len(prices) >= 2:
                max_p = max(prices.values())
                min_p = min(prices.values())
                spread = (max_p - min_p) / min_p * 10000 if min_p > 0 else 0
                
                spreads[symbol] = {
                    'prices': prices,
                    'spread_bps': spread
                }
        
        return spreads
