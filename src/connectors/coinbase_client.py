"""
Coinbase REST API Client
========================

REST client for Coinbase spot prices (US-friendly).
Replaces Binance WebSocket which blocks US IPs.
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import aiohttp

from ..core.logger import get_connector_logger

logger = get_connector_logger('coinbase')


class CoinbaseClient:
    """
    Coinbase public API client for spot prices.
    
    Uses public endpoints - no API key required for price data.
    US-friendly - no IP blocking.
    """
    
    BASE_URL = "https://api.coinbase.com/v2"
    EXCHANGE_URL = "https://api.exchange.coinbase.com"
    
    def __init__(
        self,
        symbols: List[str],
        on_ticker: Optional[Callable] = None,
    ):
        """
        Initialize Coinbase client.
        
        Args:
            symbols: List of symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
            on_ticker: Callback for ticker updates
        """
        self.symbols = [self._normalize_symbol(s) for s in symbols]
        self.on_ticker = on_ticker
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        
        # Latest data cache
        self.tickers: Dict[str, Dict] = {}
        
        logger.info(f"Coinbase client initialized for {len(self.symbols)} symbols")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Convert unified symbol to Coinbase format."""
        # 'BTC/USDT:USDT' -> 'BTC-USD'
        if ':' in symbol:
            symbol = symbol.split(':')[0]
        
        base = symbol.split('/')[0] if '/' in symbol else symbol.replace('USDT', '')
        
        # Coinbase uses USD not USDT
        return f"{base}-USD"
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get spot price for a symbol.
        
        Args:
            symbol: Coinbase symbol (e.g., 'BTC-USD')
            
        Returns:
            Ticker data or None
        """
        session = await self._get_session()
        url = f"{self.EXCHANGE_URL}/products/{symbol}/ticker"
        
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    parsed = {
                        'symbol': symbol.replace('-', ''),
                        'exchange': 'coinbase',
                        'timestamp': datetime.utcnow().isoformat(),
                        'last_price': float(data.get('price', 0)),
                        'bid_price': float(data.get('bid', 0)),
                        'ask_price': float(data.get('ask', 0)),
                        'volume_24h': float(data.get('volume', 0)),
                    }
                    
                    return parsed
                else:
                    logger.warning(f"Coinbase API error for {symbol}: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Coinbase request failed: {e}")
            return None
    
    async def get_all_tickers(self) -> Dict[str, Dict]:
        """Get tickers for all configured symbols."""
        results = {}
        
        for symbol in self.symbols:
            ticker = await self.get_ticker(symbol)
            if ticker:
                results[symbol] = ticker
                self.tickers[symbol] = ticker
        
        return results
    
    async def poll(self, interval_seconds: int = 5) -> None:
        """
        Continuously poll for price updates.
        
        Args:
            interval_seconds: Polling interval
        """
        self._running = True
        logger.info(f"Starting Coinbase price polling ({interval_seconds}s interval)")
        
        while self._running:
            try:
                for symbol in self.symbols:
                    ticker = await self.get_ticker(symbol)
                    if ticker:
                        self.tickers[symbol] = ticker
                        
                        if self.on_ticker:
                            try:
                                if asyncio.iscoroutinefunction(self.on_ticker):
                                    await self.on_ticker(ticker)
                                else:
                                    self.on_ticker(ticker)
                            except Exception as e:
                                logger.error(f"Ticker callback error: {e}")
                    
                    # Small delay between symbols to avoid rate limiting
                    await asyncio.sleep(0.2)
                    
            except Exception as e:
                logger.error(f"Coinbase polling error: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest spot price for a symbol."""
        normalized = self._normalize_symbol(symbol)
        ticker = self.tickers.get(normalized)
        return ticker['last_price'] if ticker else None
    
    @property
    def is_connected(self) -> bool:
        """Check if client is running."""
        return self._running
