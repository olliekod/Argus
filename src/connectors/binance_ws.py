"""
Binance WebSocket Connector
===========================

WebSocket client for Binance spot market data.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import websockets
from websockets.exceptions import ConnectionClosed

from ..core.logger import get_connector_logger

logger = get_connector_logger('binance')


class BinanceWebSocket:
    """
    Binance WebSocket client for spot market data.
    
    Provides:
    - Real-time spot prices
    - 24h ticker statistics
    
    Uses public endpoints (no auth needed for market data).
    """
    
    BASE_URL = "wss://stream.binance.com:9443/ws"
    
    def __init__(
        self,
        symbols: List[str],
        on_ticker: Optional[Callable] = None,
    ):
        """
        Initialize Binance WebSocket client.
        
        Args:
            symbols: List of symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
            on_ticker: Callback for ticker updates
        """
        self.symbols = [self._normalize_symbol(s) for s in symbols]
        
        # Callbacks
        self.on_ticker = on_ticker
        
        # Connection state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_delay = 5
        
        # Latest data cache
        self.tickers: Dict[str, Dict] = {}
        
        logger.info(f"Binance WebSocket initialized for {len(self.symbols)} symbols")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Convert unified symbol to Binance format."""
        # 'BTC/USDT:USDT' -> 'btcusdt' (lowercase for stream names)
        if ':' in symbol:
            symbol = symbol.split(':')[0]
        return symbol.replace('/', '').lower()
    
    def _get_stream_url(self) -> str:
        """Build combined stream URL for all symbols."""
        streams = [f"{s}@ticker" for s in self.symbols]
        return f"{self.BASE_URL}/{'/'.join(streams)}"
    
    async def connect(self) -> None:
        """Start WebSocket connection and message loop."""
        self._running = True
        retry_count = 0
        
        while self._running:
            try:
                # Build combined streams URL
                url = self._get_stream_url()
                logger.info(f"Connecting to Binance WebSocket")
                
                async with websockets.connect(
                    url,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    retry_count = 0
                    
                    logger.info("Binance WebSocket connected")
                    
                    # Message loop
                    await self._message_loop()
                    
            except ConnectionClosed as e:
                logger.warning(f"Binance WebSocket closed: {e.code}")
            except Exception as e:
                logger.error(f"Binance WebSocket error: {e}")
            
            if self._running:
                retry_count += 1
                delay = min(self._reconnect_delay * (2 ** min(retry_count, 5)), 300)
                delay += (time.time() % 1)
                logger.info(f"Reconnecting in {delay:.1f}s")
                await asyncio.sleep(delay)
        
        self._ws = None
    
    async def disconnect(self) -> None:
        """Stop WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            logger.info("Binance WebSocket disconnected")
    
    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages."""
        if not self._ws:
            return
        
        async for message in self._ws:
            try:
                data = json.loads(message)
                await self._handle_message(data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON: {message[:100]}")
            except Exception as e:
                logger.error(f"Error handling message: {e}")
    
    async def _handle_message(self, data: Dict[str, Any]) -> None:
        """Handle a parsed WebSocket message."""
        
        # 24h ticker format
        if data.get('e') == '24hrTicker':
            await self._handle_ticker(data)
    
    async def _handle_ticker(self, data: Dict[str, Any]) -> None:
        """Handle 24hr ticker message."""
        try:
            symbol = data.get('s', '').upper()
            
            parsed = {
                'symbol': symbol,
                'exchange': 'binance',
                'timestamp': datetime.utcnow().isoformat(),
                'last_price': float(data.get('c', 0)),
                'bid_price': float(data.get('b', 0)),
                'ask_price': float(data.get('a', 0)),
                'volume_24h': float(data.get('v', 0)),
                'quote_volume_24h': float(data.get('q', 0)),
                'price_change_24h': float(data.get('P', 0)),
                'high_24h': float(data.get('h', 0)),
                'low_24h': float(data.get('l', 0)),
                'open_24h': float(data.get('o', 0)),
            }
            
            # Cache
            self.tickers[symbol] = parsed
            
            # Callback
            if self.on_ticker:
                try:
                    if asyncio.iscoroutinefunction(self.on_ticker):
                        await self.on_ticker(parsed)
                    else:
                        self.on_ticker(parsed)
                except Exception as e:
                    logger.error(f"Ticker callback error: {e}")
            
        except Exception as e:
            logger.error(f"Error parsing ticker: {e}")
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get cached ticker data for a symbol."""
        # Convert to Binance format (uppercase)
        if ':' in symbol:
            symbol = symbol.split(':')[0]
        normalized = symbol.replace('/', '').upper()
        return self.tickers.get(normalized)
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest spot price for a symbol."""
        ticker = self.get_ticker(symbol)
        return ticker['last_price'] if ticker else None
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and self._ws.open
