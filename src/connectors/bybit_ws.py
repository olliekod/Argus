"""
Bybit WebSocket Connector
=========================

Public WebSocket client for Bybit perpetual futures data.
No authentication required - uses public endpoints only.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import websockets
from websockets.exceptions import ConnectionClosed

from ..core.logger import get_connector_logger

logger = get_connector_logger('bybit')


class BybitWebSocket:
    """
    Bybit public WebSocket client for perpetual futures.
    
    Provides:
    - Real-time price updates
    - Funding rate data
    - Order book depth
    
    No API key required for public data.
    """
    
    # Public WebSocket endpoints
    MAINNET_URL = "wss://stream.bybit.com/v5/public/linear"
    TESTNET_URL = "wss://stream-testnet.bybit.com/v5/public/linear"
    
    def __init__(
        self,
        symbols: List[str],
        testnet: bool = False,
        on_ticker: Optional[Callable] = None,
        on_funding: Optional[Callable] = None,
        on_orderbook: Optional[Callable] = None,
    ):
        """
        Initialize Bybit WebSocket client.
        
        Args:
            symbols: List of symbols to subscribe (e.g., ['BTCUSDT', 'ETHUSDT'])
            testnet: Use testnet endpoint
            on_ticker: Callback for ticker updates
            on_funding: Callback for funding rate updates
            on_orderbook: Callback for orderbook updates
        """
        self.url = self.TESTNET_URL if testnet else self.MAINNET_URL
        self.symbols = [self._normalize_symbol(s) for s in symbols]
        
        # Callbacks
        self.on_ticker = on_ticker
        self.on_funding = on_funding
        self.on_orderbook = on_orderbook
        
        # Connection state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_delay = 5
        self._max_reconnect_delay = 300
        self._ping_interval = 20
        
        # Latest data cache
        self.tickers: Dict[str, Dict] = {}
        self.funding_rates: Dict[str, Dict] = {}
        
        logger.info(f"Bybit WebSocket initialized for {len(self.symbols)} symbols")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Convert unified symbol to Bybit format."""
        # 'BTC/USDT:USDT' -> 'BTCUSDT'
        if ':' in symbol:
            symbol = symbol.split(':')[0]
        return symbol.replace('/', '')
    
    async def connect(self) -> None:
        """Start WebSocket connection and message loop."""
        self._running = True
        retry_count = 0
        
        while self._running:
            try:
                logger.info(f"Connecting to Bybit WebSocket: {self.url}")
                
                async with websockets.connect(
                    self.url,
                    ping_interval=self._ping_interval,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    retry_count = 0
                    self._reconnect_delay = 5
                    
                    logger.info("Bybit WebSocket connected")
                    
                    # Subscribe to channels
                    await self._subscribe()
                    
                    # Message loop
                    await self._message_loop()
                    
            except ConnectionClosed as e:
                logger.warning(f"Bybit WebSocket closed: {e.code} - {e.reason}")
            except Exception as e:
                logger.error(f"Bybit WebSocket error: {e}")
            
            if self._running:
                retry_count += 1
                delay = min(
                    self._reconnect_delay * (2 ** min(retry_count, 6)),
                    self._max_reconnect_delay
                )
                # Add jitter
                delay += (time.time() % 1)
                logger.info(f"Reconnecting in {delay:.1f}s (attempt {retry_count})")
                await asyncio.sleep(delay)
        
        self._ws = None
    
    async def disconnect(self) -> None:
        """Stop WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            logger.info("Bybit WebSocket disconnected")
    
    async def _subscribe(self) -> None:
        """Subscribe to data channels."""
        if not self._ws:
            return
        
        # Subscribe to tickers (includes funding rate info)
        ticker_topics = [f"tickers.{symbol}" for symbol in self.symbols]
        
        subscribe_msg = {
            "op": "subscribe",
            "args": ticker_topics
        }
        
        await self._ws.send(json.dumps(subscribe_msg))
        logger.debug(f"Subscribed to {len(ticker_topics)} ticker channels")
    
    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages."""
        if not self._ws:
            return
        
        async for message in self._ws:
            try:
                data = json.loads(message)
                await self._handle_message(data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {message[:100]}")
            except Exception as e:
                logger.error(f"Error handling message: {e}")
    
    async def _handle_message(self, data: Dict[str, Any]) -> None:
        """Handle a parsed WebSocket message."""
        
        # Handle subscription confirmation
        if data.get('op') == 'subscribe':
            if data.get('success'):
                logger.debug("Subscription confirmed")
            else:
                logger.warning(f"Subscription failed: {data}")
            return
        
        # Handle pong
        if data.get('op') == 'pong':
            return
        
        # Handle ticker updates
        topic = data.get('topic', '')
        
        if topic.startswith('tickers.'):
            await self._handle_ticker(data)
    
    async def _handle_ticker(self, data: Dict[str, Any]) -> None:
        """Handle ticker update message."""
        try:
            ticker_data = data.get('data', {})
            symbol = ticker_data.get('symbol', '')
            
            # Parse ticker data
            parsed = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'last_price': float(ticker_data.get('lastPrice', 0)),
                'mark_price': float(ticker_data.get('markPrice', 0)),
                'index_price': float(ticker_data.get('indexPrice', 0)),
                'bid_price': float(ticker_data.get('bid1Price', 0)),
                'ask_price': float(ticker_data.get('ask1Price', 0)),
                'volume_24h': float(ticker_data.get('volume24h', 0)),
                'turnover_24h': float(ticker_data.get('turnover24h', 0)),
                'funding_rate': float(ticker_data.get('fundingRate', 0)),
                'next_funding_time': ticker_data.get('nextFundingTime'),
                'open_interest': float(ticker_data.get('openInterest', 0)),
                'open_interest_value': float(ticker_data.get('openInterestValue', 0)),
                'price_change_24h': float(ticker_data.get('price24hPcnt', 0)) * 100,
            }
            
            # Cache latest data
            self.tickers[symbol] = parsed
            
            # Store funding rate separately
            if parsed['funding_rate'] != 0:
                self.funding_rates[symbol] = {
                    'symbol': symbol,
                    'rate': parsed['funding_rate'],
                    'next_time': parsed['next_funding_time'],
                    'timestamp': parsed['timestamp'],
                }
            
            # Trigger callbacks
            if self.on_ticker:
                try:
                    if asyncio.iscoroutinefunction(self.on_ticker):
                        await self.on_ticker(parsed)
                    else:
                        self.on_ticker(parsed)
                except Exception as e:
                    logger.error(f"Ticker callback error: {e}")
            
            if self.on_funding and parsed['funding_rate'] != 0:
                try:
                    if asyncio.iscoroutinefunction(self.on_funding):
                        await self.on_funding(self.funding_rates[symbol])
                    else:
                        self.on_funding(self.funding_rates[symbol])
                except Exception as e:
                    logger.error(f"Funding callback error: {e}")
            
        except Exception as e:
            logger.error(f"Error parsing ticker: {e}")
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get cached ticker data for a symbol."""
        normalized = self._normalize_symbol(symbol)
        return self.tickers.get(normalized)
    
    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get cached funding rate for a symbol."""
        normalized = self._normalize_symbol(symbol)
        data = self.funding_rates.get(normalized)
        return data['rate'] if data else None
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        ticker = self.get_ticker(symbol)
        return ticker['last_price'] if ticker else None
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and self._ws.open
