"""
OKX REST API Client
===================

REST client for OKX market data and funding rates.
"""

import asyncio
import hmac
import hashlib
import base64
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import aiohttp

from ..core.logger import get_connector_logger

logger = get_connector_logger('okx')


class OKXClient:
    """
    OKX REST API client for market data.
    
    Provides:
    - Funding rate data
    - Ticker/price data
    - Open interest
    """
    
    BASE_URL = "https://www.okx.com"
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
    ):
        """
        Initialize OKX client.
        
        Args:
            api_key: OKX API key
            api_secret: OKX API secret
            passphrase: OKX API passphrase
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit = 5  # requests per second
        self._last_request_time = 0
        
        logger.info("OKX client initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _sign_request(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate request signature."""
        message = timestamp + method + path + body
        mac = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict:
        """Make an API request."""
        # Rate limiting
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < (1 / self._rate_limit):
            await asyncio.sleep((1 / self._rate_limit) - elapsed)
        self._last_request_time = time.time()
        
        session = await self._get_session()
        url = f"{self.BASE_URL}{path}"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        if signed and self.api_key:
            timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.') + \
                       f'{datetime.utcnow().microsecond // 1000:03d}Z'
            signature = self._sign_request(timestamp, method.upper(), path)
            headers.update({
                "OK-ACCESS-KEY": self.api_key,
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self.passphrase,
            })
        
        try:
            async with session.request(method, url, params=params, headers=headers) as resp:
                data = await resp.json()
                
                if data.get('code') != '0':
                    logger.warning(f"OKX API error: {data.get('msg')}")
                
                return data
        except Exception as e:
            logger.error(f"OKX request failed: {e}")
            return {'code': '-1', 'msg': str(e), 'data': []}
    
    async def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Get current funding rate for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USDT-SWAP')
            
        Returns:
            Funding rate data dict or None
        """
        # Convert symbol format
        inst_id = self._to_okx_symbol(symbol)
        
        data = await self._request(
            'GET',
            '/api/v5/public/funding-rate',
            params={'instId': inst_id}
        )
        
        if data.get('data'):
            item = data['data'][0]
            return {
                'symbol': symbol,
                'exchange': 'okx',
                'funding_rate': float(item.get('fundingRate', 0)),
                'next_funding_rate': float(item.get('nextFundingRate', 0)) if item.get('nextFundingRate') else None,
                'funding_time': item.get('fundingTime'),
                'timestamp': datetime.utcnow().isoformat(),
            }
        return None
    
    async def get_funding_rate_history(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get historical funding rates.
        
        Args:
            symbol: Trading pair
            limit: Number of records
            
        Returns:
            List of funding rate records
        """
        inst_id = self._to_okx_symbol(symbol)
        
        data = await self._request(
            'GET',
            '/api/v5/public/funding-rate-history',
            params={'instId': inst_id, 'limit': str(limit)}
        )
        
        result = []
        for item in data.get('data', []):
            result.append({
                'symbol': symbol,
                'exchange': 'okx',
                'funding_rate': float(item.get('fundingRate', 0)),
                'funding_time': item.get('fundingTime'),
                'realized_rate': float(item.get('realizedRate', 0)) if item.get('realizedRate') else None,
            })
        
        return result
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get ticker data for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Ticker data dict or None
        """
        inst_id = self._to_okx_symbol(symbol)
        
        data = await self._request(
            'GET',
            '/api/v5/market/ticker',
            params={'instId': inst_id}
        )
        
        if data.get('data'):
            item = data['data'][0]
            return {
                'symbol': symbol,
                'exchange': 'okx',
                'last_price': float(item.get('last', 0)),
                'bid_price': float(item.get('bidPx', 0)),
                'ask_price': float(item.get('askPx', 0)),
                'volume_24h': float(item.get('vol24h', 0)),
                'volume_24h_ccy': float(item.get('volCcy24h', 0)),
                'high_24h': float(item.get('high24h', 0)),
                'low_24h': float(item.get('low24h', 0)),
                'timestamp': datetime.utcnow().isoformat(),
            }
        return None
    
    async def get_open_interest(self, symbol: str) -> Optional[Dict]:
        """
        Get open interest for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Open interest data or None
        """
        inst_id = self._to_okx_symbol(symbol)
        
        data = await self._request(
            'GET',
            '/api/v5/public/open-interest',
            params={'instId': inst_id}
        )
        
        if data.get('data'):
            item = data['data'][0]
            return {
                'symbol': symbol,
                'exchange': 'okx',
                'open_interest': float(item.get('oi', 0)),
                'open_interest_ccy': float(item.get('oiCcy', 0)),
                'timestamp': datetime.utcnow().isoformat(),
            }
        return None
    
    def _to_okx_symbol(self, symbol: str) -> str:
        """
        Convert unified symbol to OKX format.
        
        'BTC/USDT:USDT' -> 'BTC-USDT-SWAP'
        'BTC/USDT' -> 'BTC-USDT'
        """
        is_perp = ':' in symbol
        
        if is_perp:
            symbol = symbol.split(':')[0]
        
        base_quote = symbol.replace('/', '-')
        
        if is_perp:
            return f"{base_quote}-SWAP"
        return base_quote
    
    async def poll_funding_rates(
        self,
        symbols: List[str],
        interval_seconds: int = 300,
        callback=None
    ) -> None:
        """
        Continuously poll funding rates.
        
        Args:
            symbols: List of symbols to poll
            interval_seconds: Polling interval
            callback: Function to call with data
        """
        logger.info(f"Starting funding rate polling for {len(symbols)} symbols")
        
        while True:
            for symbol in symbols:
                try:
                    data = await self.get_funding_rate(symbol)
                    if data and callback:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
                except Exception as e:
                    logger.error(f"Error polling {symbol}: {e}")
            
            await asyncio.sleep(interval_seconds)
