"""
Deribit REST API Client
=======================

Public REST client for Deribit options data.
No authentication required for public endpoints.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import aiohttp

from ..core.logger import get_connector_logger

logger = get_connector_logger('deribit')


class DeribitClient:
    """
    Deribit public API client for options data.
    
    Provides:
    - Options IV data
    - Greeks
    - Volatility index
    
    Uses public endpoints - no API key required for US users.
    """
    
    MAINNET_URL = "https://www.deribit.com/api/v2"
    TESTNET_URL = "https://test.deribit.com/api/v2"
    
    def __init__(self, testnet: bool = True):
        """
        Initialize Deribit client.
        
        Args:
            testnet: Use testnet endpoint (recommended for testing)
        """
        self.base_url = self.TESTNET_URL if testnet else self.MAINNET_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit = 20  # public rate limit: 20/min unauthenticated
        self._request_count = 0
        self._last_reset = datetime.utcnow()
        
        logger.info(f"Deribit client initialized ({'testnet' if testnet else 'mainnet'})")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(self, method: str, params: Dict = None) -> Dict:
        """Make a public API request."""
        # Simple rate limiting
        now = datetime.utcnow()
        if (now - self._last_reset).seconds >= 60:
            self._request_count = 0
            self._last_reset = now
        
        if self._request_count >= self._rate_limit:
            wait_time = 60 - (now - self._last_reset).seconds
            logger.warning(f"Rate limit reached, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
            self._request_count = 0
            self._last_reset = datetime.utcnow()
        
        self._request_count += 1
        
        session = await self._get_session()
        url = f"{self.base_url}/public/{method}"
        
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                
                if 'error' in data:
                    logger.warning(f"Deribit API error: {data['error']}")
                
                return data
        except Exception as e:
            logger.error(f"Deribit request failed: {e}")
            return {'error': str(e)}
    
    async def get_ticker(self, instrument_name: str) -> Optional[Dict]:
        """
        Get ticker data including IV for an option.
        
        Args:
            instrument_name: Deribit instrument name (e.g., 'BTC-28JUN24-50000-C')
            
        Returns:
            Ticker data with IV and Greeks or None
        """
        data = await self._request('ticker', {'instrument_name': instrument_name})
        
        if 'result' in data:
            result = data['result']
            return {
                'instrument': instrument_name,
                'timestamp': datetime.utcnow().isoformat(),
                'last_price': result.get('last_price'),
                'mark_price': result.get('mark_price'),
                'mark_iv': result.get('mark_iv'),  # Implied volatility for mark price
                'bid_iv': result.get('bid_iv'),     # IV for best bid
                'ask_iv': result.get('ask_iv'),     # IV for best ask
                'underlying_price': result.get('underlying_price'),
                'underlying_index': result.get('underlying_index'),
                'delta': result.get('greeks', {}).get('delta'),
                'gamma': result.get('greeks', {}).get('gamma'),
                'theta': result.get('greeks', {}).get('theta'),
                'vega': result.get('greeks', {}).get('vega'),
                'rho': result.get('greeks', {}).get('rho'),
                'open_interest': result.get('open_interest'),
                'volume_24h': result.get('stats', {}).get('volume'),
            }
        return None
    
    async def get_book_summary_by_currency(
        self,
        currency: str = "BTC",
        kind: str = "option"
    ) -> List[Dict]:
        """
        Get summary of all instruments for a currency.
        
        Args:
            currency: 'BTC' or 'ETH'
            kind: 'option' or 'future'
            
        Returns:
            List of instrument summaries with IV data
        """
        data = await self._request(
            'get_book_summary_by_currency',
            {'currency': currency, 'kind': kind}
        )
        
        result = []
        for item in data.get('result', []):
            result.append({
                'instrument': item.get('instrument_name'),
                'currency': currency,
                'kind': kind,
                'mark_price': item.get('mark_price'),
                'mark_iv': item.get('mark_iv'),
                'bid_price': item.get('bid_price'),
                'ask_price': item.get('ask_price'),
                'volume_24h': item.get('volume'),
                'open_interest': item.get('open_interest'),
                'underlying_price': item.get('underlying_price'),
                'timestamp': datetime.utcnow().isoformat(),
            })
        
        return result
    
    async def get_instruments(
        self,
        currency: str = "BTC",
        kind: str = "option",
        expired: bool = False
    ) -> List[Dict]:
        """
        Get all available instruments.
        
        Args:
            currency: 'BTC' or 'ETH'
            kind: 'option' or 'future'
            expired: Include expired instruments
            
        Returns:
            List of instrument definitions
        """
        data = await self._request(
            'get_instruments',
            {'currency': currency, 'kind': kind, 'expired': str(expired).lower()}
        )
        
        result = []
        for item in data.get('result', []):
            result.append({
                'instrument': item.get('instrument_name'),
                'currency': currency,
                'kind': kind,
                'strike': item.get('strike'),
                'option_type': item.get('option_type'),  # 'call' or 'put'
                'expiration_timestamp': item.get('expiration_timestamp'),
                'is_active': item.get('is_active'),
                'min_trade_amount': item.get('min_trade_amount'),
            })
        
        return result
    
    async def get_index_price(self, index_name: str = "btc_usd") -> Optional[Dict]:
        """
        Get current index price.
        
        Args:
            index_name: Index name (e.g., 'btc_usd', 'eth_usd')
            
        Returns:
            Index price data
        """
        data = await self._request('get_index_price', {'index_name': index_name})
        
        if 'result' in data:
            return {
                'index_name': index_name,
                'index_price': data['result'].get('index_price'),
                'estimated_delivery_price': data['result'].get('estimated_delivery_price'),
                'timestamp': datetime.utcnow().isoformat(),
            }
        return None
    
    async def get_historical_volatility(
        self,
        currency: str = "BTC"
    ) -> Optional[Dict]:
        """
        Get historical volatility data.
        
        Args:
            currency: 'BTC' or 'ETH'
            
        Returns:
            Historical volatility data
        """
        data = await self._request(
            'get_historical_volatility',
            {'currency': currency}
        )
        
        if 'result' in data:
            # Returns list of [timestamp, volatility] pairs
            return {
                'currency': currency,
                'data': data['result'],
                'latest_hv': data['result'][-1][1] if data['result'] else None,
                'timestamp': datetime.utcnow().isoformat(),
            }
        return None
    
    async def get_atm_iv(self, currency: str = "BTC") -> Optional[Dict]:
        """
        Get ATM implied volatility by finding nearest strike options.
        
        Args:
            currency: 'BTC' or 'ETH'
            
        Returns:
            ATM IV data or None
        """
        # Get index price first
        index_data = await self.get_index_price(f"{currency.lower()}_usd")
        if not index_data:
            return None
        
        index_price = index_data['index_price']
        
        # Get all options
        options = await self.get_book_summary_by_currency(currency, 'option')
        if not options:
            return None
        
        # Find ATM options (closest to current price)
        atm_options = []
        for opt in options:
            if opt['mark_iv'] and opt['mark_iv'] > 0:
                # Parse instrument name to get strike
                # Format: BTC-28JUN24-50000-C
                parts = opt['instrument'].split('-')
                if len(parts) >= 3:
                    try:
                        strike = float(parts[2])
                        distance = abs(strike - index_price) / index_price
                        if distance < 0.1:  # Within 10% of ATM
                            opt['strike'] = strike
                            opt['distance_from_atm'] = distance
                            atm_options.append(opt)
                    except ValueError:
                        pass
        
        if not atm_options:
            return None
        
        # Sort by distance from ATM and get closest
        atm_options.sort(key=lambda x: x['distance_from_atm'])
        closest = atm_options[0]
        
        # Average IV of closest options
        atm_iv_values = [o['mark_iv'] for o in atm_options[:4] if o['mark_iv']]
        avg_atm_iv = sum(atm_iv_values) / len(atm_iv_values) if atm_iv_values else None
        
        return {
            'currency': currency,
            'index_price': index_price,
            'atm_iv': avg_atm_iv,
            'closest_strike': closest['strike'],
            'closest_iv': closest['mark_iv'],
            'sample_size': len(atm_iv_values),
            'timestamp': datetime.utcnow().isoformat(),
        }
    
    async def poll_options_iv(
        self,
        currency: str = "BTC",
        interval_seconds: int = 60,
        callback=None
    ) -> None:
        """
        Continuously poll ATM IV.
        
        Args:
            currency: Currency to monitor
            interval_seconds: Polling interval
            callback: Function to call with data
        """
        logger.info(f"Starting options IV polling for {currency}")
        
        while True:
            try:
                data = await self.get_atm_iv(currency)
                if data and callback:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
            except Exception as e:
                logger.error(f"Error polling IV: {e}")
            
            await asyncio.sleep(interval_seconds)
