"""
Argus Market Monitor - Main Orchestrator
=========================================

Coordinates all connectors, detectors, and alerts.
"""

import asyncio
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.config import load_all_config, validate_secrets, get_secret
from .core.database import Database
from .core.logger import setup_logger, get_logger
from .connectors.bybit_ws import BybitWebSocket
from .connectors.coinbase_client import CoinbaseClient
from .connectors.okx_client import OKXClient
from .connectors.deribit_client import DeribitClient
from .connectors.coinglass_client import CoinglassClient
from .connectors.yahoo_client import YahooFinanceClient
from .detectors.funding_detector import FundingDetector
from .detectors.basis_detector import BasisDetector
from .detectors.cross_exchange_detector import CrossExchangeDetector
from .detectors.liquidation_detector import LiquidationDetector
from .detectors.options_iv_detector import OptionsIVDetector
from .detectors.volatility_detector import VolatilityDetector
from .detectors.ibit_detector import IBITDetector
from .alerts.telegram_bot import TelegramBot


class ArgusOrchestrator:
    """
    Main Argus orchestrator.
    
    Coordinates:
    - Exchange WebSocket connections
    - Data polling clients
    - Opportunity detectors
    - Alert dispatching
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize Argus.
        
        Args:
            config_dir: Path to config directory
        """
        # Load configuration
        self.config = load_all_config(config_dir)
        self.secrets = self.config.get('secrets', {})
        
        # Setup logging
        log_level = self.config.get('system', {}).get('log_level', 'INFO')
        setup_logger('argus', level=log_level)
        self.logger = get_logger('orchestrator')
        
        # Validate secrets
        issues = validate_secrets(self.secrets)
        if issues:
            self.logger.warning("Configuration issues:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        
        # Initialize database
        db_path = Path(self.config.get('system', {}).get('database_path', 'data/argus.db'))
        self.db = Database(str(db_path))
        
        # Get symbols to monitor
        self.symbols = self.config.get('symbols', {}).get('monitored', [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
            'ARB/USDT:USDT', 'DOGE/USDT:USDT'
        ])
        
        # Components will be initialized in setup()
        self.bybit_ws: Optional[BybitWebSocket] = None
        self.coinbase_client: Optional[CoinbaseClient] = None
        self.okx_client: Optional[OKXClient] = None
        self.deribit_client: Optional[DeribitClient] = None
        self.coinglass_client: Optional[CoinglassClient] = None
        self.yahoo_client: Optional[YahooFinanceClient] = None
        self.telegram: Optional[TelegramBot] = None
        
        self.detectors: Dict[str, Any] = {}
        
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        self.logger.info("Argus Orchestrator initialized")
    
    async def setup(self) -> None:
        """Initialize all components."""
        self.logger.info("Setting up Argus components...")
        
        # Initialize database
        await self.db.connect()
        
        # Initialize connectors
        await self._setup_connectors()
        
        # Initialize detectors
        await self._setup_detectors()
        
        # Initialize Telegram
        await self._setup_telegram()
        
        self.logger.info("Setup complete!")
    
    async def _setup_connectors(self) -> None:
        """Initialize exchange connectors."""
        # Bybit WebSocket (public - no auth needed)
        bybit_symbols = [s for s in self.symbols]
        self.bybit_ws = BybitWebSocket(
            symbols=bybit_symbols,
            on_ticker=self._on_bybit_ticker,
            on_funding=self._on_funding_update,
        )
        self.logger.info(f"Bybit WS configured for {len(bybit_symbols)} symbols")
        
        # Coinbase REST client (US-friendly, replaces Binance)
        self.coinbase_client = CoinbaseClient(
            symbols=self.symbols,
            on_ticker=self._on_coinbase_ticker,
        )
        self.logger.info(f"Coinbase client configured for {len(self.symbols)} symbols")
        
        # OKX REST client
        self.okx_client = OKXClient(
            api_key=get_secret(self.secrets, 'okx', 'api_key') or '',
            api_secret=get_secret(self.secrets, 'okx', 'api_secret') or '',
            passphrase=get_secret(self.secrets, 'okx', 'passphrase') or '',
        )
        
        # Deribit REST client (public - no auth needed)
        # Use mainnet for real data
        self.deribit_client = DeribitClient(testnet=False)
        self.logger.info("Deribit client configured (mainnet)")
        
        # Yahoo Finance for IBIT ETF
        self.yahoo_client = YahooFinanceClient(
            symbols=['IBIT', 'BITO'],
            on_update=self._on_ibit_update,
        )
        self.logger.info("Yahoo Finance client configured for IBIT/BITO")
        
        # Coinglass client (optional - free tier limited)
        coinglass_key = get_secret(self.secrets, 'coinglass', 'api_key')
        if coinglass_key and not coinglass_key.startswith('PASTE_'):
            self.coinglass_client = CoinglassClient(api_key=coinglass_key)
        else:
            self.logger.warning("Coinglass not configured - liquidation detection limited")
    
    async def _setup_detectors(self) -> None:
        """Initialize all detectors."""
        thresholds = self.config.get('thresholds', {})
        
        # Funding rate detector
        funding_config = thresholds.get('funding_rate', {})
        if funding_config.get('enabled', True):
            self.detectors['funding'] = FundingDetector(funding_config, self.db)
        
        # Basis detector
        basis_config = thresholds.get('basis', {})
        if basis_config.get('enabled', True):
            self.detectors['basis'] = BasisDetector(basis_config, self.db)
        
        # Cross-exchange detector
        cross_config = thresholds.get('cross_exchange', {})
        if cross_config.get('enabled', True):
            self.detectors['cross_exchange'] = CrossExchangeDetector(cross_config, self.db)
        
        # Liquidation detector
        liq_config = thresholds.get('liquidation', {})
        if liq_config.get('enabled', True):
            self.detectors['liquidation'] = LiquidationDetector(liq_config, self.db)
        
        # Options IV detector (BTC options on Deribit - for research)
        iv_config = thresholds.get('options_iv', {})
        if iv_config.get('enabled', True):
            self.detectors['options_iv'] = OptionsIVDetector(iv_config, self.db)
        
        # Volatility detector
        vol_config = thresholds.get('volatility', {})
        if vol_config.get('enabled', True):
            self.detectors['volatility'] = VolatilityDetector(vol_config, self.db)
        
        # IBIT options detector (actionable for Robinhood)
        ibit_config = thresholds.get('ibit', {
            'enabled': True,
            'btc_iv_threshold': 70,
            'ibit_drop_threshold': -3,
            'combined_score_threshold': 1.5,
            'cooldown_hours': 4,
        })
        if ibit_config.get('enabled', True):
            self.detectors['ibit'] = IBITDetector(ibit_config, self.db)
        
        self.logger.info(f"Initialized {len(self.detectors)} detectors")
    
    async def _setup_telegram(self) -> None:
        """Initialize Telegram bot."""
        bot_token = get_secret(self.secrets, 'telegram', 'bot_token')
        chat_id = get_secret(self.secrets, 'telegram', 'chat_id')
        
        if bot_token and chat_id and not bot_token.startswith('PASTE_'):
            self.telegram = TelegramBot(
                bot_token=bot_token,
                chat_id=chat_id,
            )
            
            # Test connection
            if await self.telegram.test_connection():
                self.logger.info("Telegram bot connected successfully")
                await self.telegram.send_system_status('online', 'Argus started - monitoring 7 opportunity types')
            else:
                self.logger.error("Telegram connection failed")
                self.telegram = None
        else:
            self.logger.warning("Telegram not configured - alerts disabled")
    
    async def _on_bybit_ticker(self, data: Dict) -> None:
        """Handle Bybit ticker update."""
        data['exchange'] = 'bybit'
        
        # Run through all applicable detectors
        if 'funding' in self.detectors:
            detection = await self.detectors['funding'].analyze(data)
            if detection:
                await self._send_alert(detection)
        
        if 'volatility' in self.detectors:
            await self.detectors['volatility'].analyze(data)
        
        if 'cross_exchange' in self.detectors:
            self.detectors['cross_exchange'].update_price('bybit', data['symbol'], data['last_price'])
        
        if 'basis' in self.detectors:
            self.detectors['basis'].update_perp_price(data['symbol'], data['last_price'])
    
    async def _on_coinbase_ticker(self, data: Dict) -> None:
        """Handle Coinbase ticker update (replaces Binance)."""
        if 'cross_exchange' in self.detectors:
            self.detectors['cross_exchange'].update_price('coinbase', data['symbol'], data['last_price'])
        
        if 'basis' in self.detectors:
            self.detectors['basis'].update_spot_price(data['symbol'], data['last_price'])
            detection = await self.detectors['basis'].analyze(data)
            if detection:
                await self._send_alert(detection)
    
    async def _on_ibit_update(self, data: Dict) -> None:
        """Handle IBIT price update from Yahoo Finance."""
        if data.get('symbol') == 'IBIT' and 'ibit' in self.detectors:
            # Add source marker
            data['source'] = 'yahoo'
            detection = await self.detectors['ibit'].analyze(data)
            if detection:
                await self._send_alert(detection)
    
    async def _on_funding_update(self, data: Dict) -> None:
        """Handle funding rate update from Bybit."""
        self.logger.debug(f"Funding update: {data['symbol']} = {data['rate']:.4%}")
    
    async def _send_alert(self, detection: Dict) -> None:
        """Send alert for a detection."""
        if not self.telegram:
            return
        
        op_type = detection.get('opportunity_type')
        tier = detection.get('alert_tier', 2)
        
        self.logger.info(
            f"DETECTION: {op_type} - {detection.get('asset')} - "
            f"Edge: {detection.get('net_edge_bps', 0):.1f} bps (tier {tier})"
        )
        
        if op_type == 'funding_rate':
            await self.telegram.send_funding_alert(detection)
        elif op_type == 'options_iv':
            await self.telegram.send_iv_alert(detection)
        elif op_type == 'liquidation':
            await self.telegram.send_liquidation_alert(detection)
        elif op_type == 'basis':
            await self.telegram.send_basis_alert(detection)
        elif op_type == 'ibit_options':
            await self._send_ibit_alert(detection)
    
    async def _send_ibit_alert(self, detection: Dict) -> None:
        """Send IBIT options opportunity alert."""
        data = detection.get('detection_data', {})
        
        await self.telegram.send_alert(
            tier=1,  # High priority - actionable
            alert_type='options_iv',
            title="🎯 IBIT OPTIONS OPPORTUNITY",
            details={
                'IBIT Price': f"${data.get('ibit_price', 0):.2f}",
                'IBIT 24h Change': f"{data.get('ibit_change_24h', 0):+.1f}%",
                'BTC IV': f"{data.get('btc_iv', 0):.0f}%",
                'Market': data.get('market_state', 'CLOSED'),
                'Short Strike': f"${data.get('suggested_short_strike', 0):.0f}",
                'Long Strike': f"${data.get('suggested_long_strike', 0):.0f}",
            },
            action="Check Robinhood for put spread opportunity"
        )
    
    async def _poll_deribit(self) -> None:
        """Poll Deribit for options IV data."""
        if not self.deribit_client:
            return
        
        interval = 60  # seconds
        
        while self._running:
            try:
                for currency in ['BTC', 'ETH']:
                    data = await self.deribit_client.get_atm_iv(currency)
                    if data:
                        # Feed to options IV detector
                        if 'options_iv' in self.detectors:
                            detection = await self.detectors['options_iv'].analyze(data)
                            if detection:
                                await self._send_alert(detection)
                        
                        # Feed BTC IV to IBIT detector
                        if currency == 'BTC' and 'ibit' in self.detectors:
                            ibit_data = {
                                'source': 'deribit',
                                'atm_iv': data.get('atm_iv', 0),
                            }
                            # Update IBIT detector with BTC IV
                            self.detectors['ibit'].update_btc_iv(data.get('atm_iv', 0))
                            
            except Exception as e:
                self.logger.error(f"Deribit polling error: {e}")
            
            await asyncio.sleep(interval)
    
    async def _poll_coinglass(self) -> None:
        """Poll Coinglass for liquidation data."""
        if not self.coinglass_client:
            self.logger.info("Coinglass not available - liquidation polling disabled")
            return
        
        interval = 60  # Less frequent to avoid rate limits
        
        while self._running:
            try:
                for symbol in ['BTC', 'ETH', 'SOL']:
                    cascade = await self.coinglass_client.check_liquidation_cascade(symbol)
                    if cascade and 'liquidation' in self.detectors:
                        detection = await self.detectors['liquidation'].analyze(cascade)
                        if detection:
                            await self._send_alert(detection)
            except Exception as e:
                # Don't spam logs for known API limit issues
                if 'Upgrade plan' not in str(e):
                    self.logger.error(f"Coinglass polling error: {e}")
            
            await asyncio.sleep(interval)
    
    async def _health_check(self) -> None:
        """Periodic health check and status logging."""
        interval = 300  # 5 minutes
        
        while self._running:
            await asyncio.sleep(interval)
            
            # Log health status
            status = {
                'bybit_connected': self.bybit_ws.is_connected if self.bybit_ws else False,
                'coinbase_connected': self.coinbase_client.is_connected if self.coinbase_client else False,
                'detectors_active': len(self.detectors),
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            self.logger.info(f"Health check: {status}")
            await self.db.insert_system_health(status)
    
    async def run(self) -> None:
        """Start all components and run main loop."""
        self._running = True
        
        self.logger.info("Starting Argus...")
        
        # Start WebSocket connections
        if self.bybit_ws:
            self._tasks.append(asyncio.create_task(self.bybit_ws.connect()))
        
        # Start polling tasks
        if self.coinbase_client:
            self._tasks.append(asyncio.create_task(self.coinbase_client.poll(interval_seconds=10)))
        if self.yahoo_client:
            self._tasks.append(asyncio.create_task(self.yahoo_client.poll(interval_seconds=60)))
        
        self._tasks.append(asyncio.create_task(self._poll_deribit()))
        self._tasks.append(asyncio.create_task(self._poll_coinglass()))
        self._tasks.append(asyncio.create_task(self._health_check()))
        
        self.logger.info("Argus is running! Press Ctrl+C to stop.")
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.CancelledError:
            self.logger.info("Shutdown requested")
    
    async def stop(self) -> None:
        """Stop all components gracefully."""
        self.logger.info("Stopping Argus...")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Disconnect WebSockets
        if self.bybit_ws:
            await self.bybit_ws.disconnect()
        
        # Close REST clients
        if self.coinbase_client:
            await self.coinbase_client.close()
        if self.okx_client:
            await self.okx_client.close()
        if self.deribit_client:
            await self.deribit_client.close()
        if self.coinglass_client:
            await self.coinglass_client.close()
        if self.yahoo_client:
            await self.yahoo_client.close()
        
        # Close database
        await self.db.close()
        
        # Send shutdown notification
        if self.telegram:
            await self.telegram.send_system_status('offline', 'Argus stopped')
        
        self.logger.info("Argus stopped")


async def main() -> None:
    """Entry point for Argus."""
    argus = ArgusOrchestrator()
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        asyncio.create_task(argus.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
    
    try:
        await argus.setup()
        await argus.run()
    except KeyboardInterrupt:
        pass
    finally:
        await argus.stop()


if __name__ == "__main__":
    asyncio.run(main())
