"""
Argus Market Monitor - Main Orchestrator
=========================================

Coordinates all connectors, detectors, and alerts.
"""

import asyncio
import signal
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.config import load_all_config, validate_secrets, get_secret
from .core.database import Database
from .core.logger import setup_logger, get_logger
from .core.gap_risk_tracker import GapRiskTracker
from .core.reddit_monitor import RedditMonitor
from .core.conditions_monitor import ConditionsMonitor
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
from .analysis.daily_review import DailyReview
from .trading.paper_trader_farm import PaperTraderFarm



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
        
        # Off-hours monitoring
        self.gap_risk_tracker: Optional[GapRiskTracker] = None
        self.reddit_monitor: Optional[RedditMonitor] = None
        
        # Conditions synthesis and daily review
        self.conditions_monitor: Optional[ConditionsMonitor] = None
        self.daily_review: Optional[DailyReview] = None
        
        # Paper trader farm (752 parallel traders)
        self.paper_trader_farm: Optional[PaperTraderFarm] = None
        
        self.detectors: Dict[str, Any] = {}
        
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._start_time = datetime.now(timezone.utc)
        self._last_price_snapshot: Dict[str, datetime] = {}
        self._last_coinglass_check: Optional[datetime] = None
        self._last_health_check: Optional[datetime] = None
        
        self.logger.info("Argus Orchestrator initialized")
    
    async def setup(self) -> None:
        """Initialize all components."""
        self.logger.info("Setting up Argus components...")
        
        # Initialize database
        await self.db.connect()
        
        # Initialize connectors
        await self._setup_connectors()
        
        # Initialize Telegram
        await self._setup_telegram()
        
        # Initialize off-hours monitoring (Gap Risk, Conditions, Farm, Review)
        await self._setup_off_hours_monitoring()
        
        # Wire up Telegram callbacks after both are initialized
        self._wire_telegram_callbacks()
        
        # Initialize detectors
        await self._setup_detectors()
        
        # Send Startup Notification
        if self.telegram:
            eastern = ZoneInfo("America/New_York")
            now_et = datetime.now(eastern).strftime('%H:%M:%S %Z')
            startup_msg = f"🚀 <b>Argus Master Engine Online</b>\n"
            startup_msg += f"<i>Time: {now_et}</i>\n\n"
            startup_msg += f"✅ Detectors: {len(self.detectors)}\n"
            startup_msg += f"✅ GPU Engine: {'Enabled (CUDA)' if getattr(self.paper_trader_farm, 'trader_tensors', None) is not None else 'Disabled (CPU Fallback)'}\n"
            startup_msg += f"✅ Farm: 400,000 configurations loaded\n\n"
            startup_msg += f"Monitoring active for IBIT, BITO, and Crypto markets."
            await self.telegram.send_message(startup_msg)
        
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
            'btc_iv_threshold': 25,
            'ibit_drop_threshold': -0.5,
            'combined_score_threshold': 1.0,
            'cooldown_hours': 2,
        })
        if ibit_config.get('enabled', True):
            self.detectors['ibit'] = IBITDetector(ibit_config, self.db, symbol='IBIT')
            # Wire up Telegram for paper trade notifications
            self.detectors['ibit'].set_telegram_callback(self._send_paper_notification)
            # Wire up farm if available
            if self.paper_trader_farm:
                self.detectors['ibit'].set_paper_trader_farm(self.paper_trader_farm)
        
        # BITO options detector (same strategy, more opportunities)
        bito_config = thresholds.get('bito', {
            'enabled': True,
            'btc_iv_threshold': 25,
            'drop_threshold': -0.5,
            'combined_score_threshold': 1.0,
            'cooldown_hours': 2,
        })
        if bito_config.get('enabled', True):
            self.detectors['bito'] = IBITDetector(bito_config, self.db, symbol='BITO')
            self.detectors['bito'].set_telegram_callback(self._send_paper_notification)
            # Wire up farm if available
            if self.paper_trader_farm:
                self.detectors['bito'].set_paper_trader_farm(self.paper_trader_farm)
        
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
            else:
                self.logger.error("Telegram connection failed")
                self.telegram = None
        else:
            self.logger.warning("Telegram not configured - alerts disabled")
    
    async def _setup_off_hours_monitoring(self) -> None:
        """Initialize gap risk tracker, conditions monitor, and daily review."""
        thresholds = self.config.get('thresholds', {})
        
        # Gap Risk Tracker
        gap_config = thresholds.get('gap_risk', {})
        if gap_config.get('enabled', True):
            self.gap_risk_tracker = GapRiskTracker(self.db, gap_config)
            await self.gap_risk_tracker.initialize()
            self.logger.info("Gap Risk Tracker initialized")
        
        # Reddit Monitor (only if API keys configured)
        reddit_secrets = self.secrets.get('reddit', {})
        reddit_config = thresholds.get('reddit_sentiment', {})
        
        client_id = reddit_secrets.get('client_id', '')
        client_secret = reddit_secrets.get('client_secret', '')
        
        if client_id and client_secret and reddit_config.get('enabled', True):
            self.reddit_monitor = RedditMonitor(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=reddit_secrets.get('user_agent', 'Argus/1.0'),
            )
            self.logger.info("Reddit Monitor initialized")
        else:
            self.logger.info("Reddit Monitor not configured - sentiment tracking disabled")
        
        # Conditions Monitor (synthesis layer)
        conditions_config = thresholds.get('conditions_monitor', {})
        self.conditions_monitor = ConditionsMonitor(
            config=conditions_config,
            on_alert=self._on_conditions_alert,
        )
        
        # Wire up data sources to conditions monitor
        self.conditions_monitor.set_data_sources(
            get_btc_iv=self._get_btc_iv,
            get_funding=self._get_btc_funding,
            get_btc_price=self._get_btc_price,
        )
        self.logger.info("Conditions Monitor initialized")
        
        # Daily Review (4 PM summary)
        self.daily_review = DailyReview(
            starting_balance=5000.0,
            on_send=self._send_daily_review,
        )
        
        # Wire up data sources to daily review
        self.daily_review.set_data_sources(
            get_conditions=self.conditions_monitor.get_current_conditions,
        )
        self.logger.info("Daily Review initialized")
        
        # Paper Trader Farm (86K+ parallel traders with full coverage)
        self.paper_trader_farm = PaperTraderFarm(
            db=self.db,
            full_coverage=True,  # Generate ALL unique parameter combinations
        )
        await self.paper_trader_farm.initialize()
        
        # Wire up data sources to paper trader farm
        self.paper_trader_farm.set_data_sources(
            get_conditions=self.conditions_monitor.get_current_conditions,
        )
        self.logger.info(f"Paper Trader Farm initialized with {len(self.paper_trader_farm.trader_configs):,} traders")
        
    def _wire_telegram_callbacks(self) -> None:
        """Wire up Telegram two-way callbacks once dependencies are ready."""
        if not self.telegram:
            return
        if not self.conditions_monitor:
            return
        self.telegram.set_callbacks(
            get_conditions=self._get_status_summary,
            get_pnl=self._get_pnl_summary,
            get_positions=self._get_positions_summary,
            get_farm_status=self._get_farm_status,
        )
    
    async def _on_conditions_alert(self, snapshot) -> None:
        """Handle conditions threshold crossing alert."""
        if not self.telegram:
            return
        
        details = {
            'BTC IV': f"{snapshot.btc_iv:.0f}% ({snapshot.iv_signal})",
            'Funding': f"{snapshot.funding_rate:+.3f}% ({snapshot.funding_signal})",
            'BTC': f"{snapshot.btc_change_24h:+.1f}% ({snapshot.momentum_signal})",
            'Market': "🟢 OPEN" if snapshot.market_open else "🔴 CLOSED",
        }
        
        await self.telegram.send_conditions_alert(
            score=snapshot.score,
            label=snapshot.label,
            details=details,
            implication=snapshot.implication,
        )
    
    async def _send_daily_review(self, message: str) -> None:
        """Send daily review via Telegram."""
        if self.telegram:
            await self.telegram.send_message(message)
    
    async def _get_btc_iv(self) -> Optional[Dict]:
        """Get current BTC IV from Deribit."""
        if self.deribit_client:
            try:
                return await self.deribit_client.get_atm_iv('BTC')
            except Exception:
                pass
        return None
    
    async def _get_btc_funding(self) -> Optional[Dict]:
        """Get current BTC funding rate from Bybit."""
        if self.bybit_ws:
            rate = self.bybit_ws.get_funding_rate('BTCUSDT')
            if rate is not None:
                return {'rate': rate}
        return None
    
    async def _get_btc_price(self) -> Optional[Dict]:
        """Get current BTC price."""
        if self.bybit_ws:
            ticker = self.bybit_ws.get_ticker('BTCUSDT')
            if ticker:
                return {
                    'price': ticker.get('last_price', 0),
                    'change_24h_pct': ticker.get('price_24h_pcnt', 0) * 100,
                }
        return None
    
    async def _get_pnl_summary(self) -> Dict:
        """Get P&L summary for Telegram /pnl command."""
        # Prefer paper trader farm if available
        if self.paper_trader_farm:
            return await self.paper_trader_farm.get_pnl_for_telegram()
        if self.daily_review:
            from .analysis.daily_review import get_pnl_summary
            return await get_pnl_summary(self.daily_review)
        return {}
    
    async def _get_positions_summary(self) -> List[Dict]:
        """Get positions summary for Telegram /positions command."""
        if self.paper_trader_farm:
            return await self.paper_trader_farm.get_positions_for_telegram()
        return []

    async def _get_status_summary(self) -> Dict[str, Any]:
        """Get conditions plus data freshness for Telegram /status command."""
        conditions = {}
        if self.conditions_monitor:
            conditions = await self.conditions_monitor.get_current_conditions()
        data_status = await self._get_data_status()
        conditions['data_status'] = data_status
        return conditions

    async def _get_farm_status(self) -> Dict[str, Any]:
        """Get paper trader farm status summary for Telegram."""
        if not self.paper_trader_farm:
            return {}
        return self.paper_trader_farm.get_status_summary()

    async def _get_data_status(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Collect data freshness signals for key tables."""
        tables = {
            "Detections": ("detections", 24 * 60 * 60),
            "Funding": ("funding_rates", 2 * 60 * 60),
            "Options IV": ("options_iv", 2 * 60 * 60),
            "Liquidations": ("liquidations", 2 * 60 * 60),
            "Coinglass": ("coinglass_health", 10 * 60),
            "Prices": ("price_snapshots", 10 * 60),
            "Health": ("system_health", 10 * 60),
        }
        latest = await self.db.get_latest_timestamps(
            [t[0] for t in tables.values() if t[0] != "coinglass_health"]
        )
        now = datetime.now(timezone.utc)
        eastern = ZoneInfo("America/New_York")
        age_since_start = int((now - self._start_time).total_seconds())
        status: Dict[str, Dict[str, Optional[str]]] = {}
        for label, (table, threshold) in tables.items():
            if label == "Liquidations" and not self.coinglass_client:
                status[label] = {
                    "status": "disabled",
                    "last_seen_et": "N/A",
                    "age_human": None,
                }
                continue
            if label == "Coinglass":
                if not self.coinglass_client:
                    status[label] = {
                        "status": "disabled",
                        "last_seen_et": "N/A",
                        "age_human": None,
                    }
                    continue
                status[label] = self._format_freshness(
                    self._last_coinglass_check,
                    threshold,
                    eastern,
                    age_since_start,
                )
                continue
            ts = latest.get(table)
            if not ts:
                if age_since_start < threshold:
                    status[label] = {
                        "status": "pending",
                        "last_seen_et": "N/A",
                        "age_human": None,
                    }
                    continue
                status[label] = {
                    "status": "missing",
                    "last_seen_et": "N/A",
                    "age_human": None,
                }
                continue
            try:
                parsed = datetime.fromisoformat(ts)
            except ValueError:
                status[label] = {
                    "status": "missing",
                    "last_seen_et": "N/A",
                    "age_human": None,
                }
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age_seconds = max(0, int((now - parsed).total_seconds()))
            status[label] = {
                "status": "ok" if age_seconds <= threshold else "stale",
                "last_seen_et": parsed.astimezone(eastern).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "age_human": self._format_age(age_seconds),
            }
        return status

    def _format_freshness(
        self,
        last_seen: Optional[datetime],
        threshold: int,
        eastern: ZoneInfo,
        age_since_start: int,
    ) -> Dict[str, Optional[str]]:
        """Format freshness for in-memory timestamps."""
        if not last_seen:
            if age_since_start < threshold:
                return {
                    "status": "pending",
                    "last_seen_et": "N/A",
                    "age_human": None,
                }
            return {
                "status": "missing",
                "last_seen_et": "N/A",
                "age_human": None,
            }
        age_seconds = int((datetime.now(timezone.utc) - last_seen).total_seconds())
        return {
            "status": "ok" if age_seconds <= threshold else "stale",
            "last_seen_et": last_seen.astimezone(eastern).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "age_human": self._format_age(age_seconds),
        }

    @staticmethod
    def _format_age(age_seconds: int) -> str:
        """Format age in human-friendly units."""
        if age_seconds < 60:
            return f"{age_seconds}s ago"
        if age_seconds < 3600:
            return f"{age_seconds // 60}m ago"
        if age_seconds < 86400:
            return f"{age_seconds // 3600}h ago"
        return f"{age_seconds // 86400}d ago"
    
    async def _on_bybit_ticker(self, data: Dict) -> None:
        """Handle Bybit ticker update."""
        data['exchange'] = 'bybit'
        await self._maybe_log_price_snapshot(
            exchange='bybit',
            symbol=data.get('symbol'),
            price=data.get('last_price'),
            volume=data.get('volume_24h'),
        )
        
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
        await self._maybe_log_price_snapshot(
            exchange='coinbase',
            symbol=data.get('symbol'),
            price=data.get('last_price'),
            volume=data.get('volume_24h'),
        )
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

    async def _maybe_log_price_snapshot(
        self,
        exchange: str,
        symbol: Optional[str],
        price: Optional[float],
        volume: Optional[float],
        min_interval_seconds: int = 60,
    ) -> None:
        """Record price snapshots at a controlled cadence."""
        if not symbol or price is None:
            return
        now = datetime.now(timezone.utc)
        key = f"{exchange}:{symbol}"
        last_logged = self._last_price_snapshot.get(key)
        if last_logged and (now - last_logged).total_seconds() < min_interval_seconds:
            return
        await self.db.insert_price_snapshot(
            exchange=exchange,
            asset=symbol,
            price_type='spot',
            price=float(price),
            volume=volume,
        )
        self._last_price_snapshot[key] = now
    
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
    
    async def _send_paper_notification(self, message: str) -> None:
        """Send paper trade notification via Telegram."""
        if self.telegram:
            try:
                await self.telegram.send_message(message, parse_mode="Markdown")
            except Exception as e:
                self.logger.warning(f"Failed to send paper notification: {e}")
    
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
                self._last_coinglass_check = datetime.now(timezone.utc)
                await self.db.insert_health_check(
                    component="coinglass",
                    status="ok",
                )
            except Exception as e:
                # Don't spam logs for known API limit issues
                if 'Upgrade plan' not in str(e):
                    self.logger.error(f"Coinglass polling error: {e}")
                self._last_coinglass_check = datetime.now(timezone.utc)
                await self.db.insert_health_check(
                    component="coinglass",
                    status="error",
                    error_message=str(e),
                )
            
            await asyncio.sleep(interval)
    
    async def _health_check(self) -> None:
        """Periodic health check and status logging."""
        interval = 300  # 5 minutes
        
        while self._running:
            # Log health status
            status = {
                'bybit_connected': self.bybit_ws.is_connected if self.bybit_ws else False,
                'coinbase_connected': self.coinbase_client.is_connected if self.coinbase_client else False,
                'detectors_active': len(self.detectors),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
            
            self.logger.info(f"Health check: {status}")
            self._last_health_check = datetime.now(timezone.utc)
            await self.db.insert_health_check(
                component="bybit_ws",
                status="connected" if status['bybit_connected'] else "disconnected",
            )
            await self.db.insert_health_check(
                component="coinbase_client",
                status="connected" if status['coinbase_connected'] else "disconnected",
            )
            await self.db.insert_health_check(
                component="detectors",
                status=f"active_{status['detectors_active']}",
            )
            
            await asyncio.sleep(interval)
    
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
        
        # Start conditions monitoring (synthesis layer)
        if self.conditions_monitor:
            self._tasks.append(asyncio.create_task(self.conditions_monitor.start_monitoring()))
        
        # Start daily review monitoring (4 PM summary)
        if self.daily_review:
            self._tasks.append(asyncio.create_task(self.daily_review.start_monitoring()))
        
        # Start Telegram two-way polling
        if self.telegram:
            await self.telegram.start_polling()
        
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
        
        # Stop monitoring loops
        if self.conditions_monitor:
            self.conditions_monitor.stop_monitoring()
        if self.daily_review:
            self.daily_review.stop_monitoring()
        
        # Stop Telegram polling
        if self.telegram:
            await self.telegram.stop_polling()
        
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
