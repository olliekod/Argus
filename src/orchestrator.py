"""
Argus Market Monitor - Main Orchestrator
=========================================

Coordinates all connectors, detectors, and alerts.
"""

import asyncio
import signal
from datetime import datetime, timedelta, timezone
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
from .connectors.deribit_client import DeribitClient
from .connectors.yahoo_client import YahooFinanceClient
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
        self.deribit_client: Optional[DeribitClient] = None
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
        self._last_health_check: Optional[datetime] = None
        self._research_last_run: Optional[datetime] = None
        self._research_last_symbol: Optional[str] = None
        self._research_last_entered: int = 0
        self._research_promoted: bool = False
        self.research_config: Dict[str, Any] = self.config.get('research', {})
        self.research_enabled = self.research_config.get('enabled', False)
        self.research_alerts_enabled = self.research_config.get('alerts_enabled', False)
        self.research_daily_review_enabled = self.research_config.get('daily_review_enabled', False)
        
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
        
        # Deribit REST client (public - no auth needed)
        # Use mainnet for real data
        self.deribit_client = DeribitClient(testnet=False)
        self.logger.info("Deribit client configured (mainnet)")
        
        # Yahoo Finance for IBIT ETF
        self.yahoo_client = YahooFinanceClient(
            symbols=['IBIT', 'BITO'],
            on_update=self._on_yahoo_update,
        )
        self.logger.info("Yahoo Finance client configured for IBIT/BITO")
        
    async def _setup_detectors(self) -> None:
        """Initialize all detectors."""
        thresholds = self.config.get('thresholds', {})
        
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
            if self.research_enabled:
                self.detectors['ibit'].paper_trading_enabled = False
        
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
            if self.research_enabled:
                self.detectors['bito'].paper_trading_enabled = False
        
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

        # Wire up data sources to daily review (after farm is ready)
        self.daily_review.set_data_sources(
            get_conditions=self.conditions_monitor.get_current_conditions,
            get_positions=self.paper_trader_farm.get_positions_for_review,
            get_trade_stats=self.paper_trader_farm.get_trade_activity_summary,
            get_gap_risk=self.gap_risk_tracker.get_status if self.gap_risk_tracker else None,
        )
        self.logger.info("Daily Review initialized")
        
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
            get_signal_status=self._get_signal_status,
            get_research_status=self._get_research_status,
        )
    
    async def _on_conditions_alert(self, snapshot) -> None:
        """Handle conditions threshold crossing alert."""
        if not self.telegram:
            return
        if self.research_enabled and not self.research_alerts_enabled:
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
        if self.telegram and (not self.research_enabled or self.research_daily_review_enabled):
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
                change_5d_pct = 0.0
                if self.db:
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
                    past = await self.db.get_price_at_or_before(
                        exchange='bybit',
                        asset='BTCUSDT',
                        price_type='spot',
                        cutoff_timestamp=cutoff,
                    )
                    past_price = past.get('price') if past else None
                    current_price = ticker.get('last_price', 0)
                    if past_price and current_price:
                        change_5d_pct = ((current_price - past_price) / past_price) * 100
                return {
                    'price': ticker.get('last_price', 0),
                    'change_24h_pct': ticker.get('price_24h_pcnt', 0) * 100,
                    'change_5d_pct': change_5d_pct,
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

    async def _get_signal_status(self) -> Dict[str, Any]:
        """Get IBIT/BITO signal checklist for Telegram."""
        status: Dict[str, Any] = {}
        ibit_detector = self.detectors.get('ibit')
        if ibit_detector:
            status['IBIT'] = ibit_detector.get_signal_checklist()
        bito_detector = self.detectors.get('bito')
        if bito_detector:
            status['BITO'] = bito_detector.get_signal_checklist()
        return status

    async def _get_research_status(self) -> Dict[str, Any]:
        """Get research mode telemetry for Telegram."""
        if not self.paper_trader_farm:
            return {}
        aggregate = self.paper_trader_farm.get_aggregate_pnl()
        status = self.paper_trader_farm.get_status_summary()
        data_ready = False
        ibit_detector = self.detectors.get('ibit')
        bito_detector = self.detectors.get('bito')
        if ibit_detector:
            checklist = ibit_detector.get_signal_checklist()
            data_ready = data_ready or (checklist.get('has_btc_iv') and checklist.get('has_ibit_data'))
        if bito_detector:
            checklist = bito_detector.get_signal_checklist()
            data_ready = data_ready or (checklist.get('has_btc_iv') and checklist.get('has_ibit_data'))
        return {
            'research_enabled': self.research_enabled,
            'evaluation_interval_seconds': self.research_config.get('evaluation_interval_seconds', 60),
            'last_run': self._research_last_run.isoformat() if self._research_last_run else None,
            'last_symbol': self._research_last_symbol,
            'last_entered': self._research_last_entered,
            'aggregate': aggregate,
            'status': status,
            'data_ready': data_ready,
        }

    async def _get_data_status(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Collect data freshness signals for key tables."""
        tables = {
            "Detections": ("detections", 24 * 60 * 60),
            "Options IV": ("options_iv", 2 * 60 * 60),
            "Prices": ("price_snapshots", 10 * 60),
            "Health": ("system_health", 10 * 60),
        }
        latest = await self.db.get_latest_timestamps(
            [t[0] for t in tables.values()]
        )
        now = datetime.now(timezone.utc)
        eastern = ZoneInfo("America/New_York")
        age_since_start = int((now - self._start_time).total_seconds())
        status: Dict[str, Dict[str, Optional[str]]] = {}
        for label, (table, threshold) in tables.items():
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
        
        if 'volatility' in self.detectors:
            await self.detectors['volatility'].analyze(data)
    
    async def _on_yahoo_update(self, data: Dict) -> None:
        """Handle IBIT/BITO price update from Yahoo Finance."""
        symbol = data.get('symbol')
        if not symbol:
            return
        data['source'] = 'yahoo'
        if symbol == 'IBIT' and 'ibit' in self.detectors:
            detection = await self.detectors['ibit'].analyze(data)
            if detection:
                await self._send_alert(detection)
        elif symbol == 'BITO' and 'bito' in self.detectors:
            detection = await self.detectors['bito'].analyze(data)
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
        if self.research_enabled and not self.research_alerts_enabled:
            return
        
        op_type = detection.get('opportunity_type')
        tier = detection.get('alert_tier', 2)
        
        self.logger.info(
            f"DETECTION: {op_type} - {detection.get('asset')} - "
            f"Edge: {detection.get('net_edge_bps', 0):.1f} bps (tier {tier})"
        )
        
        if op_type == 'options_iv':
            data = detection.get('detection_data', {})
            if data.get('is_data_only'):
                return
            await self.telegram.send_iv_alert(detection)
        elif op_type == 'ibit_options':
            await self._send_ibit_alert(detection)
    
    async def _send_paper_notification(self, message: str) -> None:
        """Send paper trade notification via Telegram."""
        if self.telegram and (not self.research_enabled or self.research_alerts_enabled):
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
                        if currency == 'BTC':
                            for key in ('ibit', 'bito'):
                                detector = self.detectors.get(key)
                                if detector:
                                    detector.update_btc_iv(data.get('atm_iv', 0))
                            
            except Exception as e:
                self.logger.error(f"Deribit polling error: {e}")
            
            await asyncio.sleep(interval)
    
    async def _get_current_spread_prices(self) -> Dict[str, Dict[str, float]]:
        """Get current spread prices for open positions to evaluate exits."""
        prices: Dict[str, Dict[str, float]] = {}
        if not self.paper_trader_farm:
            return prices

        for trader_id, trader in self.paper_trader_farm.active_traders.items():
            for trade in trader.open_positions:
                symbol = trade.symbol
                if symbol not in prices:
                    prices[symbol] = {}

                current_price = None
                detector = self.detectors.get(symbol.lower())
                if detector and hasattr(detector, '_current_ibit_data') and detector._current_ibit_data:
                    current_price = detector._current_ibit_data.get('price')
                if current_price is None:
                    continue

                try:
                    if '/' not in trade.strikes:
                        continue
                    parts = trade.strikes.replace('$', '').split('/')
                    short_strike = float(parts[0])
                    long_strike = float(parts[1])
                    spread_width = short_strike - long_strike
                    entry_credit = trade.entry_credit

                    if current_price >= short_strike:
                        otm_pct = (current_price - short_strike) / current_price
                        decay_factor = max(0.0, 1.0 - otm_pct * 10)
                        current_value = entry_credit * decay_factor * 0.3
                    elif current_price <= long_strike:
                        current_value = spread_width
                    else:
                        itm_pct = (short_strike - current_price) / spread_width
                        current_value = entry_credit + (spread_width - entry_credit) * itm_pct

                    prices[symbol][trade.id] = max(0.0, current_value)
                except (ValueError, IndexError, ZeroDivisionError):
                    continue

        return prices

    async def _run_market_close_snapshot(self) -> None:
        """Take gap risk snapshots at market close (4 PM ET) on weekdays."""
        if not self.gap_risk_tracker:
            return
        eastern = ZoneInfo("America/New_York")
        last_snapshot_date = None

        while self._running:
            try:
                now_et = datetime.now(eastern)
                today = now_et.date()
                is_weekday = now_et.weekday() < 5
                past_close = now_et.hour >= 16

                if is_weekday and past_close and last_snapshot_date != today:
                    btc_price_data = await self._get_btc_price()
                    btc_price = btc_price_data.get('price', 0) if btc_price_data else 0

                    ibit_price = None
                    bito_price = None
                    ibit_det = self.detectors.get('ibit')
                    if ibit_det and hasattr(ibit_det, '_current_ibit_data') and ibit_det._current_ibit_data:
                        ibit_price = ibit_det._current_ibit_data.get('price')
                    bito_det = self.detectors.get('bito')
                    if bito_det and hasattr(bito_det, '_current_ibit_data') and bito_det._current_ibit_data:
                        bito_price = bito_det._current_ibit_data.get('price')

                    if btc_price > 0:
                        await self.gap_risk_tracker.snapshot_market_close(
                            btc_price=btc_price,
                            ibit_price=ibit_price,
                            bito_price=bito_price,
                        )
                        last_snapshot_date = today
                        self.logger.info(f"Market close snapshot taken: BTC=${btc_price:,.0f}")
            except Exception as e:
                self.logger.error(f"Market close snapshot error: {e}")
            await asyncio.sleep(300)

    async def _run_research_farm(self) -> None:
        """Continuously evaluate farm signals for research."""
        if not self.paper_trader_farm:
            return
        interval = int(self.research_config.get('evaluation_interval_seconds', 60))
        interval = max(10, interval)
        while self._running and self.research_enabled:
            try:
                # P0 fix: Check exits on every iteration
                current_prices = await self._get_current_spread_prices()
                if current_prices:
                    closed_trades = await self.paper_trader_farm.check_exits(current_prices)
                    if closed_trades:
                        self.logger.info(f"Research farm: {len(closed_trades)} trades closed via exit checks")

                # P0 fix: Check expirations
                eastern = ZoneInfo("America/New_York")
                today_et = datetime.now(eastern).strftime('%Y-%m-%d')
                expired_trades = await self.paper_trader_farm.expire_positions(today_et)
                if expired_trades:
                    self.logger.info(f"Research farm: {len(expired_trades)} trades expired")

                # Evaluate new signals
                conditions = {}
                if self.conditions_monitor:
                    conditions = await self.conditions_monitor.get_current_conditions()
                conditions_score = int(conditions.get('score', 5))
                conditions_label = conditions.get('warmth_label', 'neutral')
                btc_change = float(conditions.get('btc_change', 0))
                btc_change_5d = float(conditions.get('btc_change_5d', 0))
                timestamp = datetime.now(timezone.utc).isoformat()

                total_entered = 0
                for key in ('ibit', 'bito'):
                    detector = self.detectors.get(key)
                    if not detector:
                        continue
                    signal = detector.get_research_signal(
                        conditions_score=conditions_score,
                        conditions_label=conditions_label,
                        btc_change_24h_pct=btc_change,
                        btc_change_5d_pct=btc_change_5d,
                        timestamp=timestamp,
                    )
                    if not signal:
                        continue
                    trades = await self.paper_trader_farm.evaluate_signal(
                        symbol=signal['symbol'],
                        signal_data=signal,
                    )
                    total_entered += len(trades)
                    self._research_last_symbol = signal['symbol']

                self._research_last_run = datetime.now(timezone.utc)
                self._research_last_entered = total_entered
                await self._maybe_promote_configs()
            except Exception as e:
                self.logger.error(f"Research farm loop error: {e}")
            await asyncio.sleep(interval)

    async def _maybe_promote_configs(self) -> None:
        """Promote top-performing configs after research window."""
        if self._research_promoted:
            return
        if not self.research_config.get('auto_promote_enabled', False):
            return
        promote_after_days = int(self.research_config.get('promote_after_days', 60))
        days_since_start = (datetime.now(timezone.utc) - self._start_time).days
        if days_since_start < promote_after_days:
            return

        window_days = int(self.research_config.get('promotion_window_days', promote_after_days))
        min_trades = int(self.research_config.get('promotion_min_trades', 30))
        min_total_pnl = float(self.research_config.get('promotion_min_total_pnl', 250.0))
        min_avg_pnl = float(self.research_config.get('promotion_min_avg_pnl', 5.0))
        min_win_rate = float(self.research_config.get('promotion_min_win_rate', 55.0))
        top_n = int(self.research_config.get('promotion_top_n', 5))

        performance = await self.db.get_trader_performance(days=window_days)
        eligible = [
            p for p in performance
            if p.get('total_trades', 0) >= min_trades
            and p.get('total_pnl', 0) >= min_total_pnl
            and p.get('avg_pnl', 0) >= min_avg_pnl
            and p.get('win_rate', 0) >= min_win_rate
        ]
        if not eligible:
            return
        eligible.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)
        promoted_ids = [p['trader_id'] for p in eligible[:top_n]]
        if self.paper_trader_farm:
            self.paper_trader_farm.set_promoted_traders(promoted_ids)
        self._research_promoted = True

        if self.research_config.get('live_mode_after_promotion', False):
            self.research_enabled = False
            for key in ('ibit', 'bito'):
                detector = self.detectors.get(key)
                if detector:
                    detector.paper_trading_enabled = True
    
    async def _health_check(self) -> None:
        """Periodic health check and status logging."""
        interval = 300  # 5 minutes
        
        while self._running:
            # Log health status
            status = {
                'bybit_connected': self.bybit_ws.is_connected if self.bybit_ws else False,
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
        if self.yahoo_client:
            self._tasks.append(asyncio.create_task(self.yahoo_client.poll(interval_seconds=60)))
        
        self._tasks.append(asyncio.create_task(self._poll_deribit()))
        self._tasks.append(asyncio.create_task(self._health_check()))
        if self.research_enabled:
            self._tasks.append(asyncio.create_task(self._run_research_farm()))

        # P3: Automate gap risk snapshots at market close
        if self.gap_risk_tracker:
            self._tasks.append(asyncio.create_task(self._run_market_close_snapshot()))

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
        if self.deribit_client:
            await self.deribit_client.close()
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
