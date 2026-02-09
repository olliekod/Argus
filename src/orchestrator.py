"""
Argus Market Monitor - Module Loader / Orchestrator
====================================================

Coordinates all connectors, detectors, and alerts via a central
Pub/Sub event bus.  Reads ``ARGUS_MODE`` once at boot to decide
between **collector** (default — observe only) and **live** modes.
This is the canonical orchestrator entrypoint; legacy variants have been removed.
"""

import asyncio
import os
import signal
import time
import traceback
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.config import load_all_config, validate_secrets, get_secret
from .core.database import Database
from .core.logger import setup_logger, get_logger, uptime_seconds
from .core.bus import EventBus
from .core.events import (
    HeartbeatEvent,
    MinuteTickEvent,
    BarEvent,
    QuoteEvent,
    MetricEvent,
    SignalEvent,
    TOPIC_MARKET_BARS,
    TOPIC_MARKET_METRICS,
    TOPIC_MARKET_QUOTES,
    TOPIC_SIGNALS,
    TOPIC_SIGNALS_RAW,
    TOPIC_OPTIONS_CHAINS,
    TOPIC_SYSTEM_HEARTBEAT,
    TOPIC_SYSTEM_MINUTE_TICK,
)
from .core.status_tracker import ActivityStatusTracker
from .core.bar_builder import BarBuilder
from .core.persistence import PersistenceManager
from .core.feature_builder import FeatureBuilder
from .core.regime_detector import RegimeDetector
from .core.gap_risk_tracker import GapRiskTracker
from .core.reddit_monitor import RedditMonitor
from .core.conditions_monitor import ConditionsMonitor
from .connectors.bybit_ws import BybitWebSocket
from .connectors.deribit_client import DeribitClient
from .connectors.yahoo_client import YahooFinanceClient
from .connectors.alpaca_client import AlpacaDataClient
from .connectors.alpaca_options import AlpacaOptionsConnector, AlpacaOptionsConfig
from .strategies.spread_generator import SpreadCandidateGenerator, SpreadGeneratorConfig
from .connectors.polymarket_gamma import PolymarketGammaClient
from .connectors.polymarket_clob import PolymarketCLOBClient
from .connectors.polymarket_watchlist import PolymarketWatchlistService
from .detectors.options_iv_detector import OptionsIVDetector
from .detectors.volatility_detector import VolatilityDetector
from .detectors.ibit_detector import IBITDetector
from .alerts.telegram_bot import TelegramBot
from .analysis.daily_review import DailyReview
from .analysis.uniformity_monitor import run_uniformity_check
from .trading.paper_trader_farm import PaperTraderFarm
from .core.query_layer import QueryLayer
from .dashboard.web import ArgusWebDashboard
from .soak.guards import SoakGuardian
from .soak.tape import TapeRecorder
from .soak.resource_monitor import ResourceMonitor
from .soak.summary import build_soak_summary


class CollectorModeViolation(RuntimeError):
    """Raised when trade-execution code is invoked in collector mode."""


def _guard_collector_mode(mode: str):
    """Fail-fast if any module attempts trade execution in collector mode."""
    if mode == "collector":
        raise CollectorModeViolation(
            "Trade execution attempted while ARGUS_MODE=collector. "
            "Set ARGUS_MODE=live to enable trading."
        )



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

        # ── Global mode (read once at boot) ─────────────
        self.mode: str = os.environ.get("ARGUS_MODE", "collector").lower()

        # Recent log lines ring buffer for dashboard (must be before logger setup)
        self._recent_logs: deque = deque(maxlen=200)

        # Setup logging
        log_level = self.config.get('system', {}).get('log_level', 'INFO')
        setup_logger('argus', level=log_level, ring_buffer=self._recent_logs)
        self.logger = get_logger('orchestrator')

        # ── Collector-mode banner ───────────────────────
        if self.mode == "collector":
            banner = (
                "\n"
                "╔══════════════════════════════════════════╗\n"
                "║  TRADING DISABLED  (COLLECTOR MODE)      ║\n"
                "║  Set ARGUS_MODE=live to enable trading.   ║\n"
                "╚══════════════════════════════════════════╝"
            )
            self.logger.warning(banner)
        else:
            self.logger.info(f"ARGUS_MODE = {self.mode}")

        # Validate secrets
        issues = validate_secrets(self.secrets)
        if issues:
            self.logger.warning("Configuration issues:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")

        # Initialize database
        db_path = Path(self.config.get('system', {}).get('database_path', 'data/argus.db'))
        self.db = Database(str(db_path))

        # ── Event bus + modules ─────────────────────────
        self.event_bus = EventBus()
        self.bar_builder: Optional[BarBuilder] = None
        self.persistence: Optional[PersistenceManager] = None
        self.query_layer: Optional[QueryLayer] = None
        self.feature_builder: Optional[FeatureBuilder] = None
        self.regime_detector: Optional[RegimeDetector] = None
        self._provider_names = [
            "bybit",
            "deribit",
            "yahoo",
            "alpaca",
            "binance",
            "okx",
            "coinglass",
            "coinbase",
            "ibit_options",
            "polymarket_gamma",
            "polymarket_clob",
        ]
        self._activity_tracker = ActivityStatusTracker(
            provider_names=self._provider_names,
            boot_ts=time.time(),
        )

        # Polymarket connectors
        self.polymarket_gamma: Optional[PolymarketGammaClient] = None
        self.polymarket_clob: Optional[PolymarketCLOBClient] = None
        self.polymarket_watchlist: Optional[PolymarketWatchlistService] = None

        # Get symbols to monitor
        self.symbols = self.config.get('symbols', {}).get('monitored', [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
            'ARB/USDT:USDT', 'DOGE/USDT:USDT'
        ])
        
        # Components will be initialized in setup()
        self.bybit_ws: Optional[BybitWebSocket] = None
        self.deribit_client: Optional[DeribitClient] = None
        self.yahoo_client: Optional[YahooFinanceClient] = None
        self.alpaca_client: Optional[AlpacaDataClient] = None
        self.alpaca_options: Optional[AlpacaOptionsConnector] = None
        self.spread_generator: Optional[SpreadCandidateGenerator] = None
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
        self._research_last_error: Optional[str] = None
        self._research_consecutive_errors: int = 0
        self._exit_monitor_last_run: Optional[datetime] = None
        self._research_promoted: bool = False
        self.research_config: Dict[str, Any] = self.config.get('research', {})
        self.research_enabled = self.research_config.get('enabled', False)
        self.research_alerts_enabled = self.research_config.get('alerts_enabled', False)
        self.research_daily_review_enabled = self.research_config.get('daily_review_enabled', False)
        self._deribit_traceback_ts = 0.0

        # Market session tracking
        self._market_was_open: bool = False
        self._last_market_open_date = None
        self._last_market_close_date = None
        self._today_opened: int = 0
        self._today_closed: int = 0
        self._today_expired: int = 0

        # Boot phase timing
        self._boot_phases: Dict[str, float] = {}
        self._boot_start = time.monotonic()

        # Cached status snapshots for fast dashboard/telegram responses
        self._status_snapshot: Dict[str, Any] = {}
        self._status_snapshot_ts: float = 0.0
        self._status_snapshot_lock = asyncio.Lock()
        self._status_snapshot_interval = int(
            self.config.get('monitoring', {}).get('status_snapshot_interval', 10)
        )
        self._zombies_snapshot: Dict[str, Any] = {'zombies': [], 'total': 0, 'report': 'No data yet'}
        self._zombies_snapshot_ts: float = 0.0
        self._zombies_snapshot_interval = int(
            self.config.get('monitoring', {}).get('zombies_snapshot_interval', 120)
        )

        # Dashboard
        dash_cfg = self.config.get('dashboard', {})
        self.dashboard: Optional[ArgusWebDashboard] = None
        if dash_cfg.get('enabled', True):
            self.dashboard = ArgusWebDashboard(
                host=dash_cfg.get('host', '127.0.0.1'),
                port=dash_cfg.get('port', 8777),
            )

        # ── Soak-test hardening components ────────────────
        soak_cfg = self.config.get('soak', {})
        db_path = str(self.config.get('system', {}).get('database_path', 'data/argus.db'))
        self.resource_monitor = ResourceMonitor(
            db_path=db_path,
            log_ring=self._recent_logs,
        )
        self.soak_guardian = SoakGuardian(
            config=soak_cfg.get('guards', {}),
            alert_callback=None,  # wired after telegram setup
        )
        tape_cfg = soak_cfg.get('tape', {})
        tape_symbols = set(tape_cfg.get('symbols', [])) or None
        self.tape_recorder = TapeRecorder(
            enabled=tape_cfg.get('enabled', False),
            symbols=tape_symbols,
            maxlen=int(tape_cfg.get('maxlen', 100_000)),
        )
        # Last component heartbeat timestamps (for guard checks)
        self._component_heartbeat_ts: Dict[str, float] = {}

        # Market hours config
        self._mh_cfg = self.config.get('market_hours', {})

        self.logger.info("Argus Orchestrator initialized")
    
    def _phase(self, name: str):
        """Record a boot phase's elapsed time."""
        elapsed = time.monotonic() - self._boot_start
        self._boot_phases[name] = round(elapsed, 2)
        self.logger.info(f"[BOOT] {name}: {elapsed:.2f}s")

    def _format_boot_phases(self) -> str:
        """Format boot phases for display."""
        lines = []
        prev = 0.0
        for name, ts in self._boot_phases.items():
            delta = ts - prev
            lines.append(f"  {name}: {ts:.1f}s (delta {delta:.1f}s)")
            prev = ts
        return "\n".join(lines)

    def _note_detector_activity(self, detector: str, *, kind: str = "event") -> None:
        self._activity_tracker.record_detector_event(
            detector, event_ts=time.time(), kind=kind
        )

    def _note_detector_signal(self, event: SignalEvent) -> None:
        detector = getattr(event, "detector", None)
        if detector:
            self._activity_tracker.record_detector_signal(
                detector, event_ts=event.timestamp
            )

    def _note_provider_quote(self, event: QuoteEvent) -> None:
        provider = getattr(event, "source", None)
        if not provider:
            return
        self._activity_tracker.record_provider_event(
            provider,
            event_ts=event.event_ts,
            source_ts=event.source_ts or event.timestamp,
            kind="quote",
        )

    def _note_provider_bar(self, event: BarEvent) -> None:
        provider = getattr(event, "source", None)
        if not provider:
            return
        source_ts = (
            event.last_source_ts
            or event.first_source_ts
            or event.source_ts
            or event.timestamp
        )
        self._activity_tracker.record_provider_event(
            provider,
            event_ts=event.event_ts,
            source_ts=source_ts,
            kind="bar",
        )

    def _note_provider_metric(self, event: MetricEvent) -> None:
        provider = getattr(event, "source", None)
        if not provider:
            return
        self._activity_tracker.record_provider_event(
            provider,
            event_ts=event.event_ts,
            source_ts=event.source_ts or event.timestamp,
            kind="metric",
        )

    def _wire_activity_tracking(self) -> None:
        self.event_bus.subscribe(TOPIC_MARKET_QUOTES, self._note_provider_quote)
        self.event_bus.subscribe(TOPIC_MARKET_BARS, self._note_provider_bar)
        self.event_bus.subscribe(TOPIC_MARKET_METRICS, self._note_provider_metric)
        self.event_bus.subscribe(TOPIC_SIGNALS, self._note_detector_signal)

    def _record_detector_activity(self, detector: str, event_ts: float, kind: str) -> None:
        if kind == "signal":
            self._activity_tracker.record_detector_signal(detector, event_ts=event_ts)
        else:
            self._activity_tracker.record_detector_event(
                detector, event_ts=event_ts, kind=kind
            )

    def _sync_provider_registry(self) -> None:
        providers = {
            "bybit": self.bybit_ws,
            "deribit": self.deribit_client,
            "yahoo": self.yahoo_client,
            "alpaca": self.alpaca_client,
            "binance": getattr(self, "binance_ws", None),
            "okx": getattr(self, "okx_client", None),
            "coinglass": getattr(self, "coinglass_client", None),
            "coinbase": getattr(self, "coinbase_client", None),
            "ibit_options": getattr(self, "ibit_options_client", None),
            "polymarket_gamma": self.polymarket_gamma,
            "polymarket_clob": self.polymarket_clob,
        }
        for name in self._provider_names:
            self._activity_tracker.register_provider(
                name, configured=providers.get(name) is not None
            )

    async def setup(self) -> None:
        """Initialize all components with phase timing."""
        self.logger.info("Setting up Argus components...")
        self._boot_start = time.monotonic()

        # Phase 1: Config (already done in __init__)
        self._phase("config_loaded")

        # Phase 2: Database (full schema created on connect)
        await self.db.connect()
        self._phase("db_connected")

        # Phase 2b: Event bus modules
        loop = asyncio.get_running_loop()
        self.bar_builder = BarBuilder(self.event_bus)
        self.persistence = PersistenceManager(self.event_bus, self.db, loop)
        self.persistence.start()

        # Attach tape recorder to event bus (subscribes only if enabled)
        self.tape_recorder.attach(self.event_bus)

        # Phase 2c: Intelligence pipeline (downstream-only, safe in collector mode)
        self.feature_builder = FeatureBuilder(self.event_bus)
        self.regime_detector = RegimeDetector(self.event_bus)
        self._wire_activity_tracking()
        self._phase("event_bus_wired")

        # Phase 3: Connectors (with event bus)
        await self._setup_connectors()
        self._sync_provider_registry()
        self._phase("connectors_init")

        # Phase 4: Telegram
        await self._setup_telegram()
        self._phase("telegram_init")

        # Phase 5: Providers (gap risk, conditions, farm, review)
        # In collector mode, skip paper traders and exit monitors
        await self._setup_off_hours_monitoring()
        self._phase("providers_init")

        # Phase 6: Wire callbacks
        self._wire_telegram_callbacks()

        # Wire soak guardian alerts to telegram
        if self.telegram:
            self.soak_guardian._alert_cb = self._send_soak_alert

        # Phase 7: Detectors (with bus attachment)
        await self._setup_detectors()
        self._phase("detectors_init")

        # Phase 7b: Polymarket connectors (optional, fail-soft)
        await self._setup_polymarket()
        self._sync_provider_registry()
        self._phase("polymarket_init")

        # Phase 7c: Query layer (unified command interface)
        connectors_map: Dict[str, Any] = {
            "bybit": self.bybit_ws,
            "deribit": self.deribit_client,
            "yahoo": self.yahoo_client,
        }
        if self.polymarket_gamma:
            connectors_map["polymarket_gamma"] = self.polymarket_gamma
        if self.polymarket_clob:
            connectors_map["polymarket_clob"] = self.polymarket_clob
        self.query_layer = QueryLayer(
            bus=self.event_bus,
            db=self.db,
            detectors=self.detectors,
            connectors=connectors_map,
            bar_builder=self.bar_builder,
            persistence=self.persistence,
            feature_builder=self.feature_builder,
            regime_detector=self.regime_detector,
            provider_names=self._provider_names,
        )
        self._phase("query_layer_init")

        # Phase 7d: Start the event bus (all subscriptions registered)
        self.event_bus.start()
        self._phase("event_bus_started")

        # Phase 8: Zombie cleanup (skip full-table scan, use targeted query)
        await self._cleanup_zombie_positions()
        self._phase("zombie_cleanup")

        # Phase 9: Dashboard
        if self.dashboard:
            self.dashboard.set_callbacks(
                get_status=self._get_dashboard_system_status,
                get_pnl=self._get_pnl_summary,
                get_farm_status=self._get_farm_status,
                get_providers=self._get_provider_statuses,
                run_command=self._run_dashboard_command,
                get_recent_logs=self._get_recent_logs_text,
                get_soak_summary=self._get_soak_summary,
                export_tape=self._export_tape,
            )
            await self.dashboard.start()
            self.dashboard.set_boot_phases(self._format_boot_phases())
            self._phase("dashboard_started")

        # Total
        total = time.monotonic() - self._boot_start
        self._boot_phases["ready"] = round(total, 2)
        self.logger.info(f"[BOOT] READY in {total:.1f}s")
        self.logger.info(f"Boot phases:\n{self._format_boot_phases()}")

        # Startup notification
        if self.telegram:
            eastern = ZoneInfo("America/New_York")
            now_et = datetime.now(eastern).strftime('%H:%M:%S %Z')
            farm_count = len(self.paper_trader_farm.trader_configs) if self.paper_trader_farm else 0
            dash_port = None
            if self.dashboard:
                dash_port = self.dashboard.port
            if dash_port is None:
                dash_port = self.config.get('dashboard', {}).get('port', 8777)
            startup_msg = (
                f"<b>Argus Online</b>\n"
                f"<i>{now_et}</i>\n\n"
                f"Detectors: {len(self.detectors)}\n"
                f"Farm: {farm_count:,} configs\n"
                f"Boot: {total:.1f}s\n"
                f"Dashboard: http://127.0.0.1:{dash_port}"
            )
            await self.telegram.send_message(startup_msg)

        self.logger.info("Setup complete!")
    
    async def _setup_connectors(self) -> None:
        """Initialize exchange connectors (with event bus wiring)."""
        # Bybit WebSocket (public - no auth needed)
        bybit_symbols = [s for s in self.symbols]
        self.bybit_ws = BybitWebSocket(
            symbols=bybit_symbols,
            on_ticker=self._on_bybit_ticker,
            on_funding=self._on_funding_update,
            event_bus=self.event_bus,
        )
        self.logger.info(f"Bybit WS configured for {len(bybit_symbols)} symbols")

        # Deribit REST client (public - no auth needed)
        # Use mainnet for real data
        self.deribit_client = DeribitClient(testnet=False, event_bus=self.event_bus)
        self.logger.info("Deribit client configured (mainnet)")

        # Yahoo Finance for equity/ETF quotes -> 1m bars via BarBuilder
        yahoo_cfg = self.config.get('exchanges', {}).get('yahoo', {})
        if yahoo_cfg.get('enabled', True):
            yahoo_symbols = yahoo_cfg.get('symbols', ['IBIT', 'BITO', 'SPY', 'QQQ', 'NVDA'])
            self.yahoo_client = YahooFinanceClient(
                symbols=yahoo_symbols,
                on_update=self._on_yahoo_update,
                event_bus=self.event_bus,
            )
            self.logger.info("Yahoo Finance client configured for %s", yahoo_symbols)
        else:
            self.logger.info("Yahoo Finance client disabled (set exchanges.yahoo.enabled=true)")

        # Alpaca Data Client for IBIT/BITO bars (runs in ALL modes - data only, no execution)
        alpaca_cfg = self.config.get('exchanges', {}).get('alpaca', {})
        if alpaca_cfg.get('enabled', False):
            alpaca_api_key = get_secret(self.secrets, 'alpaca', 'api_key')
            alpaca_api_secret = get_secret(self.secrets, 'alpaca', 'api_secret')
            if alpaca_api_key and alpaca_api_secret:
                self.alpaca_client = AlpacaDataClient(
                    api_key=alpaca_api_key,
                    api_secret=alpaca_api_secret,
                    symbols=alpaca_cfg.get('symbols', ['IBIT', 'BITO']),
                    event_bus=self.event_bus,
                    db=self.db,  # For restart dedupe initialization
                    poll_interval=int(alpaca_cfg.get('poll_interval_seconds', 60)),
                    overlap_seconds=int(alpaca_cfg.get('overlap_seconds', 120)),
                )
                self.logger.info(
                    "Alpaca Data client configured for %s (fixed poll_interval=%ds, overlap=%ds)",
                    alpaca_cfg.get('symbols', ['IBIT', 'BITO']),
                    alpaca_cfg.get('poll_interval_seconds', 60),
                    alpaca_cfg.get('overlap_seconds', 120),
                )
                
                # Alpaca Options Connector (Phase 3B)
                options_cfg = alpaca_cfg.get('options', {})
                if options_cfg.get('enabled', False):
                    self.alpaca_options = AlpacaOptionsConnector(
                        config=AlpacaOptionsConfig(
                            api_key=alpaca_api_key,
                            api_secret=alpaca_api_secret,
                            cache_ttl_seconds=int(options_cfg.get('poll_interval_seconds', 60)),
                        )
                    )
                    self._alpaca_options_symbols = options_cfg.get('symbols', ['IBIT', 'BITO'])
                    self._alpaca_options_poll_interval = int(options_cfg.get('poll_interval_seconds', 60))
                    self._alpaca_options_min_dte = int(options_cfg.get('min_dte', 7))
                    self._alpaca_options_max_dte = int(options_cfg.get('max_dte', 21))
                    
                    # SpreadCandidateGenerator (subscribes to options.chains)
                    self.spread_generator = SpreadCandidateGenerator(
                        strategy_id="PUT_SPREAD_V1",
                        config=SpreadGeneratorConfig(
                            min_dte=self._alpaca_options_min_dte,
                            max_dte=self._alpaca_options_max_dte,
                        ),
                        on_signal=self._on_spread_signal,
                    )
                    # Subscribe generator to chain snapshots
                    self.event_bus.subscribe(
                        TOPIC_OPTIONS_CHAINS,
                        self.spread_generator.on_chain_snapshot,
                    )
                    self.logger.info(
                        "Alpaca Options connector + SpreadGenerator configured for %s (poll=%ds, DTE=%d-%d)",
                        self._alpaca_options_symbols,
                        self._alpaca_options_poll_interval,
                        self._alpaca_options_min_dte,
                        self._alpaca_options_max_dte,
                    )
                else:
                    self.logger.info("Alpaca Options disabled (set alpaca.options.enabled=true)")
            else:
                self.logger.warning("Alpaca enabled but API keys not found in secrets.yaml")
        else:
            self.logger.info("Alpaca Data client disabled (set exchanges.alpaca.enabled=true to activate)")

    async def _setup_polymarket(self) -> None:
        """Initialize Polymarket connectors (optional, fail-soft)."""
        pm_cfg = self.config.get('polymarket', {})
        if not pm_cfg.get('enabled', False):
            self.logger.info("Polymarket integration disabled (set polymarket.enabled=true to activate)")
            return

        try:
            self.polymarket_gamma = PolymarketGammaClient(
                event_bus=self.event_bus,
                poll_interval=pm_cfg.get('gamma_poll_interval', 60),
            )
            await self.polymarket_gamma.start()

            self.polymarket_clob = PolymarketCLOBClient(
                event_bus=self.event_bus,
                poll_interval=pm_cfg.get('clob_poll_interval', 30),
            )
            await self.polymarket_clob.start()

            self.polymarket_watchlist = PolymarketWatchlistService(
                gamma_client=self.polymarket_gamma,
                clob_client=self.polymarket_clob,
                sync_interval=pm_cfg.get('watchlist_sync_interval', 300),
                max_watchlist=pm_cfg.get('max_watchlist', 50),
                min_volume=pm_cfg.get('min_volume', 10_000),
                keywords=pm_cfg.get('keywords', []),
            )
            await self.polymarket_watchlist.start()
            self.logger.info("Polymarket integration initialised (Gamma + CLOB + Watchlist)")
        except Exception:
            self.logger.exception("Polymarket setup failed (non-fatal, continuing)")
            self.polymarket_gamma = None
            self.polymarket_clob = None
            self.polymarket_watchlist = None

    async def _setup_detectors(self) -> None:
        """Initialize all detectors and attach to event bus."""
        thresholds = self.config.get('thresholds', {})
        is_collector = self.mode == "collector"

        def _register(detector) -> None:
            detector.set_activity_callback(self._record_detector_activity)
            self._activity_tracker.register_detector(detector.name)

        # Options IV detector (BTC options on Deribit - for research)
        iv_config = thresholds.get('options_iv', {})
        if iv_config.get('enabled', True):
            self.detectors['options_iv'] = OptionsIVDetector(iv_config, self.db)
            self.detectors['options_iv'].attach_bus(self.event_bus)
            _register(self.detectors['options_iv'])

        # Volatility detector
        vol_config = thresholds.get('volatility', {})
        if vol_config.get('enabled', True):
            self.detectors['volatility'] = VolatilityDetector(vol_config, self.db)
            self.detectors['volatility'].attach_bus(self.event_bus)
            _register(self.detectors['volatility'])

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
            self.detectors['ibit'].attach_bus(self.event_bus)
            _register(self.detectors['ibit'])
            # Wire up Telegram for paper trade notifications
            self.detectors['ibit'].set_telegram_callback(self._send_paper_notification)
            # Wire up farm if available and NOT in collector mode
            if self.paper_trader_farm and not is_collector:
                self.detectors['ibit'].set_paper_trader_farm(self.paper_trader_farm)
            if self.research_enabled or is_collector:
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
            self.detectors['bito'].attach_bus(self.event_bus)
            _register(self.detectors['bito'])
            self.detectors['bito'].set_telegram_callback(self._send_paper_notification)
            # Wire up farm if available and NOT in collector mode
            if self.paper_trader_farm and not is_collector:
                self.detectors['bito'].set_paper_trader_farm(self.paper_trader_farm)
            if self.research_enabled or is_collector:
                self.detectors['bito'].paper_trading_enabled = False

        if is_collector:
            self.logger.info("Collector mode: paper trading DISABLED for all detectors")

        self.logger.info(f"Initialized {len(self.detectors)} detectors (bus-attached)")
    
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
        
        # Paper Trader Farm with guardrails from config
        # Skip in collector mode to avoid millions of trader configs and GPU uploads
        if self.mode == "collector":
            self.paper_trader_farm = None
            self.logger.info(
                "PaperTraderFarm initialization SKIPPED (ARGUS_MODE=collector)"
            )
        else:
            farm_cfg = self.config.get('farm', {})
            self.paper_trader_farm = PaperTraderFarm(
                db=self.db,
                full_coverage=True,
                starting_balance=float(farm_cfg.get('default_starting_equity', 5000.0)),
                max_traders=int(farm_cfg.get('max_traders', 2_000_000)),
                max_open_positions_total=int(farm_cfg.get('max_open_positions_total', 500_000)),
                max_trades_per_minute=int(farm_cfg.get('max_trades_per_minute', 10_000)),
            )
            await self.paper_trader_farm.initialize()

            # Wire up data sources to paper trader farm
            self.paper_trader_farm.set_data_sources(
                get_conditions=self.conditions_monitor.get_current_conditions,
            )
            # Wire Telegram callback for runaway safety alerts
            self.paper_trader_farm.set_telegram_alert_callback(self._send_paper_notification)
            self.logger.info(f"Paper Trader Farm initialized with {len(self.paper_trader_farm.trader_configs):,} traders")

        # Wire up data sources to daily review (after farm is ready, if available)
        self.daily_review.set_data_sources(
            get_conditions=self.conditions_monitor.get_current_conditions,
            get_positions=(
                self.paper_trader_farm.get_positions_for_review
                if self.paper_trader_farm else None
            ),
            get_trade_stats=(
                self.paper_trader_farm.get_trade_activity_summary
                if self.paper_trader_farm else None
            ),
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
            get_dashboard=self._get_dashboard,
            get_zombies=self._get_zombies,
            get_followed=self._get_followed_traders,
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
        cached = self._get_snapshot_section('pnl')
        if cached:
            return cached
        await self._refresh_status_snapshot(force=True)
        return self._status_snapshot.get('pnl', {})
    
    async def _get_positions_summary(self) -> List[Dict]:
        """Get positions summary for Telegram /positions command."""
        cached = self._get_snapshot_section('positions')
        if cached is not None:
            return cached
        await self._refresh_status_snapshot(force=True)
        return self._status_snapshot.get('positions', [])

    async def _get_status_summary(self) -> Dict[str, Any]:
        """Get conditions plus data freshness for Telegram /status command."""
        conditions = {}
        if self.conditions_monitor:
            conditions = await self.conditions_monitor.get_current_conditions()
        data_status = self._get_snapshot_section('data_status')
        if data_status is None:
            await self._refresh_status_snapshot(force=True)
            data_status = self._status_snapshot.get('data_status', {})
        conditions['data_status'] = data_status
        return conditions

    async def _get_farm_status(self) -> Dict[str, Any]:
        """Get paper trader farm status summary for Telegram."""
        cached = self._get_snapshot_section('farm')
        if cached:
            return cached
        await self._refresh_status_snapshot(force=True)
        return self._status_snapshot.get('farm', {})

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
            'consecutive_errors': self._research_consecutive_errors,
            'last_error': self._research_last_error,
            'aggregate': aggregate,
            'status': status,
            'data_ready': data_ready,
        }

    async def _get_zombies(self) -> Dict[str, Any]:
        """Detect zombies using the farm's 7-14 DTE-aligned detection logic."""
        if not self.paper_trader_farm:
            return {'zombies': [], 'total': 0, 'report': 'Farm not initialized'}
        await self._refresh_zombies_snapshot(force=False)
        return self._zombies_snapshot

    async def _get_followed_traders(self) -> List[Dict]:
        """Get the followed traders list from DB."""
        return await self.db.get_followed_traders()

    async def _get_dashboard(self) -> Dict[str, Any]:
        """Get full system dashboard data for Telegram /dashboard command."""
        now = datetime.now(timezone.utc)
        eastern = ZoneInfo("America/New_York")
        now_et = datetime.now(eastern)
        uptime_s = int((now - self._start_time).total_seconds())
        hours, remainder = divmod(uptime_s, 3600)
        minutes, _ = divmod(remainder, 60)

        # Task health
        def _age(ts):
            if not ts:
                return None
            return int((now - ts).total_seconds())

        research_age = _age(self._research_last_run)
        exit_age = _age(self._exit_monitor_last_run)
        health_age = _age(self._last_health_check)

        # Data freshness
        data_status = self._get_snapshot_section('data_status')
        if data_status is None:
            await self._refresh_status_snapshot(force=True)
            data_status = self._status_snapshot.get('data_status', {})

        # Farm stats
        farm = self.paper_trader_farm
        active_traders = len(farm.active_traders) if farm else 0
        open_positions = sum(
            len(t.open_positions) for t in farm.active_traders.values()
        ) if farm else 0
        total_configs = len(farm.trader_configs) if farm else 0

        # Market status
        is_weekday = now_et.weekday() < 5
        market_open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        market_open = is_weekday and market_open_time <= now_et <= market_close_time

        # Conditions
        conditions = {}
        if self.conditions_monitor:
            conditions = await self.conditions_monitor.get_current_conditions()

        return {
            'uptime': f"{hours}h {minutes}m",
            'market_open': market_open,
            'market_time_et': now_et.strftime('%H:%M %Z'),
            'conditions_score': conditions.get('score', 'N/A'),
            'conditions_label': conditions.get('warmth_label', 'N/A'),
            'data_status': data_status,
            'research_loop_age_s': research_age,
            'research_errors': self._research_consecutive_errors,
            'research_last_error': self._research_last_error,
            'exit_monitor_age_s': exit_age,
            'health_check_age_s': health_age,
            'total_configs': total_configs,
            'active_traders': active_traders,
            'open_positions': open_positions,
            'today_opened': self._today_opened,
            'today_closed': self._today_closed,
            'today_expired': self._today_expired,
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
            self._note_detector_activity(
                self.detectors['volatility'].name, kind="metric"
            )
            await self.detectors['volatility'].analyze(data)
    
    async def _on_yahoo_update(self, data: Dict) -> None:
        """Handle IBIT/BITO price update from Yahoo Finance."""
        symbol = data.get('symbol')
        if not symbol:
            return
        data['source'] = 'yahoo'
        if symbol == 'IBIT' and 'ibit' in self.detectors:
            self._note_detector_activity(
                self.detectors['ibit'].name, kind="metric"
            )
            detection = await self.detectors['ibit'].analyze(data)
            if detection:
                await self._send_alert(detection)
        elif symbol == 'BITO' and 'bito' in self.detectors:
            self._note_detector_activity(
                self.detectors['bito'].name, kind="metric"
            )
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
                            self._note_detector_activity(
                                self.detectors['options_iv'].name, kind="metric"
                            )
                            detection = await self.detectors['options_iv'].analyze(data)
                            if detection:
                                await self._send_alert(detection)
                        
                        # Feed BTC IV to IBIT detector
                        if currency == 'BTC':
                            for key in ('ibit', 'bito'):
                                detector = self.detectors.get(key)
                                if detector:
                                    self._note_detector_activity(detector.name, kind="metric")
                                    detector.update_btc_iv(data.get('atm_iv', 0))
                            
            except Exception as e:
                self.logger.error(f"Deribit polling error: {e}")
                now = time.time()
                if (now - self._deribit_traceback_ts) >= 60:
                    self._deribit_traceback_ts = now
                    self.logger.error("Deribit polling traceback:\n%s", traceback.format_exc())
            
            await asyncio.sleep(interval)
    
    def _on_spread_signal(self, signal) -> None:
        """Handle spread signal from SpreadCandidateGenerator.
        
        Emits SignalEvent to signals.raw with strategy_id=PUT_SPREAD_V1.
        """
        from .core.signals import SignalEvent as Phase3Signal, signal_to_dict
        
        # Convert to Phase 3 SignalEvent if needed
        if hasattr(signal, 'signal_id'):
            # Already a SignalEvent-like object
            self.event_bus.publish(TOPIC_SIGNALS_RAW, signal)
            self.logger.debug(
                "Spread signal emitted: %s %s credit=%.2f",
                signal.symbol if hasattr(signal, 'symbol') else "?",
                signal.direction if hasattr(signal, 'direction') else "?",
                signal.metadata.get('credit', 0) if hasattr(signal, 'metadata') else 0,
            )
    
    async def _poll_options_chains(self) -> None:
        """Poll Alpaca for options chain snapshots.
        
        Publishes OptionChainSnapshotEvent to options.chains topic.
        SpreadCandidateGenerator is already subscribed via event bus.
        """
        if not self.alpaca_options:
            return
        
        interval = getattr(self, '_alpaca_options_poll_interval', 60)
        symbols = getattr(self, '_alpaca_options_symbols', ['IBIT', 'BITO'])
        min_dte = getattr(self, '_alpaca_options_min_dte', 7)
        max_dte = getattr(self, '_alpaca_options_max_dte', 21)
        
        # Rate-limit warning aggregation
        last_warning_ts = 0
        warning_count = 0
        
        while self._running:
            try:
                for symbol in symbols:
                    # Get expirations in DTE range
                    expirations = await self.alpaca_options.get_expirations_in_range(
                        symbol, min_dte=min_dte, max_dte=max_dte
                    )
                    
                    for exp_date, dte in expirations:
                        snapshot = await self.alpaca_options.build_chain_snapshot(
                            symbol, exp_date
                        )
                        if snapshot:
                            # Publish to event bus (TapeRecorder + SpreadGenerator subscribe)
                            self.event_bus.publish(TOPIC_OPTIONS_CHAINS, snapshot)
                            self.logger.debug(
                                "Options chain: %s exp=%s DTE=%d puts=%d calls=%d",
                                symbol, exp_date, dte,
                                len(snapshot.puts), len(snapshot.calls),
                            )
                        else:
                            warning_count += 1
                            now = time.time()
                            if now - last_warning_ts > 60:
                                self.logger.warning(
                                    "Empty chain for %s exp=%s (suppressed %d similar)",
                                    symbol, exp_date, warning_count,
                                )
                                last_warning_ts = now
                                warning_count = 0
                            
            except Exception as e:
                self.logger.error("Options polling error: %s", e)
            
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

                    # Guard: skip positions with invalid spread width or credit
                    if spread_width <= 0 or not entry_credit or entry_credit <= 0:
                        continue

                    if current_price >= short_strike:
                        otm_pct = (current_price - short_strike) / current_price if current_price > 0 else 0
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

    async def _cleanup_zombie_positions(self) -> None:
        """Close orphaned positions from previous runs that are still 'open' in DB.

        Uses close_timestamp (not closed_at) and close_reason columns which
        are created by the migration in paper_trader_farm._create_tables.
        """
        try:
            row = await self.db.fetch_one(
                "SELECT COUNT(*) as cnt FROM paper_trades WHERE status = 'open'"
            )
            zombie_count = row['cnt'] if row else 0
            if zombie_count == 0:
                return

            self.logger.info(f"Found {zombie_count:,} zombie positions from previous runs, marking as expired")
            now_ts = datetime.now(timezone.utc).isoformat()

            # Use close_timestamp (the actual column name) and close_reason (added by migration)
            try:
                await self.db.execute(
                    """UPDATE paper_trades SET status = 'expired',
                       close_reason = 'system_restart_cleanup',
                       close_timestamp = ?
                       WHERE status = 'open'""",
                    (now_ts,)
                )
            except Exception:
                # Fallback if close_reason column doesn't exist yet
                await self.db.execute(
                    """UPDATE paper_trades SET status = 'expired',
                       close_timestamp = ?
                       WHERE status = 'open'""",
                    (now_ts,)
                )
            self.logger.info(f"Cleaned up {zombie_count:,} zombie positions")
        except Exception as e:
            self.logger.error(f"Failed to cleanup zombie positions: {e}")

    async def _run_exit_monitor(self) -> None:
        """Independent task: check exits and expirations every 30 seconds.

        Decoupled from the research signal loop so exits still happen even
        if signal evaluation crashes.

        On exception, reports the error to the farm's runaway safety
        tracker so repeated failures can halt new entries.
        """
        if not self.paper_trader_farm:
            return
        interval = 30

        while self._running:
            try:
                # Check exits based on current prices
                current_prices = await self._get_current_spread_prices()
                if current_prices:
                    closed_trades = await self.paper_trader_farm.check_exits(current_prices)
                    if closed_trades:
                        n = len(closed_trades)
                        self._today_closed += n
                        self.logger.info(f"Exit monitor: {n} trades closed")

                # Check expirations
                eastern = ZoneInfo("America/New_York")
                today_et = datetime.now(eastern).strftime('%Y-%m-%d')
                expired_trades = await self.paper_trader_farm.expire_positions(today_et)
                if expired_trades:
                    n = len(expired_trades)
                    self._today_expired += n
                    self.logger.info(f"Exit monitor: {n} trades expired")

                self._exit_monitor_last_run = datetime.now(timezone.utc)
            except Exception as e:
                self.logger.error(f"Exit monitor error: {e}")
                # Feed error into farm runaway safety tracker
                if self.paper_trader_farm:
                    self.paper_trader_farm.record_exit_error()
            await asyncio.sleep(interval)

    async def _run_research_farm(self) -> None:
        """Continuously evaluate farm signals for research.

        Exit checking is handled by the separate _run_exit_monitor task,
        so this loop only handles signal evaluation and new entries.
        """
        if not self.paper_trader_farm:
            return
        interval = int(self.research_config.get('evaluation_interval_seconds', 60))
        interval = max(10, interval)

        while self._running and self.research_enabled:
            # Always update the timestamp so we can see the loop is alive
            self._research_last_run = datetime.now(timezone.utc)
            try:
                # Gather market conditions
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
                    try:
                        signal = await asyncio.to_thread(
                            detector.get_research_signal,
                            conditions_score=conditions_score,
                            conditions_label=conditions_label,
                            btc_change_24h_pct=btc_change,
                            btc_change_5d_pct=btc_change_5d,
                            timestamp=timestamp,
                        )
                    except Exception as e:
                        self.logger.error(f"Signal generation failed for {key}: {e}")
                        continue
                    if not signal:
                        continue
                    trades = await self.paper_trader_farm.evaluate_signal(
                        symbol=signal['symbol'],
                        signal_data=signal,
                    )
                    entered = len(trades)
                    total_entered += entered
                    self._research_last_symbol = signal['symbol']

                    # Alert if any followed traders entered
                    if trades and self.telegram:
                        await self._alert_followed_trades(trades, signal)

                self._research_last_entered = total_entered
                self._today_opened += total_entered
                self._research_consecutive_errors = 0
                self._research_last_error = None
                await self._maybe_promote_configs()
                await self._run_uniformity_check()
            except Exception as e:
                self._research_consecutive_errors += 1
                self._research_last_error = str(e)
                self.logger.error(
                    f"Research farm error (#{self._research_consecutive_errors}): {e}"
                )
                # Alert via telegram if errors persist
                if self._research_consecutive_errors == 5 and self.telegram:
                    try:
                        await self.telegram.send_message(
                            f"⚠️ <b>Research Loop Degraded</b>\n"
                            f"5 consecutive errors.\n"
                            f"Last error: <code>{str(e)[:200]}</code>"
                        )
                    except Exception:
                        pass
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

    async def _alert_followed_trades(self, trades: list, signal: dict) -> None:
        """Send Telegram alert when followed traders enter positions."""
        if not self.telegram:
            return
        try:
            followed = await self.db.get_followed_traders()
            if not followed:
                return
            followed_ids = {t['trader_id'] for t in followed}
            matched = [t for t in trades if hasattr(t, 'trader_id') and t.trader_id in followed_ids]
            if not matched:
                return

            symbol = signal.get('symbol', '?')
            for trade in matched[:5]:  # Cap at 5 to avoid spam
                lines = [
                    f"⭐ <b>Followed Trader Entry</b>",
                    "",
                    f"Trader: <b>{trade.trader_id}</b>",
                    f"Symbol: {symbol}",
                    f"Strategy: {trade.strategy}",
                    f"Strikes: {trade.strikes}",
                    f"Expiry: {trade.expiry}",
                    f"Credit: ${trade.entry_credit:.2f}",
                    "",
                    f"<i>{datetime.now(ZoneInfo('America/New_York')).strftime('%H:%M:%S %Z')}</i>",
                ]
                await self.telegram.send_message("\n".join(lines))
        except Exception as e:
            self.logger.warning(f"Follow alert error: {e}")

    async def _run_uniformity_check(self) -> None:
        """Run uniformity check every N evaluations to detect convergence bugs."""
        if not hasattr(self, '_uniformity_check_count'):
            self._uniformity_check_count = 0
        self._uniformity_check_count += 1

        # Run every 10 evaluations to avoid overhead
        if self._uniformity_check_count % 10 != 0:
            return

        if not self.paper_trader_farm:
            return

        # Gather recent trades from active traders
        recent_trades = []
        for trader_id, trader in self.paper_trader_farm.active_traders.items():
            for pos in trader.open_positions:
                recent_trades.append({
                    'trader_id': trader_id,
                    'strategy_type': pos.strategy if hasattr(pos, 'strategy') else '',
                    'strikes': pos.strikes if hasattr(pos, 'strikes') else '',
                    'expiry': pos.expiry if hasattr(pos, 'expiry') else '',
                    'entry_credit': pos.entry_credit if hasattr(pos, 'entry_credit') else 0,
                    'contracts': pos.contracts if hasattr(pos, 'contracts') else 0,
                })

        if len(recent_trades) < 20:
            return

        try:
            results = await run_uniformity_check(
                trades=recent_trades,
                db=self.db,
            )
            alerts = [r for r in results if r.get('is_alert')]
            if alerts and self.telegram:
                from .analysis.uniformity_monitor import format_uniformity_report
                report = format_uniformity_report(results)
                msg = f"⚠️ <b>Uniformity Alert</b>\n<pre>{report[:1500]}</pre>"
                await self.telegram.send_message(msg)
        except Exception as e:
            self.logger.warning(f"Uniformity check error: {e}")

    async def _health_check(self) -> None:
        """Periodic health check (5 min) and 60-second heartbeat summary."""
        health_interval = 300
        heartbeat_interval = int(self.config.get('monitoring', {}).get('heartbeat_interval', 60))
        last_heartbeat = 0.0
        last_health = 0.0

        while self._running:
            now = time.time()

            # 60-second heartbeat line
            if now - last_heartbeat >= heartbeat_interval:
                last_heartbeat = now
                try:
                    farm = self.paper_trader_farm
                    active_traders = len(farm.active_traders) if farm else 0
                    open_positions = sum(
                        len(t.open_positions) for t in farm.active_traders.values()
                    ) if farm else 0
                    bybit_health = self.bybit_ws.get_health_status() if self.bybit_ws else {}
                    bybit_connected = bybit_health.get('extras', {}).get('connected', False)
                    bybit_str = "connected" if bybit_connected else "disconnected"
                    msg_age = bybit_health.get('seconds_since_last_message')
                    msg_age_str = f"{msg_age:.0f}s" if msg_age is not None else "N/A"
                    db_size = os.path.getsize(str(self.db.db_path)) / (1024 * 1024) if self.db.db_path.exists() else 0

                    self.logger.info(
                        f"[HEARTBEAT] uptime={uptime_seconds():.0f}s "
                        f"bybit={bybit_str} last_msg={msg_age_str} "
                        f"traders={active_traders} positions={open_positions} "
                        f"db={db_size:.0f}MB"
                    )
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {e}")

            # 5-minute health check (DB write)
            if now - last_health >= health_interval:
                last_health = now
                try:
                    bybit_connected = self.bybit_ws.is_connected if self.bybit_ws else False
                    self._last_health_check = datetime.now(timezone.utc)
                    await self.db.insert_health_check(
                        component="bybit_ws",
                        status="connected" if bybit_connected else "disconnected",
                    )
                    await self.db.insert_health_check(
                        component="detectors",
                        status=f"active_{len(self.detectors)}",
                    )
                except Exception as e:
                    self.logger.error(f"Health check error: {e}")

            try:
                await self._refresh_status_snapshot()
            except Exception as e:
                self.logger.debug(f"Status snapshot refresh error: {e}")

            try:
                await self._refresh_zombies_snapshot()
            except Exception as e:
                self.logger.debug(f"Zombies snapshot refresh error: {e}")

            await asyncio.sleep(10)  # Check every 10s for heartbeat granularity

    async def _run_market_session_monitor(self) -> None:
        """Monitor market open/close transitions and send notifications."""
        eastern = ZoneInfo("America/New_York")

        while self._running:
            try:
                now_et = datetime.now(eastern)
                today = now_et.date()
                is_weekday = now_et.weekday() < 5
                market_open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                is_open = is_weekday and market_open_time <= now_et <= market_close_time

                # Detect market OPEN transition
                if is_open and not self._market_was_open and self._last_market_open_date != today:
                    self._last_market_open_date = today
                    self._today_opened = 0
                    self._today_closed = 0
                    self._today_expired = 0
                    await self._send_market_open_notification(now_et)

                # Detect market CLOSE transition
                if not is_open and self._market_was_open and self._last_market_close_date != today:
                    self._last_market_close_date = today
                    await self._send_market_close_notification(now_et)

                self._market_was_open = is_open
            except Exception as e:
                self.logger.error(f"Market session monitor error: {e}")
            await asyncio.sleep(30)

    async def _send_market_open_notification(self, now_et: datetime) -> None:
        """Send notification when market opens."""
        if not self.telegram:
            return
        conditions = {}
        if self.conditions_monitor:
            conditions = await self.conditions_monitor.get_current_conditions()
        score = conditions.get('score', 'N/A')
        label = conditions.get('warmth_label', 'N/A')
        btc_iv = conditions.get('btc_iv', 'N/A')

        active = len(self.paper_trader_farm.active_traders) if self.paper_trader_farm else 0
        open_positions = sum(
            len(t.open_positions) for t in self.paper_trader_farm.active_traders.values()
        ) if self.paper_trader_farm else 0

        lines = [
            f"🔔 <b>Market Open</b> — {now_et.strftime('%b %d, %Y')}",
            "",
            f"Conditions: {score}/10 {str(label).upper()}",
            f"BTC IV: {btc_iv}%",
        ]

        ibit_det = self.detectors.get('ibit')
        if ibit_det and hasattr(ibit_det, '_current_ibit_data') and ibit_det._current_ibit_data:
            lines.append(f"IBIT: ${ibit_det._current_ibit_data.get('price', 0):.2f}")
        bito_det = self.detectors.get('bito')
        if bito_det and hasattr(bito_det, '_current_ibit_data') and bito_det._current_ibit_data:
            lines.append(f"BITO: ${bito_det._current_ibit_data.get('price', 0):.2f}")

        lines += [
            "",
            f"Farm: {len(self.paper_trader_farm.trader_configs):,} configs" if self.paper_trader_farm else "Farm: N/A",
            f"Active traders: {active:,}",
            f"Open positions: {open_positions:,}",
        ]
        try:
            await self.telegram.send_message("\n".join(lines))
        except Exception as e:
            self.logger.error(f"Failed to send market open notification: {e}")

    async def _send_market_close_notification(self, now_et: datetime) -> None:
        """Send end-of-day summary when market closes."""
        if not self.telegram:
            return

        farm = self.paper_trader_farm
        aggregate = farm.get_aggregate_pnl() if farm else {}
        top_gains = await farm.get_top_unrealized(n=3) if farm else []

        lines = [
            f"🔔 <b>Market Close</b> — {now_et.strftime('%b %d, %Y')}",
            "",
            "<b>Today's Activity:</b>",
            f"• Opened: {self._today_opened:,}",
            f"• Closed: {self._today_closed:,}",
            f"• Expired: {self._today_expired:,}",
            f"• Realized P&L: ${aggregate.get('realized_pnl', 0):+.2f}",
            "",
        ]

        if top_gains:
            lines.append("<b>Top 3 Unrealized Gains:</b>")
            for i, g in enumerate(top_gains, 1):
                lines.append(
                    f"{i}. {g['strategy']} {g['symbol']} {g['strikes']} — "
                    f"${g['unrealized_pnl']:+.2f} ({g['pnl_pct']:+.1f}%)"
                )
            lines.append("")

        conditions = {}
        if self.conditions_monitor:
            conditions = await self.conditions_monitor.get_current_conditions()
        lines.append(
            f"Conditions at close: {conditions.get('score', 'N/A')}/10 "
            f"{str(conditions.get('warmth_label', 'N/A')).upper()}"
        )

        try:
            await self.telegram.send_message("\n".join(lines))
        except Exception as e:
            self.logger.error(f"Failed to send market close notification: {e}")
    
    # =========================================================================
    # Dashboard helper callbacks
    # =========================================================================

    def _get_snapshot_section(self, key: str) -> Optional[Any]:
        """Fetch a cached snapshot section if available."""
        return self._status_snapshot.get(key)

    async def _refresh_status_snapshot(self, force: bool = False) -> None:
        """Refresh cached dashboard/telegram status snapshot."""
        now = time.time()
        if not force and (now - self._status_snapshot_ts) < self._status_snapshot_interval:
            return

        async with self._status_snapshot_lock:
            now = time.time()
            if not force and (now - self._status_snapshot_ts) < self._status_snapshot_interval:
                return

            started = time.perf_counter()
            snapshot: Dict[str, Any] = {}
            try:
                db_stats = await self.db.get_db_stats()
                snapshot['system'] = {
                    'db_size_mb': db_stats.get('db_size_mb', 0),
                    'boot_phases': self._format_boot_phases(),
                }
                snapshot['db_stats'] = db_stats
            except Exception as e:
                self.logger.warning(f"Status snapshot system error: {e}")
                snapshot['system'] = {'db_size_mb': 0, 'boot_phases': self._format_boot_phases()}
                snapshot['db_stats'] = {'db_size_mb': 0, 'row_counts': {}}

            try:
                snapshot['providers'] = await self._get_provider_statuses()
            except Exception as e:
                self.logger.warning(f"Status snapshot providers error: {e}")
                snapshot['providers'] = {'error': str(e)}
            try:
                snapshot['detectors'] = await self._get_detector_statuses()
            except Exception as e:
                self.logger.warning(f"Status snapshot detectors error: {e}")
                snapshot['detectors'] = {'error': str(e)}
            try:
                if self.query_layer:
                    status_v2 = await self.query_layer.status()
                    snapshot['internal'] = status_v2.get('internal', {})
                    snapshot['bus'] = status_v2.get('bus', {})
                    snapshot['db'] = status_v2.get('db', {})
                else:
                    snapshot['internal'] = {}
                    snapshot['bus'] = {}
                    snapshot['db'] = {}
            except Exception as e:
                self.logger.warning(f"Status snapshot internal error: {e}")
                snapshot['internal'] = {}
                snapshot['bus'] = {}
                snapshot['db'] = {}
            if self.telegram:
                snapshot['internal']['telegram'] = self.telegram.get_status()

            try:
                snapshot['pnl'] = await self._compute_pnl_summary()
            except Exception as e:
                self.logger.warning(f"Status snapshot pnl error: {e}")
                snapshot['pnl'] = {'error': str(e)}

            try:
                snapshot['farm'] = await self._compute_farm_status()
            except Exception as e:
                self.logger.warning(f"Status snapshot farm error: {e}")
                snapshot['farm'] = {'error': str(e)}

            try:
                snapshot['positions'] = await self._compute_positions_summary()
            except Exception as e:
                self.logger.warning(f"Status snapshot positions error: {e}")
                snapshot['positions'] = []

            try:
                snapshot['data_status'] = await self._get_data_status()
            except Exception as e:
                self.logger.warning(f"Status snapshot data_status error: {e}")
                snapshot['data_status'] = {}

            try:
                snapshot['recent_logs'] = await self._read_recent_logs_text()
            except Exception as e:
                self.logger.warning(f"Status snapshot logs error: {e}")
                snapshot['recent_logs'] = f"Error: {e}"

            self._status_snapshot = snapshot
            self._status_snapshot_ts = now
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.logger.debug(f"Status snapshot refreshed in {elapsed_ms:.1f}ms")

    async def _refresh_zombies_snapshot(self, force: bool = False) -> None:
        """Refresh cached zombies report separately (heavier query)."""
        now = time.time()
        if not force and (now - self._zombies_snapshot_ts) < self._zombies_snapshot_interval:
            return
        if not self.paper_trader_farm:
            self._zombies_snapshot = {'zombies': [], 'total': 0, 'report': 'Farm not initialized'}
            self._zombies_snapshot_ts = now
            return

        started = time.perf_counter()
        try:
            zombies = await self.paper_trader_farm.detect_zombies(stale_days=14, grace_days=2)
            report = await self.paper_trader_farm.format_zombies_report(stale_days=14, grace_days=2)
            self._zombies_snapshot = {
                'zombies': zombies,
                'total': len(zombies),
                'report': report,
            }
        except Exception as e:
            self.logger.warning(f"Zombies snapshot error: {e}")
            self._zombies_snapshot = {'zombies': [], 'total': 0, 'report': f"Error: {e}"}
        self._zombies_snapshot_ts = now
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.logger.debug(f"Zombies snapshot refreshed in {elapsed_ms:.1f}ms")

    async def _get_dashboard_system_status(self) -> Dict[str, Any]:
        """System status for the dashboard /api/status endpoint."""
        cached = self._get_snapshot_section('system')
        if cached:
            extra = {
                "internal": self._get_snapshot_section("internal") or {},
                "bus": self._get_snapshot_section("bus") or {},
                "db": self._get_snapshot_section("db") or {},
                "providers": self._get_snapshot_section("providers") or {},
                "detectors": self._get_snapshot_section("detectors") or {},
            }
            return {**cached, **extra}
        await self._refresh_status_snapshot(force=True)
        system = self._status_snapshot.get('system', {'db_size_mb': 0, 'boot_phases': self._format_boot_phases()})
        extra = {
            "internal": self._status_snapshot.get("internal", {}),
            "bus": self._status_snapshot.get("bus", {}),
            "db": self._status_snapshot.get("db", {}),
            "providers": self._status_snapshot.get("providers", {}),
            "detectors": self._status_snapshot.get("detectors", {}),
        }
        return {**system, **extra}

    async def _get_provider_statuses(self) -> Dict[str, Any]:
        """Provider health for dashboard using the activity tracker."""
        return self._activity_tracker.get_provider_statuses()

    async def _get_detector_statuses(self) -> Dict[str, Any]:
        """Detector activity status for dashboard."""
        return self._activity_tracker.get_detector_statuses()

    async def _compute_pnl_summary(self) -> Dict:
        """Compute P&L summary without cached shortcut."""
        if self.paper_trader_farm:
            return await self.paper_trader_farm.get_pnl_for_telegram()
        if self.daily_review:
            from .analysis.daily_review import get_pnl_summary
            return await get_pnl_summary(self.daily_review)
        return {}

    async def _compute_positions_summary(self) -> List[Dict]:
        """Compute positions summary without cached shortcut."""
        if self.paper_trader_farm:
            return await self.paper_trader_farm.get_positions_for_telegram()
        return []

    async def _compute_farm_status(self) -> Dict[str, Any]:
        """Compute farm status without cached shortcut."""
        if not self.paper_trader_farm:
            return {}
        return self.paper_trader_farm.get_status_summary()

    async def _run_dashboard_command(self, cmd: str) -> str:
        """Execute a / command from the web dashboard."""
        cmd = cmd.strip()
        if not cmd.startswith('/'):
            cmd = '/' + cmd

        try:
            if cmd == '/pnl':
                data = await self._compute_pnl_summary()
                return (
                    f"Today: ${data.get('today_pnl', 0):+.2f}\n"
                    f"MTD: ${data.get('month_pnl', 0):+.2f}\n"
                    f"YTD: ${data.get('year_pnl', 0):+.2f}\n"
                    f"Win rate MTD: {data.get('win_rate_mtd', 0):.0f}%\n"
                    f"Open positions: {data.get('open_positions', 0)}"
                )
            elif cmd == '/status':
                if self.query_layer:
                    data = await self.query_layer.status()
                    return str(data)
                return "Query layer not initialized"
            elif cmd == '/positions':
                data = await self._compute_positions_summary()
                return "\n".join(
                    f"{p['symbol']} ({p['strategy']}): {p['count']} positions"
                    for p in data
                ) or "No open positions"
            elif cmd == '/zombies':
                data = self._zombies_snapshot
                return data.get('report', f"Total zombies: {data.get('total', 0)}")
            elif cmd.startswith('/zombie_clean'):
                if self.paper_trader_farm:
                    n = await self.paper_trader_farm.close_zombies()
                    return f"Closed {n} zombie positions"
                return "Farm not initialized"
            elif cmd == '/dashboard':
                data = await self._get_dashboard_system_status()
                return str(data)
            elif cmd.startswith('/reset_paper'):
                parts = cmd.split()
                scope = 'all'
                mode = 'epoch'
                for p in parts[1:]:
                    if p.startswith('--scope='):
                        scope = p.split('=')[1]
                    elif p.startswith('--mode='):
                        mode = p.split('=')[1]
                if self.paper_trader_farm:
                    result = await self.paper_trader_farm.reset_paper_equity(scope=scope, mode=mode)
                    await self._refresh_status_snapshot(force=True)
                    await self._refresh_zombies_snapshot(force=True)
                    return result
                return "Farm not initialized"
            elif cmd == '/db_stats':
                stats = self._get_snapshot_section('db_stats')
                if not stats:
                    await self._refresh_status_snapshot(force=True)
                    stats = self._status_snapshot.get('db_stats', {})
                lines = [f"DB size: {stats['db_size_mb']} MB"]
                for table, count in stats.get('row_counts', {}).items():
                    lines.append(f"  {table}: {count:,} rows")
                return "\n".join(lines)
            elif cmd == '/maintenance':
                stats = await self.db.run_maintenance()
                return f"Maintenance complete. DB: {stats['db_size_mb']} MB"
            elif cmd == '/db':
                if self.query_layer:
                    data = await self.query_layer.db()
                    return str(data)
                return "Query layer not initialized"
            elif cmd.startswith('/sql '):
                query = cmd[5:].strip()
                if not query:
                    return "Usage: /sql [SELECT ...]"
                try:
                    rows = await self.db.fetch_all(query)
                    if not rows:
                        return "No rows returned."
                    
                    # Convert objects to readable strings
                    lines = []
                    headers = rows[0].keys()
                    lines.append(" | ".join(headers))
                    lines.append("-" * 40)
                    for r in rows[:50]:  # Limit to 50 rows for display
                        lines.append(" | ".join(str(r[k]) for k in headers))
                    if len(rows) > 50:
                        lines.append(f"... and {len(rows)-50} more")
                    return "\n".join(lines)
                except Exception as e:
                    return f"SQL Error: {e}"
            else:
                return f"Unknown command: {cmd}"
        except Exception as e:
            return f"Error: {e}"

    async def _get_recent_logs_text(self) -> str:
        """Return recent log lines as text for dashboard."""
        cached = self._get_snapshot_section('recent_logs')
        if cached:
            return cached
        await self._refresh_status_snapshot(force=True)
        cached = self._status_snapshot.get('recent_logs')
        if cached:
            return cached
        return await self._read_recent_logs_text()

    async def _read_recent_logs_text(self) -> str:
        """Read recent log lines directly (no cache)."""
        if self._recent_logs:
            return "\n".join(self._recent_logs)
        # Read from log file as fallback
        try:
            log_file = Path(self.config.get('logging', {}).get('log_dir', 'data/logs')) / 'argus.log'
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                return "".join(lines[-100:])
        except Exception:
            pass
        return "No logs available"

    # =========================================================================
    # Market hours gating
    # =========================================================================

    def _is_us_market_open(self) -> bool:
        """Check if US equity markets are open (Mon-Fri, 9:30-16:00 ET)."""
        eastern = ZoneInfo("America/New_York")
        now_et = datetime.now(eastern)
        if now_et.weekday() >= 5:  # Weekend
            return False
        mh = self._mh_cfg
        open_h = int(mh.get('equity_open_hour', 9))
        open_m = int(mh.get('equity_open_minute', 30))
        close_h = int(mh.get('equity_close_hour', 16))
        close_m = int(mh.get('equity_close_minute', 0))
        open_time = now_et.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
        close_time = now_et.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
        return open_time <= now_et <= close_time

    # =========================================================================
    # DB maintenance task
    # =========================================================================

    async def _run_db_maintenance(self) -> None:
        """Periodic DB maintenance: retention cleanup + PRAGMA optimize."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Retention cleanup
                retention = self.config.get('data_retention', {})
                retention_map = {
                    'price_snapshots': retention.get('price_snapshots_days', 30),
                    'system_health': retention.get('logs_days', 30),
                    'funding_rates': retention.get('price_snapshots_days', 30),
                    'liquidations': retention.get('price_snapshots_days', 30),
                    'options_iv': retention.get('detections_days', 180),
                    'detections': retention.get('detections_days', 180),
                }
                await self.db.cleanup_old_data(retention_map)

                # PRAGMA optimize
                await self.db.run_maintenance()
                self.logger.info("DB maintenance completed")
            except Exception as e:
                self.logger.error(f"DB maintenance error: {e}")

    # ── Soak-test hardening helpers ─────────────────────────────────

    async def _get_soak_summary(self) -> Dict[str, Any]:
        """Build the soak summary for /debug/soak endpoint."""
        return build_soak_summary(
            bus=self.event_bus,
            bar_builder=self.bar_builder,
            persistence=self.persistence,
            feature_builder=self.feature_builder,
            regime_detector=self.regime_detector,
            resource_monitor=self.resource_monitor,
            guardian=self.soak_guardian,
            tape_recorder=self.tape_recorder,
            providers=self._activity_tracker.get_provider_statuses(),
            detectors=self._activity_tracker.get_detector_statuses(),
            polymarket_gamma=getattr(self, 'polymarket_gamma', None),
            polymarket_clob=getattr(self, 'polymarket_clob', None),
            polymarket_watchlist=getattr(self, 'polymarket_watchlist', None),
            bybit_ws=self.bybit_ws,
        )

    async def _send_soak_alert(
        self, severity: str, guard: str, message: str
    ) -> None:
        """Send a soak guard alert via Telegram (rate-limited by guardian)."""
        if not self.telegram:
            return
        icon = "\u26a0\ufe0f" if severity == "WARN" else "\u274c"
        text = (
            f"{icon} <b>Soak Guard: {guard}</b>\n"
            f"Severity: {severity}\n"
            f"{message}"
        )
        try:
            await self.telegram.send_message(text)
        except Exception as e:
            self.logger.warning(f"Failed to send soak alert: {e}")

    async def _export_tape(self, last_n_minutes: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract a slice of recorded tape data for export."""
        if not self.tape_recorder:
            return []
        with self.tape_recorder._lock:
            snapshot = list(self.tape_recorder._tape)
        
        if not last_n_minutes:
            return snapshot
            
        now = time.time()
        cutoff = now - (last_n_minutes * 60)
        return [e for e in snapshot if e.get("timestamp", 0) >= cutoff]

    async def _run_soak_guards(self) -> None:
        """Periodically evaluate soak guards (every 30s)."""
        interval = int(
            self.config.get('soak', {}).get('guard_interval_s', 30)
        )
        while self._running:
            await asyncio.sleep(interval)
            try:
                bus_stats = self.event_bus.get_status_summary()
                bb_status = self.bar_builder.get_status() if self.bar_builder else {}
                persist_status = self.persistence.get_status() if self.persistence else {}
                resource_snap = self.resource_monitor.get_full_snapshot()

                self.soak_guardian.evaluate(
                    bus_stats=bus_stats,
                    bar_builder_status=bb_status,
                    persistence_status=persist_status,
                    resource_snapshot=resource_snap,
                    component_heartbeats=self._component_heartbeat_ts,
                )
            except Exception:
                self.logger.debug("Soak guard evaluation failed", exc_info=True)

    async def run(self) -> None:
        """Start all components and run main loop."""
        self._running = True
        is_collector = self.mode == "collector"

        self.logger.info("Starting Argus (mode=%s)...", self.mode)

        # Start WebSocket connections
        if self.bybit_ws:
            self._tasks.append(asyncio.create_task(self.bybit_ws.connect()))

        # Start polling tasks (with market-hours gating for equities)
        if self.yahoo_client:
            self._tasks.append(asyncio.create_task(self._poll_yahoo_market_hours_aware()))

        # Alpaca bars polling (runs in ALL modes - data only, no execution)
        # Uses FIXED INTERVAL polling - no market-hours gating for determinism
        if self.alpaca_client:
            self._tasks.append(asyncio.create_task(self.alpaca_client.poll()))

        # Alpaca OPTIONS chain polling (Phase 3B - runs in ALL modes)
        if self.alpaca_options:
            self._tasks.append(asyncio.create_task(self._poll_options_chains()))

        self._tasks.append(asyncio.create_task(self._poll_deribit()))
        self._tasks.append(asyncio.create_task(self._health_check()))
        self._tasks.append(asyncio.create_task(self._run_db_maintenance()))

        # Soak guard evaluation loop
        self._tasks.append(asyncio.create_task(self._run_soak_guards()))

        # Heartbeat publisher (drives persistence flush boundaries)
        self._tasks.append(asyncio.create_task(self._publish_heartbeats()))
        self._tasks.append(asyncio.create_task(self._publish_minute_ticks()))

        # ── Trading tasks: DISABLED in collector mode ───
        if not is_collector:
            # Exit monitor runs independently of research loop
            if self.paper_trader_farm:
                self._tasks.append(asyncio.create_task(self._run_exit_monitor()))

            if self.research_enabled:
                self._tasks.append(asyncio.create_task(self._run_research_farm()))
        else:
            self.logger.info("Collector mode: exit monitor and research farm SKIPPED")

        # Market session monitor (open/close notifications)
        self._tasks.append(asyncio.create_task(self._run_market_session_monitor()))

        # Component heartbeat loop (Stream 2)
        self._tasks.append(asyncio.create_task(self._publish_component_heartbeats()))

        # Periodic status snapshots (Stream 2)
        self._tasks.append(asyncio.create_task(self._publish_status_snapshots()))

        # Polymarket polling loops (Stream 3)
        if self.polymarket_gamma:
            self._tasks.append(asyncio.create_task(self.polymarket_gamma.poll_loop()))
        if self.polymarket_clob:
            self._tasks.append(asyncio.create_task(self.polymarket_clob.poll_loop()))
        if self.polymarket_watchlist:
            self._tasks.append(asyncio.create_task(self.polymarket_watchlist.sync_loop()))

        # Automate gap risk snapshots at market close
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
    
    async def _poll_yahoo_market_hours_aware(self) -> None:
        """Poll Yahoo Finance with market-hours awareness.

        During US market hours: poll every 60s.
        Off-hours: poll every off_hours_sample_interval_seconds (default 600s / 10 min).
        """
        off_interval = int(self._mh_cfg.get('off_hours_sample_interval_seconds', 600))
        on_interval = 60

        while self._running:
            try:
                market_open = self._is_us_market_open()
                interval = on_interval if market_open else off_interval

                # One poll cycle
                await self.yahoo_client.poll_once()

                if not market_open:
                    self.logger.debug(f"US market closed, next Yahoo poll in {interval}s")

                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Yahoo market-hours poll error: {e}")
                await asyncio.sleep(60)

    async def _poll_alpaca_market_hours_aware(self) -> None:
        """Poll Alpaca for equity bars with market-hours awareness.

        During US market hours: poll every poll_interval (config, default 60s).
        Off-hours: poll every off_hours_sample_interval_seconds (default 600s).

        NOTE: Runs in ALL modes (collector, paper, live) - this is DATA only, no execution.
        """
        if not self.alpaca_client:
            return
            
        alpaca_cfg = self.config.get('exchanges', {}).get('alpaca', {})
        on_interval = int(alpaca_cfg.get('poll_interval_seconds', 60))
        off_interval = int(self._mh_cfg.get('off_hours_sample_interval_seconds', 600))

        while self._running:
            try:
                market_open = self._is_us_market_open()
                interval = on_interval if market_open else off_interval

                # One poll cycle
                await self.alpaca_client.poll_once()

                if not market_open:
                    self.logger.debug(f"US market closed, next Alpaca poll in {interval}s")

                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Alpaca market-hours poll error: {e}")
                await asyncio.sleep(60)

    # ── Heartbeat publisher ───────────────────────────────

    async def _publish_heartbeats(self) -> None:
        """Emit :class:`HeartbeatEvent` every 60 s to drive persistence flushes."""
        seq = 0
        interval = self.config.get('monitoring', {}).get('heartbeat_interval', 60)
        while self._running:
            await asyncio.sleep(interval)
            seq += 1
            self.event_bus.publish(
                TOPIC_SYSTEM_HEARTBEAT,
                HeartbeatEvent(sequence=seq),
            )

    async def _publish_component_heartbeats(self) -> None:
        """Emit structured heartbeats for all components every 60s."""
        interval = self.config.get('monitoring', {}).get('heartbeat_interval', 60)
        while self._running:
            await asyncio.sleep(interval)
            try:
                if self.bar_builder:
                    self.bar_builder.emit_heartbeat()
                if self.persistence:
                    self.persistence.emit_heartbeat()
                if self.feature_builder:
                    self.feature_builder.emit_heartbeat()
                if self.regime_detector:
                    self.regime_detector.emit_heartbeat()
                # ── System-level heartbeat for uptime tracking (Phase 4A.1+) ──
                if self.db:
                    import time as _t
                    ts_ms = int(_t.time() * 1000)
                    await self.db.write_heartbeat("orchestrator", ts_ms)
            except Exception:
                self.logger.debug("Component heartbeat emission failed", exc_info=True)

    async def _publish_status_snapshots(self) -> None:
        """Periodically persist QueryLayer status snapshots to DB."""
        interval = self.config.get('monitoring', {}).get('status_snapshot_persist_interval', 300)
        while self._running:
            await asyncio.sleep(interval)
            try:
                if self.query_layer:
                    await self.query_layer.persist_snapshot()
            except Exception:
                self.logger.debug("Status snapshot persist failed", exc_info=True)

    async def _publish_minute_ticks(self) -> None:
        """Emit minute-boundary ticks aligned to UTC minute boundaries."""
        while self._running:
            now = time.time()
            next_minute = (int(now // 60) + 1) * 60
            await asyncio.sleep(max(0.0, next_minute - now))
            if not self._running:
                break
            self.event_bus.publish(
                TOPIC_SYSTEM_MINUTE_TICK,
                MinuteTickEvent(timestamp=next_minute),
            )

    async def stop(self) -> None:
        """Stop all components gracefully."""
        self.logger.info("Stopping Argus...")
        self._running = False

        # Flush bar builder partial bars → bus → persistence
        if self.bar_builder:
            self.bar_builder.flush()

        # Flush persistence buffer (writes remaining bars to DB)
        if self.persistence:
            self.persistence.shutdown()

        # Stop event bus workers
        self.event_bus.stop()

        # Stop dashboard
        if self.dashboard:
            await self.dashboard.stop()

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

        # Stop Polymarket clients
        if self.polymarket_gamma:
            await self.polymarket_gamma.stop()
        if self.polymarket_clob:
            await self.polymarket_clob.stop()
        if self.polymarket_watchlist:
            await self.polymarket_watchlist.stop()

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
