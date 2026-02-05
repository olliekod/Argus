"""
Argus Database Module
=====================

SQLite database for storing detections, market data, and system health.
Uses aiosqlite for async operations.
"""

import aiosqlite
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Database:
    """
    Async SQLite database manager for Argus.
    
    Handles:
    - Detection logging
    - Funding rate history
    - Options IV data
    - Liquidation events
    - Price snapshots
    - System health monitoring
    """
    
    def __init__(self, db_path: str):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def connect(self) -> None:
        """Establish database connection and create tables."""
        self._connection = await aiosqlite.connect(str(self.db_path))
        self._connection.row_factory = aiosqlite.Row
        await self._create_tables()
        # Enable WAL mode for long-running app
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=NORMAL")
        await self._connection.execute("PRAGMA cache_size=-64000")  # 64MB cache
        logger.info(f"Database connected: {self.db_path}")
    
    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    async def execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a SQL statement (for external use)."""
        await self._connection.execute(sql, params)
        await self._connection.commit()

    async def execute_many(self, sql: str, params_list: List[tuple]) -> None:
        """Execute a SQL statement with multiple parameter sets in a single transaction."""
        await self._connection.executemany(sql, params_list)
        await self._connection.commit()

    async def fetch_one(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        """Fetch one row from a query."""
        cursor = await self._connection.execute(sql, params)
        return await cursor.fetchone()
    
    async def fetch_all(self, sql: str, params: tuple = ()) -> List[tuple]:
        """Fetch all rows from a query."""
        cursor = await self._connection.execute(sql, params)
        return await cursor.fetchall()
    
    async def _create_tables(self) -> None:
        """Create all database tables if they don't exist."""

        # Followed traders list (for best-trader follow feature)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS followed_traders (
                trader_id TEXT PRIMARY KEY,
                followed_at TEXT NOT NULL,
                score REAL,
                scoring_method TEXT,
                window_days INTEGER,
                config_json TEXT
            )
        """)

        # Signals table (order intent / signal log)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS trade_signals (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                trader_id TEXT,
                strategy_type TEXT,
                symbol TEXT NOT NULL,
                direction TEXT,
                signal_source TEXT,
                iv_at_signal REAL,
                warmth_at_signal REAL,
                pop_at_signal REAL,
                gap_risk_at_signal REAL,
                underlying_price REAL,
                btc_price REAL,
                btc_iv REAL,
                conditions_score INTEGER,
                conditions_label TEXT,
                strikes TEXT,
                expiry TEXT,
                target_credit REAL,
                bid_price REAL,
                ask_price REAL,
                spread_width REAL,
                contracts INTEGER,
                outcome TEXT,
                outcome_reason TEXT
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_ts ON trade_signals(timestamp)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_trader ON trade_signals(trader_id)"
        )

        # Uniformity monitor snapshots
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS uniformity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_type TEXT,
                variable_name TEXT NOT NULL,
                unique_count INTEGER,
                total_count INTEGER,
                modal_value TEXT,
                modal_pct REAL,
                hhi REAL,
                entropy REAL,
                is_alert INTEGER DEFAULT 0,
                alert_reason TEXT
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_uniformity_ts ON uniformity_snapshots(timestamp)"
        )

        # Daily trader metrics (for per-trader PnL analytics)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS trader_daily_metrics (
                trader_id TEXT NOT NULL,
                date TEXT NOT NULL,
                realized_pnl REAL DEFAULT 0,
                trades_closed INTEGER DEFAULT 0,
                trades_opened INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                peak_pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                PRIMARY KEY (trader_id, date)
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_tdm_date ON trader_daily_metrics(date)"
        )

        # Main detections table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                opportunity_type TEXT NOT NULL,
                asset TEXT NOT NULL,
                exchange TEXT NOT NULL,
                current_price REAL,
                volume_24h REAL,
                volatility_1h REAL,
                volatility_24h REAL,
                detection_data TEXT,
                estimated_edge_bps REAL,
                estimated_slippage_bps REAL,
                estimated_fees_bps REAL,
                net_edge_bps REAL,
                would_trigger_entry INTEGER DEFAULT 0,
                suggested_position_size REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                resolution_timestamp TEXT,
                actual_outcome TEXT,
                hypothetical_pnl_percent REAL,
                hypothetical_pnl_usd REAL,
                alert_tier INTEGER,
                alert_sent INTEGER DEFAULT 0,
                notes TEXT
            )
        """)
        
        # Funding rates history
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS funding_rates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                exchange TEXT NOT NULL,
                asset TEXT NOT NULL,
                funding_rate REAL NOT NULL,
                predicted_rate REAL,
                open_interest REAL,
                volume_24h REAL,
                mark_price REAL,
                index_price REAL
            )
        """)
        
        # Options IV history
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS options_iv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                asset TEXT NOT NULL,
                expiry TEXT NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                implied_volatility REAL NOT NULL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                mark_price REAL,
                underlying_price REAL
            )
        """)
        
        # Liquidation events
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS liquidations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                exchange TEXT NOT NULL,
                asset TEXT NOT NULL,
                side TEXT NOT NULL,
                liquidation_amount_usd REAL NOT NULL,
                price REAL NOT NULL
            )
        """)
        
        # Price snapshots
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS price_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                exchange TEXT NOT NULL,
                asset TEXT NOT NULL,
                price_type TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL
            )
        """)
        
        # System health
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                latency_ms REAL
            )
        """)
        
        # Daily statistics rollup
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_detections INTEGER,
                detections_by_type TEXT,
                total_hypothetical_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl_percent REAL,
                total_pnl_usd REAL,
                best_trade_pnl REAL,
                worst_trade_pnl REAL,
                avg_trade_pnl REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL
            )
        """)
        
        # Create indices for common queries
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(opportunity_type)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_asset ON detections(asset)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_funding_timestamp ON funding_rates(timestamp, asset)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_options_timestamp ON options_iv(timestamp, asset)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_liquidations_timestamp ON liquidations(timestamp, asset)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON price_snapshots(timestamp, asset)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_health_timestamp ON system_health(timestamp)"
        )
        
        await self._connection.commit()

        # Paper equity epochs (for reset_paper command)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS paper_equity_epochs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch_start TEXT NOT NULL,
                starting_equity REAL NOT NULL,
                reason TEXT,
                scope TEXT DEFAULT 'all'
            )
        """)
        await self._connection.commit()

        logger.debug("Database tables created/verified")
    
    # =========================================================================
    # Detection Operations
    # =========================================================================
    
    async def insert_detection(self, detection: Dict[str, Any]) -> int:
        """
        Insert a new detection record.
        
        Args:
            detection: Detection data dictionary
            
        Returns:
            ID of inserted record
        """
        detection_data = detection.get('detection_data')
        if isinstance(detection_data, dict):
            detection_data = json.dumps(detection_data)
        
        cursor = await self._connection.execute("""
            INSERT INTO detections (
                timestamp, opportunity_type, asset, exchange,
                current_price, volume_24h, volatility_1h, volatility_24h,
                detection_data, estimated_edge_bps, estimated_slippage_bps,
                estimated_fees_bps, net_edge_bps, would_trigger_entry,
                suggested_position_size, entry_price, stop_loss, take_profit,
                alert_tier, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            detection.get('timestamp', datetime.utcnow().isoformat()),
            detection.get('opportunity_type'),
            detection.get('asset'),
            detection.get('exchange'),
            detection.get('current_price'),
            detection.get('volume_24h'),
            detection.get('volatility_1h'),
            detection.get('volatility_24h'),
            detection_data,
            detection.get('estimated_edge_bps'),
            detection.get('estimated_slippage_bps'),
            detection.get('estimated_fees_bps'),
            detection.get('net_edge_bps'),
            1 if detection.get('would_trigger_entry') else 0,
            detection.get('suggested_position_size'),
            detection.get('entry_price'),
            detection.get('stop_loss'),
            detection.get('take_profit'),
            detection.get('alert_tier'),
            detection.get('notes')
        ))
        
        await self._connection.commit()
        logger.debug(f"Inserted detection: {detection.get('opportunity_type')} - {detection.get('asset')}")
        return cursor.lastrowid
    
    async def update_detection_resolution(
        self,
        detection_id: int,
        outcome: str,
        pnl_percent: float,
        pnl_usd: float
    ) -> None:
        """
        Update a detection with resolution data.
        
        Args:
            detection_id: ID of detection to update
            outcome: 'normalized', 'stopped', 'profit_taken', 'expired'
            pnl_percent: Hypothetical P&L percentage
            pnl_usd: Hypothetical P&L in USD
        """
        await self._connection.execute("""
            UPDATE detections
            SET resolution_timestamp = ?,
                actual_outcome = ?,
                hypothetical_pnl_percent = ?,
                hypothetical_pnl_usd = ?
            WHERE id = ?
        """, (
            datetime.utcnow().isoformat(),
            outcome,
            pnl_percent,
            pnl_usd,
            detection_id
        ))
        await self._connection.commit()
        logger.debug(f"Updated detection {detection_id}: {outcome}, {pnl_percent:.2f}%")
    
    async def mark_alert_sent(self, detection_id: int) -> None:
        """Mark a detection's alert as sent."""
        await self._connection.execute(
            "UPDATE detections SET alert_sent = 1 WHERE id = ?",
            (detection_id,)
        )
        await self._connection.commit()
    
    async def get_recent_detections(
        self,
        hours: int = 24,
        opportunity_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recent detections.
        
        Args:
            hours: How many hours back to look
            opportunity_type: Filter by type (optional)
            
        Returns:
            List of detection dictionaries
        """
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        if opportunity_type:
            cursor = await self._connection.execute("""
                SELECT * FROM detections
                WHERE timestamp > ? AND opportunity_type = ?
                ORDER BY timestamp DESC
            """, (cutoff, opportunity_type))
        else:
            cursor = await self._connection.execute("""
                SELECT * FROM detections
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff,))
        
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    async def get_unresolved_detections(self) -> List[Dict]:
        """Get detections that would trigger entry but haven't been resolved."""
        cursor = await self._connection.execute("""
            SELECT * FROM detections
            WHERE would_trigger_entry = 1
              AND resolution_timestamp IS NULL
            ORDER BY timestamp DESC
        """)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # =========================================================================
    # Funding Rate Operations
    # =========================================================================
    
    async def insert_funding_rate(
        self,
        exchange: str,
        asset: str,
        funding_rate: float,
        **kwargs
    ) -> None:
        """Insert a funding rate record."""
        await self._connection.execute("""
            INSERT INTO funding_rates (
                timestamp, exchange, asset, funding_rate,
                predicted_rate, open_interest, volume_24h, mark_price, index_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            kwargs.get('timestamp', datetime.utcnow().isoformat()),
            exchange,
            asset,
            funding_rate,
            kwargs.get('predicted_rate'),
            kwargs.get('open_interest'),
            kwargs.get('volume_24h'),
            kwargs.get('mark_price'),
            kwargs.get('index_price')
        ))
        await self._connection.commit()
    
    async def get_funding_history(
        self,
        asset: str,
        periods: int = 30,
        exchange: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recent funding rate history.
        
        Args:
            asset: Asset symbol
            periods: Number of periods to retrieve
            exchange: Filter by exchange (optional)
            
        Returns:
            List of funding rate records
        """
        if exchange:
            cursor = await self._connection.execute("""
                SELECT * FROM funding_rates
                WHERE asset = ? AND exchange = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (asset, exchange, periods))
        else:
            cursor = await self._connection.execute("""
                SELECT * FROM funding_rates
                WHERE asset = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (asset, periods))
        
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # =========================================================================
    # Options IV Operations
    # =========================================================================
    
    async def insert_options_iv(
        self,
        asset: str,
        expiry: str,
        strike: float,
        option_type: str,
        iv: float,
        **kwargs
    ) -> None:
        """Insert an options IV record."""
        await self._connection.execute("""
            INSERT INTO options_iv (
                timestamp, asset, expiry, strike, option_type,
                implied_volatility, delta, gamma, theta, vega,
                mark_price, underlying_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            kwargs.get('timestamp', datetime.utcnow().isoformat()),
            asset,
            expiry,
            strike,
            option_type,
            iv,
            kwargs.get('delta'),
            kwargs.get('gamma'),
            kwargs.get('theta'),
            kwargs.get('vega'),
            kwargs.get('mark_price'),
            kwargs.get('underlying_price')
        ))
        await self._connection.commit()
    
    async def get_iv_history(
        self,
        asset: str,
        days: int = 30
    ) -> List[Dict]:
        """Get IV history for an asset."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        cursor = await self._connection.execute("""
            SELECT * FROM options_iv
            WHERE asset = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (asset, cutoff))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # =========================================================================
    # Liquidation Operations
    # =========================================================================
    
    async def insert_liquidation(
        self,
        exchange: str,
        asset: str,
        side: str,
        amount_usd: float,
        price: float,
        timestamp: Optional[str] = None
    ) -> None:
        """Insert a liquidation event."""
        await self._connection.execute("""
            INSERT INTO liquidations (timestamp, exchange, asset, side, liquidation_amount_usd, price)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            timestamp or datetime.utcnow().isoformat(),
            exchange,
            asset,
            side,
            amount_usd,
            price
        ))
        await self._connection.commit()
    
    async def get_recent_liquidations(
        self,
        asset: str,
        minutes: int = 5
    ) -> List[Dict]:
        """Get recent liquidations for cascade detection."""
        cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
        cursor = await self._connection.execute("""
            SELECT * FROM liquidations
            WHERE asset = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (asset, cutoff))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # =========================================================================
    # Price Snapshot Operations
    # =========================================================================
    
    async def insert_price_snapshot(
        self,
        exchange: str,
        asset: str,
        price_type: str,
        price: float,
        volume: Optional[float] = None
    ) -> None:
        """Insert a price snapshot."""
        await self._connection.execute("""
            INSERT INTO price_snapshots (timestamp, exchange, asset, price_type, price, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            exchange,
            asset,
            price_type,
            price,
            volume
        ))
        await self._connection.commit()
    
    async def get_price_history(
        self,
        asset: str,
        exchange: str,
        price_type: str = 'spot',
        minutes: int = 60
    ) -> List[float]:
        """Get price history for volatility calculations."""
        cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
        cursor = await self._connection.execute("""
            SELECT price FROM price_snapshots
            WHERE asset = ? AND exchange = ? AND price_type = ? AND timestamp > ?
            ORDER BY timestamp ASC
        """, (asset, exchange, price_type, cutoff))
        rows = await cursor.fetchall()
        return [row['price'] for row in rows]
    
    # =========================================================================
    # System Health Operations
    # =========================================================================
    
    async def insert_health_check(
        self,
        component: str,
        status: str,
        error_message: Optional[str] = None,
        latency_ms: Optional[float] = None
    ) -> None:
        """Record a system health check."""
        await self._connection.execute("""
            INSERT INTO system_health (timestamp, component, status, error_message, latency_ms)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            component,
            status,
            error_message,
            latency_ms
        ))
        await self._connection.commit()
    
    async def get_component_status(self, component: str) -> Optional[Dict]:
        """Get the most recent status for a component."""
        cursor = await self._connection.execute("""
            SELECT * FROM system_health
            WHERE component = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (component,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_latest_timestamps(self, tables: List[str]) -> Dict[str, Optional[str]]:
        """Fetch the most recent timestamp for each table."""
        results: Dict[str, Optional[str]] = {}
        for table in tables:
            cursor = await self._connection.execute(
                f"SELECT MAX(timestamp) AS latest FROM {table}"
            )
            row = await cursor.fetchone()
            results[table] = row["latest"] if row and row["latest"] else None
        return results

    async def get_price_at_or_before(
        self,
        exchange: str,
        asset: str,
        price_type: str,
        cutoff_timestamp: str,
    ) -> Optional[Dict[str, Any]]:
        """Fetch the most recent price at or before a cutoff timestamp."""
        cursor = await self._connection.execute("""
            SELECT price, timestamp
            FROM price_snapshots
            WHERE exchange = ? AND asset = ? AND price_type = ?
              AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (exchange, asset, price_type, cutoff_timestamp))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_per_trader_pnl(self, days: int = 30, min_trades: int = 1) -> List[Dict[str, Any]]:
        """Get per-trader realized PnL statistics for /pnl analytics.

        Computes per-trader return as realized_pnl / starting_equity.
        Each trader starts with the same notional ($5000), so return = total_pnl / 5000.
        """
        cursor = await self._connection.execute("""
            SELECT
                trader_id,
                strategy_type,
                COUNT(*) as total_trades,
                SUM(CASE WHEN status != 'open' THEN 1 ELSE 0 END) as closed_trades,
                SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN realized_pnl <= 0 AND realized_pnl IS NOT NULL THEN 1 ELSE 0 END) as losses,
                SUM(realized_pnl) as total_pnl,
                AVG(realized_pnl) as avg_pnl,
                MIN(realized_pnl) as worst_trade,
                MAX(realized_pnl) as best_trade
            FROM paper_trades
            WHERE timestamp >= datetime('now', ?)
            GROUP BY trader_id, strategy_type
            HAVING closed_trades >= ?
        """, (f"-{days} days", min_trades))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_zombie_positions(self, stale_hours: int = 48) -> List[Dict[str, Any]]:
        """Find zombie positions: open trades with no updates for stale_hours.

        A zombie position is an open position that is no longer reachable by the
        strategy lifecycle (e.g., missing close event, orphan order, expired option
        not settled, process crash before DB update).

        Detection rules:
        1. Status = 'open' AND timestamp < (now - stale_hours)
        2. Status = 'open' AND expiry date is in the past
        """
        cursor = await self._connection.execute("""
            SELECT id, trader_id, strategy_type, symbol, timestamp,
                   strikes, expiry, entry_credit, contracts, status,
                   market_conditions
            FROM paper_trades
            WHERE status = 'open'
              AND (
                  timestamp < datetime('now', ?)
                  OR (expiry IS NOT NULL AND expiry < date('now'))
              )
            ORDER BY timestamp ASC
        """, (f"-{stale_hours} hours",))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def mark_zombies(self, trade_ids: List[str], reason: str = 'zombie_detected') -> int:
        """Mark a list of trades as zombie (expired with reason)."""
        if not trade_ids:
            return 0
        now = datetime.utcnow().isoformat()
        placeholders = ','.join(['?'] * len(trade_ids))
        await self._connection.execute(f"""
            UPDATE paper_trades
            SET status = 'expired',
                close_timestamp = ?,
                market_conditions = json_set(
                    COALESCE(market_conditions, '{{}}'),
                    '$.zombie_reason', ?
                )
            WHERE id IN ({placeholders})
              AND status = 'open'
        """, (now, reason, *trade_ids))
        await self._connection.commit()
        return len(trade_ids)

    async def get_followed_traders(self) -> List[Dict[str, Any]]:
        """Get the list of followed traders."""
        cursor = await self._connection.execute(
            "SELECT * FROM followed_traders ORDER BY score DESC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def set_followed_traders(self, traders: List[Dict[str, Any]]) -> None:
        """Replace the followed traders list."""
        await self._connection.execute("DELETE FROM followed_traders")
        for t in traders:
            await self._connection.execute("""
                INSERT INTO followed_traders
                (trader_id, followed_at, score, scoring_method, window_days, config_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                t['trader_id'],
                t.get('followed_at', datetime.utcnow().isoformat()),
                t.get('score'),
                t.get('scoring_method'),
                t.get('window_days'),
                t.get('config_json'),
            ))
        await self._connection.commit()

    async def get_trader_performance(self, days: int = 60) -> List[Dict[str, Any]]:
        """Get aggregated trader performance over a window."""
        cursor = await self._connection.execute("""
            SELECT 
                trader_id,
                strategy_type,
                COUNT(*) as total_trades,
                SUM(realized_pnl) as total_pnl,
                AVG(realized_pnl) as avg_pnl,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
            FROM paper_trades
            WHERE timestamp >= datetime('now', ?)
            GROUP BY trader_id, strategy_type
        """, (f"-{days} days",))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    
    # =========================================================================
    # Statistics Operations
    # =========================================================================
    
    async def get_statistics(self, days: int = 14) -> Dict:
        """
        Get aggregated statistics for analysis.
        
        Returns dict with:
        - total_detections
        - detections_by_type
        - trade_statistics
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Total detections
        cursor = await self._connection.execute("""
            SELECT COUNT(*) as count FROM detections WHERE timestamp > ?
        """, (cutoff,))
        row = await cursor.fetchone()
        total = row['count'] if row else 0
        
        # By type
        cursor = await self._connection.execute("""
            SELECT opportunity_type, COUNT(*) as count
            FROM detections
            WHERE timestamp > ?
            GROUP BY opportunity_type
        """, (cutoff,))
        rows = await cursor.fetchall()
        by_type = {row['opportunity_type']: row['count'] for row in rows}
        
        # Trade statistics
        cursor = await self._connection.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN hypothetical_pnl_percent > 0 THEN 1 ELSE 0 END) as winners,
                AVG(hypothetical_pnl_percent) as avg_pnl,
                MIN(hypothetical_pnl_percent) as worst_pnl,
                MAX(hypothetical_pnl_percent) as best_pnl
            FROM detections
            WHERE timestamp > ?
              AND would_trigger_entry = 1
              AND resolution_timestamp IS NOT NULL
        """, (cutoff,))
        row = await cursor.fetchone()
        
        trade_stats = {
            'total_trades': row['total_trades'] or 0,
            'winners': row['winners'] or 0,
            'avg_pnl': row['avg_pnl'] or 0,
            'worst_pnl': row['worst_pnl'] or 0,
            'best_pnl': row['best_pnl'] or 0,
        }
        
        if trade_stats['total_trades'] > 0:
            trade_stats['win_rate'] = trade_stats['winners'] / trade_stats['total_trades']
        else:
            trade_stats['win_rate'] = 0
        
        return {
            'total_detections': total,
            'detections_by_type': by_type,
            'trade_statistics': trade_stats
        }
    
    # =========================================================================
    # Maintenance Operations
    # =========================================================================
    
    async def cleanup_old_data(self, retention_days: Dict[str, int]) -> None:
        """
        Delete old data based on retention policy.
        
        Args:
            retention_days: Dict mapping table names to retention days
        """
        for table, days in retention_days.items():
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            await self._connection.execute(
                f"DELETE FROM {table} WHERE timestamp < ?",
                (cutoff,)
            )
            logger.info(f"Cleaned up {table} older than {days} days")

        await self._connection.commit()

        # Clean up old closed paper trades (keep open ones)
        if 'paper_trades' in retention_days:
            days = retention_days['paper_trades']
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            await self._connection.execute(
                "DELETE FROM paper_trades WHERE status != 'open' AND close_timestamp < ?",
                (cutoff,)
            )
            logger.info(f"Cleaned up closed paper_trades older than {days} days")
            await self._connection.commit()
    
    async def vacuum(self) -> None:
        """Optimize database by reclaiming space."""
        await self._connection.execute("VACUUM")
        logger.info("Database vacuumed")

    async def run_maintenance(self) -> Dict[str, Any]:
        """Run periodic maintenance: PRAGMA optimize + retention cleanup."""
        await self._connection.execute("PRAGMA optimize")
        logger.info("PRAGMA optimize completed")
        return await self.get_db_stats()

    async def get_db_stats(self) -> Dict[str, Any]:
        """Get database size and table row counts."""
        import os
        db_size = os.path.getsize(str(self.db_path)) if self.db_path.exists() else 0
        tables = [
            'detections', 'funding_rates', 'options_iv', 'liquidations',
            'price_snapshots', 'system_health', 'daily_stats',
        ]
        # Also check paper_trades if it exists
        try:
            cursor = await self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'"
            )
            if await cursor.fetchone():
                tables.append('paper_trades')
        except Exception:
            pass

        row_counts = {}
        for table in tables:
            try:
                cursor = await self._connection.execute(f"SELECT COUNT(*) as cnt FROM {table}")
                row = await cursor.fetchone()
                row_counts[table] = row['cnt'] if row else 0
            except Exception:
                row_counts[table] = -1

        return {
            'db_size_bytes': db_size,
            'db_size_mb': round(db_size / (1024 * 1024), 1),
            'row_counts': row_counts,
        }

    async def reset_paper_epoch(self, starting_equity: float, scope: str = 'all', reason: str = 'manual_reset') -> None:
        """Start a new paper equity epoch. Old data remains but is excluded from current metrics."""
        await self._connection.execute("""
            INSERT INTO paper_equity_epochs (epoch_start, starting_equity, reason, scope)
            VALUES (?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), starting_equity, reason, scope))
        await self._connection.commit()
        logger.info(f"New paper equity epoch started: equity=${starting_equity}, scope={scope}, reason={reason}")

    async def get_current_epoch_start(self) -> Optional[str]:
        """Get the start timestamp of the current paper equity epoch."""
        cursor = await self._connection.execute(
            "SELECT epoch_start FROM paper_equity_epochs ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row['epoch_start'] if row else None

    async def backup(self, backup_path: str) -> None:
        """Create a database backup."""
        backup_db = Path(backup_path)
        backup_db.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(str(backup_db)) as backup:
            await self._connection.backup(backup)
        
        logger.info(f"Database backed up to {backup_path}")
