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
    
    async def vacuum(self) -> None:
        """Optimize database by reclaiming space."""
        await self._connection.execute("VACUUM")
        logger.info("Database vacuumed")
    
    async def backup(self, backup_path: str) -> None:
        """Create a database backup."""
        backup_db = Path(backup_path)
        backup_db.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(str(backup_db)) as backup:
            await self._connection.backup(backup)
        
        logger.info(f"Database backed up to {backup_path}")
