"""
Telegram Bot for Alerts
=======================

3-tier alert system via Telegram.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from telegram import Bot
from telegram.error import TelegramError

from ..core.logger import get_alert_logger

logger = get_alert_logger()


class TelegramBot:
    """
    Telegram notification bot with 3-tier priority system.
    
    Tiers:
    - Tier 1: Immediate (Options IV >80%, Large liquidations)
    - Tier 2: FYI (Funding extremes, Basis arb opportunities)
    - Tier 3: Background (Logged only, not sent)
    """
    
    # Emojis for different alert types
    EMOJIS = {
        1: "🚨",  # Tier 1: Urgent
        2: "📊",  # Tier 2: Informational
        3: "📝",  # Tier 3: Background
        'funding': "💰",
        'basis': "⚖️",
        'cross_exchange': "🔄",
        'liquidation': "💥",
        'options_iv': "📈",
        'volatility': "🌊",
        'system': "⚙️",
        'success': "✅",
        'warning': "⚠️",
        'error': "❌",
    }
    
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        tier_1_enabled: bool = True,
        tier_2_enabled: bool = True,
        tier_3_enabled: bool = False,
        rate_limit_seconds: int = 10,
    ):
        """
        Initialize Telegram bot.
        
        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Chat/group ID to send messages to
            tier_1_enabled: Send tier 1 alerts
            tier_2_enabled: Send tier 2 alerts
            tier_3_enabled: Send tier 3 alerts (usually disabled)
            rate_limit_seconds: Minimum seconds between same-type alerts
        """
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        
        self.tier_1_enabled = tier_1_enabled
        self.tier_2_enabled = tier_2_enabled
        self.tier_3_enabled = tier_3_enabled
        
        self.rate_limit_seconds = rate_limit_seconds
        self._last_alert_time: Dict[str, datetime] = {}
        
        logger.info("Telegram bot initialized")
    
    def _should_send(self, tier: int, alert_type: str) -> bool:
        """Check if alert should be sent based on tier and rate limiting."""
        # Check tier
        if tier == 1 and not self.tier_1_enabled:
            return False
        if tier == 2 and not self.tier_2_enabled:
            return False
        if tier == 3 and not self.tier_3_enabled:
            return False
        
        # Tier 1 always sends (no rate limiting)
        if tier == 1:
            return True
        
        # Rate limiting for tier 2+
        key = f"{tier}_{alert_type}"
        now = datetime.utcnow()
        
        if key in self._last_alert_time:
            elapsed = (now - self._last_alert_time[key]).seconds
            if elapsed < self.rate_limit_seconds:
                return False
        
        self._last_alert_time[key] = now
        return True
    
    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False
    ) -> bool:
        """
        Send a message to the configured chat.
        
        Args:
            text: Message text (HTML format supported)
            parse_mode: 'HTML' or 'Markdown'
            disable_notification: Send silently
            
        Returns:
            True if sent successfully
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_notification=disable_notification
            )
            logger.debug("Telegram message sent")
            return True
        except TelegramError as e:
            logger.error(f"Telegram send failed: {e}")
            return False
    
    async def send_alert(
        self,
        tier: int,
        alert_type: str,
        title: str,
        details: Dict[str, Any],
        action: Optional[str] = None
    ) -> bool:
        """
        Send a formatted alert message.
        
        Args:
            tier: Alert tier (1, 2, or 3)
            alert_type: Type of opportunity
            title: Alert title
            details: Details to include
            action: Suggested action (optional)
            
        Returns:
            True if sent
        """
        if not self._should_send(tier, alert_type):
            logger.debug(f"Alert suppressed: tier={tier}, type={alert_type}")
            return False
        
        # Build message
        tier_emoji = self.EMOJIS.get(tier, "📢")
        type_emoji = self.EMOJIS.get(alert_type, "📌")
        
        lines = [
            f"{tier_emoji} <b>{title}</b>",
            f"",
        ]
        
        # Add details
        for key, value in details.items():
            formatted_key = key.replace('_', ' ').title()
            lines.append(f"• {formatted_key}: <code>{value}</code>")
        
        # Add action if provided
        if action:
            lines.append("")
            lines.append(f"💡 <b>Action:</b> {action}")
        
        # Add timestamp
        lines.append("")
        lines.append(f"<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>")
        
        text = "\n".join(lines)
        
        # Tier 1 gets sound, tier 2+ is silent
        silent = tier > 1
        
        return await self.send_message(text, disable_notification=silent)
    
    async def send_funding_alert(self, detection: Dict) -> bool:
        """Send funding rate opportunity alert."""
        data = detection.get('detection_data', {})
        
        return await self.send_alert(
            tier=2,
            alert_type='funding',
            title=f"Funding Rate Alert: {detection.get('asset', 'Unknown')}",
            details={
                'Exchange': detection.get('exchange', 'Unknown'),
                'Current Rate': f"{data.get('current_funding', 0):.4%}",
                'Z-Score': f"{data.get('z_score', 0):.2f}",
                'Mean Rate': f"{data.get('mean_funding', 0):.4%}",
                'Net Edge': f"{detection.get('net_edge_bps', 0):.1f} bps",
            },
            action="Consider SHORT if positive funding, LONG if negative"
        )
    
    async def send_iv_alert(self, detection: Dict) -> bool:
        """Send options IV spike alert (Tier 1 - immediate)."""
        data = detection.get('detection_data', {})
        
        return await self.send_alert(
            tier=1,
            alert_type='options_iv',
            title=f"🚨 OPTIONS IV SPIKE: {detection.get('asset', 'BTC')}",
            details={
                'Current IV': f"{data.get('current_iv', 0):.1%}",
                'Average IV': f"{data.get('mean_iv', 0):.1%}",
                'Z-Score': f"{data.get('z_score', 0):.2f}",
                'Underlying': f"${data.get('underlying_price', 0):,.0f}",
            },
            action="Check Deribit for put spreads - HIGH IV = sell premium"
        )
    
    async def send_liquidation_alert(self, detection: Dict) -> bool:
        """Send liquidation cascade alert (Tier 1 if large)."""
        data = detection.get('detection_data', {})
        total = data.get('total_liquidations_usd', 0)
        
        tier = 1 if total >= 5_000_000 else 2
        
        return await self.send_alert(
            tier=tier,
            alert_type='liquidation',
            title=f"💥 Liquidation Cascade: {detection.get('asset', 'Unknown')}",
            details={
                'Total Liquidated': f"${total:,.0f}",
                'Long Liquidations': f"${data.get('long_liquidations_usd', 0):,.0f}",
                'Short Liquidations': f"${data.get('short_liquidations_usd', 0):,.0f}",
                'Dominant Side': data.get('dominant_side', 'Unknown'),
            },
            action="Consider fading the move after cascade completes"
        )
    
    async def send_basis_alert(self, detection: Dict) -> bool:
        """Send basis arbitrage opportunity alert."""
        data = detection.get('detection_data', {})
        
        return await self.send_alert(
            tier=2,
            alert_type='basis',
            title=f"Basis Opportunity: {detection.get('asset', 'Unknown')}",
            details={
                'Spot Price': f"${data.get('spot_price', 0):,.2f}",
                'Perp Price': f"${data.get('perp_price', 0):,.2f}",
                'Basis': f"{data.get('basis_bps', 0):.1f} bps",
                'Net Edge': f"{detection.get('net_edge_bps', 0):.1f} bps",
            },
            action="Spot-perp convergence expected"
        )
    
    async def send_daily_summary(self, stats: Dict) -> bool:
        """Send daily summary report."""
        lines = [
            f"📈 <b>Daily Market Monitor Summary</b>",
            f"<i>{datetime.utcnow().strftime('%Y-%m-%d')}</i>",
            "",
            "<b>Detections Today:</b>",
        ]
        
        by_type = stats.get('detections_by_type', {})
        for op_type, count in by_type.items():
            emoji = self.EMOJIS.get(op_type, "•")
            lines.append(f"  {emoji} {op_type.replace('_', ' ').title()}: {count}")
        
        lines.append("")
        lines.append(f"Total: {stats.get('total_detections', 0)} detections")
        
        trade_stats = stats.get('trade_statistics', {})
        if trade_stats.get('total_trades', 0) > 0:
            lines.append("")
            lines.append("<b>Hypothetical Performance:</b>")
            lines.append(f"  • Trades: {trade_stats.get('total_trades', 0)}")
            lines.append(f"  • Win Rate: {trade_stats.get('win_rate', 0):.1%}")
            lines.append(f"  • Avg P&L: {trade_stats.get('avg_pnl', 0):.2%}")
        
        text = "\n".join(lines)
        return await self.send_message(text)
    
    async def send_system_status(self, status: str, details: str = "") -> bool:
        """Send system status update."""
        emoji = self.EMOJIS.get(status, "⚙️")
        text = f"{emoji} <b>System Status: {status.upper()}</b>"
        if details:
            text += f"\n{details}"
        
        return await self.send_message(text)
    
    async def test_connection(self) -> bool:
        """Test bot connection."""
        try:
            me = await self.bot.get_me()
            logger.info(f"Telegram bot connected: @{me.username}")
            return True
        except TelegramError as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False
