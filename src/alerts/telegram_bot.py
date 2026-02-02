"""
Telegram Bot for Alerts (Two-Way)
=================================

3-tier alert system via Telegram with two-way communication.
Supports commands: /help, /status, /positions, /pnl
"""

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Callable, Dict, List, Optional
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.error import TelegramError

from ..core.logger import get_alert_logger

logger = get_alert_logger()


class TelegramBot:
    """
    Telegram notification bot with 3-tier priority system and two-way communication.
    
    Tiers:
    - Tier 1: Immediate (Options IV >80%, Large liquidations)
    - Tier 2: FYI (Funding extremes, Basis arb opportunities)
    - Tier 3: Background (Logged only, not sent)
    
    Commands:
    - /help: List all available commands
    - /status: Current system status and conditions
    - /positions: Open paper trades
    - /pnl: Today's P&L summary
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
        'warmth': "🌡️",
    }
    
    HELP_TEXT = """
<b>📋 Argus Commands</b>

<b>Status Commands:</b>
/help — Show this help message
/status — Current conditions score and system status
/positions — View open paper trading positions
/pnl — Today's P&L summary
/farm_status — Paper trader farm status
/signal_status — IBIT/BITO signal checklist
/research_status — Research mode status

<b>Trade Confirmation:</b>
Reply <code>yes</code> — Confirm you took the trade
Reply <code>no</code> — Confirm you skipped the trade

<b>Alert Tiers:</b>
🚨 Tier 1 — Immediate action needed
📊 Tier 2 — FYI, no action required
📝 Tier 3 — Logged only (not sent)
"""
    
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
        self.bot_token = bot_token
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        
        self.tier_1_enabled = tier_1_enabled
        self.tier_2_enabled = tier_2_enabled
        self.tier_3_enabled = tier_3_enabled
        
        self.rate_limit_seconds = rate_limit_seconds
        self._last_alert_time: Dict[str, datetime] = {}
        
        # Callbacks for data access (set by orchestrator)
        self._get_conditions: Optional[Callable] = None
        self._get_positions: Optional[Callable] = None
        self._get_pnl: Optional[Callable] = None
        self._get_farm_status: Optional[Callable] = None
        self._get_signal_status: Optional[Callable] = None
        self._get_research_status: Optional[Callable] = None
        self._on_trade_confirmation: Optional[Callable] = None
        
        # Track last signal for yes/no confirmation
        self._last_signal_id: Optional[str] = None
        self._last_signal_time: Optional[datetime] = None
        
        # Application for two-way communication
        self._app: Optional[Application] = None
        self._polling_task: Optional[asyncio.Task] = None
        
        logger.info("Telegram bot initialized (two-way enabled)")
    
    def set_callbacks(
        self,
        get_conditions: Optional[Callable] = None,
        get_positions: Optional[Callable] = None,
        get_pnl: Optional[Callable] = None,
        get_farm_status: Optional[Callable] = None,
        get_signal_status: Optional[Callable] = None,
        get_research_status: Optional[Callable] = None,
        on_trade_confirmation: Optional[Callable] = None,
    ):
        """Set callback functions for data access."""
        self._get_conditions = get_conditions
        self._get_positions = get_positions
        self._get_pnl = get_pnl
        self._get_farm_status = get_farm_status
        self._get_signal_status = get_signal_status
        self._get_research_status = get_research_status
        self._on_trade_confirmation = on_trade_confirmation
    
    async def start_polling(self) -> None:
        """Start listening for incoming messages."""
        try:
            self._app = Application.builder().token(self.bot_token).build()
            
            # Add command handlers
            self._app.add_handler(CommandHandler("help", self._cmd_help))
            self._app.add_handler(CommandHandler("status", self._cmd_status))
            self._app.add_handler(CommandHandler("positions", self._cmd_positions))
            self._app.add_handler(CommandHandler("pnl", self._cmd_pnl))
            self._app.add_handler(CommandHandler("farm_status", self._cmd_farm_status))
            self._app.add_handler(CommandHandler("signal_status", self._cmd_signal_status))
            self._app.add_handler(CommandHandler("research_status", self._cmd_research_status))
            
            # Add message handler for yes/no responses
            self._app.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._handle_message
            ))
            
            # Initialize and start polling
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling(drop_pending_updates=True)
            
            logger.info("Telegram bot polling started")
        except Exception as e:
            logger.error(f"Failed to start Telegram polling: {e}")
    
    async def stop_polling(self) -> None:
        """Stop listening for incoming messages."""
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
                logger.info("Telegram bot polling stopped")
            except Exception as e:
                logger.error(f"Error stopping Telegram polling: {e}")
    
    # -------------------------------------------------------------------------
    # COMMAND HANDLERS
    # -------------------------------------------------------------------------
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        await update.message.reply_text(self.HELP_TEXT, parse_mode="HTML")
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        try:
            if self._get_conditions:
                conditions = await self._get_conditions()
                score = conditions.get('score', 0)
                warmth = conditions.get('warmth_label', 'unknown')
                market_time = conditions.get('market_time_et', 'N/A')
                updated = conditions.get('last_updated_et') or "N/A"
                data_status = conditions.get('data_status', {})
                
                lines = [
                    f"🌡️ <b>CONDITIONS: {score}/10 ({warmth.upper()})</b>",
                    "",
                    f"• BTC IV: {conditions.get('btc_iv', 'N/A')}%",
                    f"• Funding: {conditions.get('funding', 'N/A')}",
                    f"• Market: {'🟢 OPEN' if conditions.get('market_open') else '🔴 CLOSED'}",
                    f"• Market Time: {market_time}",
                    f"• Updated: {updated}",
                    "",
                ]
                if data_status:
                    lines.append("<b>🧭 Data Freshness</b>")
                    for label, info in data_status.items():
                        state = info.get('status')
                        if state == "ok":
                            emoji = "✅"
                        elif state == "pending":
                            emoji = "⏳"
                        elif state == "disabled":
                            emoji = "🚫"
                        else:
                            emoji = "⚠️"
                        last_seen = info.get('last_seen_et', 'N/A')
                        age = info.get('age_human')
                        age_suffix = f" ({age})" if age else ""
                        lines.append(f"{emoji} {label}: {last_seen}{age_suffix}")
                await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            else:
                await update.message.reply_text(
                    "⚠️ Status not available. Conditions monitor not connected.",
                    parse_mode="HTML"
                )
        except Exception as e:
            logger.error(f"Error in /status: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /positions command."""
        try:
            if self._get_positions:
                positions = await self._get_positions()
                
                if not positions:
                    await update.message.reply_text("📭 No open paper positions.")
                    return
                
                lines = ["<b>📊 Open Paper Positions</b>", ""]
                
                for pos in positions:
                    # Handle both individual position and grouped formats
                    if 'count' in pos:
                        # Grouped format from farm
                        strategy = pos.get('strategy', 'unknown')
                        lines.append(
                            f"🔸 <b>{pos.get('symbol')}</b> {pos.get('sample_strikes', '')}\n"
                            f"   Strategy: {strategy} | Positions: {pos.get('count')} "
                            f"across {pos.get('traders_entered', 0)} traders"
                        )
                    else:
                        # Individual position format
                        pnl = pos.get('unrealized_pnl', 0)
                        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                        lines.append(
                            f"{pnl_emoji} {pos.get('symbol')} {pos.get('strikes', '')}\n"
                            f"   Entry: ${pos.get('entry_credit', 0):.2f} | "
                            f"P&L: ${pnl:+.2f}"
                        )
                    lines.append("")
                
                await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            else:
                await update.message.reply_text(
                    "⚠️ Positions not available. Paper trader not connected.",
                    parse_mode="HTML"
                )
        except Exception as e:
            logger.error(f"Error in /positions: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def _cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pnl command."""
        try:
            if self._get_pnl:
                pnl = await self._get_pnl()
                
                today_pnl = pnl.get('today_pnl', 0)
                month_pnl = pnl.get('month_pnl', 0)
                today_emoji = "🟢" if today_pnl >= 0 else "🔴"
                month_emoji = "🟢" if month_pnl >= 0 else "🔴"
                
                year_pnl = pnl.get('year_pnl', 0)
                year_emoji = "🟢" if year_pnl >= 0 else "🔴"
                lines = [
                    "<b>💰 P&L Summary</b>",
                    "",
                    f"{today_emoji} Today: ${today_pnl:+.2f} ({pnl.get('today_pct', 0):+.1f}%)",
                    f"{month_emoji} Month-to-Date: ${month_pnl:+.2f} ({pnl.get('month_pct', 0):+.1f}%)",
                    f"{year_emoji} Year-to-Date: ${year_pnl:+.2f} ({pnl.get('year_pct', 0):+.1f}%)",
                    "",
                    f"Opened today: {pnl.get('opened_today', 0)}",
                    f"Closed today: {pnl.get('trades_today', 0)}",
                    f"MTD closed trades: {pnl.get('trades_mtd', 0)}",
                    f"Win rate (MTD): {pnl.get('win_rate_mtd', 0):.0f}%",
                    f"Open positions: {pnl.get('open_positions', 0)}",
                    "",
                    f"<i>Paper account: ${pnl.get('account_value', 5000):.2f}</i>",
                ]
                await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            else:
                await update.message.reply_text(
                    "⚠️ P&L not available. Paper trader not connected.",
                    parse_mode="HTML"
                )
        except Exception as e:
            logger.error(f"Error in /pnl: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_farm_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /farm_status command."""
        try:
            if not self._get_farm_status:
                await update.message.reply_text(
                    "⚠️ Farm status not available. Paper trader farm not connected.",
                    parse_mode="HTML"
                )
                return
            status = await self._get_farm_status()
            if not status:
                await update.message.reply_text(
                    "⚠️ Farm status not available. Paper trader farm not connected.",
                    parse_mode="HTML"
                )
                return
            last_eval = status.get("last_evaluation_time")
            if last_eval:
                try:
                    dt = datetime.fromisoformat(last_eval)
                except ValueError:
                    dt = None
                if dt:
                    last_eval = dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
            lines = [
                "<b>🚜 Paper Trader Farm Status</b>",
                "",
                f"Configs: {status.get('total_configs', 0):,}",
                f"Active traders: {status.get('active_traders', 0):,}",
                f"Last evaluation: {last_eval or 'N/A'}",
                f"Last symbol: {status.get('last_evaluation_symbol') or 'N/A'}",
                f"Traders entered (last evaluation): {status.get('last_evaluation_entered', 0):,}",
            ]
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
        except Exception as e:
            logger.error(f"Error in /farm_status: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_signal_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /signal_status command."""
        try:
            if not self._get_signal_status:
                await update.message.reply_text(
                    "⚠️ Signal status not available. Detectors not connected.",
                    parse_mode="HTML"
                )
                return
            status = await self._get_signal_status()
            if not status:
                await update.message.reply_text(
                    "⚠️ Signal status not available. Detectors not connected.",
                    parse_mode="HTML"
                )
                return
            lines = ["<b>🧪 IBIT/BITO Signal Checklist</b>", ""]
            for symbol, checklist in status.items():
                lines.append(f"<b>{symbol}</b>")
                lines.append(f"• BTC IV: {checklist.get('btc_iv', 0):.1f}% "
                             f"(≥ {checklist.get('btc_iv_threshold', 0)}% → "
                             f"{'✅' if checklist.get('btc_iv_ok') else '❌'})")
                lines.append(f"• {symbol} Change: {checklist.get('ibit_change_pct', 0):+.2f}% "
                             f"(≤ {checklist.get('ibit_drop_threshold', 0)}% → "
                             f"{'✅' if checklist.get('ibit_drop_ok') else '❌'})")
                lines.append(f"• Combined Score: {checklist.get('combined_score', 0):.2f} "
                             f"(≥ {checklist.get('combined_score_threshold', 0)} → "
                             f"{'✅' if checklist.get('combined_score_ok') else '❌'})")
                iv_rank = checklist.get('iv_rank')
                iv_rank_str = f"{iv_rank:.1f}%" if isinstance(iv_rank, (int, float)) else "N/A"
                lines.append(f"• IV Rank: {iv_rank_str} "
                             f"(≥ {checklist.get('iv_rank_threshold', 0)} → "
                             f"{'✅' if checklist.get('iv_rank_ok') else '❌'})")
                cooldown = checklist.get('cooldown_remaining_hours')
                if cooldown is None:
                    lines.append("• Cooldown: ✅ none")
                else:
                    lines.append(f"• Cooldown: {cooldown:.2f}h remaining")
                data_ready = "✅" if checklist.get('has_btc_iv') and checklist.get('has_ibit_data') else "❌"
                lines.append(f"• Data Ready: {data_ready}")
                lines.append("")
            await update.message.reply_text("\n".join(lines).strip(), parse_mode="HTML")
        except Exception as e:
            logger.error(f"Error in /signal_status: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def _cmd_research_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /research_status command."""
        try:
            if not self._get_research_status:
                await update.message.reply_text(
                    "⚠️ Research status not available.",
                    parse_mode="HTML"
                )
                return
            status = await self._get_research_status()
            if not status:
                await update.message.reply_text(
                    "⚠️ Research status not available.",
                    parse_mode="HTML"
                )
                return
            last_run = status.get("last_run")
            if last_run:
                try:
                    dt = datetime.fromisoformat(last_run)
                except ValueError:
                    dt = None
                if dt:
                    last_run = dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
            aggregate = status.get("aggregate", {})
            farm_status = status.get("status", {})
            lines = [
                "<b>🧪 Research Mode Status</b>",
                "",
                f"Enabled: {'✅' if status.get('research_enabled') else '❌'}",
                f"Interval: {status.get('evaluation_interval_seconds', 0)}s",
                f"Last run: {last_run or 'N/A'}",
                f"Last symbol: {status.get('last_symbol') or 'N/A'}",
                f"Entered last run: {status.get('last_entered', 0):,}",
                f"Data ready: {'✅' if status.get('data_ready') else '❌'}",
                "",
                "<b>📊 Aggregate</b>",
                f"Total trades: {aggregate.get('total_trades', 0):,}",
                f"Win rate: {aggregate.get('win_rate', 0):.1f}%",
                f"Realized P&L: ${aggregate.get('realized_pnl', 0):+.2f}",
                f"Open positions: {aggregate.get('open_positions', 0)}",
                "",
                "<b>👥 Farm</b>",
                f"Active traders: {farm_status.get('active_traders', 0):,}",
                f"Promoted traders: {farm_status.get('promoted_traders', 0):,}",
            ]
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
        except Exception as e:
            logger.error(f"Error in /research_status: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle non-command messages (yes/no trade confirmations)."""
        if not update.message or not update.message.text:
            return
        
        text = update.message.text.strip().lower()
        
        # Check for yes/no response
        if text.startswith("yes") or text.startswith("no"):
            await self._handle_trade_confirmation(update, text)
        else:
            # Unknown message, provide help
            await update.message.reply_text(
                "❓ Unknown command. Type /help for available commands."
            )
    
    async def _handle_trade_confirmation(self, update: Update, text: str) -> None:
        """Handle yes/no trade confirmation."""
        # Check if there's a recent signal to confirm
        if not self._last_signal_id or not self._last_signal_time:
            await update.message.reply_text(
                "⚠️ No recent trade signal to confirm. Wait for an alert first."
            )
            return
        
        # Check if signal is still valid (within 2 hours)
        if datetime.utcnow() - self._last_signal_time > timedelta(hours=2):
            await update.message.reply_text(
                "⚠️ Last signal expired (>2 hours ago). Wait for a new alert."
            )
            return
        
        confirmed = text.startswith("yes")
        
        # Log the confirmation
        if self._on_trade_confirmation:
            try:
                await self._on_trade_confirmation(
                    signal_id=self._last_signal_id,
                    confirmed=confirmed,
                    response_text=text,
                )
            except Exception as e:
                logger.error(f"Error processing trade confirmation: {e}")
        
        # Acknowledge
        if confirmed:
            await update.message.reply_text("✅ Trade confirmed! Good luck!")
        else:
            await update.message.reply_text("📝 Trade skipped. Noted.")
        
        # Clear the signal
        self._last_signal_id = None
        self._last_signal_time = None
    
    def set_last_signal(self, signal_id: str) -> None:
        """Set the last signal ID for yes/no confirmation."""
        self._last_signal_id = signal_id
        self._last_signal_time = datetime.utcnow()
    
    # -------------------------------------------------------------------------
    # EXISTING SENDING METHODS
    # -------------------------------------------------------------------------
    
    def _should_send(self, tier: int, alert_type: str) -> bool:
        """Check if alert should be sent based on tier and rate limiting."""
        # Check tier
        if tier == 1 and not self.tier_1_enabled:
            return False
        if tier == 2 and not self.tier_2_enabled:
            return False
        if tier == 3 and not self.tier_3_enabled:
            return False
        
        # Rate limiting for ALL tiers (including Tier 1)
        # Tier 1: 30 minute minimum cooldown per alert type
        # Tier 2+: Uses configured rate_limit_seconds
        key = f"{tier}_{alert_type}"
        now = datetime.utcnow()
        
        # Tier 1 gets longer cooldown to prevent spam
        if tier == 1:
            cooldown_seconds = 30 * 60  # 30 minutes for Tier 1
        else:
            cooldown_seconds = self.rate_limit_seconds
        
        if key in self._last_alert_time:
            elapsed = (now - self._last_alert_time[key]).seconds
            if elapsed < cooldown_seconds:
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
        action: Optional[str] = None,
        signal_id: Optional[str] = None,
    ) -> bool:
        """
        Send a formatted alert message.
        
        Args:
            tier: Alert tier (1, 2, or 3)
            alert_type: Type of opportunity
            title: Alert title
            details: Details to include
            action: Suggested action (optional)
            signal_id: If provided, sets for yes/no confirmation
            
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
        
        # Add confirmation prompt for Tier 1
        if tier == 1 and signal_id:
            lines.append("")
            lines.append("━━━━━━━━━━━━━━━━━━━━━")
            lines.append("Reply <b>yes</b> if you took this trade")
            lines.append("Reply <b>no</b> if you skipped")
            self.set_last_signal(signal_id)
        
        # Add timestamp
        lines.append("")
        lines.append(f"<i>{self._now_et().strftime('%Y-%m-%d %H:%M:%S %Z')}</i>")
        
        text = "\n".join(lines)
        
        # Tier 1 gets sound, tier 2+ is silent
        silent = tier > 1
        
        return await self.send_message(text, disable_notification=silent)

    @staticmethod
    def _now_et() -> datetime:
        """Return current time in US/Eastern."""
        return datetime.now(ZoneInfo("America/New_York"))
    
    async def send_conditions_alert(
        self,
        score: int,
        label: str,
        details: Dict[str, Any],
        implication: str,
    ) -> bool:
        """Send a conditions warming/cooling alert."""
        emoji_map = {
            'cooling': "❄️",
            'neutral': "➖",
            'warming': "🔥",
            'prime': "🎯",
        }
        
        emoji = emoji_map.get(label.lower(), "🌡️")
        
        lines = [
            f"{emoji} <b>CONDITIONS: {score}/10 ({label.upper()})</b>",
            "",
        ]
        
        for key, value in details.items():
            lines.append(f"• {key}: {value}")
        
        lines.append("")
        lines.append(f"💡 <b>Implication:</b> {implication}")
        lines.append("")
        lines.append(f"<i>{self._now_et().strftime('%H:%M:%S %Z')}</i>")
        
        return await self.send_message("\n".join(lines))
    
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
    
    async def send_daily_summary(self, stats: Dict) -> bool:
        """Send daily summary report."""
        lines = [
            f"📈 <b>Daily Market Monitor Summary</b>",
            f"<i>{self._now_et().strftime('%Y-%m-%d')}</i>",
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
