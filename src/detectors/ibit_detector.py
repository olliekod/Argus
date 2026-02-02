"""
Crypto ETF Options Opportunity Detector
=======================================

Detects opportunities to sell put spreads on crypto ETFs (IBIT, BITO, etc.).
Integrates with TradeCalculator for precise recommendations.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from .base_detector import BaseDetector
from ..core.utils import calculate_z_score, calculate_mean
from ..core.economic_calendar import EconomicCalendar
from ..analysis.trade_calculator import TradeCalculator, TradeRecommendation
from ..analysis.paper_trader import PaperTrader
from ..connectors.ibit_options_client import IBITOptionsClient

logger = logging.getLogger(__name__)


class IBITDetector(BaseDetector):
    """
    Detector for IBIT options opportunities (v2).
    
    Strategy: When BTC volatility spikes and IBIT drops significantly,
    sell put spreads to collect elevated premium.
    
    This is a MANUAL trading strategy - generates precise instructions
    for executing on Robinhood.
    
    v2 Features:
    - Real options chain data via yfinance
    - Greeks calculation (Delta, Theta, Vega)
    - Dynamic position sizing based on PoP
    - Economic calendar blackout warnings
    - IV Rank validation
    """
    
    def __init__(self, config: Dict[str, Any], db, symbol: str = "IBIT"):
        super().__init__(config, db)
        
        # Symbol this detector tracks
        self.symbol = symbol.upper()
        
        # Thresholds
        self.btc_iv_threshold = config.get('btc_iv_threshold', 70)
        self.drop_threshold = config.get('drop_threshold', config.get('ibit_drop_threshold', -3))
        self.combined_score_threshold = config.get('combined_score_threshold', 1.5)
        self.iv_rank_threshold = config.get('iv_rank_threshold', 50)
        
        # Alert cooldown
        self.cooldown_hours = config.get('cooldown_hours', 4)
        self._last_alert_time: Optional[datetime] = None
        
        # Data cache
        self._btc_iv_history: List[float] = []
        self._price_history: List[Dict] = []
        self._current_btc_iv: float = 0
        self._current_data: Optional[Dict] = None
        
        # Initialize components
        self.account_size = config.get('account_size', 3000)
        self.trade_calculator = TradeCalculator(account_size=self.account_size, symbol=symbol)
        self.options_client = IBITOptionsClient(symbol=symbol)
        self.economic_calendar = EconomicCalendar()
        
        # Paper trading - Farm integration
        self.paper_trading_enabled = config.get('paper_trading', True)
        self.paper_trader: Optional[PaperTrader] = None
        self.paper_trader_farm = None  # Set by orchestrator
        
        # Telegram callback for paper trade notifications
        self._telegram_callback = None
        
        self.logger.info(
            f"{symbol}Detector initialized: BTC IV >{self.btc_iv_threshold}%, "
            f"{symbol} drop >{abs(self.drop_threshold)}%, "
            f"IV Rank >{self.iv_rank_threshold}%"
        )
    
    def set_telegram_callback(self, callback) -> None:
        """Set callback for sending Telegram notifications."""
        self._telegram_callback = callback
        
    def set_paper_trader_farm(self, farm) -> None:
        """Connect the larger scale paper trader farm."""
        self.paper_trader_farm = farm
    
    def update_btc_iv(self, iv: float) -> None:
        """Update BTC IV data from Deribit."""
        self._current_btc_iv = iv
        self._btc_iv_history.append(iv)
        if len(self._btc_iv_history) > 168:
            self._btc_iv_history = self._btc_iv_history[-168:]
    
    def update_ibit_data(self, data: Dict) -> None:
        """Update IBIT price data from Yahoo Finance."""
        self._current_ibit_data = data
        self._ibit_price_history.append({
            'price': data.get('price', 0),
            'timestamp': data.get('timestamp'),
        })
        if len(self._ibit_price_history) > 720:
            self._ibit_price_history = self._ibit_price_history[-720:]

    def get_signal_checklist(self) -> Dict[str, Any]:
        """Return a checklist of conditions for an IBIT/BITO signal."""
        btc_iv = self._current_btc_iv or 0
        ibit_change = 0.0
        ibit_price = 0.0
        if self._current_ibit_data:
            ibit_change = self._current_ibit_data.get('price_change_pct', 0)
            ibit_price = self._current_ibit_data.get('price', 0)

        iv_elevated = btc_iv >= self.btc_iv_threshold
        ibit_dropped = ibit_change <= self.ibit_drop_threshold
        iv_score = btc_iv / self.btc_iv_threshold if self.btc_iv_threshold > 0 else 0
        drop_score = abs(ibit_change) / abs(self.ibit_drop_threshold) if ibit_change < 0 else 0
        combined_score = (iv_score + drop_score) / 2 if (self.btc_iv_threshold and self.ibit_drop_threshold) else 0

        cooldown_remaining = None
        if self._last_alert_time:
            elapsed = (datetime.now(timezone.utc) - self._last_alert_time).total_seconds() / 3600
            cooldown_remaining = max(0.0, self.cooldown_hours - elapsed)

        iv_rank = None
        try:
            status = self.options_client.get_market_status()
            iv_rank = status.get('iv_rank')
        except Exception:
            iv_rank = None

        return {
            'symbol': self.symbol,
            'btc_iv': btc_iv,
            'btc_iv_threshold': self.btc_iv_threshold,
            'btc_iv_ok': iv_elevated,
            'ibit_price': ibit_price,
            'ibit_change_pct': ibit_change,
            'ibit_drop_threshold': self.ibit_drop_threshold,
            'ibit_drop_ok': ibit_dropped,
            'combined_score': combined_score,
            'combined_score_threshold': self.combined_score_threshold,
            'combined_score_ok': combined_score >= self.combined_score_threshold,
            'iv_rank': iv_rank,
            'iv_rank_threshold': self.iv_rank_threshold,
            'iv_rank_ok': iv_rank is not None and iv_rank >= self.iv_rank_threshold,
            'cooldown_remaining_hours': cooldown_remaining,
            'has_btc_iv': bool(self._current_btc_iv),
            'has_ibit_data': bool(self._current_ibit_data),
        }

    def get_research_signal(
        self,
        conditions_score: int,
        conditions_label: str,
        btc_change_24h_pct: float,
        timestamp: str,
    ) -> Optional[Dict[str, Any]]:
        """Build a research-mode signal without gating thresholds."""
        if not self._current_btc_iv or not self._current_ibit_data:
            return None

        btc_change_decimal = (btc_change_24h_pct or 0) / 100
        ibit_change_pct = self._current_ibit_data.get('price_change_pct', 0)
        ibit_change_decimal = ibit_change_pct / 100

        recommendation = self.trade_calculator.generate_recommendation(
            btc_change_24h=btc_change_decimal,
            ibit_change_24h=ibit_change_decimal,
            force=True,
        )
        if not recommendation:
            return None

        direction = 'neutral'
        if btc_change_24h_pct > 0.2:
            direction = 'bullish'
        elif btc_change_24h_pct < -0.2:
            direction = 'bearish'

        return {
            'symbol': self.symbol,
            'iv': self._current_btc_iv,
            'warmth': conditions_score,
            'dte': recommendation.dte,
            'pop': recommendation.probability_of_profit,
            'pot': recommendation.probability_of_touch_stop,
            'gap_risk': 0,
            'direction': direction,
            'strikes': f"{recommendation.short_strike}/{recommendation.long_strike}",
            'expiry': recommendation.expiration,
            'credit': recommendation.net_credit,
            'iv_rank': recommendation.iv_rank,
            'conditions_label': conditions_label,
            'btc_change_pct': btc_change_24h_pct,
            'ibit_change_pct': ibit_change_pct,
            'timestamp': timestamp,
        }
    
    def _check_market_hours(self) -> bool:
        """Check if US stock market is open (9:30 AM - 4:00 PM EST)."""
        now = datetime.now()
        is_weekday = now.weekday() < 5
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return is_weekday and market_open <= now <= market_close
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Analyze for IBIT options opportunities.
        
        Flow:
        1. Update data from market feed
        2. Check basic thresholds (BTC IV, IBIT drop)
        3. Check IV Rank threshold
        4. Check economic calendar for blackouts
        5. Generate full trade recommendation with Greeks
        
        Args:
            market_data: Either Deribit IV data or IBIT price data
            
        Returns:
            Detection with full TradeRecommendation if opportunity found
        """
        if not self.enabled:
            return None
        
        # Update data based on source
        if market_data.get('source') == 'deribit':
            iv = market_data.get('atm_iv', 0)
            if iv > 0:
                self.update_btc_iv(iv)
        elif market_data.get('source') == 'yahoo':
            self.update_ibit_data(market_data)
        
        # Need both data points
        if not self._current_btc_iv or not self._current_ibit_data:
            return None
        
        # Check cooldown
        if self._last_alert_time:
            elapsed = (datetime.now(timezone.utc) - self._last_alert_time).total_seconds() / 3600
            if elapsed < self.cooldown_hours:
                return None
        
        # === BASIC THRESHOLD CHECKS ===
        btc_iv = self._current_btc_iv
        ibit_change = self._current_ibit_data.get('price_change_pct', 0)
        ibit_price = self._current_ibit_data.get('price', 0)
        
        iv_elevated = btc_iv >= self.btc_iv_threshold
        ibit_dropped = ibit_change <= self.ibit_drop_threshold
        
        # Combined score
        iv_score = btc_iv / self.btc_iv_threshold if self.btc_iv_threshold > 0 else 0
        drop_score = abs(ibit_change) / abs(self.ibit_drop_threshold) if ibit_change < 0 else 0
        combined_score = (iv_score + drop_score) / 2
        
        if combined_score < self.combined_score_threshold:
            return None
        
        if not (iv_elevated or ibit_dropped):
            return None
        
        # === IV RANK CHECK ===
        try:
            market_status = self.options_client.get_market_status()
            iv_rank = market_status.get('iv_rank', 50)
            
            if iv_rank < self.iv_rank_threshold:
                self.logger.debug(f"IV Rank {iv_rank:.1f}% below threshold {self.iv_rank_threshold}%")
                # Still log but don't alert
                return None
        except Exception as e:
            self.logger.warning(f"Could not check IV Rank: {e}")
            iv_rank = 50  # Assume neutral
        
        # === ECONOMIC CALENDAR CHECK ===
        blackout_warning = None
        is_blackout, blackout_event = self.economic_calendar.is_blackout_period()
        if is_blackout:
            blackout_warning = self.economic_calendar.get_blackout_warning()
        
        # === GENERATE TRADE RECOMMENDATION ===
        try:
            recommendation = self.trade_calculator.generate_recommendation(
                btc_change_24h=market_data.get('btc_change_24h', 0),
                ibit_change_24h=ibit_change / 100,  # Convert to decimal
                force=True,  # We already passed threshold checks
            )
            
            if not recommendation:
                self.logger.warning("Trade calculator could not generate recommendation")
                return None
            
            # Add blackout warning to recommendation
            if blackout_warning:
                recommendation.warnings.append(blackout_warning)
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            return None
        
        # === GPU SURFACE ANALYSIS (ANOMALY DETECTION) ===
        skew_anomaly = None
        try:
            # Get the whole chain for nearest expiration
            chain = self.options_client.get_chain(recommendation.expiration)
            if chain:
                from ..analysis.gpu_engine import get_gpu_engine
                gpu = get_gpu_engine()
                
                strikes = [p['strike'] for p in chain['puts']]
                sigmas = [p['iv'] for p in chain['puts']]
                
                # Batch calculate Greeks on GPU
                T = GreeksEngine.dte_to_years(recommendation.dte)
                batch = gpu.batch_greeks(
                    ibit_price, strikes, T, sigmas, option_type='put'
                )
                
                # Look for IV 'hump' (skew deviation)
                # If a strike's IV is >3% higher than both its neighbors, it's an anomaly
                for i in range(1, len(sigmas) - 1):
                    if sigmas[i] > sigmas[i-1] + 3 and sigmas[i] > sigmas[i+1] + 3:
                        skew_anomaly = f"IV Skew Anomaly at ${strikes[i]}"
                        recommendation.warnings.append(f"🎯 SKEW ANOMALY: ${strikes[i]} strike is overpriced (GPU found {sigmas[i]:.1f}% IV vs neighbors)")
                        break
        except Exception as e:
            self.logger.debug(f"GPU Surface analysis skipped: {e}")
            
        # === BUILD DETECTION ===
        iv_z_score = 0
        if len(self._btc_iv_history) > 10:
            iv_z_score = calculate_z_score(btc_iv, self._btc_iv_history)
        
        detection = self.create_detection(
            opportunity_type='ibit_options',
            asset='IBIT',
            exchange='robinhood',
            detection_data={
                # Market conditions
                'btc_iv': btc_iv,
                'btc_iv_z_score': iv_z_score,
                'ibit_price': ibit_price,
                'ibit_change_24h': ibit_change,
                'combined_score': combined_score,
                'iv_rank': iv_rank,
                
                # Trade recommendation
                'expiration': recommendation.expiration,
                'dte': recommendation.dte,
                'short_strike': recommendation.short_strike,
                'long_strike': recommendation.long_strike,
                'spread_width': recommendation.spread_width,
                'net_credit': recommendation.net_credit,
                'max_risk': recommendation.max_risk,
                'break_even': recommendation.break_even,
                
                # Greeks
                'net_delta': recommendation.net_delta,
                'net_theta': recommendation.net_theta,
                'net_vega': recommendation.net_vega,
                'probability_of_profit': recommendation.probability_of_profit,
                
                # Position sizing
                'position_size_pct': recommendation.position_size_pct,
                'num_contracts': recommendation.num_contracts,
                'capital_at_risk': recommendation.capital_at_risk,
                
                # Blackout
                'in_blackout': is_blackout,
                'blackout_warning': blackout_warning,
                
                # Full recommendation object for Telegram
                'recommendation': recommendation,
            },
            current_price=ibit_price,
            estimated_edge_bps=int(recommendation.net_credit / recommendation.max_risk * 100) if recommendation.max_risk > 0 else 100,
            alert_tier=1,
            notes=f"IBIT put spread: ${recommendation.short_strike:.0f}/${recommendation.long_strike:.0f}, "
                  f"Credit: ${recommendation.net_credit:.2f}, PoP: {recommendation.probability_of_profit:.0f}%"
        )
        
        self._last_alert_time = datetime.now(timezone.utc)
        await self.log_detection(detection)
        
        # === AUTO-LOG PAPER TRADE ===
        if self.paper_trading_enabled:
            try:
                if self.paper_trader is None:
                    self.paper_trader = PaperTrader(self.db)
                
                paper_trade = await self.paper_trader.open_trade(
                    recommendation=recommendation,
                    btc_iv=btc_iv,
                )
                self.logger.info(f"Master paper trade #{paper_trade.id} logged")
                detection['detection_data']['paper_trade_id'] = paper_trade.id
                
                # === FARM EVALUATION ===
                farm_count = 0
                if self.paper_trader_farm:
                    # Pass signal to the 400K traders
                    farm_trades = await self.paper_trader_farm.evaluate_signal(
                        symbol=self.symbol,
                        signal_data={
                            'iv': btc_iv,
                            'warmth': combined_score,
                            'dte': recommendation.dte,
                            'pop': recommendation.probability_of_profit,
                            'gap_risk': recommendation.gap_risk_pct if hasattr(recommendation, 'gap_risk_pct') else 0,
                            'direction': 'bullish' if combined_score > 1.5 else 'neutral',
                            'strikes': f"{recommendation.short_strike}/{recommendation.long_strike}",
                            'expiry': recommendation.expiration,
                            'credit': recommendation.net_credit,
                        }
                    )
                    farm_count = len(farm_trades)
                    detection['detection_data']['farm_count'] = farm_count
                    self.logger.info(f"Farm evaluation: {farm_count} traders followed signal")
                
                # Send paper trade notification via Telegram
                if self._telegram_callback:
                    paper_msg = f"""
📝 *PAPER TRADE OPENED*

Trade #{paper_trade.id}
${paper_trade.short_strike:.0f}/${paper_trade.long_strike:.0f} Put Spread
Qty: {paper_trade.quantity} contracts//
Credit: ${paper_trade.entry_credit:.2f}

IV Rank: {paper_trade.iv_rank_at_entry:.0f}%
PoP: {paper_trade.entry_pop:.0f}%

_This is a PAPER trade for tracking._
"""
                    await self._telegram_callback(paper_msg.strip())
                    
            except Exception as e:
                self.logger.warning(f"Failed to log paper trade: {e}")
        
        return detection
    
    def format_telegram_alert(self, detection: Dict) -> str:
        """
        Format detection for Telegram.
        Uses TradeCalculator's formatting.
        """
        data = detection.get('detection_data', {})
        recommendation = data.get('recommendation')
        
        if recommendation:
            alert = self.trade_calculator.format_telegram_alert(recommendation)
            
            # Add farm confirmation
            farm_count = data.get('farm_count', 0)
            if farm_count > 0:
                alert += f"\n\n🚜 *FARM CHECK*: {farm_count:,} parallel traders entered this trade based on their individual thresholds."
            
            return alert
        
        # Fallback to basic format
        return f"""
🎯 *IBIT PUT SPREAD SIGNAL*

BTC IV: {data.get('btc_iv', 0):.0f}%
IBIT: ${data.get('ibit_price', 0):.2f} ({data.get('ibit_change_24h', 0):+.1f}%)
IV Rank: {data.get('iv_rank', 0):.0f}%

Suggested Trade:
SELL: ${data.get('short_strike', 0):.0f} Put
BUY: ${data.get('long_strike', 0):.0f} Put

Check Robinhood for current prices.
"""
    
    def calculate_edge(self, detection: Dict) -> float:
        """Calculate edge based on credit/risk ratio."""
        data = detection.get('detection_data', {})
        credit = data.get('net_credit', 0)
        risk = data.get('max_risk', 1)
        return (credit / risk * 100) if risk > 0 else 100
    
    def get_current_conditions(self) -> Dict:
        """Get current market conditions."""
        iv_rank = 50
        try:
            status = self.options_client.get_market_status()
            iv_rank = status.get('iv_rank', 50)
        except:
            pass
        
        return {
            'btc_iv': self._current_btc_iv,
            'btc_iv_avg': calculate_mean(self._btc_iv_history) if self._btc_iv_history else 0,
            'ibit_price': self._current_ibit_data.get('price', 0) if self._current_ibit_data else 0,
            'ibit_change_24h': self._current_ibit_data.get('price_change_pct', 0) if self._current_ibit_data else 0,
            'iv_rank': iv_rank,
            'market_open': self._check_market_hours(),
            'in_blackout': self.economic_calendar.is_blackout_period()[0],
        }
    
    def is_market_open(self) -> bool:
        """Check if stock market is currently open."""
        return self._check_market_hours()
