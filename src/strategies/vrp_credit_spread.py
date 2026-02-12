"""
VRP Credit Spread Strategy (Reference Strategy #2)
=================================================

Generates credit spread signals when Volatility Risk Premium (VRP) is high,
conditioned on market and symbol regimes.

Signal Logic:
1.  Monitor IV (from Greeks/Snapshots) and RV (Realized Volatility).
2.  Compute VRP = IV - RV.
3.  If VRP > Threshold AND Regime == BULLISH/NEUTRAL AND Volatility is STABLE:
    - Emit intent to SELL Put Spread.

IV Source Selection (data-source policy):
- PRIMARY: Tastytrade snapshots (``atm_iv`` / surface fields).
- FALLBACK: Derived IV computed from Tastytrade bid/ask quotes
  when ``atm_iv`` is missing.
- Alpaca snapshots are structural cross-check only and are NOT
  used for IV unless explicitly enabled via ``allow_alpaca_iv``.
"""

import logging
import math
from typing import Dict, List, Any, Optional

from src.analysis.replay_harness import ReplayStrategy, TradeIntent
from src.core.outcome_engine import BarData, OutcomeResult
from src.core.regimes import VolRegime, TrendRegime

logger = logging.getLogger("argus.strategies.vrp_credit_spread")

# Provider names (must match data_sources policy values)
_TASTYTRADE = "tastytrade"
_ALPACA = "alpaca"


def _derive_iv_from_quotes(snapshot: Any) -> Optional[float]:
    """Attempt to derive ATM implied volatility from bid/ask quotes.

    This is a simplified Newton-Raphson IV solver used as a fallback
    when the snapshot's ``atm_iv`` field is absent.  It uses the
    midpoint of the closest-to-ATM put as a proxy.

    Returns None if derivation is not possible.
    """
    try:
        import json as _json

        quotes_raw = None
        if hasattr(snapshot, "quotes_json"):
            quotes_raw = snapshot.quotes_json
        elif isinstance(snapshot, dict):
            quotes_raw = snapshot.get("quotes_json")

        if not quotes_raw:
            return None

        quotes = _json.loads(quotes_raw) if isinstance(quotes_raw, str) else quotes_raw
        puts = quotes.get("puts", [])
        if not puts:
            return None

        underlying = None
        if hasattr(snapshot, "underlying_price"):
            underlying = snapshot.underlying_price
        elif isinstance(snapshot, dict):
            underlying = snapshot.get("underlying_price")

        if not underlying or underlying <= 0:
            return None

        # Find closest-to-ATM put with a usable bid/ask
        best = None
        best_dist = float("inf")
        for p in puts:
            strike = p.get("strike")
            bid = p.get("bid")
            ask = p.get("ask")
            if strike is None or bid is None or ask is None:
                continue
            dist = abs(strike - underlying)
            if dist < best_dist and bid > 0 and ask > 0:
                best_dist = dist
                best = p

        if best is None:
            return None

        mid = (best["bid"] + best["ask"]) / 2.0
        strike = best["strike"]
        # Rough Brenner-Subrahmanyam approximation:  IV ≈ mid / (0.4 * S * sqrt(T))
        # Assume ~14 DTE (0.038 years) if not provided.
        dte_years = best.get("dte_years", 14 / 365.0)
        if dte_years <= 0:
            dte_years = 14 / 365.0
        iv_approx = mid / (0.4 * underlying * math.sqrt(dte_years))
        if 0.01 <= iv_approx <= 3.0:
            return round(iv_approx, 6)
        return None
    except Exception:
        return None


def _select_iv_from_snapshots(
    visible_snapshots: List[Any],
    allow_alpaca_iv: bool = False,
) -> Optional[float]:
    """Select the best IV value from visible snapshots.

    Selection order:
    1. Latest Tastytrade snapshot with non-null ``atm_iv``.
    2. Derived IV from latest Tastytrade snapshot's bid/ask quotes.
    3. If ``allow_alpaca_iv`` is True, latest Alpaca snapshot with
       non-null ``atm_iv`` (structural cross-check only).
    4. None if nothing is available.
    """
    if not visible_snapshots:
        return None

    def _provider(snap: Any) -> str:
        if hasattr(snap, "source"):
            return getattr(snap, "source", "")
        if hasattr(snap, "provider"):
            return getattr(snap, "provider", "")
        if isinstance(snap, dict):
            return snap.get("provider", snap.get("source", ""))
        return ""

    def _atm_iv(snap: Any) -> Optional[float]:
        if hasattr(snap, "atm_iv"):
            return snap.atm_iv
        if isinstance(snap, dict):
            return snap.get("atm_iv")
        return None

    # Pass 1: Tastytrade atm_iv (most recent first)
    for snap in reversed(visible_snapshots):
        if _provider(snap) == _TASTYTRADE:
            iv = _atm_iv(snap)
            if iv is not None:
                return iv

    # Pass 2: Derived IV from Tastytrade quotes
    for snap in reversed(visible_snapshots):
        if _provider(snap) == _TASTYTRADE:
            iv = _derive_iv_from_quotes(snap)
            if iv is not None:
                logger.debug("Using derived IV %.4f from Tastytrade quotes", iv)
                return iv

    # Pass 3: Alpaca IV (only if explicitly allowed)
    if allow_alpaca_iv:
        for snap in reversed(visible_snapshots):
            if _provider(snap) == _ALPACA:
                iv = _atm_iv(snap)
                if iv is not None:
                    logger.debug("Falling back to Alpaca IV %.4f (allow_alpaca_iv=True)", iv)
                    return iv

    return None


class VRPCreditSpreadStrategy(ReplayStrategy):
    """Ref strategy for VRP put spread selling.

    Uses Tastytrade snapshots as the authoritative IV source.
    Falls back to derived IV (Brenner-Subrahmanyam approximation)
    from Tastytrade bid/ask when ``atm_iv`` is absent.
    Alpaca IV is only used if ``allow_alpaca_iv`` is explicitly
    set in thresholds.
    """

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self._thresholds = thresholds or {
            "min_vrp": 0.05,       # 5 points of IV over RV
            "max_vol_regime": "VOL_NORMAL",
            "avoid_trend": "TREND_DOWN",
        }
        self._allow_alpaca_iv = bool(self._thresholds.get("allow_alpaca_iv", False))
        self.last_close = 0.0
        self.last_iv = None
        self.last_iv_source = None  # "tastytrade", "derived", or "alpaca"
        self.last_rv = None
        self._logged_no_iv = False
        self._logged_no_rv = False
        self._logged_vrp_or_regime = False

    @property
    def strategy_id(self) -> str:
        return "VRP_PUT_SPREAD_V1"

    def on_bar(
        self,
        bar: BarData,
        sim_ts_ms: int,
        session_regime: str,
        visible_outcomes: Dict[int, OutcomeResult],
        *,
        visible_regimes: Optional[Dict[str, Dict[str, Any]]] = None,
        visible_snapshots: Optional[List[Any]] = None,
    ) -> None:
        self.last_close = bar.close
        self.visible_regimes = visible_regimes or {}

        # Extract IV using provider-aware selection
        iv = _select_iv_from_snapshots(
            visible_snapshots or [],
            allow_alpaca_iv=self._allow_alpaca_iv,
        )
        if iv is not None:
            self.last_iv = iv

        # Extract realized_vol from latest outcome (if available)
        for ts in sorted(visible_outcomes.keys(), reverse=True):
            o = visible_outcomes[ts]
            if o.realized_vol is not None:
                self.last_rv = o.realized_vol
                break

    def generate_intents(self, sim_ts_ms: int) -> List[TradeIntent]:
        intents = []

        # 1. Check if we have data
        if self.last_iv is None:
            if not self._logged_no_iv:
                logger.warning("VRPCreditSpreadStrategy: no IV yet (need option snapshot with atm_iv). Snapshots may lack atm_iv or recv_ts_ms may be after sim time.")
                self._logged_no_iv = True
            return []

        # 2. Compute VRP
        vrp = self.last_iv - self.last_rv

        # 3. Check Regimes
        symbol_regime = self.visible_regimes.get("SPY", {})
        vol = symbol_regime.get("vol_regime", "UNKNOWN")
        trend = symbol_regime.get("trend_regime", "UNKNOWN")

        # 4. Gating Logic
        if vrp > self._thresholds["min_vrp"]:
            if vol != "VOL_SPIKE" and trend != "TREND_DOWN":
                intents.append(TradeIntent(
                    symbol="SPY",
                    side="SELL",
                    quantity=1,
                    intent_type="OPEN",
                    tag="VRP_EDGE",
                    meta={
                        "vrp": vrp,
                        "iv": self.last_iv,
                        "rv": self.last_rv,
                        "vol": vol,
                        "trend": trend,
                    }
                ))

        return intents

    def finalize(self) -> Dict[str, Any]:
        return {
            "last_iv": self.last_iv,
            "last_rv": self.last_rv,
        }
