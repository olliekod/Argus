"""
Argus Regime Types
==================

Deterministic regime event schemas for Phase 2.

All timestamps are int milliseconds (UTC epoch).
All regimes are computed from BarEvents only (no wall-clock).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict

# ═══════════════════════════════════════════════════════════════════════════
# Schema Version
# ═══════════════════════════════════════════════════════════════════════════

REGIME_SCHEMA_VERSION = 2


# ═══════════════════════════════════════════════════════════════════════════
# Data Quality Flags (bitmask)
# ═══════════════════════════════════════════════════════════════════════════

DQ_NONE = 0
DQ_REPAIRED_INPUT = 1 << 0   # input bar was repaired
DQ_GAP_WINDOW = 1 << 1       # gap detected in bar sequence
DQ_STALE_INPUT = 1 << 2      # indicator not fully warm


# ═══════════════════════════════════════════════════════════════════════════
# Regime Enums
# ═══════════════════════════════════════════════════════════════════════════

class VolRegime(IntEnum):
    """Volatility regime classification."""
    UNKNOWN = 0
    VOL_LOW = 1
    VOL_NORMAL = 2
    VOL_HIGH = 3
    VOL_SPIKE = 4


class TrendRegime(IntEnum):
    """Trend regime classification."""
    UNKNOWN = 0
    RANGE = 1
    TREND_UP = 2
    TREND_DOWN = 3


class SessionRegime(IntEnum):
    """Session regime for market hours."""
    UNKNOWN = 0
    # Equities
    PRE = 1
    RTH = 2
    POST = 3
    CLOSED = 4
    # Crypto
    ASIA = 10
    EU = 11
    US = 12
    OFFPEAK = 13


class RiskRegime(IntEnum):
    """Global risk regime (stub for future)."""
    UNKNOWN = 0
    RISK_ON = 1
    RISK_OFF = 2
    NEUTRAL = 3


# ═══════════════════════════════════════════════════════════════════════════
# String Constants (for serialization)
# ═══════════════════════════════════════════════════════════════════════════

VOL_REGIME_NAMES = {
    VolRegime.UNKNOWN: "UNKNOWN",
    VolRegime.VOL_LOW: "VOL_LOW",
    VolRegime.VOL_NORMAL: "VOL_NORMAL",
    VolRegime.VOL_HIGH: "VOL_HIGH",
    VolRegime.VOL_SPIKE: "VOL_SPIKE",
}

TREND_REGIME_NAMES = {
    TrendRegime.UNKNOWN: "UNKNOWN",
    TrendRegime.RANGE: "RANGE",
    TrendRegime.TREND_UP: "TREND_UP",
    TrendRegime.TREND_DOWN: "TREND_DOWN",
}

SESSION_REGIME_NAMES = {
    SessionRegime.UNKNOWN: "UNKNOWN",
    SessionRegime.PRE: "PRE",
    SessionRegime.RTH: "RTH",
    SessionRegime.POST: "POST",
    SessionRegime.CLOSED: "CLOSED",
    SessionRegime.ASIA: "ASIA",
    SessionRegime.EU: "EU",
    SessionRegime.US: "US",
    SessionRegime.OFFPEAK: "OFFPEAK",
}

RISK_REGIME_NAMES = {
    RiskRegime.UNKNOWN: "UNKNOWN",
    RiskRegime.RISK_ON: "RISK_ON",
    RiskRegime.RISK_OFF: "RISK_OFF",
    RiskRegime.NEUTRAL: "NEUTRAL",
}


# ═══════════════════════════════════════════════════════════════════════════
# Default Thresholds
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_REGIME_THRESHOLDS = {
    # Volatility regime thresholds (vol_z)
    "vol_spike_z": 2.5,
    "vol_high_z": 1.0,
    "vol_low_z": -0.5,
    
    # Trend regime thresholds
    "trend_slope_threshold": 0.5,
    "trend_strength_threshold": 1.0,
    
    # ATR guard
    "atr_epsilon": 1e-8,
    
    # Gap detection
    "gap_tolerance_bars": 1,
    "gap_flag_duration_bars": 2,
    
    # Warmup requirements
    "warmup_bars": 30,
}


def compute_config_hash(thresholds: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of regime configuration.
    
    Returns first 12 chars of SHA256 hex digest.
    """
    # Sort keys for determinism
    canonical = json.dumps(thresholds, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════
# Event Dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SymbolRegimeEvent:
    """
    Per-symbol regime classification.
    
    Emitted on every bar for the symbol. Contains volatility
    and trend regimes computed from incremental indicators.
    
    Published to: regimes.symbol
    """
    symbol: str
    timeframe: int              # bar_duration in seconds
    timestamp_ms: int           # bar timestamp (UTC epoch ms)
    
    # Regime classifications (string for serialization)
    vol_regime: str             # VOL_LOW | VOL_NORMAL | VOL_HIGH | VOL_SPIKE
    trend_regime: str           # TREND_UP | TREND_DOWN | RANGE
    
    # Core metrics
    atr: float
    atr_pct: float              # ATR / close (normalized)
    vol_z: float                # volatility z-score
    ema_fast: float
    ema_slow: float
    ema_slope: float            # (ema_fast - prev) / ATR
    rsi: float
    
    # Quality & confidence
    confidence: float           # 0-1
    is_warm: bool
    data_quality_flags: int     # bitmask (DQ_*)
    
    # Traceability
    config_hash: str
    
    v: int = REGIME_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class MarketRegimeEvent:
    """
    Per-market regime classification.
    
    Covers session timing and global risk state.
    Session is derived purely from bar timestamp (no wall-clock).
    
    Published to: regimes.market
    """
    market: str                 # CRYPTO | EQUITIES
    timeframe: int              # bar_duration triggering update
    timestamp_ms: int           # bar timestamp (UTC epoch ms)
    
    # Session regime (derived from timestamp)
    session_regime: str         # PRE | RTH | POST | CLOSED (equities)
                                # ASIA | EU | US | OFFPEAK (crypto)
    
    # Global risk (stub for future)
    risk_regime: str            # RISK_ON | RISK_OFF | NEUTRAL | UNKNOWN
    
    # Quality
    confidence: float
    data_quality_flags: int
    
    # Traceability
    config_hash: str
    
    v: int = REGIME_SCHEMA_VERSION


# ═══════════════════════════════════════════════════════════════════════════
# Serialization Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _round_float(val: float, decimals: int = 8) -> float:
    """Round float to fixed decimals for stable serialization."""
    return round(val, decimals)


def _to_int_ms(ts: Any) -> int:
    """
    Convert timestamp to int milliseconds.
    
    Handles backwards compatibility:
    - int: return as-is (assumed ms)
    - float < 2e10: treat as seconds, convert to ms
    - float >= 2e10: treat as ms, convert to int
    """
    if isinstance(ts, int):
        return ts
    if isinstance(ts, float):
        if ts < 2e10:  # seconds
            return int(ts * 1000)
        return int(ts)
    raise ValueError(f"Invalid timestamp type: {type(ts)}")


def canonical_metrics_json(metrics: Dict[str, Any]) -> str:
    """
    Serialize metrics dict to canonical JSON string.
    
    Deterministic: sorted keys, compact separators, rounded floats.
    """
    # Round all float values
    rounded = {}
    for k, v in metrics.items():
        if isinstance(v, float):
            rounded[k] = _round_float(v)
        else:
            rounded[k] = v
    
    return json.dumps(rounded, sort_keys=True, separators=(",", ":"))




def symbol_regime_to_dict(event: SymbolRegimeEvent) -> Dict[str, Any]:
    """Serialize SymbolRegimeEvent to dict for tape/persistence."""
    return {
        "event_type": "symbol_regime",
        "symbol": event.symbol,
        "timeframe": event.timeframe,
        "timestamp_ms": event.timestamp_ms,
        "vol_regime": event.vol_regime,
        "trend_regime": event.trend_regime,
        "atr": _round_float(event.atr),
        "atr_pct": _round_float(event.atr_pct),
        "vol_z": _round_float(event.vol_z),
        "ema_fast": _round_float(event.ema_fast),
        "ema_slow": _round_float(event.ema_slow),
        "ema_slope": _round_float(event.ema_slope),
        "rsi": _round_float(event.rsi),
        "confidence": _round_float(event.confidence),
        "is_warm": event.is_warm,
        "data_quality_flags": event.data_quality_flags,
        "config_hash": event.config_hash,
        "v": event.v,
    }


def dict_to_symbol_regime(d: Dict[str, Any]) -> SymbolRegimeEvent:
    """Deserialize dict to SymbolRegimeEvent.
    
    Backwards compatible: accepts float timestamps and converts to int ms.
    """
    return SymbolRegimeEvent(
        symbol=d["symbol"],
        timeframe=int(d["timeframe"]),
        timestamp_ms=_to_int_ms(d["timestamp_ms"]),
        vol_regime=d["vol_regime"],
        trend_regime=d["trend_regime"],
        atr=float(d["atr"]),
        atr_pct=float(d["atr_pct"]),
        vol_z=float(d["vol_z"]),
        ema_fast=float(d["ema_fast"]),
        ema_slow=float(d["ema_slow"]),
        ema_slope=float(d["ema_slope"]),
        rsi=float(d["rsi"]),
        confidence=float(d["confidence"]),
        is_warm=bool(d["is_warm"]),
        data_quality_flags=int(d["data_quality_flags"]),
        config_hash=str(d["config_hash"]),
        v=int(d.get("v", REGIME_SCHEMA_VERSION)),
    )


def market_regime_to_dict(event: MarketRegimeEvent) -> Dict[str, Any]:
    """Serialize MarketRegimeEvent to dict for tape/persistence."""
    return {
        "event_type": "market_regime",
        "market": event.market,
        "timeframe": event.timeframe,
        "timestamp_ms": event.timestamp_ms,
        "session_regime": event.session_regime,
        "risk_regime": event.risk_regime,
        "confidence": _round_float(event.confidence),
        "data_quality_flags": event.data_quality_flags,
        "config_hash": event.config_hash,
        "v": event.v,
    }


def dict_to_market_regime(d: Dict[str, Any]) -> MarketRegimeEvent:
    """Deserialize dict to MarketRegimeEvent.
    
    Backwards compatible: accepts float timestamps and converts to int ms.
    """
    return MarketRegimeEvent(
        market=d["market"],
        timeframe=int(d["timeframe"]),
        timestamp_ms=_to_int_ms(d["timestamp_ms"]),
        session_regime=d["session_regime"],
        risk_regime=d["risk_regime"],
        confidence=float(d["confidence"]),
        data_quality_flags=int(d["data_quality_flags"]),
        config_hash=str(d["config_hash"]),
        v=int(d.get("v", REGIME_SCHEMA_VERSION)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Market Classification
# ═══════════════════════════════════════════════════════════════════════════

# Symbols that belong to EQUITIES market
EQUITIES_SYMBOLS = {"IBIT", "BITO", "SPY", "QQQ"}

# All other symbols default to CRYPTO


def get_market_for_symbol(symbol: str) -> str:
    """Determine market scope for a symbol."""
    # Check if symbol starts with any equities prefix
    for eq_sym in EQUITIES_SYMBOLS:
        if symbol.upper().startswith(eq_sym):
            return "EQUITIES"
    return "CRYPTO"
