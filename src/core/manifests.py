"""
Strategy Manifest Schemas
=========================

Type-safe manifest structures for the Pantheon Intelligence Engine.

A :class:`StrategyManifest` is the structured artifact produced by Prometheus
and consumed by the Hades backtest engine.  It contains everything needed to
configure, replay, and evaluate a strategy without human intervention.

:class:`AresCritique` captures Ares's adversarial analysis.
:class:`AthenaVerdict` captures Athena's final adjudication.

All models use dataclasses with strict validation so that invalid manifests
are rejected before reaching Hades.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

MANIFEST_SCHEMA_VERSION = 1

# Indicators that are actually implemented in the Hades engine / core
HADES_INDICATOR_CATALOG = frozenset({
    "ema",
    "rsi",
    "macd",
    "vwap",
    "atr",
    "rolling_vol",
    "bollinger_bands",
    "vol_z",
    "ema_slope",
    "spread_pct",
    "volume_pctile",
    "trend_accel",
})

# Regime types available for filtering
REGIME_FILTER_CATALOG = frozenset({
    "vol_regime",
    "trend_regime",
    "liquidity_regime",
    "session_regime",
    "risk_regime",
})

# Valid directions for strategy entry/exit
VALID_DIRECTIONS = frozenset({"LONG", "SHORT", "NEUTRAL"})

# Valid entry types
VALID_ENTRY_TYPES = frozenset({"MARKET", "LIMIT", "STOP", "CONDITIONAL"})

# Valid logic operators for the DSL
VALID_LOGIC_OPS = frozenset({
    "AND", "OR", "NOT",
    "GT", "LT", "GE", "LE", "EQ", "NE",
    "CROSS_ABOVE", "CROSS_BELOW",
    "IN_REGIME", "NOT_IN_REGIME",
})


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class ManifestStatus(str, Enum):
    """Lifecycle status of a strategy manifest."""
    DRAFT = "DRAFT"                 # Prometheus initial output
    CRITIQUED = "CRITIQUED"         # After Ares review
    REVISED = "REVISED"             # After Prometheus revision
    ADJUDICATED = "ADJUDICATED"     # After Athena verdict
    PROMOTED = "PROMOTED"           # Approved for Hades backtest
    REJECTED = "REJECTED"           # Killed by Athena or Ares blockers


class CritiqueSeverity(str, Enum):
    """Severity of an Ares critique finding."""
    BLOCKER = "BLOCKER"       # Stops the case — must be resolved
    ADVISORY = "ADVISORY"     # Allows revision — should be addressed


class CritiqueCategory(str, Enum):
    """Category of an Ares critique finding."""
    OVERFITTING = "OVERFITTING"
    LOOK_AHEAD_BIAS = "LOOK_AHEAD_BIAS"
    DATA_LEAKAGE = "DATA_LEAKAGE"
    PARAMETER_FRAGILITY = "PARAMETER_FRAGILITY"
    REGIME_DEPENDENCY = "REGIME_DEPENDENCY"
    EXECUTION_RISK = "EXECUTION_RISK"
    REGULATORY_RISK = "REGULATORY_RISK"
    LIQUIDITY_RISK = "LIQUIDITY_RISK"
    DRAWDOWN_RISK = "DRAWDOWN_RISK"
    SURVIVORSHIP_BIAS = "SURVIVORSHIP_BIAS"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    OTHER = "OTHER"


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

class ManifestValidationError(Exception):
    """Raised when a strategy manifest fails structural validation."""


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(
            f"{field_name} must be a non-empty string, got {value!r}"
        )


def _validate_signals(signals: List[str]) -> None:
    if not signals:
        raise ManifestValidationError("signals must contain at least one indicator")
    unknown = set(signals) - HADES_INDICATOR_CATALOG
    if unknown:
        raise ManifestValidationError(
            f"Unknown indicators not in Hades catalog: {sorted(unknown)}. "
            f"Available: {sorted(HADES_INDICATOR_CATALOG)}"
        )


def _validate_parameters(parameters: Dict[str, Any]) -> None:
    if not isinstance(parameters, dict):
        raise ManifestValidationError("parameters must be a dict")
    for key, spec in parameters.items():
        if not isinstance(key, str):
            raise ManifestValidationError(f"Parameter key must be str, got {type(key)}")
        if isinstance(spec, dict):
            # Range spec: {"min": x, "max": y, "step": z}
            if "min" in spec and "max" in spec:
                if spec["min"] > spec["max"]:
                    raise ManifestValidationError(
                        f"Parameter '{key}' has min > max: {spec['min']} > {spec['max']}"
                    )


def _validate_logic_node(node: Dict[str, Any], path: str = "root") -> None:
    """Recursively validate a logic tree node."""
    if not isinstance(node, dict):
        raise ManifestValidationError(
            f"Logic node at '{path}' must be a dict, got {type(node)}"
        )
    op = node.get("op")
    if op is None:
        raise ManifestValidationError(
            f"Logic node at '{path}' missing 'op' field"
        )
    if op not in VALID_LOGIC_OPS:
        raise ManifestValidationError(
            f"Unknown logic operator '{op}' at '{path}'. "
            f"Valid: {sorted(VALID_LOGIC_OPS)}"
        )
    # Recursively validate children
    for child_key in ("left", "right", "operand", "condition"):
        child = node.get(child_key)
        if isinstance(child, dict) and "op" in child:
            _validate_logic_node(child, f"{path}.{child_key}")
    children = node.get("children", [])
    for i, child in enumerate(children):
        if isinstance(child, dict) and "op" in child:
            _validate_logic_node(child, f"{path}.children[{i}]")


# ═══════════════════════════════════════════════════════════════════════════
# StrategyManifest
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyManifest:
    """Structured strategy proposal generated by Prometheus.

    This is the primary artifact of the Pantheon research loop.
    It must be structurally valid for direct consumption by the
    Hades backtest engine (replay harness + experiment runner).

    Fields
    ------
    name : str
        Human-readable strategy name (e.g., "Vol-Adjusted Momentum").
    objective : str
        What the strategy aims to capture (e.g., "momentum premium in low-vol").
    signals : List[str]
        Hades-native indicators to use (must be in HADES_INDICATOR_CATALOG).
    entry_logic : Dict[str, Any]
        DSL-compatible logic tree for entry conditions.
    exit_logic : Dict[str, Any]
        DSL-compatible logic tree for exit conditions.
    parameters : Dict[str, Any]
        Tuning ranges for backtest optimization.
        Each key maps to either a fixed value or a range spec:
        ``{"min": float, "max": float, "step": float}``.
    direction : str
        Primary direction: LONG, SHORT, or NEUTRAL.
    universe : List[str]
        Target symbols (e.g., ["IBIT", "SPY", "BTCUSDT"]).
    regime_filters : Dict[str, List[str]]
        Regime conditions under which the strategy operates.
        Keys are regime types (vol_regime, trend_regime, etc.),
        values are acceptable regime states.
    timeframe : int
        Bar duration in seconds (default 60 for 1-min bars).
    holding_period : str
        Expected holding period (e.g., "intraday", "1-5 days").
    risk_per_trade_pct : float
        Maximum risk per trade as fraction of equity.
    """

    name: str
    objective: str
    signals: List[str]
    entry_logic: Dict[str, Any]
    exit_logic: Dict[str, Any]
    parameters: Dict[str, Any]
    direction: str = "LONG"
    universe: List[str] = field(default_factory=lambda: ["IBIT"])
    regime_filters: Dict[str, List[str]] = field(default_factory=dict)
    timeframe: int = 60
    holding_period: str = "intraday"
    risk_per_trade_pct: float = 0.02
    status: ManifestStatus = ManifestStatus.DRAFT
    version: int = MANIFEST_SCHEMA_VERSION

    def validate(self) -> None:
        """Validate all fields. Raises ManifestValidationError on failure."""
        _validate_non_empty_string(self.name, "name")
        _validate_non_empty_string(self.objective, "objective")
        _validate_signals(self.signals)
        _validate_logic_node(self.entry_logic, "entry_logic")
        _validate_logic_node(self.exit_logic, "exit_logic")
        _validate_parameters(self.parameters)

        if self.direction not in VALID_DIRECTIONS:
            raise ManifestValidationError(
                f"direction must be one of {sorted(VALID_DIRECTIONS)}, "
                f"got {self.direction!r}"
            )

        if not self.universe:
            raise ManifestValidationError("universe must contain at least one symbol")

        for regime_type in self.regime_filters:
            if regime_type not in REGIME_FILTER_CATALOG:
                raise ManifestValidationError(
                    f"Unknown regime filter '{regime_type}'. "
                    f"Valid: {sorted(REGIME_FILTER_CATALOG)}"
                )

        if self.timeframe <= 0:
            raise ManifestValidationError(
                f"timeframe must be positive, got {self.timeframe}"
            )

        if not (0.0 < self.risk_per_trade_pct <= 1.0):
            raise ManifestValidationError(
                f"risk_per_trade_pct must be in (0, 1], got {self.risk_per_trade_pct}"
            )

    def compute_hash(self) -> str:
        """Compute deterministic hash for manifest identity."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "name": self.name,
            "objective": self.objective,
            "signals": sorted(self.signals),
            "entry_logic": self.entry_logic,
            "exit_logic": self.exit_logic,
            "parameters": self.parameters,
            "direction": self.direction,
            "universe": sorted(self.universe),
            "regime_filters": {
                k: sorted(v) for k, v in sorted(self.regime_filters.items())
            },
            "timeframe": self.timeframe,
            "holding_period": self.holding_period,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "status": self.status.value,
            "version": self.version,
        }

    def to_json(self) -> str:
        """Serialize to deterministic JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyManifest:
        """Deserialize from dict. Raises ManifestValidationError on bad data."""
        try:
            manifest = cls(
                name=str(data["name"]),
                objective=str(data["objective"]),
                signals=list(data["signals"]),
                entry_logic=dict(data["entry_logic"]),
                exit_logic=dict(data["exit_logic"]),
                parameters=dict(data["parameters"]),
                direction=str(data.get("direction", "LONG")),
                universe=list(data.get("universe", ["IBIT"])),
                regime_filters={
                    str(k): list(v) for k, v in data.get("regime_filters", {}).items()
                },
                timeframe=int(data.get("timeframe", 60)),
                holding_period=str(data.get("holding_period", "intraday")),
                risk_per_trade_pct=float(data.get("risk_per_trade_pct", 0.02)),
                status=ManifestStatus(data.get("status", "DRAFT")),
                version=int(data.get("version", MANIFEST_SCHEMA_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestValidationError(
                f"Failed to parse StrategyManifest: {exc}"
            ) from exc
        return manifest

    @classmethod
    def from_json(cls, json_str: str) -> StrategyManifest:
        """Parse from JSON string. Raises ManifestValidationError on bad JSON."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(
                f"Invalid JSON in manifest: {exc}"
            ) from exc
        return cls.from_dict(data)

    def to_backtest_config(self) -> Dict[str, Any]:
        """Convert to a config dict compatible with the Hades experiment runner.

        Returns a dict that can be used as ``strategy_params`` in
        :class:`~src.analysis.research_loop_config.StrategySpec`.
        """
        return {
            "strategy_class": f"PantheonGenerated_{self.name.replace(' ', '_')}",
            "params": {
                **{k: v for k, v in self.parameters.items()
                   if not isinstance(v, dict)},
                "signals": self.signals,
                "entry_logic": self.entry_logic,
                "exit_logic": self.exit_logic,
                "direction": self.direction,
                "regime_filters": self.regime_filters,
                "risk_per_trade_pct": self.risk_per_trade_pct,
                "holding_period": self.holding_period,
            },
            "sweep": {
                k: v for k, v in self.parameters.items()
                if isinstance(v, dict) and "min" in v and "max" in v
            },
            "universe": self.universe,
            "timeframe": self.timeframe,
        }


# ═══════════════════════════════════════════════════════════════════════════
# AresCritique
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CritiqueFinding:
    """A single finding from Ares's adversarial analysis."""
    category: CritiqueCategory
    severity: CritiqueSeverity
    description: str
    evidence: str = ""
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CritiqueFinding:
        return cls(
            category=CritiqueCategory(data["category"]),
            severity=CritiqueSeverity(data["severity"]),
            description=str(data["description"]),
            evidence=str(data.get("evidence", "")),
            recommendation=str(data.get("recommendation", "")),
        )


@dataclass
class AresCritique:
    """Structured adversarial critique from Ares.

    Contains prioritized findings split into blockers and advisories.
    A manifest with unresolved blockers cannot be promoted.
    """
    manifest_hash: str
    findings: List[CritiqueFinding] = field(default_factory=list)
    summary: str = ""

    @property
    def blockers(self) -> List[CritiqueFinding]:
        return [f for f in self.findings if f.severity == CritiqueSeverity.BLOCKER]

    @property
    def advisories(self) -> List[CritiqueFinding]:
        return [f for f in self.findings if f.severity == CritiqueSeverity.ADVISORY]

    @property
    def has_blockers(self) -> bool:
        return len(self.blockers) > 0

    def validate(self) -> None:
        """Ensure critique has at least three failure vectors analyzed."""
        if len(self.findings) < 3:
            raise ManifestValidationError(
                f"Ares must analyze at least 3 failure vectors, "
                f"found {len(self.findings)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_hash": self.manifest_hash,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "blocker_count": len(self.blockers),
            "advisory_count": len(self.advisories),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AresCritique:
        return cls(
            manifest_hash=str(data["manifest_hash"]),
            findings=[
                CritiqueFinding.from_dict(f) for f in data.get("findings", [])
            ],
            summary=str(data.get("summary", "")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> AresCritique:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(
                f"Invalid JSON in critique: {exc}"
            ) from exc
        return cls.from_dict(data)


# ═══════════════════════════════════════════════════════════════════════════
# AthenaVerdict
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AthenaVerdict:
    """Final adjudication from Athena.

    Contains a confidence score, the validated research packet
    (cleaned StrategyManifest), and the rationale for the decision.
    """
    confidence: float                     # 0.0–1.0
    decision: str                         # "PROMOTE" or "REJECT"
    rationale: str
    research_packet: Optional[Dict[str, Any]] = None  # Validated manifest dict
    unresolved_blockers: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)  # Conditions for promotion

    # Scoring rubric breakdown
    rubric_scores: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        """Ensure verdict is structurally sound."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ManifestValidationError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if self.decision not in ("PROMOTE", "REJECT"):
            raise ManifestValidationError(
                f"decision must be PROMOTE or REJECT, got {self.decision!r}"
            )
        if self.decision == "PROMOTE" and self.research_packet is None:
            raise ManifestValidationError(
                "PROMOTE decision requires a non-null research_packet"
            )
        _validate_non_empty_string(self.rationale, "rationale")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": round(self.confidence, 4),
            "decision": self.decision,
            "rationale": self.rationale,
            "research_packet": self.research_packet,
            "unresolved_blockers": self.unresolved_blockers,
            "conditions": self.conditions,
            "rubric_scores": {
                k: round(v, 4) for k, v in self.rubric_scores.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AthenaVerdict:
        return cls(
            confidence=float(data["confidence"]),
            decision=str(data["decision"]),
            rationale=str(data["rationale"]),
            research_packet=data.get("research_packet"),
            unresolved_blockers=list(data.get("unresolved_blockers", [])),
            conditions=list(data.get("conditions", [])),
            rubric_scores={
                str(k): float(v) for k, v in data.get("rubric_scores", {}).items()
            },
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> AthenaVerdict:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(
                f"Invalid JSON in verdict: {exc}"
            ) from exc
        return cls.from_dict(data)


# ═══════════════════════════════════════════════════════════════════════════
# JSON Extraction Helper
# ═══════════════════════════════════════════════════════════════════════════

def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from an LLM response that may contain prose.

    Tries, in order:
    1. Fenced ``<manifest>...</manifest>`` or ``<critique>...</critique>``
       or ``<verdict>...</verdict>`` tags
    2. Fenced ```json ... ``` blocks
    3. First balanced ``{...}`` in the text

    Returns None if no valid JSON found.
    """
    # 1. Try tagged blocks
    for tag in ("manifest", "critique", "verdict"):
        pattern = rf"<{tag}>\s*(\{{.*?\}})\s*</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    # 2. Try fenced code blocks
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Try first balanced braces
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1
                    continue

    return None
