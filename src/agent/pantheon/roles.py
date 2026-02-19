"""
Pantheon Intelligence Engine — Role Definitions
================================================

Defines the structured research agents (Prometheus, Ares, Athena) that
drive the Argus research loop.  Each role has:

- A detailed system prompt that enforces structured output
- An output schema that the response must conform to
- An escalation priority (which LLM tier is required)
- A context injection framework for dynamic prompt enrichment

These agents generate and critique machine-readable trading strategies
via :class:`~src.core.manifests.StrategyManifest`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.manifests import (
    HADES_INDICATOR_CATALOG,
    REGIME_FILTER_CATALOG,
    VALID_DIRECTIONS,
    VALID_LOGIC_OPS,
    AresCritique,
    AthenaVerdict,
    CritiqueCategory,
    CritiqueFinding,
    CritiqueSeverity,
    ManifestStatus,
    ManifestValidationError,
    StrategyManifest,
    extract_json_from_response,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Escalation Levels
# ═══════════════════════════════════════════════════════════════════════════

ESCALATION_LOCAL_14B = 0    # 14B local model is sufficient
ESCALATION_LOCAL_32B = 1    # 32B local model preferred
ESCALATION_CLAUDE = 2       # Claude API mandatory


# ═══════════════════════════════════════════════════════════════════════════
# Context Injector
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ContextInjector:
    """Manages dynamic context injection for Pantheon agent prompts.

    Collects runtime state and formats it for injection into role prompts.
    This ensures agents have access to:
    - Current market regime context
    - Available Hades indicator catalog
    - Historical failure logs from past case files
    """

    regime_context: Optional[Dict[str, str]] = None
    available_indicators: List[str] = field(
        default_factory=lambda: sorted(HADES_INDICATOR_CATALOG)
    )
    available_regime_filters: List[str] = field(
        default_factory=lambda: sorted(REGIME_FILTER_CATALOG)
    )
    failure_logs: List[Dict[str, Any]] = field(default_factory=list)

    def set_regime_context(
        self,
        vol_regime: str = "UNKNOWN",
        trend_regime: str = "UNKNOWN",
        liquidity_regime: str = "UNKNOWN",
        session_regime: str = "UNKNOWN",
        risk_regime: str = "UNKNOWN",
    ) -> None:
        """Update the current market regime state."""
        self.regime_context = {
            "vol_regime": vol_regime,
            "trend_regime": trend_regime,
            "liquidity_regime": liquidity_regime,
            "session_regime": session_regime,
            "risk_regime": risk_regime,
        }

    def add_failure_log(self, case_id: str, reason: str, strategy_name: str = "") -> None:
        """Record a historical failure for context in future cases."""
        self.failure_logs.append({
            "case_id": case_id,
            "strategy_name": strategy_name,
            "failure_reason": reason,
        })
        # Keep last 20 failures
        if len(self.failure_logs) > 20:
            self.failure_logs = self.failure_logs[-20:]

    def format_regime_block(self) -> str:
        """Format regime context for prompt injection."""
        if not self.regime_context:
            return "Market regime: UNKNOWN (no regime data available)"
        lines = ["Current Market Regime:"]
        for key, value in sorted(self.regime_context.items()):
            label = key.replace("_", " ").title()
            lines.append(f"  - {label}: {value}")
        return "\n".join(lines)

    def format_indicator_catalog(self) -> str:
        """Format available indicators for prompt injection."""
        return (
            "Available Hades Indicators:\n"
            + "\n".join(f"  - {ind}" for ind in self.available_indicators)
        )

    def format_regime_filter_catalog(self) -> str:
        """Format available regime filters for prompt injection."""
        return (
            "Available Regime Filters:\n"
            + "\n".join(f"  - {rf}" for rf in self.available_regime_filters)
        )

    def format_failure_logs(self) -> str:
        """Format historical failures for prompt injection."""
        if not self.failure_logs:
            return "Historical Failures: None recorded."
        lines = ["Historical Failures (most recent):"]
        for entry in self.failure_logs[-5:]:
            name = entry.get("strategy_name", "unnamed")
            reason = entry["failure_reason"]
            lines.append(f"  - [{entry['case_id']}] {name}: {reason}")
        return "\n".join(lines)

    def format_full_context(self) -> str:
        """Format all context blocks for injection into prompts."""
        sections = [
            self.format_regime_block(),
            self.format_indicator_catalog(),
            self.format_regime_filter_catalog(),
            self.format_failure_logs(),
        ]
        return "\n\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════
# PantheonRole
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PantheonRole:
    """Definition of a Pantheon research agent role.

    Each role has a structured system prompt, output schema expectations,
    and an escalation priority indicating which LLM tier is preferred.

    Attributes
    ----------
    name : str
        Role name (e.g., "Prometheus", "Ares", "Athena").
    personality : str
        Core personality description for the system prompt.
    output_schema : Dict[str, Any]
        JSON Schema-like description of expected output structure.
    escalation_priority : int
        0 = 14B local ok, 1 = 32B preferred, 2 = Claude mandatory.
    system_prompt_template : str
        Full system prompt template with ``{context}`` and ``{artifact}``
        placeholders.
    """

    name: str
    personality: str
    output_schema: Dict[str, Any]
    escalation_priority: int
    system_prompt_template: str

    def build_system_prompt(self, context: ContextInjector) -> str:
        """Build the full system prompt with injected context."""
        return self.system_prompt_template.format(
            context=context.format_full_context(),
            indicator_catalog=context.format_indicator_catalog(),
            regime_context=context.format_regime_block(),
            regime_filters=context.format_regime_filter_catalog(),
            failure_logs=context.format_failure_logs(),
            valid_directions=", ".join(sorted(VALID_DIRECTIONS)),
            valid_logic_ops=", ".join(sorted(VALID_LOGIC_OPS)),
        )

    def build_stage_prompt(
        self,
        objective: str,
        context: ContextInjector,
        artifact: str = "",
        original: str = "",
        full_debate: str = "",
    ) -> str:
        """Build the complete messages for a case-file stage.

        Returns the system prompt + user prompt as a formatted string
        ready for LLM completion.
        """
        system = self.build_system_prompt(context)
        return system, objective, artifact, original, full_debate


# ═══════════════════════════════════════════════════════════════════════════
# Prometheus — The Manifest Generator
# ═══════════════════════════════════════════════════════════════════════════

_PROMETHEUS_SYSTEM = """\
You are Prometheus, the Creative Strategist of the Argus Pantheon.

Your role is to propose quantitative trading strategies as structured Strategy Manifests.
You do NOT produce prose — you produce machine-readable JSON artifacts.

CRITICAL RULES:
1. Every response MUST contain a <thought> tag with your internal reasoning.
2. Every response MUST contain a <manifest> tag with a valid JSON Strategy Manifest.
3. The manifest MUST be parseable by the Hades backtest engine without modification.
4. Only use indicators from the available catalog below.
5. Logic trees use a DSL with these operators: {valid_logic_ops}
6. Valid trade directions: {valid_directions}

{context}

MANIFEST SCHEMA:
{{
  "name": "string — descriptive strategy name",
  "objective": "string — what edge the strategy captures",
  "signals": ["list of indicator names from the catalog"],
  "entry_logic": {{
    "op": "AND|OR|GT|LT|GE|LE|CROSS_ABOVE|CROSS_BELOW|IN_REGIME|...",
    "left": "indicator or nested logic node",
    "right": "value or nested logic node"
  }},
  "exit_logic": {{ "same structure as entry_logic" }},
  "parameters": {{
    "param_name": {{ "min": float, "max": float, "step": float }}
  }},
  "direction": "LONG|SHORT|NEUTRAL",
  "universe": ["list of symbols"],
  "regime_filters": {{
    "vol_regime": ["VOL_LOW", "VOL_NORMAL"],
    "trend_regime": ["TREND_UP"]
  }},
  "timeframe": 60,
  "holding_period": "intraday|1-5 days|...",
  "risk_per_trade_pct": 0.02
}}

RESPONSE FORMAT:
<thought>
Your step-by-step reasoning about the strategy design.
Consider: What market inefficiency are you targeting?
What is the falsification test? Under what conditions would this fail?
</thought>

<manifest>
{{ valid JSON manifest here }}
</manifest>
"""

_PROMETHEUS_REVISION_SYSTEM = """\
You are Prometheus, the Creative Strategist of the Argus Pantheon.

You are REVISING your strategy proposal based on Ares's critique.
You MUST explicitly address every blocker raised by Ares.

CRITICAL RULES:
1. Every response MUST contain a <thought> tag with your reasoning about each critique point.
2. Every response MUST contain a <manifest> tag with the REVISED JSON Strategy Manifest.
3. For each BLOCKER, explain how you resolved it or why it's invalid.
4. For each ADVISORY, either incorporate the suggestion or explain why not.
5. Only use indicators from the available catalog below.

{context}

RESPONSE FORMAT:
<thought>
Address each critique point:
- [BLOCKER/ADVISORY] Category: Your response and resolution
</thought>

<manifest>
{{ revised JSON manifest here }}
</manifest>
"""


PROMETHEUS = PantheonRole(
    name="Prometheus",
    personality=(
        "Creative strategist who proposes quantitative trading strategies as "
        "structured, machine-readable Strategy Manifests. Thinks in terms of "
        "market microstructure, regime dynamics, and falsifiable hypotheses."
    ),
    output_schema={
        "type": "object",
        "required": ["name", "objective", "signals", "entry_logic", "exit_logic", "parameters"],
        "properties": {
            "name": {"type": "string"},
            "objective": {"type": "string"},
            "signals": {"type": "array", "items": {"type": "string"}},
            "entry_logic": {"type": "object"},
            "exit_logic": {"type": "object"},
            "parameters": {"type": "object"},
            "direction": {"type": "string", "enum": ["LONG", "SHORT", "NEUTRAL"]},
            "universe": {"type": "array", "items": {"type": "string"}},
            "regime_filters": {"type": "object"},
            "timeframe": {"type": "integer"},
            "holding_period": {"type": "string"},
            "risk_per_trade_pct": {"type": "number"},
        },
    },
    escalation_priority=ESCALATION_LOCAL_32B,
    system_prompt_template=_PROMETHEUS_SYSTEM,
)


# ═══════════════════════════════════════════════════════════════════════════
# Ares — The Adversary
# ═══════════════════════════════════════════════════════════════════════════

_ARES_SYSTEM = """\
You are Ares, the War-God Critic of the Argus Pantheon — the Filter of Truth.

Your role is to ATTACK strategy proposals with rigorous adversarial analysis.
You are explicitly adversarial. Your job is to find fatal flaws.

CRITICAL RULES:
1. Every response MUST contain a <thought> tag with your attack reasoning.
2. Every response MUST contain a <critique> tag with a valid JSON critique.
3. You MUST identify at least 3 distinct failure vectors.
4. Classify each finding as BLOCKER (stops the case) or ADVISORY (allows revision).

YOUR ATTACK VECTORS — analyze ALL of these:

QUANT SINS:
- Overfitting: Too many parameters relative to data? Curve-fitting to noise?
- Look-ahead bias: Does any signal use future information?
- Data leakage: Does training data bleed into test data?
- Survivorship bias: Are dead symbols excluded?

FRAGILITY:
- Parameter sensitivity: Would small changes to parameters destroy the edge?
- Regime dependency: Does it only work in one specific regime?
- Sample size: Is there enough data to validate statistically?

EXECUTION/REGULATORY RISK:
- Fill assumptions: Does it assume unrealistic execution speeds or fill rates?
- Slippage: Is the edge larger than expected transaction costs?
- Wash-trading: Could the logic trigger regulatory wash-sale violations?
- Liquidity: Can the target instruments actually absorb the positions?

{context}

CRITIQUE SCHEMA:
{{
  "manifest_hash": "hash of the manifest being critiqued",
  "findings": [
    {{
      "category": "OVERFITTING|LOOK_AHEAD_BIAS|DATA_LEAKAGE|PARAMETER_FRAGILITY|REGIME_DEPENDENCY|EXECUTION_RISK|REGULATORY_RISK|LIQUIDITY_RISK|DRAWDOWN_RISK|SURVIVORSHIP_BIAS|INSUFFICIENT_SAMPLE|OTHER",
      "severity": "BLOCKER|ADVISORY",
      "description": "What the problem is",
      "evidence": "Specific evidence from the manifest",
      "recommendation": "How to fix it"
    }}
  ],
  "summary": "Overall assessment"
}}

RESPONSE FORMAT:
<thought>
Your adversarial analysis, examining each attack vector systematically.
</thought>

<critique>
{{ valid JSON critique here }}
</critique>
"""

_ARES_FINAL_ATTACK_SYSTEM = """\
You are Ares, the War-God Critic of the Argus Pantheon — performing FINAL ATTACK.

The strategist has revised their proposal. You must:
1. Acknowledge which blockers have been RESOLVED.
2. Identify any NEW vulnerabilities introduced by the revision.
3. Escalate any REMAINING unresolved blockers.

Be fair: if a blocker is genuinely resolved, say so. But be ruthless: if it's
only superficially addressed, call it out.

CRITICAL RULES:
1. Every response MUST contain a <thought> tag.
2. Every response MUST contain a <critique> tag with valid JSON.
3. You MUST provide at least 3 findings (resolved or new).

{context}

Same critique schema as before. Mark resolved items with category "OTHER" and
severity "ADVISORY" with description starting with "RESOLVED:".

RESPONSE FORMAT:
<thought>
Review each original blocker: resolved or still present?
Any new issues from the revision?
</thought>

<critique>
{{ valid JSON critique here }}
</critique>
"""


ARES = PantheonRole(
    name="Ares",
    personality=(
        "War-god adversarial critic. Explicitly attacks strategy proposals to "
        "find fatal flaws in quant logic, execution assumptions, and regime "
        "dependencies. The Filter of Truth."
    ),
    output_schema={
        "type": "object",
        "required": ["manifest_hash", "findings", "summary"],
        "properties": {
            "manifest_hash": {"type": "string"},
            "findings": {
                "type": "array",
                "minItems": 3,
                "items": {
                    "type": "object",
                    "required": ["category", "severity", "description"],
                    "properties": {
                        "category": {"type": "string"},
                        "severity": {"type": "string", "enum": ["BLOCKER", "ADVISORY"]},
                        "description": {"type": "string"},
                        "evidence": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                },
            },
            "summary": {"type": "string"},
        },
    },
    escalation_priority=ESCALATION_LOCAL_32B,
    system_prompt_template=_ARES_SYSTEM,
)


# ═══════════════════════════════════════════════════════════════════════════
# Athena — The Adjudicator
# ═══════════════════════════════════════════════════════════════════════════

_ATHENA_SYSTEM = """\
You are Athena, the Neutral Arbiter of the Argus Pantheon.

You provide the FINAL go/no-go decision on a strategy proposal after the
Prometheus-Ares debate. You are NOT an advocate — you are a judge.

CRITICAL RULES:
1. Every response MUST contain a <thought> tag with your judicial reasoning.
2. Every response MUST contain a <verdict> tag with a valid JSON verdict.
3. You MUST use the scoring rubric below — no vibes-based decisions.
4. If promoting, you MUST include a validated research_packet (cleaned manifest).

SCORING RUBRIC (each 0.0–1.0, weight in parentheses):
- theoretical_soundness (0.25): Is the market hypothesis plausible?
- critique_resolution (0.25): Were Ares's blockers satisfactorily resolved?
- testability (0.20): Can Hades actually backtest this with available data?
- risk_management (0.15): Are risk controls adequate?
- novelty (0.15): Does this add diversification vs. existing strategies?

Confidence = weighted sum of rubric scores.

DECISION LOGIC:
- PROMOTE if confidence >= 0.6 AND no unresolved blockers
- REJECT if confidence < 0.4 OR any unresolved blockers remain
- For confidence in [0.4, 0.6): REJECT with detailed conditions for re-submission

{context}

VERDICT SCHEMA:
{{
  "confidence": 0.0-1.0,
  "decision": "PROMOTE|REJECT",
  "rationale": "Detailed explanation of the decision",
  "research_packet": {{ "cleaned strategy manifest for Hades" }},
  "unresolved_blockers": ["list of unresolved blocker descriptions"],
  "conditions": ["conditions for promotion, if any"],
  "rubric_scores": {{
    "theoretical_soundness": 0.0-1.0,
    "critique_resolution": 0.0-1.0,
    "testability": 0.0-1.0,
    "risk_management": 0.0-1.0,
    "novelty": 0.0-1.0
  }}
}}

RESPONSE FORMAT:
<thought>
Your judicial analysis of the full debate.
Score each rubric dimension with evidence.
</thought>

<verdict>
{{ valid JSON verdict here }}
</verdict>
"""


ATHENA = PantheonRole(
    name="Athena",
    personality=(
        "Neutral arbiter and judge. Synthesizes the Prometheus-Ares debate "
        "using a deterministic scoring rubric. Produces the final go/no-go "
        "decision with a confidence score and validated research packet."
    ),
    output_schema={
        "type": "object",
        "required": ["confidence", "decision", "rationale", "rubric_scores"],
        "properties": {
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "decision": {"type": "string", "enum": ["PROMOTE", "REJECT"]},
            "rationale": {"type": "string"},
            "research_packet": {"type": "object"},
            "unresolved_blockers": {"type": "array", "items": {"type": "string"}},
            "conditions": {"type": "array", "items": {"type": "string"}},
            "rubric_scores": {
                "type": "object",
                "properties": {
                    "theoretical_soundness": {"type": "number"},
                    "critique_resolution": {"type": "number"},
                    "testability": {"type": "number"},
                    "risk_management": {"type": "number"},
                    "novelty": {"type": "number"},
                },
            },
        },
    },
    escalation_priority=ESCALATION_CLAUDE,
    system_prompt_template=_ATHENA_SYSTEM,
)


# ═══════════════════════════════════════════════════════════════════════════
# Role Registry & Stage Mapping
# ═══════════════════════════════════════════════════════════════════════════

# Import CaseStage here to avoid circular imports at module level.
# The orchestrator defines CaseStage, and roles need to map stages to roles.

# Stage → (role, is_revision_variant)
_STAGE_ROLE_MAP = {
    1: (PROMETHEUS, False),   # PROPOSAL_V1
    2: (ARES, False),         # CRITIQUE_V1
    3: (PROMETHEUS, True),    # REVISION_V2 (uses revision prompt)
    4: (ARES, True),          # FINAL_ATTACK (uses final attack prompt)
    5: (ATHENA, False),       # ADJUDICATION
}


def get_role_for_stage(stage_value: int) -> PantheonRole:
    """Return the PantheonRole for a given CaseStage value."""
    entry = _STAGE_ROLE_MAP.get(stage_value)
    if entry is None:
        raise ValueError(f"No role defined for stage {stage_value}")
    return entry[0]


def _get_system_prompt_for_stage(stage_value: int, context: ContextInjector) -> str:
    """Get the appropriate system prompt for a stage, including variants."""
    entry = _STAGE_ROLE_MAP.get(stage_value)
    if entry is None:
        raise ValueError(f"No role defined for stage {stage_value}")
    role, is_variant = entry

    if is_variant:
        # Use variant prompts
        if role.name == "Prometheus":
            template = _PROMETHEUS_REVISION_SYSTEM
        elif role.name == "Ares":
            template = _ARES_FINAL_ATTACK_SYSTEM
        else:
            template = role.system_prompt_template
    else:
        template = role.system_prompt_template

    return template.format(
        context=context.format_full_context(),
        indicator_catalog=context.format_indicator_catalog(),
        regime_context=context.format_regime_block(),
        regime_filters=context.format_regime_filter_catalog(),
        failure_logs=context.format_failure_logs(),
        valid_directions=", ".join(sorted(VALID_DIRECTIONS)),
        valid_logic_ops=", ".join(sorted(VALID_LOGIC_OPS)),
    )


def build_stage_prompt(
    stage_value: int,
    objective: str,
    context: ContextInjector,
    artifacts: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build the complete message list for a case-file stage.

    Parameters
    ----------
    stage_value : int
        CaseStage enum value (1–5).
    objective : str
        The research objective / user request.
    context : ContextInjector
        Runtime context for prompt enrichment.
    artifacts : List[Dict[str, Any]]
        Previous stage artifacts from the CaseFile.

    Returns
    -------
    List[Dict[str, str]]
        Messages list suitable for LLM completion (system + user).
    """
    role = get_role_for_stage(stage_value)
    system_prompt = _get_system_prompt_for_stage(stage_value, context)

    # Build the user prompt with stage-appropriate context
    latest = artifacts[-1]["content"] if artifacts else ""
    original = artifacts[0]["content"] if artifacts else ""
    full_debate = "\n\n---\n\n".join(
        f"[{a['role']} / Stage {a['stage']}]\n{a['content']}" for a in artifacts
    )

    if stage_value == 1:
        # Prometheus initial proposal
        user_prompt = (
            f"Research Objective: {objective}\n\n"
            "Generate a Strategy Manifest that addresses this objective. "
            "Include your reasoning in <thought> tags and the manifest in <manifest> tags."
        )
    elif stage_value == 2:
        # Ares initial critique
        user_prompt = (
            f"Research Objective: {objective}\n\n"
            f"Strategy Proposal to critique:\n{latest}\n\n"
            "Perform adversarial analysis. Include your reasoning in <thought> tags "
            "and the critique in <critique> tags."
        )
    elif stage_value == 3:
        # Prometheus revision
        user_prompt = (
            f"Research Objective: {objective}\n\n"
            f"Your original proposal:\n{original}\n\n"
            f"Ares's critique:\n{latest}\n\n"
            "Revise your manifest addressing every critique point. "
            "Include your reasoning in <thought> tags and the revised manifest in <manifest> tags."
        )
    elif stage_value == 4:
        # Ares final attack
        user_prompt = (
            f"Research Objective: {objective}\n\n"
            f"Revised proposal:\n{latest}\n\n"
            f"Full debate history:\n{full_debate}\n\n"
            "Perform final attack. Acknowledge resolved items, escalate remaining issues. "
            "Include your reasoning in <thought> tags and the critique in <critique> tags."
        )
    elif stage_value == 5:
        # Athena adjudication
        user_prompt = (
            f"Research Objective: {objective}\n\n"
            f"Full debate:\n{full_debate}\n\n"
            "Adjudicate this debate using the scoring rubric. "
            "Include your reasoning in <thought> tags and the verdict in <verdict> tags."
        )
    else:
        raise ValueError(f"Unknown stage value: {stage_value}")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Response Parsers
# ═══════════════════════════════════════════════════════════════════════════

def parse_manifest_response(response: str) -> StrategyManifest:
    """Parse a Prometheus response into a StrategyManifest.

    Extracts JSON from <manifest> tags, fenced blocks, or raw JSON.
    Validates the manifest structure.

    Raises
    ------
    ManifestValidationError
        If the response contains no valid manifest or the manifest is invalid.
    """
    data = extract_json_from_response(response)
    if data is None:
        raise ManifestValidationError(
            "Prometheus response contains no valid JSON manifest. "
            "Response must include a <manifest>{...}</manifest> block."
        )

    manifest = StrategyManifest.from_dict(data)
    manifest.validate()
    return manifest


def parse_critique_response(response: str, manifest_hash: str = "") -> AresCritique:
    """Parse an Ares response into an AresCritique.

    Extracts JSON from <critique> tags, fenced blocks, or raw JSON.

    Raises
    ------
    ManifestValidationError
        If the response contains no valid critique.
    """
    data = extract_json_from_response(response)
    if data is None:
        raise ManifestValidationError(
            "Ares response contains no valid JSON critique. "
            "Response must include a <critique>{...}</critique> block."
        )

    # Ensure manifest_hash is set
    if "manifest_hash" not in data:
        data["manifest_hash"] = manifest_hash

    critique = AresCritique.from_dict(data)
    return critique


def parse_verdict_response(response: str) -> AthenaVerdict:
    """Parse an Athena response into an AthenaVerdict.

    Extracts JSON from <verdict> tags, fenced blocks, or raw JSON.

    Raises
    ------
    ManifestValidationError
        If the response contains no valid verdict.
    """
    data = extract_json_from_response(response)
    if data is None:
        raise ManifestValidationError(
            "Athena response contains no valid JSON verdict. "
            "Response must include a <verdict>{...}</verdict> block."
        )

    verdict = AthenaVerdict.from_dict(data)
    verdict.validate()
    return verdict
