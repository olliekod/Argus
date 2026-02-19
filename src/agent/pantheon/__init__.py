"""Pantheon Intelligence Engine — structured research agents."""

from .roles import (
    PantheonRole,
    ContextInjector,
    PROMETHEUS,
    ARES,
    ATHENA,
    get_role_for_stage,
    build_stage_prompt,
)

__all__ = [
    "PantheonRole",
    "ContextInjector",
    "PROMETHEUS",
    "ARES",
    "ATHENA",
    "get_role_for_stage",
    "build_stage_prompt",
]
