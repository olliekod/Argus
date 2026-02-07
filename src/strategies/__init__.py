"""
Argus Strategy Modules
======================

Phase 3 deterministic strategy framework.

- BaseStrategy: Abstract base class for all strategies
- SignalRouter: Collects, scores, and ranks signals
"""

from .base import BaseStrategy
from .router import SignalRouter, DEFAULT_RANKER_CONFIG

__all__ = [
    "BaseStrategy",
    "SignalRouter",
    "DEFAULT_RANKER_CONFIG",
]
