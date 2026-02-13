"""
Shared test fixtures and environment stubs.

Stubs out the ``multitasking`` package (which cannot be built in some
CI environments) so that ``yfinance`` and its transitive importers
(e.g. ``src.connectors.ibit_options_client``) can import cleanly.
"""

import sys
import types

# Stub multitasking before any test module triggers an import of yfinance.
if "multitasking" not in sys.modules:
    _mt = types.ModuleType("multitasking")
    _mt.__version__ = "0.0.11"
    _mt.task = lambda f: f  # no-op decorator
    sys.modules["multitasking"] = _mt
