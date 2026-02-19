"""
Shared test fixtures and environment stubs.

Stubs out packages that cannot be built in some CI environments
(``multitasking``, ``curl_cffi``, etc.) so that ``yfinance`` and
its transitive importers can import cleanly.
"""

import sys
import types

# Stub multitasking before any test module triggers an import of yfinance.
if "multitasking" not in sys.modules:
    _mt = types.ModuleType("multitasking")
    _mt.__version__ = "0.0.11"
    _mt.task = lambda f: f  # no-op decorator
    sys.modules["multitasking"] = _mt

# Stub curl_cffi (yfinance >= 1.2.0 dependency) if not installed.
if "curl_cffi" not in sys.modules:
    _cc = types.ModuleType("curl_cffi")
    _cc_req = types.ModuleType("curl_cffi.requests")
    _cc.requests = _cc_req
    sys.modules["curl_cffi"] = _cc
    sys.modules["curl_cffi.requests"] = _cc_req

# Stub yfinance entirely if its transitive deps (bs4, protobuf, etc.) are
# missing.  The analysis module's __init__.py pulls in trade_calculator →
# ibit_options_client → yfinance, but our Phase 5 / sizing / allocation
# tests don't need any of those.
try:
    import yfinance  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _yf = types.ModuleType("yfinance")
    _yf.__version__ = "0.0.0"

    class _FakeTicker:
        def __init__(self, *a, **kw):
            pass

    _yf.Ticker = _FakeTicker
    sys.modules["yfinance"] = _yf
