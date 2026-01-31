"""
Argus - Crypto Market Monitor
=============================

24/7 market monitoring system for detecting trading opportunities.
"""

__version__ = "0.1.0"
__author__ = "Argus"

from .orchestrator import ArgusOrchestrator, main

__all__ = ["ArgusOrchestrator", "main"]
