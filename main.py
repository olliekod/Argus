#!/usr/bin/env python3
"""
Argus - Crypto Market Monitor
=============================

Main entry point for running Argus.

Usage:
    python main.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import main


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nArgus stopped by user")
