#!/usr/bin/env python3
"""
Argus - Entry Point Script
==========================

The All-Seeing Market Monitor

Usage:
    python run.py              # Start normally
    python run.py --test       # Run in test mode (no alerts)
    python run.py --dry-run    # Test configuration without connecting
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import main


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Argus - The All-Seeing Market Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py              Start Argus normally
    python run.py --test       Run without sending alerts
    python run.py --dry-run    Validate config without connecting
        """
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (disables Telegram alerts)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without connecting to exchanges"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   👁️  ARGUS - The All-Seeing Market Monitor                  ║
    ║                                                               ║
    ║   Named after Argus Panoptes, the hundred-eyed giant         ║
    ║   of Greek mythology who never slept.                        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(main(
            test_mode=args.test,
            dry_run=args.dry_run,
            config_path=args.config
        ))
    except KeyboardInterrupt:
        print("\n\n👁️  Argus is closing its eyes... Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
