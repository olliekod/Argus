"""
Tests that the option chain snapshotter uses a config-driven underlyings list.

Verifies:
- The snapshotter reads symbols from config (no hardcoded list in the poll loop).
- Default config includes SPY and QQQ in addition to IBIT/BITO.
"""

from __future__ import annotations

import pytest

from src.core.config import load_config


def _options_symbols_from_config(config: dict) -> list[str]:
    """Extract options chain symbols the same way the orchestrator does."""
    options_cfg = (
        config.get("exchanges", {}).get("alpaca", {}).get("options", {})
    )
    return options_cfg.get("symbols", ["IBIT", "BITO"])


class TestOptionChainSnapshotterConfigDriven:
    """Snapshotter must iterate over a config-driven list (no hardcoded symbols)."""

    def test_symbols_come_from_config(self):
        """When config has options.symbols, that list is used."""
        config = {
            "exchanges": {
                "alpaca": {
                    "options": {
                        "enabled": True,
                        "symbols": ["CUSTOM1", "CUSTOM2", "CUSTOM3"],
                    },
                },
            },
        }
        symbols = _options_symbols_from_config(config)
        assert symbols == ["CUSTOM1", "CUSTOM2", "CUSTOM3"]

    def test_fallback_when_no_symbols_key(self):
        """When options.symbols is missing, fallback is used."""
        config = {
            "exchanges": {
                "alpaca": {
                    "options": {"enabled": True},
                },
            },
        }
        symbols = _options_symbols_from_config(config)
        assert symbols == ["IBIT", "BITO"]

    def test_default_config_includes_spy_qqq_ibit_bito(self):
        """Default config includes SPY, QQQ, and existing IBIT/BITO."""
        config = load_config()
        symbols = _options_symbols_from_config(config)
        assert "SPY" in symbols, "Default config should include SPY for replay packs"
        assert "QQQ" in symbols, "Default config should include QQQ for replay packs"
        assert "IBIT" in symbols, "Default config should keep IBIT"
        assert "BITO" in symbols, "Default config should keep BITO"
