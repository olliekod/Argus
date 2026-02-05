"""
Argus Logger Module
===================

Structured logging with size-based rotating file handlers,
configurable console/file levels, and uptime tracking.

Config variables (env or config.yaml):
  LOG_LEVEL_CONSOLE  (default INFO)
  LOG_LEVEL_FILE     (default DEBUG)
  LOG_DIR            (default data/logs)
  LOG_MAX_BYTES      (default 50MB)
  LOG_BACKUP_COUNT   (default 10)
"""

import logging
import os
import sys
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler

try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Global boot timestamp (set once at import time)
# ---------------------------------------------------------------------------
_BOOT_TS = time.monotonic()

# ---------------------------------------------------------------------------
# Configurable defaults (overridden via env vars or setup_logger kwargs)
# ---------------------------------------------------------------------------
LOG_LEVEL_CONSOLE = os.environ.get("LOG_LEVEL_CONSOLE", "INFO")
LOG_LEVEL_FILE = os.environ.get("LOG_LEVEL_FILE", "DEBUG")
LOG_DIR = os.environ.get("LOG_DIR", "data/logs")
LOG_MAX_BYTES = int(os.environ.get("LOG_MAX_BYTES", str(50 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", "10"))


def _uptime() -> str:
    """Human-readable uptime since process start."""
    secs = int(time.monotonic() - _BOOT_TS)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s"
    hours = secs // 3600
    mins = (secs % 3600) // 60
    return f"{hours}h{mins:02d}m"


def uptime_seconds() -> float:
    """Raw uptime in seconds."""
    return time.monotonic() - _BOOT_TS


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class StructuredFormatter(logging.Formatter):
    """File formatter with uptime, no colours."""

    def format(self, record):
        record.uptime = _uptime()
        return super().format(record)


class ColoredFormatter(logging.Formatter):
    """Console formatter with ANSI colors and uptime."""

    COLORS = {
        'DEBUG': Fore.CYAN if COLORS_AVAILABLE else '',
        'INFO': Fore.GREEN if COLORS_AVAILABLE else '',
        'WARNING': Fore.YELLOW if COLORS_AVAILABLE else '',
        'ERROR': Fore.RED if COLORS_AVAILABLE else '',
        'CRITICAL': (Fore.RED + Style.BRIGHT) if COLORS_AVAILABLE else '',
    }
    RESET = Style.RESET_ALL if COLORS_AVAILABLE else ''

    def format(self, record):
        record.uptime = _uptime()
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname:8}{self.RESET}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_ROOT_LOGGER_CONFIGURED = False


def setup_logger(
    name: str = 'argus',
    level: str = 'INFO',
    log_dir: str = None,
    console_level: str = None,
    file_level: str = None,
    max_bytes: int = None,
    backup_count: int = None,
) -> logging.Logger:
    """
    Set up root Argus logger with console + size-rotated file handlers.

    Call once at startup.  Subsequent calls return the existing logger.
    """
    global _ROOT_LOGGER_CONFIGURED

    logger = logging.getLogger(name)
    if _ROOT_LOGGER_CONFIGURED:
        return logger

    log_dir = log_dir or LOG_DIR
    console_level = console_level or LOG_LEVEL_CONSOLE
    file_level = file_level or LOG_LEVEL_FILE
    max_bytes = max_bytes or LOG_MAX_BYTES
    backup_count = backup_count or LOG_BACKUP_COUNT

    logger.setLevel(logging.DEBUG)  # handlers do their own filtering

    # --- Console (less verbose by default) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s [%(uptime)s] %(levelname)s %(name)s  %(message)s',
        datefmt='%H:%M:%S',
    ))
    logger.addHandler(console_handler)

    # --- Rotating file (more verbose, size-based) ---
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_path / 'argus.log',
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8',
    )
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    file_handler.setFormatter(StructuredFormatter(
        '%(asctime)s [%(uptime)s] %(levelname)-8s %(name)s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    logger.addHandler(file_handler)

    _ROOT_LOGGER_CONFIGURED = True
    return logger


# ---------------------------------------------------------------------------
# Convenience getters (unchanged API)
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific component."""
    return logging.getLogger(f'argus.{name}')


def get_connector_logger(connector_name: str) -> logging.Logger:
    """Get logger for an exchange connector."""
    return get_logger(f'connectors.{connector_name}')


def get_detector_logger(detector_name: str) -> logging.Logger:
    """Get logger for a detector."""
    return get_logger(f'detectors.{detector_name}')


def get_alert_logger() -> logging.Logger:
    """Get logger for alert system."""
    return get_logger('alerts')
