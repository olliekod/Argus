"""
Argus Logger Module
===================

Colored console logging with file rotation.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': Fore.CYAN if COLORS_AVAILABLE else '',
        'INFO': Fore.GREEN if COLORS_AVAILABLE else '',
        'WARNING': Fore.YELLOW if COLORS_AVAILABLE else '',
        'ERROR': Fore.RED if COLORS_AVAILABLE else '',
        'CRITICAL': Fore.RED + Style.BRIGHT if COLORS_AVAILABLE else '',
    }
    RESET = Style.RESET_ALL if COLORS_AVAILABLE else ''
    
    def format(self, record):
        # Add color to level name
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname:8}{self.RESET}"
        
        # Add color to specific message patterns
        message = super().format(record)
        
        # Highlight important keywords
        if COLORS_AVAILABLE:
            # Detection alerts
            if 'DETECTION' in message or 'ALERT' in message:
                message = message.replace('DETECTION', f'{Fore.MAGENTA}DETECTION{self.RESET}')
                message = message.replace('ALERT', f'{Fore.MAGENTA}ALERT{self.RESET}')
            
            # Price/money
            if '$' in message:
                pass  # Could highlight money amounts
        
        return message


class PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no colors)."""
    pass


def setup_logger(
    name: str = 'argus',
    level: str = 'INFO',
    log_dir: str = 'data/logs'
) -> logging.Logger:
    """
    Set up logging with console and file handlers.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = ColoredFormatter(
        '%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler with daily rotation
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    file_handler = TimedRotatingFileHandler(
        filename=log_path / 'argus.log',
        when='midnight',
        interval=1,
        backupCount=30,  # Keep 30 days of logs
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = PlainFormatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger for a specific component.
    
    Args:
        name: Component name (e.g., 'bybit_ws', 'funding_detector')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'argus.{name}')


# Pre-configured loggers for common components
def get_connector_logger(connector_name: str) -> logging.Logger:
    """Get logger for an exchange connector."""
    return get_logger(f'connectors.{connector_name}')


def get_detector_logger(detector_name: str) -> logging.Logger:
    """Get logger for a detector."""
    return get_logger(f'detectors.{detector_name}')


def get_alert_logger() -> logging.Logger:
    """Get logger for alert system."""
    return get_logger('alerts')
