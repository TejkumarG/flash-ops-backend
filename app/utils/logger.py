"""
Logging utility.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from app.config import settings


# Global file handler shared by all loggers
_file_handler = None


def _get_file_handler() -> logging.Handler:
    """Get or create the shared file handler for all loggers."""
    global _file_handler

    if _file_handler is None:
        log_path = Path(settings.LOG_PATH)
        log_path.mkdir(parents=True, exist_ok=True)

        # Single consolidated log file with rotation
        log_file = log_path / "app.log"

        # Rotating file handler: 10MB per file, keep 5 backup files
        _file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        _file_handler.setLevel(logging.DEBUG)

        # Format includes module name to distinguish different loggers
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        _file_handler.setFormatter(file_format)

    return _file_handler


def setup_logger(name: str) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    All loggers write to the same consolidated log file.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Add shared file handler
    logger.addHandler(_get_file_handler())

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    return logger
