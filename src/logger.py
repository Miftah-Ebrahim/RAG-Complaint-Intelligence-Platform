"""Centralized logging for the CrediTrust RAG platform.

Provides a singleton ``logger`` instance with both file and console
output, ensuring consistent log formatting across the entire application.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "CrediTrust_RAG", log_file: str = "app.log"
) -> logging.Logger:
    """Configure and return a logger with console and file handlers.

    Creates a ``logs/`` directory in the current working directory if it
    does not already exist, and attaches a file handler and a console
    handler with a unified format.

    Args:
        name: Name assigned to the logger instance.
        log_file: Filename for the log file inside the ``logs/``
            directory.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    # Create logs directory if it doesn't exist
    log_dir: Path = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File Handler
    file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Singleton logger instance
logger: logging.Logger = setup_logger()
