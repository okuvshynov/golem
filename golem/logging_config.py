"""Logging configuration for golem."""

import logging
import sys


def setup_logging(level: str = "INFO", name: str = "golem") -> logging.Logger:
    """Set up logging with a consistent format.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Don't add handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Create console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_logger(name: str = "golem") -> logging.Logger:
    """Get a logger instance.

    If the logger hasn't been set up yet, it will be configured with INFO level.

    Args:
        name: Logger name (typically module name)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # Set up with defaults if not already configured
    if not logger.handlers:
        setup_logging("INFO", name)

    return logger
