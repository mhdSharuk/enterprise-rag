"""Logging configuration for EnterpriseRAG services."""

import logging
import os
import sys

from dotenv import load_dotenv

# Load .env
load_dotenv()

# Get log level from environment
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


def get_logger(name: str = "enterpriserag") -> logging.Logger:
    """Get a logger instance."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set log level
    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


# Default logger
logger = get_logger("enterpriserag")