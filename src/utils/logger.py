"""Logging configuration for EnterpriseRAG services."""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(os.getcwd()) / ".env"
load_dotenv(dotenv_path=env_path, override=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def get_logger(name: str = "enterpriserag") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger

logger = get_logger("enterpriserag")