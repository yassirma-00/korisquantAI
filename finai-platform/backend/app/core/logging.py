"""Structured logging configuration shared by API, workers and CLI scripts."""

from __future__ import annotations

import logging
import sys
from logging.config import dictConfig

from app.core.config import settings

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = (level or settings.LOG_LEVEL).upper()
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)-32s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": sys.stdout,
                },
            },
            "root": {"handlers": ["console"], "level": log_level},
            "loggers": {
                "uvicorn.access": {"level": "WARNING", "propagate": True},
                "yfinance": {"level": "ERROR", "propagate": False},
                "urllib3": {"level": "WARNING", "propagate": False},
                "peewee": {"level": "ERROR", "propagate": False},
            },
        }
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
