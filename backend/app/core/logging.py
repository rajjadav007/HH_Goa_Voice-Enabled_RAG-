"""Structured logging configuration for backend application."""

import logging
import sys
from app.core.config import settings


def setup_logging() -> None:
    """Configures application-wide logging format and level."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Silence overly verbose third-party loggers if needed
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


logger = logging.getLogger("hh_goa_rag")
