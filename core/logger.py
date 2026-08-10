"""Structured logging (contract §5, guideline C).

Log *events with fields*, not sentences:

    log.info("vector_retrieved", count=12, lead_engine="both")

not:

    log.info(f"got 12 results for a both-lead query")

The first is queryable — on Day 7 you will want "every request where
lead_engine=both and citation_count=0", which is a filter over fields and
essentially impossible over prose. Every node emits its key decisions this way
as it is written; observability is not retrofitted at the end.
"""

from __future__ import annotations

import logging
import sys

import structlog

from core.config import settings

_configured = False


def configure_logging() -> None:
    """Idempotent global logging setup. Safe to call from any entrypoint."""
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    # Third-party libraries that log one line per HTTP call would drown our own
    # events during a 5,000-request ingest. Raise their floor to WARNING.
    for noisy in ("httpx", "httpcore", "urllib3", "neo4j", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # The processor chain runs on every log call, in order. The last processor
    # renders: JSON lines for machines, coloured key=value for humans.
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # request-scoped fields
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger bound to a component name, e.g. get_logger("tmdb_pull")."""
    configure_logging()
    return structlog.get_logger(component=name)
