"""Logging setup shared by the service, the CLI and the scripts."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logging once; safe to call from several entry points."""
    root = logging.getLogger()
    if getattr(root, "_sichuan_configured", False):
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATEFMT)
    root._sichuan_configured = True  # type: ignore[attr-defined]
