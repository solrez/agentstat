"""Logging to both console and a file under ``logs/`` for progress + debugging.

Long benchmark runs need visible progress and a persistent record. ``get_logger``
returns a logger that writes to stderr (so you see it live) and appends to a
per-run file ``logs/<name>-<run_id>.log``. The run id is passed in by the caller
(scripts stamp it once at startup) rather than generated here, because
``Date.now``-style time calls should live at the edge, not in library code.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

DEFAULT_LOG_DIR = Path("logs")

_CONFIGURED: set[str] = set()


def get_logger(
    name: str = "agentstat",
    run_id: str | None = None,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger that tees to console and ``logs/<name>-<run_id>.log``.

    Idempotent per (name, run_id): repeated calls don't stack duplicate handlers.
    If ``run_id`` is None, only the console handler is attached (no file).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    tag = f"{name}:{run_id}"
    if tag in _CONFIGURED:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (added once per name).
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(fmt)
        logger.addHandler(console)

    # File handler (only when a run_id is given, so we get one file per run).
    if run_id is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{name}-{run_id}.log"
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("Logging to %s", log_path)

    _CONFIGURED.add(tag)
    return logger
