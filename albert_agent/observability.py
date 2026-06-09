"""Observability: a structured run log plus optional LangSmith tracing.

Two complementary layers, matching the spec's "LangSmith works well; a simple
structured log is also acceptable":

- The **structured log** is always on. Every node entry, routing decision and
  tool call (name, arguments, outcome, duration) is written to both stderr and
  ``albert_agent.log`` as a single ``key=value`` line, so the agent's behaviour
  is auditable even with no third-party service.
- **LangSmith** is opt-in: if a LangSmith API key is present in the
  environment, full traces (prompts, token usage, latencies, the graph
  trajectory) are exported automatically by LangChain. No key → no-op.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager

from . import config

_LOGGER_NAME = "albert_agent"
_configured = False


def get_logger() -> logging.Logger:
    """Return the package logger, configuring handlers exactly once."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    try:  # file handler is best-effort; never let logging break the app
        file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        pass

    logger.propagate = False
    _configured = True
    return logger


def _kv(**fields: object) -> str:
    """Render fields as a compact, log-friendly ``key=value`` string."""
    parts = []
    for key, value in fields.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        text = str(value)
        if len(text) > 300:  # keep log lines bounded
            text = text[:297] + "..."
        if " " in text or "=" in text:
            text = f'"{text}"'
        parts.append(f"{key}={text}")
    return " ".join(parts)


def log_event(event: str, **fields: object) -> None:
    """Emit a single structured log line, e.g. ``event=route decision=tools``."""
    get_logger().info(_kv(event=event, **fields))


@contextmanager
def log_tool_call(name: str, args: dict):
    """Context manager timing a tool call and logging its outcome.

    Logs ``status=ok`` with a duration on success, or ``status=error`` with the
    exception type on failure (the exception is re-raised for the caller to
    turn into a graceful ToolMessage).
    """
    start = time.perf_counter()
    log_event("tool_start", tool=name, args=args)
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - we log then re-raise
        ms = round((time.perf_counter() - start) * 1000)
        log_event("tool_end", tool=name, status="error",
                  error=type(exc).__name__, ms=ms)
        raise
    else:
        ms = round((time.perf_counter() - start) * 1000)
        log_event("tool_end", tool=name, status="ok", ms=ms)


def setup_langsmith() -> bool:
    """Enable LangSmith tracing if an API key is configured.

    Accepts either the modern ``LANGSMITH_*`` or the legacy ``LANGCHAIN_*``
    variable names. Returns ``True`` when tracing was switched on.
    """
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not key:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", key)
    os.environ.setdefault("LANGCHAIN_PROJECT", "albert-student-assistant")
    log_event("langsmith", status="enabled",
              project=os.environ["LANGCHAIN_PROJECT"])
    return True
