"""EvoInfer MCP package."""

from __future__ import annotations

import logging
from typing import Any

__version__ = "0.1.0"


class _BraceLogger:
    """Small compatibility logger for extracted code that used loguru-style braces."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("evoinfer_mcp")

    def warning(self, message: str, *args: Any) -> None:
        if args:
            try:
                message = message.format(*args)
            except Exception:
                message = f"{message} {' '.join(str(arg) for arg in args)}"
        self._logger.warning(message)


logger = _BraceLogger()

__all__ = ["__version__", "logger"]
