"""Shared filesystem locations for EvoInfer MCP."""

from __future__ import annotations

import os
from pathlib import Path


def get_share_dir() -> Path:
    """Return the EvoInfer share directory.

    `EVOINFER_SHARE_DIR` is the canonical variable. If it is not set, EvoInfer
    stores data under `~/.evoinfer`.
    """

    configured = os.getenv("EVOINFER_SHARE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".evoinfer"
