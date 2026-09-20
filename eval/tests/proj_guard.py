"""The marker that skips tests whose data files exist only under a project root."""

from __future__ import annotations

import os

import pytest

needs_proj = pytest.mark.skipif(
    not os.environ.get("SCRM_PROJ"),
    reason="SCRM_PROJ names no project root, so the data files these assert on are absent",
)
