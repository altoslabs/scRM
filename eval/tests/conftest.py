"""Suite-wide isolation: `eval/tests` must not write to shared storage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True, scope="session")
def _panel_cache_is_never_the_shared_one(tmp_path_factory):
    from scrm_eval import targets

    real = targets.CACHE_DIR
    targets.CACHE_DIR = tmp_path_factory.mktemp("target_cache")
    yield
    targets.CACHE_DIR = real
