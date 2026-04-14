"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from myelin.config import MyelinSettings
from myelin.recall.activation import HebbianTracker
from myelin.store.hippocampus import Hippocampus


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> MyelinSettings:
    """MyelinSettings pointing at a disposable temp directory."""
    return MyelinSettings(data_dir=tmp_path / ".myelin")


@pytest.fixture()
def hippocampus(tmp_settings: MyelinSettings) -> Hippocampus:
    """Hippocampus backed by a temp ChromaDB, isolated per test."""
    return Hippocampus(cfg=tmp_settings)


@pytest.fixture()
def hebbian_tracker(tmp_path: Path) -> Generator[HebbianTracker]:
    """HebbianTracker backed by a temp SQLite DB, auto-closed after test."""
    tracker = HebbianTracker(db_path=tmp_path / "test.db")
    yield tracker
    tracker.close()
