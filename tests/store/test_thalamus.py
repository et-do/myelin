"""Tests for the ThalamicBuffer — working memory relay."""

from __future__ import annotations

from pathlib import Path

import pytest

from myelin.config import MyelinSettings
from myelin.store.thalamus import ThalamicBuffer


@pytest.fixture()
def thalamus(tmp_path: Path) -> ThalamicBuffer:
    cfg = MyelinSettings(data_dir=tmp_path / ".myelin", thalamus_recency_limit=5)
    cfg.ensure_dirs()
    buf = ThalamicBuffer(cfg=cfg)
    yield buf
    buf.close()


class TestPin:
    def test_pin_and_get(self, thalamus: ThalamicBuffer) -> None:
        thalamus.pin("m1", priority=0, label="identity")
        thalamus.pin("m2", priority=1, label="fact")
        pinned = thalamus.get_pinned()
        assert len(pinned) == 2
        assert pinned[0]["memory_id"] == "m1"
        assert pinned[0]["priority"] == 0
        assert pinned[0]["label"] == "identity"
        assert pinned[1]["memory_id"] == "m2"

    def test_pin_upsert(self, thalamus: ThalamicBuffer) -> None:
        thalamus.pin("m1", priority=1, label="old")
        thalamus.pin("m1", priority=0, label="new")
        pinned = thalamus.get_pinned()
        assert len(pinned) == 1
        assert pinned[0]["priority"] == 0
        assert pinned[0]["label"] == "new"

    def test_unpin_returns_true(self, thalamus: ThalamicBuffer) -> None:
        thalamus.pin("m1")
        assert thalamus.unpin("m1") is True

    def test_unpin_missing_returns_false(self, thalamus: ThalamicBuffer) -> None:
        assert thalamus.unpin("nonexistent") is False

    def test_pinned_count(self, thalamus: ThalamicBuffer) -> None:
        assert thalamus.pinned_count() == 0
        thalamus.pin("m1")
        thalamus.pin("m2")
        assert thalamus.pinned_count() == 2
        thalamus.unpin("m1")
        assert thalamus.pinned_count() == 1

    def test_get_pinned_filters_by_priority(self, thalamus: ThalamicBuffer) -> None:
        thalamus.pin("m0", priority=0)
        thalamus.pin("m1", priority=1)
        thalamus.pin("m2", priority=2)
        result = thalamus.get_pinned(max_priority=1)
        ids = [r["memory_id"] for r in result]
        assert "m0" in ids
        assert "m1" in ids
        assert "m2" not in ids


class TestRecency:
    def test_touch_and_get_recent(self, thalamus: ThalamicBuffer) -> None:
        thalamus.touch(["a", "b", "c"])
        recent = thalamus.get_recent()
        assert recent == ["c", "b", "a"]

    def test_touch_moves_to_end(self, thalamus: ThalamicBuffer) -> None:
        thalamus.touch(["a", "b", "c"])
        thalamus.touch(["a"])  # re-touch 'a'
        recent = thalamus.get_recent()
        assert recent[0] == "a"

    def test_recency_limit_trims(self, thalamus: ThalamicBuffer) -> None:
        """Buffer should trim to thalamus_recency_limit (5)."""
        thalamus.touch(["a", "b", "c", "d", "e", "f", "g"])
        recent = thalamus.get_recent()
        assert len(recent) == 5
        # Oldest entries trimmed
        assert "a" not in recent
        assert "b" not in recent
        assert "g" in recent

    def test_get_recent_with_n(self, thalamus: ThalamicBuffer) -> None:
        thalamus.touch(["a", "b", "c", "d"])
        recent = thalamus.get_recent(n=2)
        assert len(recent) == 2
        assert recent == ["d", "c"]


class TestDominantRegion:
    def test_returns_majority_region(self, thalamus: ThalamicBuffer) -> None:
        thalamus.touch(["a", "b", "c", "d", "e"])
        lookup = {
            "a": "security",
            "b": "security",
            "c": "security",
            "d": "data",
            "e": "security",
        }
        assert thalamus.dominant_region(lookup) == "security"

    def test_returns_none_below_threshold(self, thalamus: ThalamicBuffer) -> None:
        thalamus.touch(["a", "b", "c", "d", "e"])
        lookup = {
            "a": "security",
            "b": "data",
            "c": "frontend",
            "d": "data",
            "e": "security",
        }
        assert thalamus.dominant_region(lookup) is None

    def test_returns_none_when_empty(self, thalamus: ThalamicBuffer) -> None:
        assert thalamus.dominant_region({}) is None

    def test_returns_none_below_minimum_total(self, thalamus: ThalamicBuffer) -> None:
        thalamus.touch(["a", "b"])
        lookup = {"a": "security", "b": "security"}
        assert thalamus.dominant_region(lookup) is None


class TestCleanup:
    def test_removes_stale_pins(self, thalamus: ThalamicBuffer) -> None:
        thalamus.pin("m1")
        thalamus.pin("m2")
        thalamus.pin("m3")
        removed = thalamus.cleanup(valid_ids={"m1"})
        assert removed == 2
        assert thalamus.pinned_count() == 1
        assert thalamus.get_pinned()[0]["memory_id"] == "m1"

    def test_removes_stale_recency(self, thalamus: ThalamicBuffer) -> None:
        thalamus.touch(["a", "b", "c"])
        thalamus.cleanup(valid_ids={"b"})
        recent = thalamus.get_recent()
        assert recent == ["b"]
