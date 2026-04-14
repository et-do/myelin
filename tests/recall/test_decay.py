"""Tests for the decay (TTL pruning) module."""

from datetime import UTC, datetime, timedelta

from myelin.recall.decay import find_stale


class TestFindStale:
    def _make_meta(self, id_: str, days_ago: int, access_count: int) -> dict:
        last = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
        return {"id": id_, "last_accessed": last, "access_count": access_count}

    def test_prunes_old_low_access(self) -> None:
        meta = [self._make_meta("a", days_ago=120, access_count=1)]
        assert find_stale(meta, max_idle_days=90, min_access_count=2) == ["a"]

    def test_keeps_old_high_access(self) -> None:
        """High-access memories survive normal threshold but not absolute."""
        meta = [self._make_meta("a", days_ago=120, access_count=5)]
        assert (
            find_stale(
                meta,
                max_idle_days=90,
                min_access_count=2,
                max_idle_days_absolute=365,
            )
            == []
        )

    def test_keeps_recent_low_access(self) -> None:
        meta = [self._make_meta("a", days_ago=10, access_count=0)]
        assert find_stale(meta, max_idle_days=90, min_access_count=2) == []

    def test_mixed_batch(self) -> None:
        meta = [
            self._make_meta("keep1", days_ago=5, access_count=0),
            self._make_meta("prune1", days_ago=100, access_count=1),
            self._make_meta("keep2", days_ago=200, access_count=10),
            self._make_meta("prune2", days_ago=95, access_count=0),
        ]
        stale = find_stale(
            meta,
            max_idle_days=90,
            min_access_count=2,
            max_idle_days_absolute=365,
        )
        assert sorted(stale) == ["prune1", "prune2"]

    def test_absolute_threshold_prunes_high_access(self) -> None:
        """Memories idle beyond absolute max are pruned regardless of access count."""
        meta = [self._make_meta("a", days_ago=400, access_count=100)]
        stale = find_stale(
            meta,
            max_idle_days=90,
            min_access_count=2,
            max_idle_days_absolute=365,
        )
        assert stale == ["a"]

    def test_absolute_threshold_keeps_recently_accessed(self) -> None:
        """High-access memory under absolute threshold survives."""
        meta = [self._make_meta("a", days_ago=300, access_count=50)]
        stale = find_stale(
            meta,
            max_idle_days=90,
            min_access_count=2,
            max_idle_days_absolute=365,
        )
        assert stale == []

    def test_mixed_batch_with_absolute(self) -> None:
        meta = [
            self._make_meta("keep", days_ago=200, access_count=10),
            self._make_meta("prune_low", days_ago=100, access_count=0),
            self._make_meta("prune_abs", days_ago=500, access_count=50),
        ]
        stale = find_stale(
            meta,
            max_idle_days=90,
            min_access_count=2,
            max_idle_days_absolute=365,
        )
        assert sorted(stale) == ["prune_abs", "prune_low"]
