"""Tests for the decay (TTL pruning) module."""

from datetime import UTC, datetime, timedelta

from myelin.recall.decay import find_lru, find_stale


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

    def test_skips_row_with_missing_last_accessed(self) -> None:
        """Rows missing last_accessed are skipped, not crashed."""
        good = self._make_meta("good", days_ago=120, access_count=1)
        bad = {"id": "bad"}  # no last_accessed key
        result = find_stale(
            [good, bad],
            max_idle_days=90,
            min_access_count=2,
            max_idle_days_absolute=365,
        )
        assert result == ["good"]  # bad row silently skipped

    def test_skips_row_with_invalid_last_accessed(self) -> None:
        """Rows with unparseable timestamps are skipped."""
        meta = [{"id": "x", "last_accessed": "not-a-date", "access_count": 0}]
        result = find_stale(meta, max_idle_days=90, min_access_count=2)
        assert result == []

    def test_tolerates_missing_access_count(self) -> None:
        """Missing access_count defaults to 0 (prune-eligible)."""
        last = (datetime.now(UTC) - timedelta(days=120)).isoformat()
        meta = [{"id": "a", "last_accessed": last}]  # no access_count
        result = find_stale(meta, max_idle_days=90, min_access_count=2)
        assert result == ["a"]


class TestFindLru:
    def _meta(self, id_: str, days_ago: int) -> dict:
        ts = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
        return {"id": id_, "last_accessed": ts}

    def test_empty_list_returns_empty(self) -> None:
        assert find_lru([], 3) == []

    def test_n_zero_returns_empty(self) -> None:
        meta = [self._meta("a", 5), self._meta("b", 10)]
        assert find_lru(meta, 0) == []

    def test_n_negative_returns_empty(self) -> None:
        meta = [self._meta("a", 5)]
        assert find_lru(meta, -1) == []

    def test_returns_oldest_first(self) -> None:
        meta = [
            self._meta("recent", 1),
            self._meta("old", 30),
            self._meta("ancient", 90),
        ]
        result = find_lru(meta, 2)
        assert result == ["ancient", "old"]

    def test_excludes_pinned_ids(self) -> None:
        meta = [
            self._meta("ancient", 90),
            self._meta("medium", 30),
            self._meta("recent", 1),
        ]
        result = find_lru(meta, 1, exclude_ids={"ancient"})
        assert result == ["medium"]

    def test_n_greater_than_available_returns_all(self) -> None:
        meta = [self._meta("a", 10), self._meta("b", 5)]
        result = find_lru(meta, 10)
        assert set(result) == {"a", "b"}

    def test_excludes_multiple_pinned(self) -> None:
        meta = [self._meta(str(i), i * 10) for i in range(5)]
        # Exclude the 2 oldest (40 and 30 days ago, i.e. ids "4" and "3")
        result = find_lru(meta, 2, exclude_ids={"4", "3"})
        assert result == ["2", "1"]

    def test_all_excluded_returns_empty(self) -> None:
        meta = [self._meta("a", 10), self._meta("b", 20)]
        assert find_lru(meta, 5, exclude_ids={"a", "b"}) == []
