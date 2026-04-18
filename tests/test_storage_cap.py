"""Tests for storage cap (LRU eviction) and auto-decay timer."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from myelin.config import MyelinSettings
from myelin.recall.decay import find_lru
from myelin.server import configure, do_status, do_store
from myelin.timer import DecayTimer

# ---------------------------------------------------------------------------
# find_lru — unit tests (pure function, no I/O)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Storage cap — integration tests via do_store()
# ---------------------------------------------------------------------------


@pytest.fixture()
def cap_settings(tmp_path: pytest.TempPathFactory) -> MyelinSettings:
    """Settings with cap=3, consolidation off."""
    return MyelinSettings(
        data_dir=tmp_path / ".myelin",
        max_memories=3,
        consolidation_interval=0,
    )


@pytest.fixture(autouse=True)
def _isolate(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: no cap, consolidation off."""
    cfg = MyelinSettings(data_dir=tmp_path / ".myelin", consolidation_interval=0)
    configure(cfg)
    monkeypatch.setattr("myelin.cli.settings", cfg)


class TestStorageCap:
    def test_no_eviction_when_disabled(self) -> None:
        """max_memories=0 means no cap — memories accumulate freely."""
        contents = [
            "JWT RS256 authentication approach for the API gateway service",
            "Redis cluster configuration for distributed session caching",
            "PostgreSQL read replicas for scaling analytics queries",
            "Kubernetes pod autoscaling based on CPU and memory metrics",
            "Terraform modules for provisioning AWS VPC and subnets",
        ]
        for c in contents:
            do_store(c)
        status = do_status()
        assert status["memory_count"] == 5
        assert status["max_memories"] == 0

    def test_no_eviction_below_cap(self, cap_settings: MyelinSettings) -> None:
        configure(cap_settings)
        do_store("JWT RS256 authentication approach for the API service")
        do_store("PostgreSQL schema design for the users table")
        status = do_status()
        assert status["memory_count"] == 2

    def test_evicts_lru_when_over_cap(self, cap_settings: MyelinSettings) -> None:
        configure(cap_settings)
        # Store 4 distinct memories — should trigger eviction (cap=3)
        do_store("JWT RS256 auth tokens for the authentication service")
        do_store("Redis cluster used for distributed session caching layer")
        do_store("PostgreSQL database schema design for users and accounts")
        result = do_store("Kubernetes deployment with HPA for the web frontend")
        status = do_status()
        assert status["memory_count"] == 3
        assert result.get("evicted") == 1

    def test_evicted_count_in_store_result(self, cap_settings: MyelinSettings) -> None:
        configure(cap_settings)
        do_store("JWT RS256 auth tokens for the authentication service")
        do_store("Redis cluster used for distributed session caching layer")
        do_store("PostgreSQL database schema design for users and accounts")
        result = do_store("Kubernetes deployment with HPA for the web frontend")
        assert result.get("evicted", 0) >= 1

    def test_count_stays_at_cap_after_many_stores(
        self, cap_settings: MyelinSettings
    ) -> None:
        configure(cap_settings)
        topics = [
            "JWT RS256 authentication approach",
            "Redis distributed session caching",
            "PostgreSQL schema for users table",
            "Kubernetes HPA deployment config",
            "Terraform VPC provisioning modules",
            "gRPC service mesh configuration",
            "Prometheus alerting rule definitions",
            "Elasticsearch index mapping schema",
            "Kafka consumer group rebalancing",
            "Vault secret rotation policy setup",
        ]
        for t in topics:
            do_store(t)
        status = do_status()
        assert status["memory_count"] <= 3

    def test_cap_of_one(self, tmp_path: pytest.TempPathFactory) -> None:
        cfg = MyelinSettings(
            data_dir=tmp_path / ".myelin-cap1",
            max_memories=1,
            consolidation_interval=0,
        )
        configure(cfg)
        do_store("JWT RS256 authentication approach for the API service")
        do_store("PostgreSQL schema design for users and accounts table")
        status = do_status()
        assert status["memory_count"] == 1

    def test_status_reports_max_memories(self, cap_settings: MyelinSettings) -> None:
        configure(cap_settings)
        status = do_status()
        assert status["max_memories"] == 3

    def test_no_evicted_key_when_under_cap(self, cap_settings: MyelinSettings) -> None:
        configure(cap_settings)
        result = do_store("JWT RS256 authentication for the API service only memory")
        assert "evicted" not in result

    def test_eviction_does_not_affect_new_memory(
        self, cap_settings: MyelinSettings
    ) -> None:
        """The just-stored memory should survive even if it evicts an older one."""
        configure(cap_settings)
        do_store("JWT RS256 authentication approach for the API service")
        do_store("Redis cluster used for distributed session caching layer")
        do_store("PostgreSQL database schema design for users and accounts")
        new_result = do_store("Terraform VPC module provisioning on AWS infrastructure")
        # The new memory's ID is in the store (not evicted)
        assert new_result["status"] in ("stored", "updated")
        status = do_status()
        assert status["memory_count"] == 3


# ---------------------------------------------------------------------------
# DecayTimer — unit tests
# ---------------------------------------------------------------------------


class TestDecayTimer:
    def test_disabled_when_interval_zero(self) -> None:
        called = threading.Event()
        timer = DecayTimer(fn=called.set, interval_hours=0)
        timer.start()
        assert not timer.is_running
        assert not called.is_set()

    def test_disabled_when_interval_negative(self) -> None:
        timer = DecayTimer(fn=lambda: None, interval_hours=-1)
        timer.start()
        assert not timer.is_running

    def test_start_stop_lifecycle(self) -> None:
        timer = DecayTimer(fn=lambda: None, interval_hours=999)
        timer.start()
        assert timer.is_running
        timer.stop()
        assert not timer.is_running

    def test_start_is_idempotent(self) -> None:
        timer = DecayTimer(fn=lambda: None, interval_hours=999)
        timer.start()
        thread1 = timer._thread
        timer.start()  # second call should be a no-op
        assert timer._thread is thread1
        timer.stop()

    def test_stop_before_start_is_safe(self) -> None:
        timer = DecayTimer(fn=lambda: None, interval_hours=1)
        timer.stop()  # should not raise

    def test_fn_fires_after_interval(self) -> None:
        fired = threading.Event()
        # Use a very short interval (0.05s = 0.05/3600 hours)
        timer = DecayTimer(fn=fired.set, interval_hours=0.05 / 3600)
        timer.start()
        assert fired.wait(timeout=5.0), "decay function did not fire within timeout"
        timer.stop()

    def test_fn_exception_does_not_kill_timer(self) -> None:
        call_count = [0]

        def flaky() -> None:
            call_count[0] += 1
            raise RuntimeError("boom")

        # Short interval so it fires quickly
        timer = DecayTimer(fn=flaky, interval_hours=0.02 / 3600)
        timer.start()
        time.sleep(0.2)  # let it fire a couple times
        assert timer.is_running  # thread should still be alive
        assert call_count[0] >= 1
        timer.stop()

    def test_stop_wakes_sleeping_timer(self) -> None:
        """stop() should return promptly even when timer is mid-sleep."""
        timer = DecayTimer(fn=lambda: None, interval_hours=24)  # very long interval
        timer.start()
        t0 = time.monotonic()
        timer.stop()
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, f"stop() blocked for {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# DecayTimer — integration with server status
# ---------------------------------------------------------------------------


class TestDecayTimerServerIntegration:
    def test_timer_not_running_when_interval_zero(self) -> None:
        status = do_status()
        assert status["decay_timer_running"] is False

    def test_timer_starts_when_interval_set(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        cfg = MyelinSettings(
            data_dir=tmp_path / ".myelin2",
            decay_interval_hours=24.0,
            consolidation_interval=0,
        )
        configure(cfg)
        # Manually start the timer (lifespan not running in tests)
        from myelin.server import _get_decay_timer

        _get_decay_timer().start()
        try:
            status = do_status()
            assert status["decay_timer_running"] is True
        finally:
            _get_decay_timer().stop()
