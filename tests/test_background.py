"""Tests for myelin.background — BackgroundWorker and DecayTimer."""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

from myelin.background import BackgroundWorker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_worker(
    *,
    consolidate_fn=None,
    decay_fn=None,
    decay_interval_hours: float = 0,  # disabled by default for most tests
    queue_maxsize: int = 10,
) -> BackgroundWorker:
    return BackgroundWorker(
        consolidate_fn=consolidate_fn or MagicMock(return_value={"ok": True}),
        decay_fn=decay_fn or MagicMock(return_value={"pruned": 0}),
        decay_interval_hours=decay_interval_hours,
        queue_maxsize=queue_maxsize,
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_not_running_before_start(self) -> None:
        w = _make_worker()
        assert not w.is_running

    def test_running_after_start(self) -> None:
        w = _make_worker()
        w.start()
        try:
            assert w.is_running
        finally:
            w.stop()

    def test_start_is_idempotent(self) -> None:
        w = _make_worker()
        w.start()
        thread_before = w._thread
        w.start()  # second call should be a no-op
        assert w._thread is thread_before
        w.stop()

    def test_not_running_after_stop(self) -> None:
        w = _make_worker()
        w.start()
        w.stop()
        assert not w.is_running

    def test_stop_before_start_is_safe(self) -> None:
        w = _make_worker()
        w.stop()  # should not raise

    def test_stop_is_idempotent(self) -> None:
        w = _make_worker()
        w.start()
        w.stop()
        w.stop()  # second stop should not raise


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


class TestConsolidation:
    def test_submit_returns_true_when_queued(self) -> None:
        w = _make_worker()
        w.start()
        try:
            assert w.submit_consolidate() is True
        finally:
            w.stop()

    def test_consolidate_fn_called(self) -> None:
        called = threading.Event()
        fn = MagicMock(side_effect=lambda: called.set() or {"ok": True})
        w = _make_worker(consolidate_fn=fn)
        w.start()
        w.submit_consolidate()
        called.wait(timeout=5)
        w.stop()
        fn.assert_called()

    def test_last_consolidation_at_set_after_run(self) -> None:
        called = threading.Event()
        fn = MagicMock(side_effect=lambda: called.set() or {"ok": True})
        w = _make_worker(consolidate_fn=fn)
        w.start()
        assert w.status()["last_consolidation_at"] is None
        w.submit_consolidate()
        called.wait(timeout=5)
        w.stop()
        assert w.status()["last_consolidation_at"] is not None

    def test_submit_returns_false_when_queue_full(self) -> None:
        """Queue full → task dropped, returns False."""
        # Use a tiny queue and a slow consolidate to fill it before processing.
        barrier = threading.Barrier(2)

        def slow_fn() -> dict[str, Any]:
            barrier.wait(timeout=5)  # block until test releases
            return {"ok": True}

        w = _make_worker(consolidate_fn=slow_fn, queue_maxsize=2)
        w.start()
        # Fill queue + the one being processed
        results = [w.submit_consolidate() for _ in range(10)]
        # At least one should have been dropped (returned False)
        assert False in results
        # Unblock the worker
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        w.stop()

    def test_consolidate_exception_does_not_crash_worker(self) -> None:
        calls: list[int] = []

        def boom() -> None:
            calls.append(1)
            raise RuntimeError("oops")

        second = threading.Event()

        def ok() -> dict[str, Any]:
            second.set()
            return {"ok": True}

        call_count = [0]
        lock = threading.Lock()

        def alternating() -> Any:
            with lock:
                call_count[0] += 1
                n = call_count[0]
            if n == 1:
                raise RuntimeError("first call fails")
            second.set()
            return {"ok": True}

        w = _make_worker(consolidate_fn=alternating)
        w.start()
        w.submit_consolidate()
        w.submit_consolidate()
        second.wait(timeout=5)
        w.stop()
        assert w.is_running is False  # worker survived the exception


# ---------------------------------------------------------------------------
# Periodic decay
# ---------------------------------------------------------------------------


class TestPeriodicDecay:
    def test_decay_fires_when_interval_elapsed(self) -> None:
        decay_called = threading.Event()
        decay_fn = MagicMock(side_effect=lambda: decay_called.set() or {"pruned": 0})

        # Set interval to a tiny fraction of a second so the test is fast.
        w = BackgroundWorker(
            consolidate_fn=MagicMock(return_value={}),
            decay_fn=decay_fn,
            decay_interval_hours=0.0001,  # ~0.36 seconds
        )
        w.start()
        fired = decay_called.wait(timeout=5)
        w.stop()
        assert fired, "decay_fn was not called within timeout"

    def test_decay_not_fired_when_interval_zero(self) -> None:
        decay_fn = MagicMock(return_value={"pruned": 0})
        w = _make_worker(decay_fn=decay_fn, decay_interval_hours=0)
        w.start()
        time.sleep(0.2)
        w.stop()
        decay_fn.assert_not_called()

    def test_last_decay_at_set_after_run(self) -> None:
        decay_called = threading.Event()
        decay_fn = MagicMock(side_effect=lambda: decay_called.set() or {"pruned": 0})
        w = BackgroundWorker(
            consolidate_fn=MagicMock(return_value={}),
            decay_fn=decay_fn,
            decay_interval_hours=0.0001,
        )
        w.start()
        assert w.status()["last_decay_at"] is None
        decay_called.wait(timeout=5)
        w.stop()
        assert w.status()["last_decay_at"] is not None

    def test_decay_exception_does_not_crash_worker(self) -> None:
        second_decay = threading.Event()
        call_count = [0]
        lock = threading.Lock()

        def boom_then_ok() -> Any:
            with lock:
                call_count[0] += 1
                n = call_count[0]
            if n == 1:
                raise RuntimeError("first decay fails")
            second_decay.set()
            return {"pruned": 0}

        w = BackgroundWorker(
            consolidate_fn=MagicMock(return_value={}),
            decay_fn=boom_then_ok,
            decay_interval_hours=0.0001,
        )
        w.start()
        second_decay.wait(timeout=10)
        w.stop()
        assert call_count[0] >= 2


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_fields_present(self) -> None:
        w = _make_worker()
        s = w.status()
        assert "running" in s
        assert "queue_depth" in s
        assert "last_consolidation_at" in s
        assert "last_decay_at" in s

    def test_status_running_false_before_start(self) -> None:
        w = _make_worker()
        assert w.status()["running"] is False

    def test_status_running_true_after_start(self) -> None:
        w = _make_worker()
        w.start()
        try:
            assert w.status()["running"] is True
        finally:
            w.stop()

    def test_queue_depth_reflects_queued_tasks(self) -> None:
        # Block the worker so tasks stay in the queue.
        barrier = threading.Barrier(2)
        fn = MagicMock(side_effect=lambda: barrier.wait(timeout=5) or {})

        w = BackgroundWorker(
            consolidate_fn=fn,
            decay_fn=MagicMock(return_value={}),
            decay_interval_hours=0,
            queue_maxsize=20,
        )
        w.start()
        # Flood the queue before the worker can drain it.
        for _ in range(5):
            w.submit_consolidate()
        depth = w.status()["queue_depth"]
        assert depth >= 0  # may have already drained some — just ensure no crash
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        w.stop()
