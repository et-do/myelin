"""Tests for DataDirLock — process-level exclusive lock on the data directory."""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path

import pytest

from myelin.lock import DataDirLock, DataDirLockedError


class TestDataDirLockBasic:
    def test_acquire_and_release(self, tmp_path: Path) -> None:
        lock = DataDirLock(tmp_path)
        lock.acquire()
        lock.release()
        # Lock file should exist after release
        assert (tmp_path / "myelin.lock").exists()

    def test_context_manager(self, tmp_path: Path) -> None:
        with DataDirLock(tmp_path):
            pass  # Should not raise

    def test_creates_data_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        with DataDirLock(nested):
            assert nested.exists()

    def test_double_release_is_safe(self, tmp_path: Path) -> None:
        lock = DataDirLock(tmp_path)
        lock.acquire()
        lock.release()
        lock.release()  # Should not raise

    def test_reacquire_after_release(self, tmp_path: Path) -> None:
        lock = DataDirLock(tmp_path)
        lock.acquire()
        lock.release()
        # Can re-acquire the same lock after releasing
        lock.acquire()
        lock.release()


class TestDataDirLockExclusion:
    def test_second_acquire_raises(self, tmp_path: Path) -> None:
        """Two DataDirLock instances on the same dir — second must fail."""
        lock1 = DataDirLock(tmp_path)
        lock2 = DataDirLock(tmp_path)
        lock1.acquire()
        try:
            with pytest.raises(DataDirLockedError):
                lock2.acquire()
        finally:
            lock1.release()

    def test_context_manager_exclusion(self, tmp_path: Path) -> None:
        with DataDirLock(tmp_path):
            with pytest.raises(DataDirLockedError):
                DataDirLock(tmp_path).acquire()

    def test_released_lock_can_be_taken(self, tmp_path: Path) -> None:
        lock1 = DataDirLock(tmp_path)
        lock1.acquire()
        lock1.release()

        lock2 = DataDirLock(tmp_path)
        lock2.acquire()  # Should succeed now
        lock2.release()

    def test_different_dirs_dont_interfere(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        with DataDirLock(dir_a):
            with DataDirLock(dir_b):  # Different dir — should not raise
                pass

    @pytest.mark.skipif(sys.platform == "win32", reason="fork not available on Windows")
    def test_cross_process_exclusion(self, tmp_path: Path) -> None:
        """A child process must not be able to acquire a lock held by the parent."""

        def child_try_lock(path: str, result_queue: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
            try:
                DataDirLock(Path(path)).acquire()
                result_queue.put("acquired")
            except DataDirLockedError:
                result_queue.put("locked")

        lock = DataDirLock(tmp_path)
        lock.acquire()
        try:
            q: multiprocessing.Queue = multiprocessing.Queue()  # type: ignore[type-arg]
            p = multiprocessing.Process(target=child_try_lock, args=(str(tmp_path), q))
            p.start()
            p.join(timeout=5)
            assert not p.is_alive(), "child process timed out"
            result = q.get_nowait()
            assert result == "locked", f"expected 'locked', got {result!r}"
        finally:
            lock.release()

    @pytest.mark.skipif(sys.platform == "win32", reason="fork not available on Windows")
    def test_lock_released_on_process_death(self, tmp_path: Path) -> None:
        """After a child process dies (crash), the lock must be releaseable."""

        def child_acquire_and_die(path: str) -> None:
            # Acquire the lock, then exit without releasing it
            lock = DataDirLock(Path(path))
            lock.acquire()
            # Intentional abrupt exit — simulates crash, no atexit
            os._exit(0)

        p = multiprocessing.Process(target=child_acquire_and_die, args=(str(tmp_path),))
        p.start()
        p.join(timeout=5)
        assert p.exitcode == 0

        # The OS should have released the flock on process exit
        # Wait a tiny moment for the OS to clean up
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                lock = DataDirLock(tmp_path)
                lock.acquire()
                lock.release()
                break
            except DataDirLockedError:
                time.sleep(0.05)
        else:
            pytest.fail("Lock was not released after child process death")
