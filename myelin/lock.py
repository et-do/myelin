"""Per-process exclusive lock on a Myelin data directory.

Available as a utility for callers that need to prevent concurrent writes
(e.g., a batch-import script that wants sole ownership while importing a
large snapshot).  The MCP server itself does **not** acquire this lock on
startup: multiple ``myelin serve`` processes may open the same data
directory simultaneously.  Write safety relies on SQLite's WAL mode,
which serialises concurrent writers at the database level and allows
unlimited concurrent readers.

Implementation
--------------
We use a lock *file* (``<data_dir>/myelin.lock``) rather than a
database-level mechanism so it covers all backends at once (ChromaDB,
neocortex.db, hebbian.db, thalamus.db).

- **Unix**: ``fcntl.flock(fd, LOCK_EX | LOCK_NB)`` — released
  automatically by the OS when the process exits, even on crash.
- **Windows**: ``msvcrt.locking`` on a byte of the file — also released
  on process death.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import TracebackType


class DataDirLockedError(RuntimeError):
    """Raised when another process already holds the data-directory lock."""


class DataDirLock:
    """Exclusive per-process lock on a Myelin data directory.

    Usage::

        with DataDirLock(data_dir):
            run_server()

    Or acquire/release manually::

        lock = DataDirLock(data_dir)
        lock.acquire()   # raises DataDirLockedError if already taken
        ...
        lock.release()
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "myelin.lock"
        self._fd: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """Acquire the exclusive lock.  Raises DataDirLockedError if unavailable."""
        data_dir = self._path.parent
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR)
        except OSError as exc:
            raise DataDirLockedError(
                f"Cannot open lock file {self._path}: {exc}"
            ) from exc

        try:
            _lock_exclusive_nonblocking(fd)
        except OSError:
            os.close(fd)
            raise DataDirLockedError(
                f"Another myelin-mcp process is already running against "
                f"{self._path.parent}.\n"
                "Only one process may open a data directory at a time.\n"
                "Stop the other process first, or use a different MYELIN_DATA_DIR."
            )
        self._fd = fd

    def release(self) -> None:
        """Release the lock and close the file descriptor."""
        if self._fd is None:
            return
        try:
            _unlock(self._fd)
        finally:
            os.close(self._fd)
            self._fd = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> DataDirLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()


# ------------------------------------------------------------------
# Platform-specific locking primitives
# ------------------------------------------------------------------

if sys.platform == "win32":
    import msvcrt

    def _lock_exclusive_nonblocking(fd: int) -> None:
        """Attempt a non-blocking exclusive lock on Windows."""
        # Move to byte 0 and lock exactly 1 byte.
        os.lseek(fd, 0, os.SEEK_SET)
        # LK_NBLCK raises OSError immediately if already locked.
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]

else:
    import fcntl

    def _lock_exclusive_nonblocking(fd: int) -> None:
        """Attempt a non-blocking exclusive lock on Unix."""
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
