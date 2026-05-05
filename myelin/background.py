"""Background maintenance — periodic decay timer and off-hot-path worker thread.

Merges two previously separate modules (``timer.py`` and ``worker.py``) into a
single coherent module so that all background thread management lives in one
place.

Architecture
------------
Two classes serve different use cases:

:class:`DecayTimer`
    Simple periodic callback — fires a caller-supplied function every N hours
    on a daemon thread.  Used in the MCP server lifespan for auto-decay sweeps.

:class:`BackgroundWorker`
    Queue-driven daemon thread with *both* queue tasks (consolidation) and a
    built-in periodic decay sweep.  The single thread handles both concerns so
    maintenance never blocks the hot path.

    1. **Queue-driven** — consolidation requests submitted via
       :meth:`BackgroundWorker.submit_consolidate`.  Bounded queue; full-queue
       drops are safe because consolidation is idempotent.
    2. **Timer-driven** — decay sweep fires at ``decay_interval_hours`` cadence
       inside the same loop, so the thread never sleeps longer than ~1 second
       while still honouring the interval.

Lifecycle (both classes)
------------------------
- ``start()`` — idempotent; starts the daemon thread.
- ``stop()`` — signals STOP; joins thread.
- Integrated with server lifespan: started during startup, stopped on shutdown.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


# ===========================================================================
# DecayTimer
# ===========================================================================


class DecayTimer:
    """Calls *fn* repeatedly on a background thread every *interval_hours* hours.

    - ``interval_hours <= 0`` disables the timer; ``start()`` is a no-op.
    - Exceptions raised by *fn* are logged and swallowed so the loop continues.
    - ``stop()`` signals the thread and waits up to 2 s for a clean shutdown.
    """

    def __init__(self, fn: Callable[[], Any], interval_hours: float) -> None:
        self._fn = fn
        self._interval = interval_hours * 3600.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background timer thread (idempotent)."""
        if self._interval <= 0:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="myelin-decay-timer",
        )
        self._thread.start()
        logger.debug("decay timer started (interval=%.1fh)", self._interval / 3600.0)

    def stop(self) -> None:
        """Signal the timer to stop and wait for the thread to exit."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def is_running(self) -> bool:
        """True while the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(timeout=self._interval):
            try:
                self._fn()
            except Exception:
                logger.exception("decay timer: error during sweep")


# ===========================================================================
# BackgroundWorker
# ===========================================================================


class _Task(Enum):
    CONSOLIDATE = auto()
    STOP = auto()


class BackgroundWorker:
    """Daemon thread that handles off-hot-path memory maintenance."""

    def __init__(
        self,
        *,
        consolidate_fn: Callable[[], Any],
        decay_fn: Callable[[], Any],
        decay_interval_hours: float = 24.0,
        queue_maxsize: int = 10,
    ) -> None:
        self._consolidate_fn = consolidate_fn
        self._decay_fn = decay_fn
        self._decay_interval_seconds = decay_interval_hours * 3600.0
        self._queue: queue.Queue[_Task] = queue.Queue(maxsize=queue_maxsize)
        self._thread: threading.Thread | None = None
        self._last_consolidation_at: datetime | None = None
        self._last_decay_at: datetime | None = None
        self._ts_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="myelin-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "background worker started (decay_interval=%.1fh)",
            self._decay_interval_seconds / 3600,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for it to finish."""
        if self._thread is None or not self._thread.is_alive():
            return
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(_Task.STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("background worker stopped")

    def submit_consolidate(self) -> bool:
        """Queue a consolidation run (non-blocking, idempotent on drop)."""
        try:
            self._queue.put_nowait(_Task.CONSOLIDATE)
            return True
        except queue.Full:
            logger.debug("consolidation task dropped (worker queue full)")
            return False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        """Return worker health info for ``do_status()``."""
        with self._ts_lock:
            return {
                "running": self.is_running,
                "queue_depth": self._queue.qsize(),
                "last_consolidation_at": (
                    self._last_consolidation_at.isoformat()
                    if self._last_consolidation_at
                    else None
                ),
                "last_decay_at": (
                    self._last_decay_at.isoformat() if self._last_decay_at else None
                ),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        next_decay = (
            _now().timestamp() + self._decay_interval_seconds
            if self._decay_interval_seconds > 0
            else None
        )

        while True:
            if next_decay is not None:
                wait = max(0.0, min(1.0, next_decay - _now().timestamp()))
            else:
                wait = 1.0

            try:
                task = self._queue.get(timeout=wait)
            except queue.Empty:
                task = None

            if task is _Task.STOP:
                break

            if task is _Task.CONSOLIDATE:
                self._run_consolidate()

            if next_decay is not None and _now().timestamp() >= next_decay:
                self._run_decay()
                next_decay = _now().timestamp() + self._decay_interval_seconds

    def _run_consolidate(self) -> None:
        try:
            logger.info("background consolidation started")
            result = self._consolidate_fn()
            with self._ts_lock:
                self._last_consolidation_at = _now()
            logger.info("background consolidation done: %s", result)
        except Exception:
            logger.exception("background consolidation failed")

    def _run_decay(self) -> None:
        try:
            logger.info("background decay sweep started")
            result = self._decay_fn()
            with self._ts_lock:
                self._last_decay_at = _now()
            logger.info("background decay done: %s", result)
        except Exception:
            logger.exception("background decay sweep failed")


def _now() -> datetime:
    return datetime.now(UTC)


__all__ = ["DecayTimer", "BackgroundWorker"]
