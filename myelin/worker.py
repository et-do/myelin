"""Background worker — async consolidation and scheduled decay.

Moves memory maintenance off the hot path so store/recall tool calls
are never blocked by consolidation or decay work.

Architecture
------------
A single daemon thread processes work in two ways:

1. **Queue-driven** — consolidation requests are submitted via
   ``submit_consolidate()`` (called by ``do_store`` every N stores).
   The queue is bounded; if full, the task is silently dropped.
   Consolidation is idempotent so drops are always safe.

2. **Timer-driven** — if ``decay_interval_hours > 0``, a periodic decay
   sweep fires on that cadence.  The worker checks whether decay is due
   after each queue wait timeout, so it never sleeps longer than one
   second while still respecting the interval.

Lifecycle
---------
- ``start()`` — idempotent, starts the daemon thread.
- ``stop(timeout)`` — signals STOP, joins thread.
- Integrated with server lifespan: started during MCP server startup,
  stopped during graceful shutdown.
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
        self._ts_lock = threading.Lock()  # protects last_* timestamps

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
        # Drain the queue and insert STOP so it is processed promptly.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(_Task.STOP)
        except queue.Full:
            pass  # thread will exit on next timeout tick
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("background worker stopped")

    def submit_consolidate(self) -> bool:
        """Queue a consolidation run.

        Non-blocking.  Returns True if queued, False if queue was full
        (task dropped — consolidation is idempotent so this is safe).
        """
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
        """Return worker health info for inclusion in ``do_status()``."""
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
        """Worker loop — drain queue tasks and fire periodic decay."""
        next_decay = (
            _now().timestamp() + self._decay_interval_seconds
            if self._decay_interval_seconds > 0
            else None
        )

        while True:
            # Wait at most 1 second before checking if decay is due.
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

            # Fire periodic decay if the interval has elapsed.
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
