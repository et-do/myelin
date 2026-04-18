"""Background decay timer — periodic auto-decay sweep.

Runs a caller-supplied function on a daemon thread at a fixed interval.
Uses threading.Event for clean shutdown without waiting the full interval.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class DecayTimer:
    """Calls *fn* repeatedly on a background thread every *interval_hours* hours.

    - ``interval_hours <= 0`` disables the timer entirely; ``start()`` is a no-op.
    - Exceptions raised by *fn* are logged and swallowed so the loop continues.
    - ``stop()`` signals the thread and waits up to 2 s for a clean shutdown.
    """

    def __init__(self, fn: Callable[[], None], interval_hours: float) -> None:
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
        # Wait for the interval (or the stop signal, whichever comes first).
        # This means the first sweep fires *after* one full interval, not
        # immediately on startup — intentional: let the server settle first.
        while not self._stop.wait(timeout=self._interval):
            try:
                self._fn()
            except Exception:
                logger.exception("decay timer: error during sweep")
