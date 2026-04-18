"""Hebbian reinforcement — memories that fire together wire together.

Tracks co-retrieval patterns between memories and uses that signal to
boost future recall scores. Lightweight SQLite storage.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
from itertools import combinations
from pathlib import Path

import sqlite_utils

from ..config import MyelinSettings, settings
from ..models import RecallResult

logger = logging.getLogger(__name__)


class HebbianTracker:
    """Tracks co-access patterns between memories."""

    def __init__(
        self,
        db_path: Path | None = None,
        cfg: MyelinSettings | None = None,
    ) -> None:
        self._settings = cfg or settings
        path = db_path or (self._settings.data_dir / "hebbian.db")
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5.0)
        self.db = sqlite_utils.Database(conn)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.RLock()
        self._ensure_table()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self.db.conn.close()

    def _ensure_table(self) -> None:
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS co_access (
                id_a TEXT NOT NULL,
                id_b TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (id_a, id_b)
            )"""
        )

    def reinforce(self, memory_ids: list[str]) -> None:
        """Strengthen connections between co-retrieved memories."""
        delta = self._settings.hebbian_delta
        with self._lock:
            for a, b in combinations(sorted(memory_ids), 2):
                self.db.execute(
                    """INSERT INTO co_access (id_a, id_b, weight) VALUES (?, ?, ?)
                       ON CONFLICT(id_a, id_b) DO UPDATE SET weight = weight + ?""",
                    [a, b, delta, delta],
                )

    def boost(self, results: list[RecallResult]) -> list[RecallResult]:
        """Re-rank results using co-access history (multiplicative log boost).

        Uses logarithmic scaling to model LTP saturation: the first co-accesses
        give the biggest boost, with diminishing returns as weight accumulates.
        This prevents Hebbian reinforcement from overriding semantic relevance.

        Formula: score *= (1 + hebbian_scale * log(1 + total_co_weight))
        """
        if len(results) < 2:
            return results

        ids = [r.memory.id for r in results]
        pairs = list(combinations(sorted(set(ids)), 2))
        if not pairs:
            return results

        with self._lock:
            # Batch-fetch co-access weights in a single query
            placeholders = " OR ".join(["(id_a = ? AND id_b = ?)"] * len(pairs))
            params: list[str] = [v for pair in pairs for v in pair]
            weights: dict[tuple[str, str], float] = {}
            for row in self.db.execute(
                f"SELECT id_a, id_b, weight FROM co_access WHERE {placeholders}",
                params,
            ).fetchall():
                weights[(row[0], row[1])] = row[2]

        # Multiplicative logarithmic boost (LTP saturation model)
        scale = self._settings.hebbian_scale
        for result in results:
            total_weight = 0.0
            for other_id in ids:
                if other_id == result.memory.id:
                    continue
                a, b = min(result.memory.id, other_id), max(result.memory.id, other_id)
                total_weight += weights.get((a, b), 0.0)
            if total_weight > 0:
                result.score *= 1.0 + scale * math.log1p(total_weight)

        return sorted(results, key=lambda r: r.score, reverse=True)

    def cleanup(self, valid_ids: set[str]) -> int:
        """Remove co-access entries referencing deleted memories."""
        with self._lock:
            stale = [
                (row[0], row[1])
                for row in self.db.execute(
                    "SELECT id_a, id_b FROM co_access"
                ).fetchall()
                if row[0] not in valid_ids or row[1] not in valid_ids
            ]
            for a, b in stale:
                self.db.execute(
                    "DELETE FROM co_access WHERE id_a = ? AND id_b = ?",
                    [a, b],
                )
            return len(stale)

    def lookup_weights(self, ids: list[str]) -> dict[str, float]:
        """Return the total Hebbian co-weight for each ID in *ids*.

        For each memory, sums the co_access weights with every other
        memory in the set.  Used by debug-recall to show how much each
        result benefits from reinforcement before recall re-ranks it.
        """
        if len(ids) < 2:
            return {mid: 0.0 for mid in ids}
        pairs = list(combinations(sorted(set(ids)), 2))
        with self._lock:
            placeholders = " OR ".join(["(id_a = ? AND id_b = ?)"] * len(pairs))
            params: list[str] = [v for pair in pairs for v in pair]
            weights: dict[tuple[str, str], float] = {}
            for row in self.db.execute(
                f"SELECT id_a, id_b, weight FROM co_access WHERE {placeholders}",
                params,
            ).fetchall():
                weights[(row[0], row[1])] = row[2]

        result: dict[str, float] = {}
        for mid in ids:
            total = 0.0
            for other in ids:
                if other == mid:
                    continue
                a, b = min(mid, other), max(mid, other)
                total += weights.get((a, b), 0.0)
            result[mid] = round(total, 4)
        return result
