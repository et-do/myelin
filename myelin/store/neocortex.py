"""Neocortex — semantic memory network with spreading activation.

Neuroscience basis: The temporal neocortex maintains a distributed
semantic network — concepts as nodes, associations as weighted edges
(Collins & Loftus, 1975). Activating a concept spreads activation to
connected nodes, priming related memories for retrieval.

Unlike MemPalace's static knowledge graph (explicit triples at ingest),
the neocortex learns associations from *co-occurrence* across stored
memories. Entities that appear in the same memory form edges.
Repeated co-occurrence strengthens the edge (Hebbian learning).

The semantic network supports:
- Entity registration with temporal validity (valid_from / valid_to)
- Co-occurrence edges with learned weights
- Spreading activation: given seed entities, find related concepts
  to expand or filter hippocampal recall

References:
    Collins, A. M. & Loftus, E. F. (1975). A spreading-activation
        theory of semantic processing. Psychological Review, 82(6).
    McClelland, J. L. & Rogers, T. T. (2003). The parallel distributed
        processing approach to semantic cognition. Nature Reviews
        Neuroscience, 4.
    Patterson, K. et al. (2007). Where do you know what you know?
        The representation of semantic knowledge in the human brain.
        Nature Reviews Neuroscience, 8.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlite_utils

from ..config import MyelinSettings, settings

logger = logging.getLogger(__name__)


class SemanticNetwork:
    """SQLite-backed semantic network with spreading activation."""

    def __init__(
        self,
        db_path: Path | None = None,
        cfg: MyelinSettings | None = None,
    ) -> None:
        self._settings = cfg or settings
        path = db_path or (self._settings.data_dir / "neocortex.db")
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5.0)
        self.db = sqlite_utils.Database(conn)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.RLock()
        self._ensure_tables()

    def close(self) -> None:
        """Commit pending changes and close the underlying SQLite connection."""
        self.db.conn.commit()
        self.db.conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL DEFAULT 'concept',
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL
            )"""
        )
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS relationships (
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL DEFAULT 'co_occurs',
                object TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                valid_from TEXT,
                valid_to TEXT,
                PRIMARY KEY (subject, predicate, object)
            )"""
        )
        self.db.execute(
            """CREATE INDEX IF NOT EXISTS idx_rel_subject
               ON relationships(subject)"""
        )
        self.db.execute(
            """CREATE INDEX IF NOT EXISTS idx_rel_object
               ON relationships(object)"""
        )

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        entity_type: str = "concept",
    ) -> None:
        """Register an entity (idempotent — updates last_seen on conflict)."""
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self.db.execute(
                """INSERT INTO entities (name, entity_type, first_seen, last_seen)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET last_seen = ?""",
                [name.lower(), entity_type, now, now, now],
            )

    def get_entity(self, name: str) -> dict[str, Any] | None:
        """Look up an entity by name."""
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM entities WHERE name = ?", [name.lower()]
            ).fetchone()
        if row is None:
            return None
        return {
            "name": row[0],
            "entity_type": row[1],
            "first_seen": row[2],
            "last_seen": row[3],
        }

    def entity_count(self) -> int:
        with self._lock:
            row = self.db.execute("SELECT COUNT(*) FROM entities").fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    def add_relationship(
        self,
        subject: str,
        object_: str,
        predicate: str = "co_occurs",
        weight: float = 1.0,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> None:
        """Add or strengthen a relationship edge.

        On conflict (same subject-predicate-object), the weight is
        incremented — Hebbian "fire together, wire together".
        """
        with self._lock:
            # Ensure both entities exist
            self.add_entity(subject)
            self.add_entity(object_)

            s, o = subject.lower(), object_.lower()
            self.db.execute(
                """INSERT INTO relationships
                       (subject, predicate, object, weight, valid_from, valid_to)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(subject, predicate, object)
                   DO UPDATE SET weight = weight + ?""",
                [s, predicate, o, weight, valid_from, valid_to, weight],
            )

    def get_relationships(
        self,
        entity: str,
        predicate: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all relationships involving an entity (as subject or object)."""
        with self._lock:
            return self._get_relationships_unlocked(entity, predicate)

    def _get_relationships_unlocked(
        self,
        entity: str,
        predicate: str | None = None,
    ) -> list[dict[str, Any]]:
        """Internal: fetch relationships without acquiring the lock."""
        name = entity.lower()
        if predicate:
            rows = self.db.execute(
                """SELECT subject, predicate, object, weight, valid_from, valid_to
                   FROM relationships
                   WHERE (subject = ? OR object = ?) AND predicate = ?""",
                [name, name, predicate],
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT subject, predicate, object, weight, valid_from, valid_to
                   FROM relationships
                   WHERE subject = ? OR object = ?""",
                [name, name],
            ).fetchall()

        return [
            {
                "subject": r[0],
                "predicate": r[1],
                "object": r[2],
                "weight": r[3],
                "valid_from": r[4],
                "valid_to": r[5],
            }
            for r in rows
        ]

    def relationship_count(self) -> int:
        with self._lock:
            row = self.db.execute("SELECT COUNT(*) FROM relationships").fetchone()
            return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Spreading activation
    # ------------------------------------------------------------------

    def spread(
        self,
        seeds: list[str],
        *,
        max_depth: int = 2,
        min_weight: float = 0.5,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Spread activation from seed entities through the network.

        Returns up to top_k related entities sorted by accumulated
        activation, filtered by min_weight. Activation decays by 0.5x
        at each hop (distance-dependent decay).

        This models the brain's spreading activation: activating "auth"
        primes "JWT", "OAuth", "security" — concepts that co-occurred
        with auth across stored memories.
        """
        with self._lock:
            activation: dict[str, float] = {}
            visited: set[str] = set()
            frontier = [(s.lower(), 1.0) for s in seeds]

            for _depth in range(max_depth):
                next_frontier: list[tuple[str, float]] = []
                for entity, energy in frontier:
                    if entity in visited:
                        continue
                    visited.add(entity)
                    activation[entity] = activation.get(entity, 0.0) + energy

                    # Spread to neighbors
                    rels = self._get_relationships_unlocked(entity)
                    for rel in rels:
                        neighbor = (
                            rel["object"]
                            if rel["subject"] == entity
                            else rel["subject"]
                        )
                        edge_weight = rel["weight"]
                        propagated = energy * 0.5 * min(edge_weight, 3.0) / 3.0
                        if propagated >= 0.01:
                            next_frontier.append((neighbor, propagated))

                frontier = next_frontier

            # Remove seeds from results (we already know about them)
            seed_set = {s.lower() for s in seeds}
            results = [
                (entity, score)
                for entity, score in activation.items()
                if entity not in seed_set and score >= min_weight
            ]
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    # ------------------------------------------------------------------
    # Invalidation (temporal validity)
    # ------------------------------------------------------------------

    def invalidate(
        self,
        subject: str,
        predicate: str,
        object_: str,
        ended: str | None = None,
    ) -> bool:
        """Mark a relationship as ended (set valid_to).

        Returns True if the relationship existed and was updated.
        """
        end_date = ended or datetime.now(UTC).isoformat()
        s, o = subject.lower(), object_.lower()
        with self._lock:
            cursor = self.db.execute(
                """UPDATE relationships SET valid_to = ?
                   WHERE subject = ? AND predicate = ? AND object = ?
                   AND valid_to IS NULL""",
                [end_date, s, predicate, o],
            )
        return bool(cursor.rowcount > 0)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all entities and relationships."""
        with self._lock:
            self.db.execute("DELETE FROM relationships")
            self.db.execute("DELETE FROM entities")
