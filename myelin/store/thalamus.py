"""Thalamus — working memory relay and attention buffer.

Neuroscience basis: The thalamus is the brain's central relay station.
Almost all sensory input passes through it before reaching the cortex.
Key thalamic nuclei relevant to memory:

- **Mediodorsal nucleus** → connects to prefrontal cortex, supports
  *working memory* — the small set of currently active information.
- **Anterior nucleus** → connects to hippocampus, supports
  *spatial/contextual memory* and episodic retrieval.
- **Pulvinar** → sustained attention, maintains focus on important stimuli.

In Myelin, the Thalamus maintains two memory tiers:

- **Pinned memories** (L0/L1): Critical facts that should always be
  available — identity context, key preferences, active projects.
  Analogous to the dorsomedial thalamus keeping PFC representations
  active via sustained thalamocortical loops.

- **Recency buffer** (L2): Recently accessed memories with rapid
  decay. Analogous to sensory thalamic relay — recent activations
  echo for a short period before fading. This captures the
  "working set" of currently relevant memories.

MemPalace equivalence:
    L0 (Identity)         → Thalamic pinned (priority=0, always loaded)
    L1 (Critical facts)   → Thalamic pinned (priority=1, always loaded)
    L2 (Room recall)      → Thalamic recency buffer

References:
    Saalmann, Y. B. (2014). Intralaminar and medial thalamic influence
        on cortical synchrony, information transmission, and cognition.
        Frontiers in Systems Neuroscience, 8.
    Mitchell, A. S. & Chakraborty, S. (2013). What does the mediodorsal
        thalamus do? Frontiers in Systems Neuroscience, 7.
    Aggleton, J. P. et al. (2010). Thalamic pathways for memory,
        space, and time. Annals of the NY Academy of Sciences, 1199.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlite_utils

from ..config import MyelinSettings, settings


class ThalamicBuffer:
    """SQLite-backed working memory buffer with pinned and recency tiers."""

    def __init__(
        self,
        db_path: Path | None = None,
        cfg: MyelinSettings | None = None,
    ) -> None:
        self._settings = cfg or settings
        path = db_path or (self._settings.data_dir / "thalamus.db")
        self.db = sqlite_utils.Database(str(path))
        self._ensure_tables()
        self._recency: OrderedDict[str, float] = OrderedDict()

    def close(self) -> None:
        self.db.conn.commit()
        self.db.conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS pinned (
                memory_id TEXT PRIMARY KEY,
                priority INTEGER NOT NULL DEFAULT 1,
                label TEXT,
                pinned_at TEXT NOT NULL
            )"""
        )

    # ------------------------------------------------------------------
    # Pinned memories (L0/L1 — persistent)
    # ------------------------------------------------------------------

    def pin(
        self,
        memory_id: str,
        priority: int = 1,
        label: str | None = None,
    ) -> None:
        """Pin a memory for always-available retrieval.

        Priority 0 = identity/system context (L0).
        Priority 1 = critical facts (L1).
        """
        now = datetime.now(UTC).isoformat()
        self.db.execute(
            """INSERT INTO pinned (memory_id, priority, label, pinned_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(memory_id)
               DO UPDATE SET priority = ?, label = ?""",
            [memory_id, priority, label, now, priority, label],
        )

    def unpin(self, memory_id: str) -> bool:
        """Remove a pinned memory. Returns True if it existed."""
        cursor = self.db.execute("DELETE FROM pinned WHERE memory_id = ?", [memory_id])
        return bool(cursor.rowcount > 0)

    def get_pinned(self, max_priority: int | None = None) -> list[dict[str, Any]]:
        """Get all pinned memory IDs, ordered by priority (ascending).

        If max_priority is set, only return pins with priority <= max_priority.
        """
        if max_priority is not None:
            rows = self.db.execute(
                "SELECT memory_id, priority, label FROM pinned "
                "WHERE priority <= ? ORDER BY priority",
                [max_priority],
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT memory_id, priority, label FROM pinned ORDER BY priority"
            ).fetchall()
        return [{"memory_id": r[0], "priority": r[1], "label": r[2]} for r in rows]

    def pinned_count(self) -> int:
        row = self.db.execute("SELECT COUNT(*) FROM pinned").fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Recency buffer (L2 — in-memory, volatile)
    # ------------------------------------------------------------------

    def touch(self, memory_ids: list[str]) -> None:
        """Update recency buffer with recently accessed memories."""
        now = datetime.now(UTC).timestamp()
        for mid in memory_ids:
            # Move to end (most recent)
            self._recency.pop(mid, None)
            self._recency[mid] = now

        # Trim to limit
        limit = self._settings.thalamus_recency_limit
        while len(self._recency) > limit:
            self._recency.popitem(last=False)

    def get_recent(self, n: int | None = None) -> list[str]:
        """Get recently accessed memory IDs (most recent first)."""
        limit = n or self._settings.thalamus_recency_limit
        items = list(self._recency.keys())
        items.reverse()
        return items[:limit]

    def dominant_region(self, metadata_lookup: dict[str, str]) -> str | None:
        """Infer the dominant ec_region from recent buffer entries.

        Takes a dict mapping memory_id → ec_region for the current
        recency buffer contents. Returns the region that accounts for
        ≥ 60% of recent entries, or None if no clear signal.

        This models the thalamic reticular nucleus (TRN): when recent
        activity clusters in one cortical territory, the TRN suppresses
        unrelated territories to sharpen attentional focus.
        """
        if not self._recency:
            return None

        region_counts: dict[str, int] = {}
        total = 0
        for mid in self._recency:
            region = metadata_lookup.get(mid)
            if region:
                region_counts[region] = region_counts.get(region, 0) + 1
                total += 1

        if total < 3:
            return None

        # Find majority region (≥ 60%)
        for region, count in sorted(
            region_counts.items(), key=lambda x: x[1], reverse=True
        ):
            if count / total >= 0.6:
                return region
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self, valid_ids: set[str]) -> int:
        """Remove pinned entries referencing deleted memories."""
        stale = [
            row[0]
            for row in self.db.execute("SELECT memory_id FROM pinned").fetchall()
            if row[0] not in valid_ids
        ]
        for mid in stale:
            self.db.execute("DELETE FROM pinned WHERE memory_id = ?", [mid])

        # Also clean recency buffer
        for mid in list(self._recency):
            if mid not in valid_ids:
                del self._recency[mid]

        return len(stale)
