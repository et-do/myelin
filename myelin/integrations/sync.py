"""Sync registry — tracks export/import history to enable incremental operations.

Stored at ``data_dir/sync.db`` (alongside the other SQLite databases).

Two tables
----------
``export_log``
    Records which memories were exported to a destination, keyed by
    ``(integration, dest, memory_id)``.  The ``content_hash`` field allows
    re-exports to skip notes whose content has not changed.

``import_log``
    Records which files were imported from a source, keyed by
    ``(integration, src, file_path)``.  The ``file_hash`` field allows
    re-imports to skip files that have not changed on disk.

Usage
-----
::

    from myelin.integrations.sync import SyncRegistry
    from myelin.config import settings

    reg = SyncRegistry(settings.data_dir / "sync.db")

    # --- export side ---
    new_memories = reg.filter_for_export("obsidian", vault_path, all_memories)
    skip_ids = {m["id"] for m in all_memories} - {m["id"] for m in new_memories}
    count = exporter.export(all_memories, vault_path, skip_ids=skip_ids)
    reg.record_exports("obsidian", vault_path, new_memories)

    # --- import side ---
    all_files = list((vault_path / "Memories").rglob("*.md"))
    new_files = reg.filter_for_import("obsidian", vault_path, all_files)
    pairs = importer.import_(vault_path, only_files=frozenset(new_files))
    reg.record_imports("obsidian", vault_path, new_files)
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

_HASH_LEN = 16  # hex chars (= 64-bit prefix of SHA-256, sufficient for dedup)


def _hash_memory(memory: dict[str, Any]) -> str:
    """Return a short content hash for *memory*.

    Combines ``id`` (stable) and ``content`` (changes when memory is updated)
    so the hash changes if and only if the note would need to be rewritten.
    """
    raw = (str(memory.get("id", "")) + str(memory.get("content", ""))).encode()
    return hashlib.sha256(raw).hexdigest()[:_HASH_LEN]


def _hash_file(path: Path) -> str:
    """Return a short hash of the bytes of *path*."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:_HASH_LEN]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS export_log (
        integration  TEXT NOT NULL,
        dest         TEXT NOT NULL,
        memory_id    TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        exported_at  TEXT NOT NULL,
        PRIMARY KEY (integration, dest, memory_id)
    );
    CREATE TABLE IF NOT EXISTS import_log (
        integration  TEXT NOT NULL,
        src          TEXT NOT NULL,
        file_path    TEXT NOT NULL,
        file_hash    TEXT NOT NULL,
        imported_at  TEXT NOT NULL,
        PRIMARY KEY (integration, src, file_path)
    );
"""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SyncRegistry:
    """SQLite-backed registry of export/import history for incremental operations.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  The parent directory is created if absent.
        Typically ``settings.data_dir / "sync.db"``.
    """

    def __init__(self, db_path: Path) -> None:
        self._db = str(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db) as con:
            con.executescript(_SCHEMA)

    # ------------------------------------------------------------------ export

    def filter_for_export(
        self,
        integration: str,
        dest: Path,
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return the subset of *memories* that need to be (re)written.

        A memory is included when its ``id`` has not been recorded for this
        ``(integration, dest)`` pair, or its content hash has changed since
        the last export.
        """
        known = self._exported_hashes(integration, dest)
        return [m for m in memories if known.get(str(m["id"])) != _hash_memory(m)]

    def record_exports(
        self,
        integration: str,
        dest: Path,
        memories: list[dict[str, Any]],
    ) -> None:
        """Upsert an export record for each memory in *memories*."""
        now = datetime.now(UTC).isoformat()
        dest_s = str(dest.resolve())
        with sqlite3.connect(self._db) as con:
            con.executemany(
                "INSERT OR REPLACE INTO export_log VALUES (?,?,?,?,?)",
                [
                    (integration, dest_s, str(m["id"]), _hash_memory(m), now)
                    for m in memories
                ],
            )

    def remove_export_records(
        self, integration: str, dest: Path, memory_ids: list[str]
    ) -> None:
        """Remove stale export records (e.g. after memories are deleted)."""
        dest_s = str(dest.resolve())
        with sqlite3.connect(self._db) as con:
            con.executemany(
                "DELETE FROM export_log WHERE integration=? AND dest=? AND memory_id=?",
                [(integration, dest_s, mid) for mid in memory_ids],
            )

    def export_summary(self, integration: str, dest: Path) -> dict[str, Any]:
        """Return ``{"total_tracked": int, "last_exported": str | None}``."""
        dest_s = str(dest.resolve())
        with sqlite3.connect(self._db) as con:
            row = con.execute(
                "SELECT COUNT(*), MAX(exported_at) FROM export_log "
                "WHERE integration=? AND dest=?",
                (integration, dest_s),
            ).fetchone()
        return {"total_tracked": row[0] or 0, "last_exported": row[1]}

    # ------------------------------------------------------------------ import

    def filter_for_import(
        self, integration: str, src: Path, files: list[Path]
    ) -> list[Path]:
        """Return files in *files* that are new or changed since last import.

        Each file is hashed; files whose hash matches the last recorded value
        for ``(integration, src, relative_path)`` are skipped.
        """
        known = self._imported_hashes(integration, src)
        src_r = src.resolve()
        result: list[Path] = []
        for f in files:
            rel = str(f.resolve().relative_to(src_r))
            if known.get(rel) != _hash_file(f):
                result.append(f)
        return result

    def record_imports(self, integration: str, src: Path, files: list[Path]) -> None:
        """Upsert an import record for each file in *files*."""
        now = datetime.now(UTC).isoformat()
        src_s = str(src.resolve())
        src_r = src.resolve()
        with sqlite3.connect(self._db) as con:
            con.executemany(
                "INSERT OR REPLACE INTO import_log VALUES (?,?,?,?,?)",
                [
                    (
                        integration,
                        src_s,
                        str(f.resolve().relative_to(src_r)),
                        _hash_file(f),
                        now,
                    )
                    for f in files
                ],
            )

    def import_summary(self, integration: str, src: Path) -> dict[str, Any]:
        """Return ``{"total_tracked": int, "last_imported": str | None}``."""
        src_s = str(src.resolve())
        with sqlite3.connect(self._db) as con:
            row = con.execute(
                "SELECT COUNT(*), MAX(imported_at) FROM import_log "
                "WHERE integration=? AND src=?",
                (integration, src_s),
            ).fetchone()
        return {"total_tracked": row[0] or 0, "last_imported": row[1]}

    # -------------------------------------------------------- item-based import
    # For non-file-based integrations (e.g. git commits, API items) where
    # "already imported" is determined by item identity (SHA / ID), not by
    # on-disk file hash.

    def filter_new_items(
        self, integration: str, src_key: str, item_ids: list[str]
    ) -> list[str]:
        """Return item IDs in *item_ids* that have not yet been imported.

        Uses the same ``import_log`` table as :meth:`filter_for_import` but
        keyed by an arbitrary string *src_key* (e.g. a repo URL or path) and
        string item IDs (e.g. git commit SHAs) rather than file paths.
        """
        known = self._imported_item_ids(integration, src_key)
        return [i for i in item_ids if i not in known]

    def record_items(self, integration: str, src_key: str, item_ids: list[str]) -> None:
        """Mark *item_ids* as imported for ``(integration, src_key)``."""
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(self._db) as con:
            con.executemany(
                "INSERT OR REPLACE INTO import_log VALUES (?,?,?,?,?)",
                # Store the item_id as both file_path and file_hash so the
                # "hash" never changes (identity-based dedup).
                [(integration, src_key, iid, iid, now) for iid in item_ids],
            )

    def item_summary(self, integration: str, src_key: str) -> dict[str, Any]:
        """Return ``{"total_tracked": int, "last_imported": str | None}``."""
        with sqlite3.connect(self._db) as con:
            row = con.execute(
                "SELECT COUNT(*), MAX(imported_at) FROM import_log "
                "WHERE integration=? AND src=?",
                (integration, src_key),
            ).fetchone()
        return {"total_tracked": row[0] or 0, "last_imported": row[1]}

    # ------------------------------------------------------------------ private

    def _exported_hashes(self, integration: str, dest: Path) -> dict[str, str]:
        dest_s = str(dest.resolve())
        with sqlite3.connect(self._db) as con:
            rows = con.execute(
                "SELECT memory_id, content_hash FROM export_log "
                "WHERE integration=? AND dest=?",
                (integration, dest_s),
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def _imported_hashes(self, integration: str, src: Path) -> dict[str, str]:
        src_s = str(src.resolve())
        with sqlite3.connect(self._db) as con:
            rows = con.execute(
                "SELECT file_path, file_hash FROM import_log "
                "WHERE integration=? AND src=?",
                (integration, src_s),
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def _imported_item_ids(self, integration: str, src_key: str) -> set[str]:
        with sqlite3.connect(self._db) as con:
            rows = con.execute(
                "SELECT file_path FROM import_log WHERE integration=? AND src=?",
                (integration, src_key),
            ).fetchall()
        return {row[0] for row in rows}
