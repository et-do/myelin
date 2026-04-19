"""Tests for the MCP server logic (do_* functions, not MCP transport)."""

from __future__ import annotations

import pytest

from myelin.config import MyelinSettings
from myelin.server import (
    configure,
    do_consolidate,
    do_decay_sweep,
    do_forget,
    do_recall,
    do_status,
    do_store,
)


@pytest.fixture(autouse=True)
def _isolate_server(tmp_settings: MyelinSettings) -> None:
    """Point the server singletons at a temp directory for each test."""
    configure(tmp_settings)


class TestDoStore:
    def test_stores_valid_content(self) -> None:
        result = do_store("The auth service uses JWT tokens")
        assert result["status"] == "stored"
        assert result["id"] is not None
        assert result["reason"] is None

    def test_rejects_short_content(self) -> None:
        result = do_store("hi")
        assert result["status"] == "rejected"
        assert result["id"] is None
        assert result["reason"] is not None

    def test_rejects_oversized_content(self) -> None:
        result = do_store("x" * 500_001)
        assert result["status"] == "rejected"
        assert result["id"] is None
        assert "limit" in result["reason"]

    def test_response_always_has_stable_keys(self) -> None:
        """Every store response has id, status, and reason regardless of outcome."""
        for content in ["hi", "The auth service uses JWT tokens for sessions"]:
            r = do_store(content)
            assert "id" in r
            assert "status" in r
            assert "reason" in r

    def test_stores_with_metadata(self) -> None:
        result = do_store(
            "Uses PostgreSQL for persistence",
            project="backend",
            language="python",
            scope="database",
            tags="db,postgres",
            source="copilot",
        )
        assert result["status"] == "stored"

    def test_new_memory_access_count_is_one(self) -> None:
        """Stored memories start with access_count=1 (the store itself counts)."""
        from myelin.models import Memory

        m = Memory(content="test")
        assert m.access_count == 1


class TestDoRecall:
    def test_recalls_stored_memory(self) -> None:
        do_store("The auth service uses JWT tokens for sessions")
        results = do_recall("authentication")
        assert len(results) >= 1
        assert "JWT" in results[0]["content"]

    def test_recalls_empty_returns_empty(self) -> None:
        results = do_recall("anything at all")
        assert results == []

    def test_oversized_query_returns_empty(self) -> None:
        results = do_recall("x" * 10_001)
        assert results == []

    def test_recall_includes_score(self) -> None:
        do_store("The CI pipeline uses GitHub Actions with pytest")
        results = do_recall("continuous integration")
        assert len(results) >= 1
        assert "score" in results[0]
        assert results[0]["score"] > 0

    def test_recall_filters_by_project(self) -> None:
        do_store("Auth uses JWT tokens for sessions", project="api")
        do_store("Auth uses OAuth tokens for sessions", project="web")
        results = do_recall("authentication", project="api")
        assert len(results) == 1
        assert results[0]["project"] == "api"


class TestDoForget:
    def test_forgets_stored_memory(self) -> None:
        store_result = do_store("Something to forget eventually here")
        forget_result = do_forget(store_result["id"])
        assert forget_result["status"] == "forgotten"

        # Verify it's gone
        status = do_status()
        assert status["memory_count"] == 0


class TestDoDecaySweep:
    def test_sweep_on_empty_store(self) -> None:
        result = do_decay_sweep()
        assert result["pruned"] == 0
        assert result["remaining"] == 0

    def test_sweep_keeps_recent_memories(self) -> None:
        do_store("Fresh memory that was just stored today")
        result = do_decay_sweep()
        assert result["pruned"] == 0
        assert result["remaining"] == 1


class TestDoStatus:
    def test_returns_status_fields(self) -> None:
        result = do_status()
        assert "memory_count" in result
        assert "entity_count" in result
        assert "relationship_count" in result
        assert "data_dir" in result
        assert "embedding_model" in result
        assert result["memory_count"] == 0


class TestDoConsolidate:
    def test_consolidate_empty(self) -> None:
        result = do_consolidate()
        assert result["memories_replayed"] == 0
        assert result["entities_found"] == 0
        assert result["relationships_created"] == 0

    def test_consolidate_extracts_entities(self) -> None:
        do_store("Kai Tanaka reviewed the OAuth migration for Project Alpha")
        do_store("Project Alpha uses JWT tokens designed by Kai Tanaka")
        result = do_consolidate()
        assert result["memories_replayed"] >= 1
        assert result["entities_found"] >= 1


class TestEndToEnd:
    def test_store_recall_forget_cycle(self) -> None:
        # Store
        s1 = do_store("PostgreSQL is the primary database")
        s2 = do_store("Redis is used for caching sessions")
        assert s1["status"] == "stored"
        assert s2["status"] == "stored"

        # Recall
        results = do_recall("database")
        assert len(results) >= 1
        contents = [r["content"] for r in results]
        assert any("PostgreSQL" in c for c in contents)

        # Forget one
        do_forget(s1["id"])
        assert do_status()["memory_count"] == 1

        # Remaining recall should find Redis, not PostgreSQL
        results = do_recall("caching")
        assert len(results) == 1
        assert "Redis" in results[0]["content"]


class TestAutoConsolidation:
    def test_triggers_after_interval_inline(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """When the worker is not running, consolidation falls back to inline."""
        cfg = MyelinSettings(data_dir=tmp_path / ".myelin", consolidation_interval=3)  # type: ignore[arg-type]
        configure(cfg)
        # Worker is not started — inline path is taken.

        # Store 2 — no consolidation yet
        r1 = do_store("Kai Tanaka works on Project Alpha backend")
        assert "consolidation" not in r1
        r2 = do_store("Project Alpha uses JWT tokens for authentication")
        assert "consolidation" not in r2

        # Store 3 — should trigger inline consolidation (worker not running)
        r3 = do_store("Kai Tanaka deployed the OAuth service for Project Alpha")
        assert "consolidation" in r3
        assert r3["consolidation"]["memories_replayed"] >= 1

    def test_disabled_when_interval_zero(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Setting consolidation_interval=0 disables auto-consolidation."""
        cfg = MyelinSettings(data_dir=tmp_path / ".myelin", consolidation_interval=0)  # type: ignore[arg-type]
        configure(cfg)

        for i in range(5):
            result = do_store(f"Memory number {i} about some topic here")
            assert "consolidation" not in result
