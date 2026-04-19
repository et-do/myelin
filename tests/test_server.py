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

    def test_relations_writes_edges_to_neocortex(self) -> None:
        """Explicit relations are stored as neocortex edges immediately."""
        from myelin.server import _get_neocortex

        result = do_store(
            "AuthService depends on JWTHelper for token validation",
            relations='[["AuthService", "depends_on", "JWTHelper"]]',
        )
        assert result["status"] == "stored"
        assert result.get("relationships") == 1
        net = _get_neocortex()
        rels = net.get_relationships("AuthService", predicate="depends_on")
        objects = [r["object"] for r in rels]
        assert "jwthelper" in objects

    def test_relations_multiple_triples(self) -> None:
        """Multiple triples are all written."""
        result = do_store(
            "The API gateway routes to AuthService and PaymentService",
            relations=(
                '[["APIGateway","routes_to","AuthService"],'
                '["APIGateway","routes_to","PaymentService"]]'
            ),
        )
        assert result["status"] == "stored"
        assert result.get("relationships") == 2

    def test_relations_invalid_json_is_ignored(self) -> None:
        """Malformed JSON in relations never blocks the store."""
        result = do_store(
            "The cache layer wraps Redis for session storage",
            relations="not valid json at all",
        )
        assert result["status"] == "stored"
        assert result.get("relationships") is None

    def test_relations_not_a_list_is_ignored(self) -> None:
        result = do_store(
            "The cache layer uses Redis for session data persistence",
            relations='{"subject": "A", "predicate": "rel", "object": "B"}',
        )
        assert result["status"] == "stored"
        assert result.get("relationships") is None

    def test_relations_malformed_triple_skipped(self) -> None:
        """Triples with wrong length or non-string values are silently skipped."""
        result = do_store(
            "Scheduler dispatches tasks using Celery for async processing",
            relations='[["OnlyTwo", "items"], ["ValidA", "uses", "ValidB"], [1, 2, 3]]',
        )
        assert result["status"] == "stored"
        # Only the valid triple ("ValidA","uses","ValidB") should be counted
        assert result.get("relationships") == 1

    def test_relations_empty_string_field_skipped(self) -> None:
        """Triples containing empty strings are silently skipped."""
        result = do_store(
            "Worker processes jobs from the task queue using async handlers",
            relations='[["", "uses", "Redis"], ["Worker", "uses", ""]]',
        )
        assert result["status"] == "stored"
        assert result.get("relationships") is None

    def test_relations_on_rejected_store_not_written(self) -> None:
        """Relations must not be written if the store itself is rejected."""
        from myelin.server import _get_neocortex

        do_store("hi", relations='[["A", "rel", "B"]]')
        net = _get_neocortex()
        rels = net.get_relationships("A")
        assert rels == []

    def test_store_without_relations_has_no_relationships_key(self) -> None:
        result = do_store("The search service indexes documents using Elasticsearch")
        assert "relationships" not in result


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


class TestDoPinUnpin:
    def test_pin_returns_pinned_status(self) -> None:
        from myelin.server import do_pin

        result = do_pin("mem-123", priority=2, label="important")
        assert result["status"] == "pinned"
        assert result["memory_id"] == "mem-123"
        assert result["priority"] == 2

    def test_unpin_existing_memory(self) -> None:
        from myelin.server import do_pin, do_unpin

        do_pin("mem-456")
        result = do_unpin("mem-456")
        assert result["status"] == "unpinned"

    def test_unpin_nonexistent_returns_not_found(self) -> None:
        from myelin.server import do_unpin

        result = do_unpin("does-not-exist")
        assert result["status"] == "not_found"

    def test_pinned_memories_injected_into_recall(self) -> None:
        from myelin.server import do_pin, do_recall

        r = do_store("The load balancer uses nginx with round-robin policy")
        memory_id = r["id"]
        do_pin(memory_id, priority=1)

        # Recall with unrelated query — pinned memory should still appear
        results = do_recall("completely unrelated query about cooking recipes")
        ids = [res["id"] for res in results]
        assert memory_id in ids


class TestDoDecaySweepWithStale:
    def test_sweep_prunes_stale_memories(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Memories with ancient last_accessed dates are pruned."""
        from unittest.mock import patch

        from myelin.server import do_decay_sweep

        do_store("Ancient ephemeral fact that should be pruned by decay")
        memory_id = do_recall("ephemeral fact")[0]["id"]

        # Make find_stale return it as stale
        with patch("myelin.server.find_stale", return_value=[memory_id]):
            result = do_decay_sweep()

        assert result["pruned"] == 1
        assert result["remaining"] == 0


class TestDoStoreChunksAndReplace:
    def test_store_long_content_reports_chunks(self) -> None:
        """Multi-chunk content returns chunks count in result."""
        # Use paragraph-split content so chunk_text generates multiple segments
        para1 = "The authentication service processes login requests. " * 10
        para2 = "JWT tokens are validated using RS256 asymmetric keys. " * 10
        long_content = para1 + "\n\n" + para2
        result = do_store(long_content)
        assert result["status"] == "stored"
        assert "chunks" in result
        assert result["chunks"] > 1

    def test_store_overwrite_returns_replaced(self) -> None:
        """store with overwrite=True on near-duplicate returns replaced key."""
        original = "The payment service uses Stripe for card processing"
        updated = "The payment service uses Stripe for all payment processing"
        do_store(original)
        result = do_store(updated, overwrite=True)
        assert result["status"] in ("updated", "stored")
        if result["status"] == "updated":
            assert "replaced" in result


class TestShutdown:
    def test_shutdown_clears_singletons(self) -> None:
        """shutdown() closes and resets all server singletons."""
        import myelin.server as srv

        # Trigger initialization
        do_store("Initialize the server singletons by storing something here")
        assert srv._hippocampus is not None

        srv.shutdown()
        assert srv._hippocampus is None
        assert srv._hebbian is None
        assert srv._neocortex is None
        assert srv._thalamus is None

    def test_shutdown_idempotent(self) -> None:
        """Calling shutdown twice does not raise."""
        import myelin.server as srv

        srv.shutdown()
        srv.shutdown()  # should not raise


class TestWarmUp:
    def test_warm_up_runs_without_error(self) -> None:
        """warm_up() pre-warms models; must not raise."""
        warm_up()  # covers server.py line 112


class TestSignalHandler:
    def test_signal_handler_raises_system_exit(self) -> None:
        """_signal_handler raises SystemExit on any signal."""
        import pytest as _pytest

        with _pytest.raises(SystemExit):
            _signal_handler(15, None)  # covers server.py line 135


class TestAgentNamespace:
    """Verify agent_id namespace isolation across store and recall."""

    def test_agent_scoped_memory_not_returned_for_other_agent(self) -> None:
        """Memories stored under agent-a must not appear for agent-b."""
        do_store(
            "The OIDC service handles all SSO login flows for agent-a",
            agent_id="agent-a",
        )
        results = do_recall("OIDC single sign-on", agent_id="agent-b")
        ids = [r["id"] for r in results]
        agent_a_result = do_recall("OIDC single sign-on", agent_id="agent-a")
        agent_a_ids = [r["id"] for r in agent_a_result]
        # agent-a's memory must not be visible to agent-b
        for aid in agent_a_ids:
            assert aid not in ids

    def test_agent_scoped_memory_returned_for_same_agent(self) -> None:
        """Memories stored with agent_id are returned for the same agent_id."""
        result = do_store(
            "The rate limiter uses a sliding-window token bucket algorithm",
            agent_id="copilot-bot",
        )
        stored_id = result["id"]
        results = do_recall("rate limiting sliding window", agent_id="copilot-bot")
        assert any(r["id"] == stored_id for r in results)

    def test_empty_agent_id_stores_in_global_namespace(self) -> None:
        """Memories with no agent_id are accessible without an agent filter."""
        result = do_store("Global shared memory about the API gateway design")
        stored_id = result["id"]
        results = do_recall("API gateway design")
        assert any(r["id"] == stored_id for r in results)

    def test_agent_id_stored_in_metadata(self) -> None:
        """agent_id is persisted in ChromaDB and round-trips correctly."""
        from myelin.server import _get_hippocampus

        do_store(
            "The feature flag service uses LaunchDarkly for all products",
            agent_id="scout-bot",
        )
        hc = _get_hippocampus()
        all_meta = hc.get_all_metadata()
        scout_mems = [m for m in all_meta if m.get("agent_id") == "scout-bot"]
        assert len(scout_mems) == 1

    def test_no_agent_id_filter_returns_global_memories(self) -> None:
        """Recalling without agent_id should return memories with no agent_id set."""
        result = do_store(
            "The deployment pipeline uses ArgoCD for GitOps continuous delivery"
        )
        stored_id = result["id"]
        results = do_recall("deployment pipeline ArgoCD GitOps")
        assert any(r["id"] == stored_id for r in results)
