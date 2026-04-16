"""Tests for the Hippocampus episodic memory store."""

from __future__ import annotations

from myelin.config import MyelinSettings
from myelin.models import MemoryMetadata
from myelin.store.hippocampus import Hippocampus


class TestStore:
    def test_stores_and_returns_memory(self, hippocampus: Hippocampus) -> None:
        memory = hippocampus.store("The auth service uses JWT tokens")
        assert memory is not None
        assert memory.content == "The auth service uses JWT tokens"
        assert memory.id
        assert hippocampus.count() == 1

    def test_rejects_short_content(self, hippocampus: Hippocampus) -> None:
        result = hippocampus.store("hi")
        assert result is None
        assert hippocampus.count() == 0

    def test_rejects_near_duplicate(self, hippocampus: Hippocampus) -> None:
        hippocampus.store("The auth service uses JWT tokens for sessions")
        dup = hippocampus.store("The auth service uses JWT tokens for sessions")
        assert dup is None
        assert hippocampus.count() == 1

    def test_stores_with_metadata(self, hippocampus: Hippocampus) -> None:
        meta = MemoryMetadata(project="myelin", language="python", scope="auth")
        memory = hippocampus.store("JWT refresh token logic", meta)
        assert memory is not None
        assert memory.metadata.project == "myelin"
        assert memory.metadata.language == "python"

    def test_stores_multiple_distinct(self, hippocampus: Hippocampus) -> None:
        hippocampus.store("The database uses PostgreSQL with pgvector")
        hippocampus.store("Frontend is built with React and TypeScript")
        hippocampus.store("CI pipeline runs on GitHub Actions")
        assert hippocampus.count() == 3


class TestRecall:
    def test_recalls_by_semantic_similarity(self, hippocampus: Hippocampus) -> None:
        hippocampus.store("The auth service uses JWT tokens for sessions")
        hippocampus.store("The database is PostgreSQL with pgvector")
        hippocampus.store("CI runs on GitHub Actions with ruff and pytest")

        results = hippocampus.recall("authentication and tokens")
        assert len(results) > 0
        assert "JWT" in results[0].memory.content

    def test_recalls_empty_store(self, hippocampus: Hippocampus) -> None:
        results = hippocampus.recall("anything")
        assert results == []

    def test_respects_n_results(self, hippocampus: Hippocampus) -> None:
        for i in range(5):
            hippocampus.store(f"Distinct memory number {i} about topic {i * 7}")
        results = hippocampus.recall("memory", n_results=2)
        assert len(results) == 2

    def test_filters_by_project(self, hippocampus: Hippocampus) -> None:
        hippocampus.store(
            "Auth uses JWT tokens",
            MemoryMetadata(project="backend"),
        )
        hippocampus.store(
            "Auth uses OAuth tokens",
            MemoryMetadata(project="frontend"),
        )

        results = hippocampus.recall("auth tokens", project="backend")
        assert len(results) == 1
        assert results[0].memory.metadata.project == "backend"

    def test_filters_by_language(self, hippocampus: Hippocampus) -> None:
        hippocampus.store(
            "Use dataclasses for models",
            MemoryMetadata(language="python"),
        )
        hippocampus.store(
            "Use interfaces for models",
            MemoryMetadata(language="typescript"),
        )

        results = hippocampus.recall("models", language="python")
        assert len(results) == 1
        assert results[0].memory.metadata.language == "python"

    def test_returns_scores(self, hippocampus: Hippocampus) -> None:
        hippocampus.store("The cat sat on the mat with a hat")
        results = hippocampus.recall("cat on mat")
        assert len(results) == 1
        assert results[0].score > 0.0

    def test_bumps_access_count(self, hippocampus: Hippocampus) -> None:
        hippocampus.store("The auth service uses JWT tokens for sessions")
        hippocampus.recall("JWT auth")
        hippocampus.recall("JWT auth")
        # After two recalls the access_count in metadata should be >= 1
        meta = hippocampus.get_all_metadata()
        assert meta[0]["access_count"] >= 1


class TestForget:
    def test_forgets_by_id(self, hippocampus: Hippocampus) -> None:
        memory = hippocampus.store("Something to forget eventually")
        assert memory is not None
        assert hippocampus.count() == 1

        ok = hippocampus.forget(memory.id)
        assert ok
        assert hippocampus.count() == 0

    def test_forget_nonexistent_returns_true(self, hippocampus: Hippocampus) -> None:
        # ChromaDB delete on missing IDs doesn't raise
        ok = hippocampus.forget("nonexistent_id")
        assert ok

    def test_forget_batch(self, hippocampus: Hippocampus) -> None:
        m1 = hippocampus.store("First thing to remember for a while")
        m2 = hippocampus.store("Second thing to remember for a while")
        hippocampus.store("Third thing to remember for a while")
        assert hippocampus.count() == 3

        assert m1 is not None
        assert m2 is not None
        count = hippocampus.forget_batch([m1.id, m2.id])
        assert count == 2
        assert hippocampus.count() == 1


class TestGetAllMetadata:
    def test_returns_metadata_for_all(self, hippocampus: Hippocampus) -> None:
        hippocampus.store("Memory one with enough text to store")
        hippocampus.store("Memory two with enough text to store")
        meta = hippocampus.get_all_metadata()
        assert len(meta) == 2
        assert all("id" in m for m in meta)
        assert all("created_at" in m for m in meta)
        assert all("access_count" in m for m in meta)


class TestChunking:
    def test_long_conversation_is_chunked(self, tmp_path: object) -> None:
        """Long conversation content should be split into multiple chunks."""
        from pathlib import Path

        cfg = MyelinSettings(
            data_dir=Path(str(tmp_path)) / ".myelin",
            chunk_max_chars=200,
            chunk_overlap_chars=50,
        )
        hc = Hippocampus(cfg=cfg)

        turns = []
        for i in range(10):
            turns.append(f"user: Question {i} about authentication and tokens")
            turns.append(f"assistant: Answer {i} about JWT refresh token rotation")
        text = "\n".join(turns)

        memory = hc.store(text)
        assert memory is not None
        # Should have stored multiple chunks, not just one
        assert hc.count() > 1

    def test_chunks_share_parent_id(self, tmp_path: object) -> None:
        """All chunks from a single store() call should share a parent_id."""
        from pathlib import Path

        cfg = MyelinSettings(
            data_dir=Path(str(tmp_path)) / ".myelin",
            chunk_max_chars=200,
            chunk_overlap_chars=50,
        )
        hc = Hippocampus(cfg=cfg)

        turns = []
        for i in range(10):
            turns.append(f"user: Question {i} about databases and indexes")
            turns.append(f"assistant: Answer {i} about PostgreSQL B-tree indexes")
        text = "\n".join(turns)

        hc.store(text)
        meta = hc.get_all_metadata()
        parent_ids = {m.get("parent_id") for m in meta}
        parent_ids.discard(None)
        # All chunks should share exactly one parent_id
        assert len(parent_ids) == 1

    def test_short_content_not_chunked(self, hippocampus: Hippocampus) -> None:
        """Content within chunk_max_chars should store as a single memory."""
        memory = hippocampus.store("Short memory about authentication tokens")
        assert memory is not None
        assert hippocampus.count() == 1
        meta = hippocampus.get_all_metadata()
        # Single-chunk memories get parent_id = memory.id for uniform
        # gist-guided filtering (no separate parent group).
        assert meta[0].get("parent_id") == memory.id

    def test_chunked_recall_finds_specific_chunk(self, tmp_path: object) -> None:
        """Recall should find the specific chunk matching the query."""
        from pathlib import Path

        cfg = MyelinSettings(
            data_dir=Path(str(tmp_path)) / ".myelin", chunk_max_chars=300
        )
        hc = Hippocampus(cfg=cfg)

        text = (
            "user: What database should we use?\n"
            "assistant: PostgreSQL is great for relational data with pgvector.\n"
            "user: What about caching?\n"
            "assistant: Redis is the standard choice for session caching.\n"
            "user: How should we handle auth?\n"
            "assistant: Use JWT tokens with refresh rotation every 15 minutes.\n"
            "user: What about the CI pipeline?\n"
            "assistant: GitHub Actions with pytest and ruff for linting."
        )
        hc.store(text)
        assert hc.count() > 1  # should be chunked

        results = hc.recall("database choice")
        assert len(results) > 0
        assert "PostgreSQL" in results[0].memory.content


class TestUpsert:
    """Tests for store(overwrite=True) — replace near-duplicate memories."""

    def test_overwrite_replaces_near_duplicate(self, hippocampus: Hippocampus) -> None:
        original = hippocampus.store("The auth service uses JWT tokens for sessions")
        assert original is not None
        assert hippocampus.count() == 1

        updated = hippocampus.store(
            "The auth service uses JWT tokens for sessions",
            overwrite=True,
        )
        assert updated is not None
        assert hippocampus.count() == 1
        assert updated.replaced_id == original.id

    def test_overwrite_false_still_rejects_duplicate(
        self, hippocampus: Hippocampus
    ) -> None:
        hippocampus.store("The auth service uses JWT tokens for sessions")
        dup = hippocampus.store(
            "The auth service uses JWT tokens for sessions",
            overwrite=False,
        )
        assert dup is None
        assert hippocampus.count() == 1

    def test_overwrite_stores_new_content(self, tmp_path: object) -> None:
        from pathlib import Path

        cfg = MyelinSettings(
            data_dir=Path(str(tmp_path)) / ".myelin",
            dedup_similarity_threshold=0.5,
        )
        hc = Hippocampus(cfg=cfg)
        hc.store("The auth service uses JWT tokens for sessions with HS256 signing")
        updated = hc.store(
            "The auth service uses JWT tokens for sessions with RS256 signing",
            overwrite=True,
        )
        assert updated is not None
        assert "RS256" in updated.content
        assert updated.replaced_id is not None
        assert hc.count() == 1

    def test_overwrite_no_existing_stores_normally(
        self, hippocampus: Hippocampus
    ) -> None:
        memory = hippocampus.store(
            "Completely new memory with no existing near-duplicate",
            overwrite=True,
        )
        assert memory is not None
        assert memory.replaced_id is None
        assert hippocampus.count() == 1

    def test_overwrite_does_not_replace_on_short_content(
        self, hippocampus: Hippocampus
    ) -> None:
        result = hippocampus.store("hi", overwrite=True)
        assert result is None
        assert hippocampus.count() == 0

    def test_replaced_id_not_set_on_normal_store(
        self, hippocampus: Hippocampus
    ) -> None:
        memory = hippocampus.store("A distinct memory about database indexing")
        assert memory is not None
        assert memory.replaced_id is None

    def test_overwrite_multi_chunk_replaces_all_chunks(self, tmp_path: object) -> None:
        """Upserting long content should remove all chunks of the old memory."""
        from pathlib import Path

        cfg = MyelinSettings(
            data_dir=Path(str(tmp_path)) / ".myelin",
            chunk_max_chars=200,
            chunk_overlap_chars=50,
            # Lower threshold so the full-content embedding query against
            # individual chunks still triggers the overwrite path.
            dedup_similarity_threshold=0.5,
        )
        hc = Hippocampus(cfg=cfg)

        turns = []
        for i in range(8):
            turns.append(f"user: Question {i} about JWT authentication tokens auth")
            turns.append(f"assistant: Answer {i} about JWT refresh token rotation")
        original_text = "\n".join(turns)

        hc.store(original_text)
        original_count = hc.count()
        assert original_count > 1  # multiple chunks

        # Slightly modified but semantically near-duplicate
        turns[-1] = "assistant: Updated answer about JWT rotation with RS256 signing"
        updated_text = "\n".join(turns)

        first_chunk = hc.store(updated_text, overwrite=True)
        assert first_chunk is not None
        assert first_chunk.replaced_id is not None
        # All original chunks replaced; only new chunks remain
        assert hc.count() <= original_count
