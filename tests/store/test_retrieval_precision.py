"""Retrieval precision tests — functional correctness of store→recall pipeline.

These tests assert *which* memory surfaces at which rank, not just that
recall returns something.  They catch regressions in embedding quality,
filter logic, scoring, and the gist/chunk pathways.

All tests use the ephemeral Hippocampus (no reranker) from conftest so
they run in the normal pytest suite without downloading extra models.
"""

from __future__ import annotations

from myelin.models import MemoryMetadata
from myelin.store.hippocampus import Hippocampus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _top(hc: Hippocampus, query: str, **kwargs: object) -> str:
    """Return the content of the top-ranked recall result."""
    results = hc.recall(query, **kwargs)  # type: ignore[arg-type]
    assert results, f"No results for query: {query!r}"
    return results[0].memory.content


def _ids(hc: Hippocampus, query: str, n: int = 5, **kwargs: object) -> list[str]:
    """Return content of top-n results (for rank assertions)."""
    results = hc.recall(query, n_results=n, **kwargs)  # type: ignore[arg-type]
    return [r.memory.content for r in results]


# ---------------------------------------------------------------------------
# Single-domain precision — the most-relevant memory must rank first
# ---------------------------------------------------------------------------


class TestTopRankPrecision:
    def test_exact_topic_ranks_first(self, hippocampus: Hippocampus) -> None:
        hippocampus.store("The auth service uses JWT tokens with RS256 signing")
        hippocampus.store("The database is PostgreSQL 16 with pgvector extension")
        hippocampus.store("CI pipeline uses GitHub Actions with ruff and pytest")
        hippocampus.store("Frontend is React 18 with TypeScript and Vite")

        top = _top(hippocampus, "authentication and JWT signing")
        assert "JWT" in top

    def test_database_query_ranks_db_memory_first(
        self, hippocampus: Hippocampus
    ) -> None:
        hippocampus.store("The auth service uses JWT tokens with RS256 signing")
        hippocampus.store("The database is PostgreSQL 16 with pgvector extension")
        hippocampus.store("CI pipeline uses GitHub Actions with ruff and pytest")

        top = _top(hippocampus, "which database are we using")
        assert "PostgreSQL" in top

    def test_infra_query_ranks_infra_memory_first(
        self, hippocampus: Hippocampus
    ) -> None:
        hippocampus.store("The auth service uses JWT tokens with RS256 signing")
        hippocampus.store("Deployment uses Kubernetes with Helm charts on EKS")
        hippocampus.store("The database is PostgreSQL 16 with pgvector extension")

        top = _top(hippocampus, "kubernetes deployment infrastructure")
        assert "Kubernetes" in top


# ---------------------------------------------------------------------------
# Filter precision — metadata filters must narrow results correctly
# ---------------------------------------------------------------------------


class TestFilterPrecision:
    def test_project_filter_excludes_other_projects(
        self, hippocampus: Hippocampus
    ) -> None:
        hippocampus.store(
            "Auth uses RS256 JWT tokens",
            MemoryMetadata(project="backend"),
        )
        hippocampus.store(
            "Auth uses OAuth2 with PKCE flow",
            MemoryMetadata(project="mobile"),
        )

        results = hippocampus.recall("authentication", project="backend")
        assert all(r.memory.metadata.project == "backend" for r in results)
        contents = [r.memory.content for r in results]
        assert any("RS256" in c for c in contents)
        assert not any("PKCE" in c for c in contents)

    def test_scope_filter_isolates_domain(self, hippocampus: Hippocampus) -> None:
        hippocampus.store(
            "Use dataclasses for internal data models",
            MemoryMetadata(scope="architecture"),
        )
        hippocampus.store(
            "Use Pydantic for API request/response validation",
            MemoryMetadata(scope="api"),
        )

        results = hippocampus.recall("data models", scope="architecture")
        assert all(r.memory.metadata.scope == "architecture" for r in results)

    def test_memory_type_filter_returns_only_that_type(
        self, hippocampus: Hippocampus
    ) -> None:
        hippocampus.store(
            "We decided to use event sourcing for the audit log",
            MemoryMetadata(memory_type="semantic"),
        )
        hippocampus.store(
            "Deployed v2.3.1 to production on Friday with zero downtime",
            MemoryMetadata(memory_type="episodic"),
        )

        results = hippocampus.recall("deployment", memory_type="episodic")
        assert all(r.memory.metadata.memory_type == "episodic" for r in results)

    def test_language_filter_excludes_other_languages(
        self, hippocampus: Hippocampus
    ) -> None:
        hippocampus.store(
            "Use type aliases and Protocol for structural typing",
            MemoryMetadata(language="python"),
        )
        hippocampus.store(
            "Use interface and type for structural typing",
            MemoryMetadata(language="typescript"),
        )

        py_results = hippocampus.recall("structural typing", language="python")
        ts_results = hippocampus.recall("structural typing", language="typescript")

        assert all(r.memory.metadata.language == "python" for r in py_results)
        assert all(r.memory.metadata.language == "typescript" for r in ts_results)


# ---------------------------------------------------------------------------
# Multi-memory corpus recall — relevant items surface within top-k
# ---------------------------------------------------------------------------


class TestCorpusRecall:
    def test_relevant_memory_in_top_3_from_10(self, hippocampus: Hippocampus) -> None:
        """With 10 mixed memories, the on-topic one must appear in top-3."""
        topics = [
            "The CI pipeline triggers on every PR using GitHub Actions",
            "We use Terraform for infrastructure-as-code on AWS",
            "Database migrations use Alembic with SQLAlchemy models",
            "Frontend state management uses Zustand over Redux",
            "API rate limiting is enforced at the Nginx ingress layer",
            "Secrets are stored in AWS Secrets Manager, never in git",
            "The mobile app is React Native targeting iOS 16+ and Android 12+",
            "Logging uses structured JSON sent to Datadog via FluentBit",
            "The auth service issues RS256 JWT tokens valid for 15 minutes",
            "Code review requires two approvals before merge to main",
        ]
        for t in topics:
            hippocampus.store(t)

        top3 = _ids(hippocampus, "JWT token authentication expiry", n=3)
        assert any("JWT" in c or "auth" in c.lower() for c in top3), (
            f"Auth memory not in top 3: {top3}"
        )

    def test_two_relevant_memories_both_surface(self, hippocampus: Hippocampus) -> None:
        """When two memories are relevant, both should appear in top-5."""
        hippocampus.store("PostgreSQL uses B-tree indexes for range queries")
        hippocampus.store("PostgreSQL query planner uses statistics for planning")
        hippocampus.store("Redis stores session data with 24h TTL")
        hippocampus.store("CI pipeline triggers on push to main and PRs")
        hippocampus.store("The frontend uses Vite for hot module replacement")

        top5 = _ids(hippocampus, "PostgreSQL query performance", n=5)
        pg_hits = [c for c in top5 if "PostgreSQL" in c]
        assert len(pg_hits) >= 2, f"Expected 2 PostgreSQL results in top-5: {top5}"


# ---------------------------------------------------------------------------
# Negative recall — irrelevant queries must not surface wrong memories
# ---------------------------------------------------------------------------


class TestNegativePrecision:
    def test_unrelated_query_gets_low_score(self, hippocampus: Hippocampus) -> None:
        """A query completely unrelated to stored content returns low scores."""
        hippocampus.store("The auth service uses JWT tokens with RS256 signing")
        hippocampus.store("The database is PostgreSQL 16 with pgvector extension")

        results = hippocampus.recall("baking bread sourdough recipe flour")
        # Results may be returned (recall is fuzzy) but scores should be low
        if results:
            assert results[0].score < 0.5, (
                f"Irrelevant query returned high score: {results[0].score}"
            )

    def test_filter_mismatch_returns_empty(self, hippocampus: Hippocampus) -> None:
        """A project filter that matches nothing returns empty results."""
        hippocampus.store(
            "The database is PostgreSQL",
            MemoryMetadata(project="backend"),
        )
        results = hippocampus.recall("database", project="nonexistent_project_xyz")
        assert results == []


# ---------------------------------------------------------------------------
# Forget precision — deleted memories must not surface in recall
# ---------------------------------------------------------------------------


class TestForgetPrecision:
    def test_forgotten_memory_absent_from_recall(
        self, hippocampus: Hippocampus
    ) -> None:
        m = hippocampus.store("The auth service uses JWT tokens with RS256 signing")
        hippocampus.store("The database is PostgreSQL 16 with pgvector extension")
        assert m is not None

        hippocampus.forget(m.id)

        results = hippocampus.recall("JWT authentication tokens")
        contents = [r.memory.content for r in results]
        assert not any("JWT" in c for c in contents), (
            "Forgotten memory still appearing in recall"
        )

    def test_forgotten_batch_absent_from_recall(self, hippocampus: Hippocampus) -> None:
        m1 = hippocampus.store("We use Redis for distributed caching layer")
        m2 = hippocampus.store("Redis TTL is set to 3600 seconds for sessions")
        hippocampus.store("The database is PostgreSQL 16")
        assert m1 is not None and m2 is not None

        hippocampus.forget_batch([m1.id, m2.id])

        results = hippocampus.recall("Redis caching sessions")
        contents = [r.memory.content for r in results]
        assert not any("Redis" in c for c in contents), (
            "Batch-forgotten memories still appearing in recall"
        )
