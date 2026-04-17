"""Latency benchmark tests — pytest-benchmark micro-timings for store/recall.

Run the full suite (with timing output):

    uv run pytest tests/benchmarks/test_latency.py -p no:xdist -v

Compare against a saved baseline:

    uv run pytest tests/benchmarks/test_latency.py -p no:xdist --benchmark-compare

Save results for future comparison:

    uv run pytest tests/benchmarks/test_latency.py -p no:xdist \\
        --benchmark-json=benchmarks/perf/results.json

When running the full test suite (which uses -n auto), pytest-benchmark
automatically disables timing but the tests still run as correctness checks.
"""

from __future__ import annotations

import pytest

from myelin.config import MyelinSettings
from myelin.models import MemoryMetadata
from myelin.store.hippocampus import Hippocampus

# ---------------------------------------------------------------------------
# Representative corpus for benchmarking — diverse topics, realistic length
# ---------------------------------------------------------------------------

_MEMORIES = [
    # Authentication / security
    "We use JWT tokens signed with RS256; verification is done at the API gateway.",
    "Secrets are stored in AWS Secrets Manager and rotated every 90 days.",
    "OAuth2 with PKCE is the flow for mobile and SPA clients.",
    "Session tokens are stored in HTTP-only, Secure, SameSite=Strict cookies.",
    "Rate limiting is enforced at 100 req/min per API key via Nginx.",
    # Data / storage
    "PostgreSQL 16 is the primary database; pgvector is used for embeddings.",
    "Redis 7 handles session cache and pub/sub for real-time notifications.",
    "Database migrations use Alembic; never modify a migration once deployed.",
    "We shard writes across 3 Postgres replicas with pgBouncer for pooling.",
    "S3 is used for blob storage with lifecycle rules to Glacier after 90 days.",
    # Infrastructure
    "Kubernetes 1.30 on EKS; Helm charts live in the infra/helm directory.",
    "ArgoCD handles GitOps deployments — merge to main triggers staging deploy.",
    "Terraform state is stored in S3 with DynamoDB locking.",
    "Observability stack: Prometheus + Grafana + Loki + Tempo (LGTM).",
    "Circuit breaker pattern is implemented via Resilience4j in the Java services.",
    # Frontend / mobile
    "Frontend is Next.js 14 with App Router and React Server Components.",
    "State management uses Zustand for local state, React Query for server state.",
    "Mobile targets iOS 16+ and Android 12+ with React Native.",
    "i18n uses react-intl; translations are managed in Phrase.",
    "Bundle size limit is 200 KB gzipped — enforced in CI via bundlesize.",
    # ML / data science
    "ML training runs on AWS SageMaker with A100 instances.",
    "MLflow on Databricks tracks experiments; models are promoted via Model Registry.",
    "Feature store is Tecton backed by DynamoDB for online serving.",
    "Data versioning uses DVC; large files are stored in S3.",
    "Embeddings are generated with sentence-transformers all-MiniLM-L6-v2.",
    # Process / team
    "Code review requires two approvals; one must be from a senior engineer.",
    "We follow Conventional Commits for all commit messages.",
    "Sprint length is two weeks; planning is Monday, retrospective is Friday.",
    "On-call rotation is weekly; PagerDuty is the alerting platform.",
    "Tech debt items are tracked in the #tech-debt Jira board with priority tags.",
    # Architecture decisions
    "Event-driven architecture via Kafka; topics follow <service>.<entity>.<event>.",
    "Service mesh is Istio; mTLS is enforced between all services in the cluster.",
    "API-first design — OpenAPI specs are committed before implementation.",
    "CQRS is used in the billing service; event sourcing for the audit log.",
    "The auth service is stateless; JWTs carry all necessary claims.",
    # Testing / quality
    "Unit tests use pytest with pytest-asyncio for async code.",
    "Integration tests run against a dockerized test database seeded with fixtures.",
    "E2E tests use Playwright; they run nightly against staging.",
    "Coverage gate is 80% for new code; measured with pytest-cov.",
    "Flaky tests are tracked in a Notion database and must be fixed within one sprint.",
    # Developer experience
    "Local dev uses devcontainers with VS Code; docker-compose for dependencies.",
    "Pre-commit hooks run ruff, mypy, and pytest (unit tests only).",
    "Secrets for local dev are managed with direnv + .envrc (gitignored).",
    "The internal docs portal is Backstage; service catalog is kept up to date.",
    "Onboarding takes 3 days: day 1 setup, day 2 architecture tour, day 3 first PR.",
    # Performance
    "Target LCP is 2.5s on desktop, 3s on mobile (Core Web Vitals).",
    "API p99 latency budget is 500ms; alerting fires at 1s.",
    "Database query timeout is 30s; long queries are logged and reviewed weekly.",
    "CDN is Cloudflare; cache TTL for static assets is 1 year.",
    "Load testing uses k6; baseline is 1000 concurrent users at 5% error rate.",
]


def _make_hippocampus(n: int, tmp_path: object) -> Hippocampus:
    """Build an ephemeral Hippocampus pre-populated with *n* memories."""
    cfg = MyelinSettings(data_dir=tmp_path / ".myelin")  # type: ignore[operator]
    hc = Hippocampus(cfg=cfg, ephemeral=True)

    # Cycle through the corpus until we have n memories
    corpus_len = len(_MEMORIES)
    for i in range(n):
        content = _MEMORIES[i % corpus_len]
        if n > corpus_len:
            # Disambiguate duplicates when n > corpus size
            content = f"[variant-{i // corpus_len}] {content}"
        hc.store(
            content,
            MemoryMetadata(
                project=f"project-{i % 3}",
                scope=["chat", "code", "docs"][i % 3],
                memory_type=["episodic", "semantic", "procedural"][i % 3],
            ),
        )
    return hc


# ---------------------------------------------------------------------------
# Module-scoped fixtures — expensive setup runs once per pytest-xdist worker
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hc_100(tmp_path_factory: pytest.TempPathFactory) -> Hippocampus:
    tmp = tmp_path_factory.mktemp("hc100")
    return _make_hippocampus(100, tmp)


@pytest.fixture(scope="module")
def hc_500(tmp_path_factory: pytest.TempPathFactory) -> Hippocampus:
    tmp = tmp_path_factory.mktemp("hc500")
    return _make_hippocampus(500, tmp)


# ---------------------------------------------------------------------------
# Store benchmarks — how long does a single store() call take?
# ---------------------------------------------------------------------------


class TestStoreBenchmarks:
    def test_store_latency_n100(self, benchmark: object, tmp_path: object) -> None:
        """Store a single memory into a 100-item collection."""
        hc = _make_hippocampus(100, tmp_path)
        benchmark(  # type: ignore[operator]
            hc.store,
            "Benchmark probe memory — measuring store latency.",
            MemoryMetadata(project="bench"),
        )

    def test_store_latency_n500(self, benchmark: object, tmp_path: object) -> None:
        """Store a single memory into a 500-item collection."""
        hc = _make_hippocampus(500, tmp_path)
        benchmark(  # type: ignore[operator]
            hc.store,
            "Benchmark probe memory — measuring store latency at scale.",
            MemoryMetadata(project="bench"),
        )


# ---------------------------------------------------------------------------
# Recall benchmarks — how long does recall() take across corpus sizes?
# ---------------------------------------------------------------------------


class TestRecallBenchmarks:
    def test_recall_latency_n100(self, benchmark: object, hc_100: Hippocampus) -> None:
        """Recall from a 100-item collection."""
        benchmark(hc_100.recall, "JWT authentication token expiry")  # type: ignore[operator]

    def test_recall_latency_n500(self, benchmark: object, hc_500: Hippocampus) -> None:
        """Recall from a 500-item collection."""
        benchmark(hc_500.recall, "JWT authentication token expiry")  # type: ignore[operator]

    def test_recall_with_project_filter_n100(
        self, benchmark: object, hc_100: Hippocampus
    ) -> None:
        """Recall with a project filter — measures filter-path overhead."""
        benchmark(hc_100.recall, "database performance indexing", project="project-0")  # type: ignore[operator]

    def test_recall_with_scope_filter_n100(
        self, benchmark: object, hc_100: Hippocampus
    ) -> None:
        """Recall with a scope filter — measures metadata-filter path."""
        benchmark(hc_100.recall, "deployment infrastructure kubernetes", scope="code")  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Throughput sanity — verify results are returned (not just timing overhead)
# ---------------------------------------------------------------------------


class TestRecallCorrectnessDuringBenchmark:
    def test_recall_returns_results_n100(
        self, benchmark: object, hc_100: Hippocampus
    ) -> None:
        """Benchmark recall and verify at least one result is returned."""
        result = benchmark(hc_100.recall, "JWT authentication")  # type: ignore[operator]
        assert result  # non-empty list

    def test_recall_returns_results_n500(
        self, benchmark: object, hc_500: Hippocampus
    ) -> None:
        """Benchmark recall at scale and verify results come back."""
        result = benchmark(hc_500.recall, "PostgreSQL database indexing")  # type: ignore[operator]
        assert result
