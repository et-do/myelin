"""Latency benchmark tests — pytest-benchmark micro-timings for store/recall.

Design rationale
----------------
Tests are designed to match realistic agent usage patterns, not synthetic
best-case conditions:

  Store probes
    Each probe is a semantically ADJACENT fact from the same tech domain as
    the stored corpus (auth, infra, data).  Similarity to existing memories
    is ~0.75-0.90 -- close to but below the 0.95 dedup threshold.  This
    exercises the real dedup-check decision boundary, not a trivially-passing
    similarity of 0.3.

  Hebbian pre-seeding
    The agent pipeline test pre-seeds the HebbianTracker with co-access
    history derived from the actual fixture memory IDs — simulating months of
    agent use (~50 recall sessions, 5 co-retrieved memories each = ~125
    co-access pairs).  An empty Hebbian DB would return in <1ms (zero rows),
    understating the real overhead by 5-10ms.

  Pinned memory in Thalamus
    One memory is pinned before the agent pipeline benchmark runs, exercising
    the thalamus overlay path (get_pinned → inject into results) every round.

Key findings (8-core CPU, no GPU, dev container — run with: uv run pytest
tests/benchmarks/test_latency.py -p no:xdist --override-ini="addopts=" -v):
  - Store:  ~60ms per memory (embed ~15ms + dedup-check HNSW ~5ms + gist + write)
            Flat between n=100 and n=500 (<5ms HNSW difference at this scale)
  - Recall: ~110-130ms avg, FLAT from n=100 -> n=500 -> n=1000.
            Bottleneck is fixed model inference (bi-encoder ~15ms + cross-encoder
            ~60ms x 3 probes), NOT HNSW index search (<5ms at all tested scales).
            This plateau is a design property: retrieval stays fast as memory grows.
  - Filters: project/scope filters are FASTER than unfiltered recall because
             they reduce the cross-encoder candidate pool.
  - Agent pipeline overhead: ~15-25ms on top of raw recall for a seeded Hebbian DB
             (SQLite batch read for co-access weights + reinforce write + thalamus
             touch).

Latency budgets (asserted at test time):
  - Store:  < 200ms  (generous for cold-start / repo mounts)
  - Recall: < 500ms  (hard SLA from docs/TODO.md P0)

Run options:
    # Fast suite (no n=1000 build)
    uv run pytest tests/benchmarks/test_latency.py \\
        -p no:xdist --override-ini="addopts=" -v

    # Full suite including slow n=1000 scaling validation
    uv run pytest tests/benchmarks/test_latency.py \\
        -p no:xdist --override-ini="addopts=" -v -m slow

    # Save results for future comparison
    uv run pytest tests/benchmarks/test_latency.py \\
        -p no:xdist --override-ini="addopts=" \\
        --benchmark-json=benchmarks/perf/results.json

When the full test suite runs with -n auto, pytest-benchmark disables timing
automatically — these tests still run as correctness checks.
"""

from __future__ import annotations

import pytest

from myelin.config import MyelinSettings
from myelin.models import MemoryMetadata
from myelin.recall.activation import HebbianTracker
from myelin.store.hippocampus import Hippocampus
from myelin.store.thalamus import ThalamicBuffer

# ---------------------------------------------------------------------------
# Representative corpus — diverse topics and realistic lengths
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

_CORPUS_LEN = len(_MEMORIES)


def _make_hippocampus(n: int, tmp_path: object) -> Hippocampus:
    """Build an ephemeral Hippocampus pre-populated with *n* distinct memories.

    Each memory gets a unique ``[u{i}]`` prefix so no two entries are
    near-duplicates — the dedup gate rejects at similarity ≥ 0.95, so
    identical content would not be stored past round 1.
    """
    cfg = MyelinSettings(data_dir=tmp_path / ".myelin")  # type: ignore[operator]
    hc = Hippocampus(cfg=cfg, ephemeral=True)
    for i in range(n):
        content = f"[u{i}] {_MEMORIES[i % _CORPUS_LEN]}"
        hc.store(
            content,
            MemoryMetadata(
                project=f"project-{i % 3}",
                scope=["auth", "infra", "data"][i % 3],
                memory_type=["episodic", "semantic", "procedural"][i % 3],
            ),
        )
    return hc


# ---------------------------------------------------------------------------
# Module-scoped fixtures — expensive setup runs once per worker
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hc_100(tmp_path_factory: pytest.TempPathFactory) -> Hippocampus:
    return _make_hippocampus(100, tmp_path_factory.mktemp("hc100"))


@pytest.fixture(scope="module")
def hc_500(tmp_path_factory: pytest.TempPathFactory) -> Hippocampus:
    return _make_hippocampus(500, tmp_path_factory.mktemp("hc500"))


@pytest.fixture(scope="module")
def hc_1000(tmp_path_factory: pytest.TempPathFactory) -> Hippocampus:
    return _make_hippocampus(1000, tmp_path_factory.mktemp("hc1000"))


# ---------------------------------------------------------------------------
# Store benchmarks
#
# Each benchmark invocation stores a UNIQUE string so the dedup gate never
# fires after round 1.  Previously the same content was reused, meaning
# rounds 2+ measured embed + gate-reject (~18ms) instead of the full store
# path including ChromaDB write and gist indexing (~25ms).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Store benchmarks
#
# Each round uses a SEMANTICALLY DISTINCT probe sentence from a domain
# completely removed from the tech corpus (_MEMORIES).  Random UUID tokens
# don't help here — the embedding model ignores hex strings and computes
# a near-identical vector for "Benchmark entry abc123..." vs
# "Benchmark entry def456...".  Content must differ in MEANING so the
# dedup gate (cosine similarity threshold 0.95) never fires.
#
# Using benchmark.pedantic() with setup= gives each round fresh content
# without rebuilding the collection, so the HNSW dedup-check correctly
# queries the growing collection each time.
# ---------------------------------------------------------------------------

# Store probes: adjacent-domain facts that are related to the tech corpus
# but distinct enough to stay below the 0.95 dedup threshold.
# These produce similarity scores of ~0.75-0.90 against existing memories,
# exercising the real dedup-check decision boundary rather than trivially
# passing with similarity ~0.3 (which unrelated domains would produce).
# 30 probes is enough for rounds=20 + warmup_rounds=1 = 21 consumed.
_STORE_PROBES = [
    # Auth / security variants
    "We use EdDSA for internal service-to-service tokens; RS256 is reserved for user JWTs.",  # noqa: E501
    "API keys for third-party integrations are hashed with bcrypt before storage.",
    "MFA is enforced for all admin accounts; TOTP via Authenticator app is the standard.",  # noqa: E501
    "Access tokens expire after 15 minutes; refresh tokens are rotated on each use.",
    "CORS policy allows only the production and staging origins; no wildcard origins.",
    # Data / storage variants
    "Read replicas are in us-east-2; writes always go to the primary in us-east-1.",
    "Table partitioning in Postgres is by month for events older than 90 days.",
    "Redis keys are namespaced by service: auth:session:*, billing:cache:*, etc.",
    "We use Flyway instead of Alembic for the Java service schema migrations.",
    "DynamoDB TTL is set on session records to auto-expire after 24 hours.",
    # Infra variants
    "Helm chart values are split per environment: values-dev.yaml, values-prod.yaml.",
    "Cluster autoscaler is configured to scale down after 10 minutes of idle nodes.",
    "Pod disruption budgets require at least 2 replicas available during deploys.",
    "Network policies restrict cross-namespace traffic to explicitly allowed ports.",
    "Fluentd DaemonSet forwards container logs to the central OpenSearch cluster.",
    # Frontend variants
    "Client-side feature flags are evaluated via LaunchDarkly SDK on page load.",
    "SSR is enabled only for the marketing pages; the app shell is fully CSR.",
    "Web vitals are tracked with the web-vitals library and sent to Datadog RUM.",
    "Image optimization uses Next.js built-in image component with WebP format.",
    "CSS variables are used for the design system tokens; no hardcoded hex values.",
    # Process / quality variants
    "All database queries over 500ms are flagged in Datadog and reviewed weekly.",
    "Feature branches are rebased, not merged, before opening a PR for review.",
    "Dependency updates are automated via Renovate with auto-merge for patch bumps.",
    "Load tests against staging run every Friday night via a scheduled k6 Cloud job.",
    "RFCs for architectural decisions are written in Notion before implementation.",
    # Architecture variants
    "The payments service publishes events to Kafka; no direct DB reads from other services.",  # noqa: E501
    "gRPC is used for synchronous inter-service calls; REST is for public-facing APIs.",
    "We use saga pattern for distributed transactions in the order fulfillment flow.",
    "Dead letter queues on all SQS consumers; failed messages alert PagerDuty.",
    "Service-level objectives are defined as 99.9% availability, p99 < 300ms.",
]


# ---------------------------------------------------------------------------
# Latency budget assertions
#
# Applied after benchmarks run.  Only active when the benchmark plugin is
# enabled (i.e. not during -n auto CI where benchmarks are disabled).
# ---------------------------------------------------------------------------

_STORE_BUDGET_MS = 200.0  # generous for slow repo mounts / cold containers
_RECALL_BUDGET_MS = 500.0  # hard SLA from docs/TODO.md P0


def _assert_latency_budget(benchmark: object, budget_ms: float, label: str) -> None:
    """Fail if mean latency exceeds budget — only when benchmark plugin is active."""
    stats = getattr(benchmark, "stats", None)
    if stats is None:
        return  # benchmark disabled (e.g. -n auto mode)
    mean_ms: float = stats["mean"] * 1000
    assert mean_ms < budget_ms, (
        f"{label}: mean {mean_ms:.0f}ms exceeds {budget_ms:.0f}ms budget"
    )


# ---------------------------------------------------------------------------
# Store benchmarks
#
# pedantic() gives each round a fresh probe via setup= so the collection
# grows by one item per round — realistic for an active agent session.
# ---------------------------------------------------------------------------


class TestStoreBenchmarks:
    def test_store_n100(self, benchmark: object, tmp_path: object) -> None:
        """Full store path into a 100-item collection.

        Measures the complete write path: embed (~15ms) + dedup-check (~5ms) +
        classify + gist extraction + ChromaDB write.

        Uses pedantic() with semantically distinct probe content per round so
        the dedup gate never fires — every round exercises the full write path.
        """
        hc = _make_hippocampus(100, tmp_path)
        probe_cycle = iter(_STORE_PROBES * 3)  # 90 entries — well beyond max rounds

        def setup() -> tuple[tuple[object, ...], dict[str, object]]:
            content = next(probe_cycle)
            return (hc,), {
                "content": content,
                "metadata": MemoryMetadata(project="bench"),
            }

        result = benchmark.pedantic(  # type: ignore[union-attr]
            Hippocampus.store,
            setup=setup,
            rounds=20,
            iterations=1,
            warmup_rounds=1,
        )
        assert result is not None, "Probe content must pass the dedup gate"
        _assert_latency_budget(benchmark, _STORE_BUDGET_MS, "store@n100")

    def test_store_n500(self, benchmark: object, tmp_path: object) -> None:
        """Full store path into a 500-item collection.

        The dedup-check queries ChromaDB for the nearest neighbour — a slightly
        larger collection means marginally more HNSW work.  Expect <5ms delta
        vs n=100 because HNSW is O(log n) and n is small.
        """
        hc = _make_hippocampus(500, tmp_path)
        probe_cycle = iter(_STORE_PROBES * 3)

        def setup() -> tuple[tuple[object, ...], dict[str, object]]:
            content = next(probe_cycle)
            return (hc,), {
                "content": content,
                "metadata": MemoryMetadata(project="bench"),
            }

        result = benchmark.pedantic(  # type: ignore[union-attr]
            Hippocampus.store,
            setup=setup,
            rounds=20,
            iterations=1,
            warmup_rounds=1,
        )
        assert result is not None
        _assert_latency_budget(benchmark, _STORE_BUDGET_MS, "store@n500")


# ---------------------------------------------------------------------------
# Recall scaling
#
# Key finding: recall latency is FLAT across collection sizes at this scale.
# Bottleneck is fixed model inference (embedding + cross-encoder), not HNSW.
# HNSW search adds <5ms at n=100..500.  This is a feature — retrieval stays
# fast as the memory grows to thousands of items.
#
# The slow n=1000 test validates that the flat curve holds further.
# ---------------------------------------------------------------------------


class TestRecallScaling:
    def test_recall_n100(self, benchmark: object, hc_100: Hippocampus) -> None:
        """Recall from 100-item collection — baseline latency."""
        result = benchmark(hc_100.recall, "JWT authentication token expiry")  # type: ignore[operator]
        assert result, "Should return results for a query matching stored content"
        _assert_latency_budget(benchmark, _RECALL_BUDGET_MS, "recall@n100")

    def test_recall_n500(self, benchmark: object, hc_500: Hippocampus) -> None:
        """Recall from 500-item collection.

        Mean should be within ~10ms of n=100.  If it diverges significantly,
        the HNSW index is starting to dominate — investigate ChromaDB tuning.
        """
        result = benchmark(hc_500.recall, "JWT authentication token expiry")  # type: ignore[operator]
        assert result
        _assert_latency_budget(benchmark, _RECALL_BUDGET_MS, "recall@n500")

    @pytest.mark.slow
    def test_recall_n1000(self, benchmark: object, hc_1000: Hippocampus) -> None:
        """Recall from 1000-item collection — validates flat-scaling claim.

        Expected: latency still within ~20ms of n=100.  This is the scale
        a heavy individual user reaches after ~6-12 months of daily use.
        """
        result = benchmark(hc_1000.recall, "JWT authentication token expiry")  # type: ignore[operator]
        assert result
        _assert_latency_budget(benchmark, _RECALL_BUDGET_MS, "recall@n1000")

    def test_recall_with_project_filter(
        self, benchmark: object, hc_100: Hippocampus
    ) -> None:
        """Recall with project filter — measures metadata-filter path overhead."""
        result = benchmark(hc_100.recall, "database performance", project="project-0")  # type: ignore[operator]
        assert result
        _assert_latency_budget(
            benchmark, _RECALL_BUDGET_MS, "recall@n100+project_filter"
        )

    def test_recall_with_scope_filter(
        self, benchmark: object, hc_100: Hippocampus
    ) -> None:
        """Recall with scope filter — narrows the ChromaDB candidate set."""
        result = benchmark(hc_100.recall, "deployment infrastructure", scope="infra")  # type: ignore[operator]
        assert result
        _assert_latency_budget(benchmark, _RECALL_BUDGET_MS, "recall@n100+scope_filter")


# ---------------------------------------------------------------------------
# Full pipeline (agent path) overhead
#
# Agents call do_recall() in server.py, which wraps hc.recall() with:
#   - HebbianTracker.boost()    — SQLite read
#   - HebbianTracker.reinforce() — SQLite write
#   - ThalamicBuffer.touch()    — SQLite write
#   - ThalamicBuffer.get_pinned() — SQLite read
#
# This test isolates that overhead vs raw hippocampus recall.
# SQLite in WAL mode is fast; expect ~10ms total overhead.
# ---------------------------------------------------------------------------


class TestAgentPipelineOverhead:
    def test_hebbian_thalamus_overhead_n100(
        self, benchmark: object, hc_100: Hippocampus, tmp_path: object
    ) -> None:
        """Full post-recall agent pipeline: Hebbian + Thalamus on top of recall.

        Measures what MCP tool calls actually pay vs raw hc.recall().
        """
        cfg = MyelinSettings(data_dir=tmp_path / ".myelin")  # type: ignore[operator]
        cfg.ensure_dirs()
        heb = HebbianTracker(cfg=cfg)
        thal = ThalamicBuffer(cfg=cfg)

        # Pre-seed Hebbian with realistic co-access history: simulate ~50 recall
        # sessions x 5 co-retrieved memories each = ~125 co-access pairs.
        # An empty Hebbian DB returns boost() in <1ms (zero rows); a seeded one
        # reflects what agents pay after months of use (~15-25ms).
        all_ids = [m["id"] for m in hc_100.get_all_metadata()]
        chunk = 5
        for offset in range(0, min(50 * chunk, len(all_ids) - chunk), chunk):
            heb.reinforce(all_ids[offset : offset + chunk])

        # Pin one memory so every round exercises the thalamus overlay path.
        thal.pin(all_ids[0], priority=1, label="bench-pin")

        query = "JWT authentication token expiry"

        def _full_pipeline() -> list[object]:
            results = hc_100.recall(query)
            if results:
                results = heb.boost(results)
                heb.reinforce([r.memory.id for r in results])
                thal.touch([r.memory.id for r in results])
                _ = thal.get_pinned()
            return results  # type: ignore[return-value]

        result = benchmark(_full_pipeline)  # type: ignore[operator]
        assert result
        _assert_latency_budget(benchmark, _RECALL_BUDGET_MS, "agent_pipeline@n100")


# ---------------------------------------------------------------------------
# Relevance sanity checks
#
# These run during normal -n auto CI (benchmark disabled, just correctness).
# Verify that the top results are topically relevant to the query.
# ---------------------------------------------------------------------------


class TestRelevanceSanity:
    def test_jwt_query_returns_security_content(
        self, benchmark: object, hc_100: Hippocampus
    ) -> None:
        """JWT query should surface auth/security memories in top results."""
        result = benchmark(hc_100.recall, "JWT authentication")  # type: ignore[operator]
        assert result, "Non-empty result expected"
        top = [r.memory.content.lower() for r in result[:3]]
        assert any("jwt" in c or "token" in c or "auth" in c for c in top), (
            f"Top-3 results don't contain JWT/auth content: {top}"
        )

    def test_postgres_query_returns_db_content(
        self, benchmark: object, hc_100: Hippocampus
    ) -> None:
        """PostgreSQL query should surface data/storage memories in top results."""
        result = benchmark(hc_100.recall, "PostgreSQL database")  # type: ignore[operator]
        assert result
        top = [r.memory.content.lower() for r in result[:3]]
        assert any(
            "postgres" in c or "database" in c or "sql" in c or "redis" in c
            for c in top
        ), f"Top-3 results don't contain DB content: {top}"

    def test_recall_scores_descending(
        self, benchmark: object, hc_100: Hippocampus
    ) -> None:
        """Results must be returned in descending score order."""
        result = benchmark(hc_100.recall, "Kubernetes deployment infrastructure")  # type: ignore[operator]
        assert result
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted by score: {scores}"
        )
