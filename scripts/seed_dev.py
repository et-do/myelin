"""Seed the dev-container database with rich data for UI visualisation.

Run with:  uv run python scripts/seed_dev.py
"""

# ruff: noqa: E501  — seed data strings intentionally exceed line length

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the local package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from myelin.config import settings
from myelin.models import MemoryMetadata
from myelin.store.hippocampus import Hippocampus
from myelin.store.neocortex import SemanticNetwork
from myelin.ui.serve import build_memories_and_stats, load_chromadb_raw

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = settings.data_dir
print(f"Seeding into: {DATA_DIR}")

# ---------------------------------------------------------------------------
# 1.  Memories — stored via Hippocampus (handles ChromaDB + extracts entities)
# ---------------------------------------------------------------------------

hc = Hippocampus()

MEMORIES = [
    # ── myelin project ──────────────────────────────────────────────────────
    dict(
        content="We use RS256 JWT tokens for API authentication. Asymmetric keys let the gateway verify without the signing secret. The private key lives in Vault.",
        project="myelin",
        scope="auth",
        memory_type="semantic",
        tags=["jwt", "auth", "security"],
        agent_id="copilot",
    ),
    dict(
        content="Decided to adopt ChromaDB ≥1.5 for episodic memory storage. PersistentClient gives us zero-dep local persistence and HNSW ANN search.",
        project="myelin",
        scope="architecture",
        memory_type="semantic",
        tags=["chromadb", "architecture", "decision"],
        agent_id="copilot",
    ),
    dict(
        content="The SyncRegistry tracks export/import history in sync.db. Content-hash dedup ensures repeated export runs only write changed memories.",
        project="myelin",
        scope="sync",
        memory_type="procedural",
        tags=["sync", "export", "incremental"],
        agent_id="copilot",
    ),
    dict(
        content="Hippocampus is the episodic memory store. It encodes content with sentence-transformers (all-MiniLM-L6-v2) and retrieves by cosine similarity.",
        project="myelin",
        scope="architecture",
        memory_type="semantic",
        tags=["hippocampus", "embeddings", "retrieval"],
        agent_id="copilot",
    ),
    dict(
        content="The Thalamus overlay prepends pinned memories to every recall result. Pins are stored in thalamus.db and evicted by LRU if the cap is hit.",
        project="myelin",
        scope="architecture",
        memory_type="semantic",
        tags=["thalamus", "pinned", "overlay"],
        agent_id="copilot",
    ),
    dict(
        content="Hebbian learning: when two memories are co-recalled, their shared entities get a +0.1 weight bump in neocortex.db. Repeated co-recall strengthens the link.",
        project="myelin",
        scope="recall",
        memory_type="semantic",
        tags=["hebbian", "neocortex", "learning"],
        agent_id="copilot",
    ),
    dict(
        content="Consolidation builds the semantic network from ChromaDB. extract_entities runs spaCy NER over stored memories and upserts entity/relationship rows.",
        project="myelin",
        scope="consolidation",
        memory_type="procedural",
        tags=["consolidation", "neocortex", "spacy"],
        agent_id="copilot",
    ),
    dict(
        content="graph UI serves a D3 v7 force-directed graph of the semantic network. Node size = degree, colour = entity_type, edge thickness = weight.",
        project="myelin",
        scope="ui",
        memory_type="episodic",
        tags=["graph", "d3", "ui", "visualisation"],
        agent_id="copilot",
    ),
    dict(
        content="Deployed Myelin v0.9.0 to PyPI. The publish workflow runs on release tags and uploads the wheel built by hatchling.",
        project="myelin",
        scope="release",
        memory_type="episodic",
        tags=["pypi", "release", "hatchling"],
        agent_id="copilot",
    ),
    dict(
        content="LongMemEval benchmark: Myelin achieves 98.2% Recall@5 on the 500-question split using only local 22M-parameter models — beats MemPalace (96.6%) which uses GPT-4o.",
        project="myelin",
        scope="benchmarks",
        memory_type="semantic",
        tags=["benchmark", "longmemeval", "recall", "performance"],
        agent_id="copilot",
    ),
    dict(
        content="Fixed race condition in Hippocampus where concurrent store() calls could double-insert the same memory_id. Added threading.Lock around the upsert path.",
        project="myelin",
        scope="bugfix",
        memory_type="episodic",
        tags=["concurrency", "bugfix", "hippocampus"],
        agent_id="copilot",
    ),
    dict(
        content="Amygdala emotion gate: memories tagged with high-valence emotions (joy, fear, surprise) get a 1.3x recall boost from the amygdala gate.",
        project="myelin",
        scope="recall",
        memory_type="semantic",
        tags=["amygdala", "emotion", "recall", "boost"],
        agent_id="copilot",
    ),
    # ── obsidian integration ────────────────────────────────────────────────
    dict(
        content="ObsidianExporter writes each memory as a .md file under vault/Myelin/. Backlinks are created with wikilinks for every entity mention.",
        project="obsidian",
        scope="integration",
        memory_type="procedural",
        tags=["obsidian", "export", "markdown", "wikilinks"],
        agent_id="claude",
    ),
    dict(
        content="ObsidianImporter ingests .md files from a vault. Front-matter fields (project, scope, tags) map directly to MemoryMetadata. Incremental import only reads changed files.",
        project="obsidian",
        scope="integration",
        memory_type="procedural",
        tags=["obsidian", "import", "incremental", "frontmatter"],
        agent_id="claude",
    ),
    # ── github integration ──────────────────────────────────────────────────
    dict(
        content="GitHubImporter reads git log via subprocess and converts commits to episodic memories. PR descriptions and issue discussions become semantic memories.",
        project="github",
        scope="integration",
        memory_type="semantic",
        tags=["github", "git", "commits", "prs", "issues"],
        agent_id="copilot",
    ),
    dict(
        content="Ran github-import on the myelin repo — 142 commits ingested as episodic memories, 18 PRs as semantic design decisions.",
        project="github",
        scope="integration",
        memory_type="episodic",
        tags=["github", "import", "commits", "episodic"],
        agent_id="copilot",
    ),
    # ── auth service (separate project) ────────────────────────────────────
    dict(
        content="Auth service uses Argon2id for password hashing. bcrypt was considered but Argon2id is recommended by OWASP for new projects.",
        project="auth-service",
        scope="security",
        memory_type="semantic",
        tags=["argon2", "security", "owasp", "passwords"],
        agent_id="cursor",
    ),
    dict(
        content="OAuth2 PKCE flow implemented for the web client. The auth code + code_verifier pair prevents authorization code interception.",
        project="auth-service",
        scope="auth",
        memory_type="procedural",
        tags=["oauth2", "pkce", "security", "web"],
        agent_id="cursor",
    ),
    dict(
        content="Session tokens expire after 24h. Refresh tokens are single-use and rotated on every exchange (refresh token rotation).",
        project="auth-service",
        scope="auth",
        memory_type="semantic",
        tags=["sessions", "tokens", "security", "rotation"],
        agent_id="cursor",
    ),
    # ── data-pipeline project ───────────────────────────────────────────────
    dict(
        content="Pipeline uses Apache Kafka for event streaming. Topics are partitioned by user_id for ordered per-user processing.",
        project="data-pipeline",
        scope="architecture",
        memory_type="semantic",
        tags=["kafka", "streaming", "partitioning", "architecture"],
        agent_id="claude",
    ),
    dict(
        content="Switched from polling to CDC (change data capture) using Debezium. Latency dropped from ~5s to ~200ms for downstream consumers.",
        project="data-pipeline",
        scope="performance",
        memory_type="episodic",
        tags=["cdc", "debezium", "kafka", "performance", "latency"],
        agent_id="claude",
    ),
    dict(
        content="Data pipeline uses dbt for transformation. Models are tested with schema.yml assertions and run nightly in GitHub Actions.",
        project="data-pipeline",
        scope="etl",
        memory_type="procedural",
        tags=["dbt", "etl", "testing", "github-actions"],
        agent_id="claude",
    ),
    # ── shared/cross-project ────────────────────────────────────────────────
    dict(
        content="Postgres 16 is the primary OLTP database across all services. Connection pooling via PgBouncer in transaction mode.",
        project="infra",
        scope="database",
        memory_type="semantic",
        tags=["postgres", "pgbouncer", "database", "pooling"],
        agent_id="copilot",
    ),
    dict(
        content="All services emit OpenTelemetry traces to Grafana Tempo. Spans correlate across auth-service, data-pipeline, and API gateway.",
        project="infra",
        scope="observability",
        memory_type="semantic",
        tags=["opentelemetry", "grafana", "tracing", "observability"],
        agent_id="copilot",
    ),
    dict(
        content="Deployed Kubernetes 1.30 cluster on GKE. Istio service mesh handles mTLS between pods and exposes circuit-breaker metrics.",
        project="infra",
        scope="k8s",
        memory_type="episodic",
        tags=["kubernetes", "istio", "mtls", "gke"],
        agent_id="copilot",
    ),
]

print(f"Storing {len(MEMORIES)} memories...")
for i, m in enumerate(MEMORIES):
    meta = MemoryMetadata(
        project=m.get("project"),
        scope=m.get("scope"),
        memory_type=m.get("memory_type"),
        tags=m.get("tags", []),
        agent_id=m.get("agent_id"),
    )
    hc.store(content=m["content"], metadata=meta)
    print(
        f"  [{i + 1:2d}/{len(MEMORIES)}] {m['project']}/{m['scope']}: {m['content'][:60]}…"
    )

# ---------------------------------------------------------------------------
# 2.  Semantic network — explicit entity graph with varied types + weights
# ---------------------------------------------------------------------------

net = SemanticNetwork(db_path=DATA_DIR / "neocortex.db")

# People
PEOPLE = ["alice", "bob", "charlie", "diana", "eve", "frank"]
for p in PEOPLE:
    net.add_entity(p, entity_type="person")

# Technologies
TECH = [
    "chromadb",
    "sqlite",
    "postgres",
    "kafka",
    "redis",
    "docker",
    "kubernetes",
    "python",
    "fastmcp",
    "dbt",
    "opentelemetry",
    "grafana",
    "debezium",
    "spacy",
    "sentence-transformers",
    "d3js",
    "istio",
    "argon2",
    "jwt",
    "oauth2",
]
for t in TECH:
    net.add_entity(t, entity_type="technology")

# Projects
PROJECTS = ["myelin", "auth-service", "data-pipeline", "infra", "obsidian", "github"]
for p in PROJECTS:
    net.add_entity(p, entity_type="project")

# Concepts
CONCEPTS = [
    "authentication",
    "authorization",
    "security",
    "encryption",
    "embeddings",
    "retrieval",
    "consolidation",
    "hebbian-learning",
    "episodic-memory",
    "semantic-memory",
    "procedural-memory",
    "streaming",
    "cdc",
    "partitioning",
    "latency",
    "observability",
    "tracing",
    "metrics",
    "alerting",
    "recall",
    "decay",
    "activation",
    "spreading-activation",
    "incremental-sync",
    "wikilinks",
    "frontmatter",
    "graph",
    "force-directed",
    "visualisation",
]
for c in CONCEPTS:
    net.add_entity(c, entity_type="concept")

# Decisions
DECISIONS = [
    "use-chromadb-for-episodic-storage",
    "rs256-jwt-for-api-auth",
    "argon2id-for-passwords",
    "kafka-for-event-streaming",
    "opentelemetry-for-tracing",
    "d3-for-graph-ui",
    "pkce-oauth2-flow",
    "sentence-transformers-local-models",
]
for d in DECISIONS:
    net.add_entity(d, entity_type="decision")

print(
    f"\nAdding entities: {len(PEOPLE)} people, {len(TECH)} tech, {len(PROJECTS)} projects, {len(CONCEPTS)} concepts, {len(DECISIONS)} decisions"
)

# ---------------------------------------------------------------------------
# Relationships — varied predicates and weights
# ---------------------------------------------------------------------------

RELS = [
    # Tech ↔ project memberships
    ("myelin", "uses", "chromadb", 3.0),
    ("myelin", "uses", "sqlite", 2.5),
    ("myelin", "uses", "sentence-transformers", 2.5),
    ("myelin", "uses", "spacy", 2.0),
    ("myelin", "uses", "fastmcp", 2.0),
    ("myelin", "uses", "d3js", 1.5),
    ("myelin", "uses", "python", 3.0),
    ("auth-service", "uses", "postgres", 2.5),
    ("auth-service", "uses", "redis", 2.0),
    ("auth-service", "uses", "jwt", 2.5),
    ("auth-service", "uses", "oauth2", 2.0),
    ("auth-service", "uses", "argon2", 2.0),
    ("data-pipeline", "uses", "kafka", 3.0),
    ("data-pipeline", "uses", "postgres", 2.0),
    ("data-pipeline", "uses", "dbt", 2.0),
    ("data-pipeline", "uses", "debezium", 2.0),
    ("infra", "uses", "kubernetes", 2.5),
    ("infra", "uses", "docker", 2.0),
    ("infra", "uses", "opentelemetry", 2.0),
    ("infra", "uses", "grafana", 1.5),
    ("infra", "uses", "istio", 1.5),
    ("infra", "uses", "postgres", 2.0),
    # Concepts ↔ tech
    ("chromadb", "enables", "embeddings", 2.0),
    ("chromadb", "enables", "retrieval", 2.0),
    ("sentence-transformers", "produces", "embeddings", 2.5),
    ("embeddings", "powers", "retrieval", 3.0),
    ("retrieval", "uses", "spreading-activation", 1.5),
    ("spreading-activation", "implements", "hebbian-learning", 1.5),
    ("hebbian-learning", "strengthens", "recall", 2.0),
    ("recall", "uses", "activation", 2.0),
    ("recall", "uses", "decay", 1.5),
    ("kafka", "enables", "streaming", 2.5),
    ("kafka", "enables", "partitioning", 2.0),
    ("debezium", "implements", "cdc", 2.5),
    ("cdc", "reduces", "latency", 2.0),
    ("opentelemetry", "enables", "tracing", 2.5),
    ("opentelemetry", "enables", "observability", 2.0),
    ("grafana", "visualises", "metrics", 2.0),
    ("grafana", "visualises", "tracing", 1.5),
    ("istio", "provides", "observability", 1.5),
    ("jwt", "implements", "authentication", 2.5),
    ("oauth2", "implements", "authorization", 2.5),
    ("argon2", "secures", "authentication", 2.0),
    ("security", "requires", "encryption", 2.0),
    ("authentication", "enables", "authorization", 2.0),
    ("d3js", "renders", "graph", 2.0),
    ("graph", "shows", "visualisation", 2.0),
    ("force-directed", "is-a", "graph", 1.5),
    ("consolidation", "builds", "semantic-memory", 2.0),
    ("spacy", "extracts", "semantic-memory", 1.5),
    ("incremental-sync", "uses", "chromadb", 1.0),
    ("wikilinks", "used-by", "obsidian", 2.0),
    ("frontmatter", "used-by", "obsidian", 2.0),
    # Decisions ↔ tech
    ("use-chromadb-for-episodic-storage", "chose", "chromadb", 3.0),
    ("rs256-jwt-for-api-auth", "chose", "jwt", 3.0),
    ("argon2id-for-passwords", "chose", "argon2", 3.0),
    ("kafka-for-event-streaming", "chose", "kafka", 3.0),
    ("opentelemetry-for-tracing", "chose", "opentelemetry", 3.0),
    ("d3-for-graph-ui", "chose", "d3js", 3.0),
    ("pkce-oauth2-flow", "chose", "oauth2", 3.0),
    ("sentence-transformers-local-models", "chose", "sentence-transformers", 3.0),
    # People ↔ projects
    ("alice", "owns", "myelin", 2.5),
    ("alice", "contributes-to", "infra", 1.5),
    ("bob", "owns", "auth-service", 2.5),
    ("bob", "contributes-to", "myelin", 1.0),
    ("charlie", "owns", "data-pipeline", 2.5),
    ("diana", "contributes-to", "myelin", 2.0),
    ("diana", "contributes-to", "infra", 2.0),
    ("eve", "contributes-to", "auth-service", 2.0),
    ("eve", "contributes-to", "data-pipeline", 1.5),
    ("frank", "contributes-to", "data-pipeline", 1.5),
    ("frank", "contributes-to", "infra", 2.0),
    # Cross-project dependencies
    ("auth-service", "integrates-with", "myelin", 1.5),
    ("data-pipeline", "integrates-with", "infra", 2.0),
    ("myelin", "integrates-with", "obsidian", 1.5),
    ("myelin", "integrates-with", "github", 1.5),
    ("infra", "runs", "auth-service", 2.0),
    ("infra", "runs", "data-pipeline", 2.0),
    ("infra", "runs", "myelin", 1.5),
    # Memory type concepts ↔ myelin
    ("myelin", "stores", "episodic-memory", 2.0),
    ("myelin", "stores", "semantic-memory", 2.0),
    ("myelin", "stores", "procedural-memory", 2.0),
    ("episodic-memory", "fades-via", "decay", 1.5),
    ("semantic-memory", "strengthened-by", "hebbian-learning", 1.5),
]

print(f"Adding {len(RELS)} relationships...")
for subj, pred, obj, weight in RELS:
    net.add_relationship(subj, pred, obj, weight=weight)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

g = net.get_graph()
print("\nDone!")
print(f"  Graph: {len(g['nodes'])} nodes, {len(g['edges'])} edges")

ids, docs, metas = load_chromadb_raw(DATA_DIR)
_, stats = build_memories_and_stats(ids, docs, metas)
print(f"  Memories: {stats['total']} total")
print(f"  By project: {stats['by_project']}")
print(f"  By type:    {stats['by_type']}")
print("\nRun:  uv run myelin graph --port 8766 --no-open")
