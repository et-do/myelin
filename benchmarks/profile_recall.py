"""Profile recall latency — phase-level breakdown.

Creates a Hippocampus with enough data to exercise the full pipeline
(embedding, search, gist, boosting, cross-encoder, lateral inhibition),
then times each phase.

Usage:
    uv run python -m benchmarks.profile_recall
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from myelin.config import MyelinSettings
from myelin.models import MemoryMetadata
from myelin.recall.reranker import Neocortex
from myelin.store.hippocampus import Hippocampus


def _build_hippocampus(tmp: Path) -> Hippocampus:
    """Create a Hippocampus with Neocortex wired in."""
    s = MyelinSettings(data_dir=tmp)
    s.ensure_dirs()
    reranker = Neocortex(model_name=s.cross_encoder_model)
    return Hippocampus(cfg=s, reranker=reranker)


def _populate(hc: Hippocampus) -> None:
    """Store ~30 memories across several sessions/projects."""
    sessions = [
        {
            "project": "alpha",
            "scope": "chat",
            "memory_type": "episodic",
            "turns": [
                "Alice: We decided to use PostgreSQL for the main database.",
                "Bob: I'll set up the connection pooling with pgBouncer.",
                "Alice: Let's also add Redis for caching hot queries.",
                "Bob: Good idea. I'll benchmark Redis vs Memcached first.",
                "Alice: Our deadline is end of March for the MVP launch.",
            ],
        },
        {
            "project": "alpha",
            "scope": "code",
            "memory_type": "procedural",
            "turns": [
                "Implemented retry logic with exponential backoff for HTTP calls.",
                "Added circuit breaker pattern to prevent cascade failures.",
                "Refactored authentication middleware to support OAuth2 and API keys.",
            ],
        },
        {
            "project": "beta",
            "scope": "chat",
            "memory_type": "episodic",
            "turns": [
                "Carol: The ML pipeline needs GPU support for training.",
                "Dave: We can use AWS SageMaker or run on-prem with our A100s.",
                "Carol: Let's compare costs. Also need to handle data versioning.",
                "Dave: DVC or MLflow for experiment tracking?",
                "Carol: MLflow. It integrates better with our stack.",
                "Dave: I prefer the hosted MLflow on Databricks.",
            ],
        },
        {
            "project": "beta",
            "scope": "docs",
            "memory_type": "semantic",
            "turns": [
                "Architecture decision: event-driven microservices with Kafka.",
                "Service boundaries: auth, billing, notifications, core-api.",
                "Deployment model: Kubernetes with Helm charts, ArgoCD for GitOps.",
            ],
        },
        {
            "project": "gamma",
            "scope": "chat",
            "memory_type": "episodic",
            "turns": [
                "Eve: We need to support i18n for the Japanese market launch.",
                "Frank: I'll add react-intl. Also need RTL support later.",
                "Eve: Performance budget is 3s LCP on mobile.",
                "Frank: That means we need code splitting and lazy loading.",
                "Eve: Let's use Next.js for SSR. It handles both well.",
            ],
        },
    ]

    for i, sess in enumerate(sessions):
        content = "\n".join(sess["turns"])
        meta = MemoryMetadata(
            project=sess["project"],
            scope=sess["scope"],
            memory_type=sess["memory_type"],
            source=f"session-{i}",
        )
        hc.store(content=content, metadata=meta)


def _timed(label: str, fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run fn, return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


def _profile_recall(hc: Hippocampus, query: str, label: str) -> None:
    """Profile a single recall call with phase breakdown."""
    print(f"\n{'=' * 60}")
    print(f"Query: {query!r}")
    print(f"Label: {label}")
    print(f"{'=' * 60}")

    # Full end-to-end recall
    results, total_ms = _timed("total", hc.recall, query)
    print(f"\n  Total recall:  {total_ms:7.1f} ms  ({len(results)} results)")

    # Phase breakdown via isolated calls
    # 1. Embedding
    _, emb_ms = _timed("embed", hc._embedder.encode, query)
    print(f"  Embedding:     {emb_ms:7.1f} ms")

    # 2. ChromaDB search
    embedding = hc._embedder.encode(query).tolist()
    n_retrieve = min(
        hc._settings.default_n_results * hc._settings.recall_over_factor,
        hc._collection.count(),
    )

    def _search() -> Any:
        return hc._collection.query(
            query_embeddings=[embedding],
            n_results=n_retrieve,
            include=["documents", "metadatas", "distances"],
        )

    _, search_ms = _timed("search", _search)
    print(f"  ChromaDB query:{search_ms:7.1f} ms")

    # 3. Gist search
    if hc._summaries.count() > 0:
        gist_k = min(hc._settings.perirhinal_top_k, hc._summaries.count())
        _, gist_ms = _timed("gist", hc._summaries.search, embedding, n_results=gist_k)
        print(f"  Gist search:   {gist_ms:7.1f} ms")

    # 4. Cross-encoder (if wired)
    if hc._reranker is not None and results:
        passages = [r.memory.content for r in results]
        _, ce_ms = _timed("ce", hc._reranker.rerank, query, passages)
        print(f"  Cross-encoder: {ce_ms:7.1f} ms  ({len(passages)} passages)")

    print(f"  {'─' * 40}")
    gist_adj = gist_ms if hc._summaries.count() > 0 else 0
    ce_adj = ce_ms if hc._reranker and results else 0
    phase_sum = emb_ms + search_ms + gist_adj + ce_adj
    overhead = total_ms - phase_sum
    print(f"  Sum of phases: {phase_sum:7.1f} ms")
    print(
        f"  Overhead:      {overhead:7.1f} ms"
        f"  (boosts, merging, planning, lateral inhibition)"
    )


def main() -> None:
    print("Myelin Recall Latency Profiler")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Build + warm up
        print("\nBuilding Hippocampus with Neocortex...")
        t0 = time.perf_counter()
        hc = _build_hippocampus(tmp_path)
        build_ms = (time.perf_counter() - t0) * 1000
        print(f"  Init: {build_ms:.0f} ms (includes model loading)")

        # Populate
        print("\nPopulating with test data...")
        t0 = time.perf_counter()
        _populate(hc)
        pop_ms = (time.perf_counter() - t0) * 1000
        n_chunks = hc._collection.count()
        n_gists = hc._summaries.count()
        print(f"  Store: {pop_ms:.0f} ms ({n_chunks} chunks, {n_gists} gists)")

        # Warm-up recall (first call loads model caches)
        print("\nWarm-up recall...")
        _, warmup_ms = _timed("warmup", hc.recall, "test query")
        print(f"  Warm-up: {warmup_ms:.0f} ms")

        # Profile different query types
        queries = [
            ("What database did we choose?", "simple-fact"),
            ("What did Alice say about the deadline?", "speaker-specific"),
            ("Compare the ML pipeline options", "complex-multi-session"),
            ("What architecture decisions were made?", "broad-topic"),
            ("Tell me about the Japanese market launch", "narrow-topic"),
        ]

        for query, label in queries:
            _profile_recall(hc, query, label)

        # Summary: average over all queries
        print(f"\n{'=' * 60}")
        print("Average over all queries:")
        total_times = []
        for query, _ in queries:
            _, ms = _timed("", hc.recall, query)
            total_times.append(ms)
        avg = sum(total_times) / len(total_times)
        print(f"  Average recall: {avg:.1f} ms")
        print(f"  Min:            {min(total_times):.1f} ms")
        print(f"  Max:            {max(total_times):.1f} ms")


if __name__ == "__main__":
    main()
