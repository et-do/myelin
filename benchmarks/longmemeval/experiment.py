"""Small-scale A/B experiments for candidate features.

Runs a small question set through variants of the recall pipeline and
compares R@5 and per-question hit/miss changes.  Each experiment is
self-contained: store sessions, build any auxiliary structures (e.g.
semantic network), recall with the variant, score.

Usage:
    uv run python -m benchmarks.longmemeval.experiment \
        benchmarks/longmemeval/data/longmemeval_s_test50.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer

from myelin.config import MyelinSettings
from myelin.models import Memory, MemoryMetadata, RecallResult
from myelin.reranker import Neocortex
from myelin.store.chunking import chunk
from myelin.store.consolidation import extract_entities
from myelin.store.entorhinal import extract_keywords
from myelin.store.hippocampus import Hippocampus
from myelin.store.neocortex import SemanticNetwork
from myelin.store.perirhinal import summarise
from myelin.store.prefrontal import classify_memory_type

from .run import (
    _parse_question_date,
    flatten_session,
    load_cache,
)

# ------------------------------------------------------------------
# Scoring helpers
# ------------------------------------------------------------------


def _score_results(
    results: list[dict[str, Any]],
    ground_truth: dict[str, set[str]],
) -> dict[str, Any]:
    """Compute R@5, per-category R@5, and per-question hit/miss."""
    hits = 0
    per_q: dict[str, bool] = {}
    by_type: dict[str, list[bool]] = defaultdict(list)

    for r in results:
        qid = r["question_id"]
        qtype = r.get("question_type", "unknown")
        ranked_sids = [x["session_id"] for x in r.get("ranked", [])]
        # Deduplicate to unique sessions
        seen: set[str] = set()
        unique: list[str] = []
        for sid in ranked_sids:
            if sid not in seen:
                seen.add(sid)
                unique.append(sid)
        hit = bool(set(unique[:5]) & ground_truth.get(qid, set()))
        per_q[qid] = hit
        by_type[qtype].append(hit)
        if hit:
            hits += 1

    total = len(results)
    cat_scores = {t: sum(v) / len(v) for t, v in sorted(by_type.items())}
    return {
        "r_at_5": hits / total if total else 0,
        "hits": hits,
        "total": total,
        "per_question": per_q,
        "per_category": cat_scores,
    }


def _diff_scores(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    label: str,
) -> None:
    """Print comparison between baseline and variant."""
    b_pq = baseline["per_question"]
    v_pq = variant["per_question"]
    gained = [qid for qid, hit in v_pq.items() if hit and not b_pq.get(qid, False)]
    lost = [qid for qid, hit in v_pq.items() if not hit and b_pq.get(qid, True)]

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(
        f"  R@5: {baseline['r_at_5']:.1%} → {variant['r_at_5']:.1%}  "
        f"(Δ {variant['r_at_5'] - baseline['r_at_5']:+.1%})"
    )
    hits_b = f"{baseline['hits']}/{baseline['total']}"
    hits_v = f"{variant['hits']}/{variant['total']}"
    print(f"  Hits: {hits_b} -> {hits_v}")
    if gained:
        print(f"  GAINED ({len(gained)}): {gained}")
    if lost:
        print(f"  LOST   ({len(lost)}):   {lost}")
    if not gained and not lost:
        print("  No per-question changes.")

    # Per-category
    for cat in sorted(
        set(list(baseline["per_category"]) + list(variant["per_category"]))
    ):
        b = baseline["per_category"].get(cat, 0)
        v = variant["per_category"].get(cat, 0)
        delta = v - b
        marker = " ***" if abs(delta) > 0.001 else ""
        print(f"  {cat:30s}  {b:.1%} → {v:.1%}  (Δ {delta:+.1%}){marker}")


# ------------------------------------------------------------------
# Store helper — build a fresh Hippocampus + SemanticNetwork per instance
# ------------------------------------------------------------------


def _store_instance(
    instance: dict[str, Any],
    cfg: MyelinSettings,
    embedder: SentenceTransformer,
    client: Any,
    cache_data: tuple[Any, Any, dict[str, Any]] | None,
    reranker: Neocortex | None = None,
) -> tuple[Hippocampus, SemanticNetwork]:
    """Store all sessions; also build a semantic network via consolidation."""
    hc = Hippocampus(
        cfg=cfg,
        embedder=embedder,
        ephemeral=True,
        client=client,
        reranker=reranker,
    )

    session_ids = instance["haystack_session_ids"]
    dates = instance["haystack_dates"]
    sessions = instance["haystack_sessions"]

    all_texts: list[str] = []  # for consolidation

    if cache_data:
        cache_emb, cache_gist, cache_idx = cache_data
        # Bulk-insert from cache (same as run_instance_cached)
        all_ids: list[str] = []
        all_docs: list[str] = []
        all_embeddings: list[list[float]] = []
        all_metadatas: list[dict[str, Any]] = []
        gist_entries: list[tuple[str, str, list[float]]] = []

        for i, session in enumerate(sessions):
            sid = str(session_ids[i])
            text = flatten_session(session)
            all_texts.append(text)

            session_cache = cache_idx["sessions"].get(sid)
            if session_cache is None:
                metadata = MemoryMetadata(
                    scope=f"session_{sid}",
                    tags=[dates[i]] if i < len(dates) else [],
                )
                hc.store(text, metadata)
                continue

            segments = chunk(
                text,
                max_chars=cfg.chunk_max_chars,
                overlap_chars=cfg.chunk_overlap_chars,
            )
            chunk_start, chunk_end = session_cache["chunk_range"]
            if len(segments) != (chunk_end - chunk_start):
                metadata = MemoryMetadata(
                    scope=f"session_{sid}",
                    tags=[dates[i]] if i < len(dates) else [],
                )
                hc.store(text, metadata)
                continue

            mt = classify_memory_type(text)
            base_meta = MemoryMetadata(
                scope=f"session_{sid}",
                tags=[dates[i]] if i < len(dates) else [],
                memory_type=mt,
            )
            parent_id = uuid4().hex if len(segments) > 1 else None

            for j, seg in enumerate(segments):
                seg_meta = base_meta
                if parent_id:
                    seg_meta = base_meta.model_copy(update={"parent_id": parent_id})
                memory = Memory(content=seg, metadata=seg_meta)
                emb = cache_emb[chunk_start + j].tolist()
                chroma_meta = Hippocampus._build_chroma_meta(memory)
                Hippocampus._attach_ec_coords(chroma_meta, seg)
                Hippocampus._attach_session_date(chroma_meta, base_meta.tags)
                if parent_id:
                    chroma_meta["parent_id"] = parent_id
                else:
                    chroma_meta["parent_id"] = memory.id

                all_ids.append(memory.id)
                all_docs.append(seg)
                all_embeddings.append(emb)
                all_metadatas.append(chroma_meta)

            gist = summarise(text)
            gist_key = parent_id if parent_id else all_ids[-1]
            if gist:
                gist_idx = session_cache["gist_idx"]
                gist_emb = cache_gist[gist_idx].tolist()
                gist_entries.append((gist_key, gist, gist_emb))

        if all_ids:
            batch = 5000
            for start in range(0, len(all_ids), batch):
                end = start + batch
                hc._collection.add(
                    ids=all_ids[start:end],
                    embeddings=all_embeddings[start:end],
                    documents=all_docs[start:end],
                    metadatas=all_metadatas[start:end],
                )
        if gist_entries:
            hc._summaries._collection.upsert(
                ids=[f"summary_{e[0]}" for e in gist_entries],
                embeddings=[e[2] for e in gist_entries],
                documents=[e[1] for e in gist_entries],
                metadatas=[{"parent_id": e[0]} for e in gist_entries],
            )
    else:
        for i, session in enumerate(sessions):
            text = flatten_session(session)
            all_texts.append(text)
            metadata = MemoryMetadata(
                scope=f"session_{sid}",
                tags=[dates[i]] if i < len(dates) else [],
            )
            hc.store(text, metadata)

    # Build semantic network from stored sessions (consolidation)
    import tempfile

    neo_path = Path(tempfile.mktemp(suffix=".db"))
    neo = SemanticNetwork(db_path=neo_path, cfg=cfg)
    for text in all_texts:
        entities = extract_entities(text)
        for e in entities:
            neo.add_entity(e, entity_type="auto")
        for a, b in combinations(entities, 2):
            neo.add_relationship(a, b, predicate="co_occurs", weight=1.0)

    return hc, neo


# ------------------------------------------------------------------
# Experiment: Spreading Activation query expansion
# ------------------------------------------------------------------


def _recall_with_spread(
    hc: Hippocampus,
    neo: SemanticNetwork,
    query: str,
    ref_date: datetime | None,
    n_results: int = 5,
) -> list[RecallResult]:
    """Recall with spreading activation boosting results that
    contain related entities found via the semantic network.

    This tests the neuroscience hypothesis: activating related concepts
    in the semantic network should help surface memories that share
    associations with the query even when direct similarity is low.
    """
    # Standard recall first
    results = hc.recall(query, n_results=n_results * 3, reference_date=ref_date)

    if neo.entity_count() == 0 or not results:
        return results[:n_results]

    # Extract query keywords as seed entities
    query_kws = extract_keywords(query, top_n=5)
    if not query_kws:
        return results[:n_results]

    # Spread activation from query keywords
    related = neo.spread(query_kws, max_depth=2, min_weight=0.3, top_k=10)
    if not related:
        return results[:n_results]

    # Boost results containing related entities
    for r in results:
        content_lower = r.memory.content.lower()
        for entity, act_score in related:
            if entity in content_lower:
                r.score *= 1.0 + 0.15 * min(act_score, 2.0)

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:n_results]


# ------------------------------------------------------------------
# Experiment: Multi-probe recall (query reformulation)
# ------------------------------------------------------------------


def _recall_with_multiprobe(
    hc: Hippocampus,
    neo: SemanticNetwork,
    query: str,
    ref_date: datetime | None,
    n_results: int = 5,
) -> list[RecallResult]:
    """Multi-probe recall: run the query multiple times with different
    angles and merge results.

    Probes:
    1. Original query (baseline)
    2. Keyword-focused probe: top keywords from query
    3. Entity-expanded probe: related entities from semantic network
       appended to query
    """
    # Probe 1: original query
    r1 = hc.recall(query, n_results=n_results * 2, reference_date=ref_date)

    # Probe 2: keyword-focused (just the keywords, no filler)
    query_kws = extract_keywords(query, top_n=8)
    kw_query = " ".join(query_kws) if query_kws else query
    r2 = hc.recall(kw_query, n_results=n_results * 2, reference_date=ref_date)

    # Probe 3: entity-expanded query
    if neo.entity_count() > 0 and query_kws:
        related = neo.spread(query_kws, max_depth=1, min_weight=0.3, top_k=5)
        if related:
            expansion = " ".join(ent for ent, _ in related)
            expanded_query = f"{query} {expansion}"
            r3 = hc.recall(
                expanded_query, n_results=n_results * 2, reference_date=ref_date
            )
        else:
            r3 = []
    else:
        r3 = []

    # Merge: reciprocal rank fusion
    id_score: dict[str, float] = {}
    id_result: dict[str, RecallResult] = {}
    k = 60  # RRF constant

    for probe_results in [r1, r2, r3]:
        for rank, r in enumerate(probe_results):
            mid = r.memory.id
            rrf = 1.0 / (k + rank + 1)
            id_score[mid] = id_score.get(mid, 0.0) + rrf
            if mid not in id_result or r.score > id_result[mid].score:
                id_result[mid] = r

    # Sort by fused RRF score
    merged = sorted(id_score.items(), key=lambda x: x[1], reverse=True)
    final: list[RecallResult] = []
    for mid, fused_score in merged[:n_results]:
        rr = id_result[mid]
        rr.score = fused_score
        final.append(rr)

    return final


# ------------------------------------------------------------------
# Experiment: Combined (spread + multiprobe)
# ------------------------------------------------------------------


def _recall_combined(
    hc: Hippocampus,
    neo: SemanticNetwork,
    query: str,
    ref_date: datetime | None,
    n_results: int = 5,
) -> list[RecallResult]:
    """Combined: multiprobe with spreading activation boost on the merged set."""
    # Run multiprobe to get wide candidate pool
    results = _recall_with_multiprobe(hc, neo, query, ref_date, n_results=n_results * 3)

    if neo.entity_count() == 0 or not results:
        return results[:n_results]

    query_kws = extract_keywords(query, top_n=5)
    related = neo.spread(query_kws, max_depth=2, min_weight=0.3, top_k=10)
    if related:
        for r in results:
            content_lower = r.memory.content.lower()
            for entity, act_score in related:
                if entity in content_lower:
                    r.score *= 1.0 + 0.15 * min(act_score, 2.0)
        results.sort(key=lambda r: r.score, reverse=True)

    return results[:n_results]


# ------------------------------------------------------------------
# Main runner
# ------------------------------------------------------------------


def _extract_ranked(results: list[RecallResult]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for r in results:
        scope = r.memory.metadata.scope or ""
        sid = scope.removeprefix("session_") if scope.startswith("session_") else ""
        ranked.append({"session_id": sid, "score": round(r.score, 4)})
    return ranked


def main(data_file: str) -> None:
    data = json.loads(Path(data_file).read_text())
    gt = {e["question_id"]: set(e["answer_session_ids"]) for e in data}

    cfg = MyelinSettings()
    embedder = SentenceTransformer(cfg.embedding_model)
    shared_client = chromadb.EphemeralClient()

    # Load cross-encoder
    reranker: Neocortex | None = None
    if cfg.neocortex_rerank:
        print("Loading cross-encoder...", flush=True)
        reranker = Neocortex(model_name=cfg.cross_encoder_model)

    # Load cache
    # The cache is keyed to longmemeval_s_cleaned — we need to use
    # that same cache even for our subset.
    cache_path = Path(data_file).parent / "longmemeval_s_cleaned.json"
    cache = load_cache(str(cache_path))
    cache_data = cache if cache else None

    total = len(data)
    print(f"Running experiments on {total} questions...\n", flush=True)

    # Collect results for each variant
    baseline_out: list[dict[str, Any]] = []
    spread_out: list[dict[str, Any]] = []
    multiprobe_out: list[dict[str, Any]] = []
    combined_out: list[dict[str, Any]] = []

    for i, instance in enumerate(data):
        qid = instance["question_id"]
        qtype = instance["question_type"]
        ref_date = _parse_question_date(instance.get("question_date", ""))

        t0 = time.time()

        # Build fresh store + semantic network for this instance
        hc, neo = _store_instance(
            instance, cfg, embedder, shared_client, cache_data, reranker=reranker
        )

        entity_count = neo.entity_count()
        rel_count = neo.relationship_count()

        # --- Baseline (v7 pipeline) ---
        baseline_results = hc.recall(
            instance["question"], n_results=5, reference_date=ref_date
        )
        baseline_ranked = _extract_ranked(baseline_results)
        baseline_out.append(
            {"question_id": qid, "question_type": qtype, "ranked": baseline_ranked}
        )

        # --- Spreading activation ---
        spread_results = _recall_with_spread(
            hc, neo, instance["question"], ref_date, n_results=5
        )
        spread_ranked = _extract_ranked(spread_results)
        spread_out.append(
            {"question_id": qid, "question_type": qtype, "ranked": spread_ranked}
        )

        # --- Multi-probe ---
        mp_results = _recall_with_multiprobe(
            hc, neo, instance["question"], ref_date, n_results=5
        )
        mp_ranked = _extract_ranked(mp_results)
        multiprobe_out.append(
            {"question_id": qid, "question_type": qtype, "ranked": mp_ranked}
        )

        # --- Combined ---
        comb_results = _recall_combined(
            hc, neo, instance["question"], ref_date, n_results=5
        )
        comb_ranked = _extract_ranked(comb_results)
        combined_out.append(
            {"question_id": qid, "question_type": qtype, "ranked": comb_ranked}
        )

        elapsed = time.time() - t0
        print(
            f"[{i + 1}/{total}] {qid} [{qtype}] "
            f"entities={entity_count} rels={rel_count} "
            f"{elapsed:.1f}s",
            flush=True,
        )

        # Clean up
        neo.close()

    # Score all variants
    baseline_score = _score_results(baseline_out, gt)
    spread_score = _score_results(spread_out, gt)
    mp_score = _score_results(multiprobe_out, gt)
    comb_score = _score_results(combined_out, gt)

    print(f"\n{'=' * 60}")
    print("  BASELINE (v7)")
    print(f"{'=' * 60}")
    b_hits = baseline_score["hits"]
    b_total = baseline_score["total"]
    print(f"  R@5: {baseline_score['r_at_5']:.1%}  ({b_hits}/{b_total})")
    for cat, val in baseline_score["per_category"].items():
        print(f"  {cat:30s}  {val:.1%}")

    _diff_scores(baseline_score, spread_score, "SPREADING ACTIVATION")
    _diff_scores(baseline_score, mp_score, "MULTI-PROBE RECALL")
    _diff_scores(baseline_score, comb_score, "COMBINED (spread + multiprobe)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m benchmarks.longmemeval.experiment <data_file>")
        sys.exit(1)
    main(sys.argv[1])
