"""Hippocampus — episodic memory store via ChromaDB vector embeddings.

The core of Myelin. Stores memories as vector embeddings with context
metadata for filtered retrieval. Handles:
- Embedding and storage
- Similarity search with context metadata filtering
- Access tracking (timestamps + counts) for decay and reinforcement
"""

from __future__ import annotations

import logging
import math
import threading
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer

from ..config import MyelinSettings, settings
from ..log import suppress_noisy_loggers
from ..models import Memory, MemoryMetadata, RecallResult
from ..recall.query_planner import plan as plan_query
from ..recall.time_cells import (
    has_relative_temporal_expression,
    parse_session_date,
    parse_temporal_reference,
    recency_boost,
)
from ..recall.reranker import Neocortex
from .amygdala import passes_gate
from .chunking import chunk
from .entorhinal import (
    assign_region,
    extract_keywords,
    extract_speakers,
    topic_overlap,
)
from .neocortex import SemanticNetwork
from .perirhinal import SummaryIndex, summarise
from .prefrontal import classify_memory_type

suppress_noisy_loggers()

logger = logging.getLogger(__name__)


def _matches_where(meta: dict[str, Any], where: dict[str, Any]) -> bool:
    """Check if a metadata dict satisfies a ChromaDB-style where clause.

    Supports simple equality and $and clauses (the only forms we build
    via _build_where).  Used to filter direct-ID gist injection results
    that bypass ChromaDB's where parameter.
    """
    if "$and" in where:
        return all(_matches_where(meta, clause) for clause in where["$and"])
    for key, val in where.items():
        if meta.get(key) != val:
            return False
    return True


class Hippocampus:
    """Vector-based episodic memory store backed by ChromaDB."""

    def __init__(
        self,
        cfg: MyelinSettings | None = None,
        embedder: SentenceTransformer | None = None,
        *,
        ephemeral: bool = False,
        client: Any | None = None,
        reranker: Neocortex | None = None,
        semantic_network: SemanticNetwork | None = None,
    ) -> None:
        self._settings = cfg or settings
        if client is not None:
            self._client = client
            coll_name = f"memories_{uuid4().hex[:12]}"
        elif ephemeral:
            self._client = chromadb.EphemeralClient()
            # Each ephemeral instance gets its own collection to avoid
            # cross-contamination — EphemeralClient shares in-memory state.
            coll_name = f"memories_{uuid4().hex[:12]}"
        else:
            self._settings.ensure_dirs()
            self._client = chromadb.PersistentClient(
                path=str(self._settings.data_dir / "chroma"),
            )
            coll_name = "memories"
        self._collection = self._client.get_or_create_collection(
            name=coll_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self._settings.embedding_model,
            },
        )
        self._embedder = embedder or SentenceTransformer(self._settings.embedding_model)
        # Perirhinal cortex — gist summaries for two-stage recall
        if coll_name == "memories":
            summary_coll = "summaries"
        else:
            summary_coll = coll_name.replace("memories", "summaries")
        self._summaries = SummaryIndex(
            self._client,
            self._embedder,
            collection_name=summary_coll,
        )
        # Neocortex — optional cross-encoder for deliberative re-ranking
        self._reranker = reranker
        # Semantic network — optional entity graph for spreading activation
        self._semantic_network = semantic_network
        # Concurrency safety — RLock allows re-entrant calls from same thread
        self._lock = threading.RLock()

    def warm_up(self) -> None:
        """Pre-warm embedding and reranker models with dummy inference."""
        self._embedder.encode("warmup")
        if self._reranker is not None:
            self._reranker.rerank("warmup", ["warmup"])

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        metadata: MemoryMetadata | None = None,
        overwrite: bool = False,
    ) -> Memory | None:
        """Embed and store a memory, chunking long content automatically.

        Long content is split into focused segments (pattern separation)
        so each embedding stays within the model's attention window.
        Returns the first chunk's Memory, or None if the gate rejects it.

        When *overwrite* is True and content is a near-duplicate of an
        existing memory, the old memory (all its chunks and gists) is
        deleted before the new content is stored.  The returned Memory
        will have ``replaced_id`` set to the parent_id of the deleted
        memory so callers can surface the replacement to users.
        """
        with self._lock:
            return self._store_impl(content, metadata, overwrite=overwrite)

    def _store_impl(
        self,
        content: str,
        metadata: MemoryMetadata | None = None,
        overwrite: bool = False,
    ) -> Memory | None:
        # Gate check on full content (dedup + length)
        embedding = self._embedder.encode(content).tolist()

        existing_sim: list[float] = []
        nearest_parent_id: str | None = None
        if self._collection.count() > 0:
            hits = self._collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["distances", "metadatas"],
            )
            if hits["distances"] and hits["distances"][0]:
                existing_sim = [1.0 - d for d in hits["distances"][0]]
            if overwrite and hits["metadatas"] and hits["metadatas"][0]:
                nearest_meta = hits["metadatas"][0][0]
                nearest_parent_id = str(nearest_meta.get("parent_id", "")) or None

        ok, reason = passes_gate(content, existing_sim, cfg=self._settings)
        replaced_id: str | None = None
        if not ok:
            if overwrite and reason.startswith("near-duplicate") and nearest_parent_id:
                # Remove the stale memory so the replacement can be stored.
                self._delete_memory_by_parent_id(nearest_parent_id)
                replaced_id = nearest_parent_id
            else:
                return None

        meta = metadata or MemoryMetadata()

        # PFC schema-consistent encoding: auto-classify if no type provided
        if meta.memory_type is None:
            mt = classify_memory_type(content)
            meta = meta.model_copy(update={"memory_type": mt})

        # Chunk if content exceeds the model's effective window
        segments = chunk(
            content,
            max_chars=self._settings.chunk_max_chars,
            overlap_chars=self._settings.chunk_overlap_chars,
        )

        if len(segments) <= 1:
            # Short content — store as single memory (original path)
            memory = Memory(content=content, metadata=meta)
            chroma_meta = self._build_chroma_meta(memory)
            self._attach_ec_coords(chroma_meta, content)
            self._attach_session_date(chroma_meta, meta.tags)
            # Set parent_id = memory.id so gist-guided recall can
            # filter on parent_id uniformly (single + multi chunk).
            chroma_meta["parent_id"] = memory.id
            self._collection.add(
                ids=[memory.id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[chroma_meta],
            )
            # Perirhinal gist for single-chunk memories
            try:
                gist = summarise(content)
                if gist:
                    self._summaries.add(memory.id, gist)
            except Exception:
                logger.warning("Gist indexing failed for %s", memory.id, exc_info=True)
            if replaced_id:
                memory.replaced_id = replaced_id
            return memory

        # Long content — store each chunk with shared parent_id
        parent_id = uuid4().hex

        # Batch-encode all chunks at once (much faster than one-at-a-time)
        seg_embeddings = self._embedder.encode(segments)

        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        first_memory: Memory | None = None

        for seg, seg_emb in zip(segments, seg_embeddings):
            chunk_meta = meta.model_copy(update={"parent_id": parent_id})
            memory = Memory(content=seg, metadata=chunk_meta)

            ids.append(memory.id)
            documents.append(seg)
            embeddings.append(seg_emb.tolist())
            chroma_m = self._build_chroma_meta(memory)
            self._attach_ec_coords(chroma_m, seg)
            self._attach_session_date(chroma_m, meta.tags)
            metadatas.append(chroma_m)

            if first_memory is None:
                first_memory = memory

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # Perirhinal gist — session-level summary under parent_id.
        # Wrapped in try/except so chunks remain stored even if gist
        # indexing fails (partial data is better than rollback).
        try:
            gist = summarise(content)
            if gist:
                self._summaries.add(parent_id, gist)

            # Per-chunk gists: each chunk gets its own gist in the summary
            # index.  This gives the perirhinal pathway finer-grained
            # coverage — a specific fact buried in one chunk can be
            # matched even if the session-level gist misses it.
            for chunk_id, seg in zip(ids, segments):
                chunk_gist = summarise(seg)
                if chunk_gist:
                    self._summaries.add(chunk_id, chunk_gist)
        except Exception:
            logger.warning(
                "Gist indexing failed for parent %s",
                parent_id,
                exc_info=True,
            )

        if first_memory is not None and replaced_id:
            first_memory.replaced_id = replaced_id
        return first_memory

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        n_results: int | None = None,
        project: str | None = None,
        language: str | None = None,
        scope: str | None = None,
        memory_type: str | None = None,
        agent_id: str | None = None,
        auto_filter: bool = True,
        reference_date: datetime | None = None,
    ) -> list[RecallResult]:
        """Retrieve memories by semantic similarity with multi-signal re-ranking.

        Retrieval pipeline:
        1. PFC query planning — auto-infer type/scope filters
        2. Perirhinal gist search — wide session-level familiarity scores
        3. Dual-pathway chunk retrieval — filtered + unfiltered, merged
        4. Entorhinal re-ranking — topic keyword overlap boost
        5. Perirhinal re-ranking — gist-matched session boost
        6. Sort by composite score
        7. Gist retrieval pathway — inject sessions from gist matches
        8. Cross-encoder score fusion — CE evaluates full pool
        9. Time-cell boost — temporal reference matching (post-CE)
        10. Lateral inhibition — enforce diversity on ranked list
        11. Multi-probe — CE-score pool merge of keyword + entity probes
        12. Spreading activation — entity-graph post-retrieval boost
        13. Trim to n
        """
        with self._lock:
            return self._recall_impl(
                query,
                n_results=n_results,
                project=project,
                language=language,
                scope=scope,
                memory_type=memory_type,
                agent_id=agent_id,
                auto_filter=auto_filter,
                reference_date=reference_date,
            )

    def _recall_impl(
        self,
        query: str,
        *,
        n_results: int | None = None,
        project: str | None = None,
        language: str | None = None,
        scope: str | None = None,
        memory_type: str | None = None,
        agent_id: str | None = None,
        auto_filter: bool = True,
        reference_date: datetime | None = None,
    ) -> list[RecallResult]:
        if self._collection.count() == 0:
            return []

        # Auto-infer reference_date: when no explicit date is provided,
        # check if the query contains a relative temporal expression
        # (e.g. "3 days ago", "last Tuesday").  If so, use "now" as the
        # reference date so that time-cell and recency features activate.
        # Named months ("in June") are excluded — they need calendar-year
        # context that the caller must supply.
        if reference_date is None and has_relative_temporal_expression(query):
            reference_date = datetime.now(UTC)

        n = min(
            n_results or self._settings.default_n_results,
            self._collection.count(),
        )

        def _probe(q: str, n_res: int) -> list[RecallResult]:
            return self._recall_single(
                q,
                n_results=n_res,
                project=project,
                language=language,
                scope=scope,
                memory_type=memory_type,
                agent_id=agent_id,
                auto_filter=auto_filter,
                reference_date=reference_date,
            )

        # Multi-probe recall: run multiple query variants and merge.
        # Each probe runs the full single-query pipeline including
        # cross-encoder re-ranking (EC boost, perirhinal, time cells, CE).
        # We use the probes to assemble a wider candidate pool, but keep
        # each result's cross-encoder score for ranking — RRF would
        # flatten the score distribution and destroy ranking precision.
        if self._settings.multiprobe:
            pool_n = n * 3  # match non-multiprobe retrieval depth

            # Probe 1: original query (full pipeline)
            r1 = _probe(query, pool_n)

            # Probe 2: keyword-focused (just extracted keywords)
            query_kws = extract_keywords(query, top_n=8)
            kw_query = " ".join(query_kws) if query_kws else query
            r2 = _probe(kw_query, pool_n)

            # Probe 3: entity-expanded query (if semantic network available)
            r3: list[RecallResult] = []
            neo = self._semantic_network
            if neo is not None and self._settings.spreading_activation and query_kws:
                try:
                    related = neo.spread(
                        query_kws,
                        max_depth=1,
                        min_weight=0.3,
                        top_k=5,
                    )
                    if related:
                        expansion = " ".join(ent for ent, _ in related)
                        expanded_query = f"{query} {expansion}"
                        r3 = _probe(expanded_query, pool_n)
                except Exception:
                    logger.warning("Spreading activation probe failed", exc_info=True)

            # Merge: collect unique candidates across all probes,
            # keeping the highest per-probe score as a composite signal.
            seen: dict[str, RecallResult] = {}
            for probe_results in [r1, r2, r3]:
                for r in probe_results:
                    mid = r.memory.id
                    if mid not in seen or r.score > seen[mid].score:
                        seen[mid] = r
            recall_results = list(seen.values())

            logger.debug(
                "multi-probe merge: p1=%d p2=%d p3=%d unique=%d",
                len(r1),
                len(r2),
                len(r3),
                len(recall_results),
            )

            # Unified CE re-ranking: probes 2+3 ran CE against keyword
            # or entity-expanded queries, producing scores that aren't
            # comparable to probe 1's.  Re-score the entire merged pool
            # with CE against the *original* query so every candidate
            # has a calibrated relevance score.
            if (
                self._reranker is not None
                and self._settings.neocortex_rerank
                and len(recall_results) > 1
            ):
                passages = [r.memory.content for r in recall_results]
                ce_scores = self._reranker.rerank(query, passages)
                ce_min = min(ce_scores)
                ce_max = max(ce_scores)
                ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
                # Use existing merged scores as composite bi-encoder
                # signal (preserves EC, perirhinal, temporal boosts).
                # Lower alpha than per-probe blending: the bi component
                # already contains per-probe CE, so full alpha would
                # make the effective CE weight ~84%.  Using alpha/2
                # keeps the re-calibration benefit while respecting
                # per-probe scoring signals.
                bi_scores = [r.score for r in recall_results]
                bi_min = min(bi_scores)
                bi_max = max(bi_scores)
                bi_range = bi_max - bi_min if bi_max > bi_min else 1.0
                merge_alpha = self._settings.neocortex_weight * 0.5
                for r, ce_raw, bi_raw in zip(recall_results, ce_scores, bi_scores):
                    ce_norm = (ce_raw - ce_min) / ce_range if ce_max > ce_min else 0.5
                    bi_norm = (bi_raw - bi_min) / bi_range if bi_max > bi_min else 0.5
                    r.score = merge_alpha * ce_norm + (1.0 - merge_alpha) * bi_norm

            # Re-apply temporal boost after CE re-scoring so the time-
            # cell signal survives the second normalization pass (same
            # rationale as applying it after per-probe CE in
            # _recall_single).
            if reference_date is not None:
                date_range = parse_temporal_reference(query, reference_date)
                if date_range:
                    t_start, t_end = date_range
                    for r in recall_results:
                        for tag in r.memory.metadata.tags:
                            session_dt = parse_session_date(tag)
                            if session_dt and t_start <= session_dt <= t_end:
                                r.score += self._settings.temporal_boost
                                break

            recall_results.sort(key=lambda r: r.score, reverse=True)
        else:
            recall_results = _probe(query, n * 3)

        # Session evidence aggregation: when multiple chunks from the
        # same session appear in the candidate pool, the session has
        # broader relevance and is more likely to be a true match.
        # Boost the top chunk per session proportionally to the number
        # of contributing chunks (logarithmic to avoid over-weighting).
        agg_boost = self._settings.session_aggregation_boost
        if agg_boost > 0 and len(recall_results) > 1:
            session_chunks: dict[str | None, list[RecallResult]] = {}
            for r in recall_results:
                key = r.memory.metadata.scope or r.memory.metadata.parent_id
                session_chunks.setdefault(key, []).append(r)
            for chunks in session_chunks.values():
                if len(chunks) > 1:
                    chunks.sort(key=lambda r: r.score, reverse=True)
                    bonus = math.log1p(len(chunks) - 1) * agg_boost
                    chunks[0].score *= 1.0 + bonus
            recall_results.sort(key=lambda r: r.score, reverse=True)

        # Spreading activation: boost results containing entities related
        # to the query via the semantic network (Collins & Loftus 1975).
        neo = self._semantic_network
        if neo is not None and self._settings.spreading_activation and recall_results:
            query_kws = extract_keywords(query, top_n=5)
            if query_kws:
                try:
                    related = neo.spread(
                        query_kws,
                        max_depth=self._settings.spreading_max_depth,
                        min_weight=0.3,
                        top_k=self._settings.spreading_top_k,
                    )
                    if related:
                        boost = self._settings.spreading_boost
                        for r in recall_results:
                            content_lower = r.memory.content.lower()
                            for entity, act_score in related:
                                if entity in content_lower:
                                    r.score *= 1.0 + boost * min(act_score, 2.0)
                        recall_results.sort(key=lambda r: r.score, reverse=True)
                except Exception:
                    logger.warning("Spreading activation boost failed", exc_info=True)

        # Final lateral inhibition: enforce session diversity on the
        # merged pool.  Per-probe lateral_k was already applied, but
        # merging can reunite chunks from the same session across probes.
        lat_k = self._settings.lateral_k
        if lat_k > 0:
            scope_counts: dict[str | None, int] = {}
            inhibited: list[RecallResult] = []
            for r in recall_results:
                key = r.memory.metadata.scope or r.memory.metadata.parent_id
                count = scope_counts.get(key, 0)
                if count < lat_k:
                    inhibited.append(r)
                    scope_counts[key] = count + 1
            recall_results = inhibited

        recall_results = recall_results[:n]

        # Update access tracking for recalled memories
        self._bump_access([r.memory for r in recall_results])

        logger.debug("recall returning %d results", len(recall_results))

        return recall_results

    def _recall_single(
        self,
        query: str,
        *,
        n_results: int | None = None,
        project: str | None = None,
        language: str | None = None,
        scope: str | None = None,
        memory_type: str | None = None,
        agent_id: str | None = None,
        auto_filter: bool = True,
        reference_date: datetime | None = None,
    ) -> list[RecallResult]:
        """Single-probe retrieval pipeline (steps 1-10 of recall)."""
        if self._collection.count() == 0:
            return []

        # PFC inhibitory gating — auto-infer filters from query
        auto_memory_type: str | None = None
        auto_scope: str | None = None
        if auto_filter:
            qplan = plan_query(query)
            if memory_type is None and qplan.memory_type is not None:
                auto_memory_type = qplan.memory_type
            if scope is None and qplan.scope_hint is not None:
                auto_scope = qplan.scope_hint

        n = min(
            n_results or self._settings.default_n_results,
            self._collection.count(),
        )
        embedding = self._embedder.encode(query).tolist()

        # Entorhinal cortex — extract query context coordinates
        query_kws = extract_keywords(query, top_n=5)

        # Over-retrieve for re-ranking headroom (configurable factor)
        n_retrieve = min(
            n * self._settings.recall_over_factor,
            self._collection.count(),
        )

        # Perirhinal cortex — wide gist search for session familiarity
        summary_parent_scores: dict[str, float] = {}
        if self._summaries.count() > 0:
            gist_k = min(
                self._settings.perirhinal_top_k,
                self._summaries.count(),
            )
            summary_hits = self._summaries.search(embedding, n_results=gist_k)
            summary_parent_scores = dict(summary_hits)

        # Base where clause from explicit parameters only
        base_where = self._build_where(
            project=project,
            language=language,
            scope=scope,
            memory_type=memory_type,
            agent_id=agent_id,
        )

        # Dual-pathway retrieval: when auto-filters are active, run both
        # filtered and unfiltered queries and merge the best results.
        # This prevents auto-inferred filters from excluding
        # high-relevance documents that don't match the predicted type.
        has_auto = bool(auto_memory_type or auto_scope)
        if has_auto:
            auto_where = self._build_where(
                project=project,
                language=language,
                scope=scope or auto_scope,
                memory_type=memory_type or auto_memory_type,
                agent_id=agent_id,
            )
            filtered = self._collection.query(
                query_embeddings=[embedding],
                n_results=n_retrieve,
                where=auto_where if auto_where else None,
                include=["documents", "metadatas", "distances"],
            )
            unfiltered = self._collection.query(
                query_embeddings=[embedding],
                n_results=n_retrieve,
                where=base_where if base_where else None,
                include=["documents", "metadatas", "distances"],
            )
            results: Any = self._merge_results(filtered, unfiltered, n_retrieve)
        else:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=n_retrieve,
                where=base_where if base_where else None,
                include=["documents", "metadatas", "distances"],
            )

        # Temporal dual-pathway: when a temporal reference is detected,
        # run a second query constrained to sessions in the date range.
        # This gives the correct session a chance to be found in a much
        # smaller haystack — like hippocampal time cells narrowing the
        # search to a specific temporal context.
        if reference_date is not None:
            date_range = parse_temporal_reference(query, reference_date)
            if date_range:
                t_start, t_end = date_range
                start_ord = t_start.toordinal()
                end_ord = t_end.toordinal()
                temporal_where: dict[str, Any] = {
                    "$and": [
                        {"session_date": {"$gte": start_ord}},
                        {"session_date": {"$lte": end_ord}},
                    ]
                }
                try:
                    temporal_results = self._collection.query(
                        query_embeddings=[embedding],
                        n_results=n_retrieve,
                        where=temporal_where,
                        include=["documents", "metadatas", "distances"],
                    )
                    results = self._merge_results(results, temporal_results, n_retrieve)
                except Exception:
                    logger.warning("Temporal recall filter failed", exc_info=True)

        _docs = results["documents"]
        _metas = results["metadatas"]
        _dists = results["distances"]

        if _docs is None or _metas is None or _dists is None:
            return []

        result_docs = _docs[0]
        result_metas = _metas[0]
        result_dists = _dists[0]

        # Reference ordinal for recency gradient (0 = disabled)
        ref_ordinal = reference_date.toordinal() if reference_date is not None else 0

        recall_results: list[RecallResult] = []
        # Source monitoring: detect known speakers mentioned in the query.
        # Collect all unique speakers from retrieved chunks, then check
        # which ones the query references.
        query_speakers: list[str] = []
        if self._settings.speaker_boost > 0:
            all_speakers: set[str] = set()
            for m in result_metas:
                sp_raw = str(m.get("ec_speakers", ""))
                if sp_raw:
                    all_speakers.update(
                        s.strip() for s in sp_raw.split(",") if s.strip()
                    )
            if all_speakers:
                from .entorhinal import detect_query_speakers

                query_speakers = detect_query_speakers(query, sorted(all_speakers))

        for i, doc in enumerate(result_docs):
            chroma_meta = dict(result_metas[i])
            distance = result_dists[i]
            score = 1.0 - distance  # cosine distance → similarity

            # Entorhinal re-ranking: boost by topic keyword overlap
            if query_kws:
                mem_topics_raw = str(chroma_meta.get("ec_topics", ""))
                mem_kws = mem_topics_raw.split(",") if mem_topics_raw else []
                overlap = topic_overlap(query_kws, mem_kws)
                if overlap > 0:
                    score *= 1.0 + self._settings.entorhinal_boost * overlap

            # Source monitoring boost: when the query mentions a speaker,
            # prefer chunks where that speaker is present (the "who"
            # coordinate in the EC context system).
            if query_speakers:
                mem_speakers_raw = str(chroma_meta.get("ec_speakers", ""))
                if mem_speakers_raw:
                    mem_speakers = {
                        s.strip().lower() for s in mem_speakers_raw.split(",")
                    }
                    for qs in query_speakers:
                        if qs.lower() in mem_speakers:
                            score *= 1.0 + self._settings.speaker_boost
                            break

            # Perirhinal gist boost: session familiarity signal
            if summary_parent_scores:
                pid = str(chroma_meta.get("parent_id", ""))
                mid = results["ids"][0][i]
                s_score = summary_parent_scores.get(pid) or summary_parent_scores.get(
                    mid, 0.0
                )
                if s_score > 0:
                    score *= 1.0 + self._settings.perirhinal_boost * s_score

            # Soft recency gradient: gentle boost for newer memories.
            # Uses the session_date ordinal if available; this implements
            # the temporal context model's recency bias as a continuous
            # signal rather than a step function (time cells only fire
            # on explicit temporal queries).
            half_life = self._settings.recency_half_life_days
            if half_life > 0 and ref_ordinal > 0:
                sess_ord = int(chroma_meta.get("session_date") or 0)
                if sess_ord > 0:
                    r_boost = recency_boost(sess_ord, ref_ordinal, half_life)
                    # Apply as gentle multiplicative factor: max ~10% boost
                    # for very recent memories tapering to ~0% for old ones
                    score *= 1.0 + 0.1 * r_boost

            memory = Memory(
                id=results["ids"][0][i],
                content=doc,
                metadata=MemoryMetadata.from_chroma(chroma_meta),
                access_count=int(chroma_meta.get("access_count") or 0),
                created_at=datetime.fromisoformat(str(chroma_meta["created_at"])),
                last_accessed=datetime.fromisoformat(str(chroma_meta["last_accessed"])),
            )
            recall_results.append(RecallResult(memory=memory, score=score))

        # Sort by re-ranked score (before CE, which re-sorts)
        recall_results.sort(key=lambda r: r.score, reverse=True)

        # Gist retrieval pathway: for each high-scoring gist session not
        # already represented in the candidate pool, pull the best chunk
        # from that session and inject it.  This mirrors perirhinal
        # familiarity — recognising the gist of an episode can trigger
        # retrieval of its constituent details.
        #
        # Batched: single $or query for all parent_ids, then single
        # ids=[...] fallback for per-chunk gists. Avoids N serial lookups.
        if summary_parent_scores:
            existing_pids = {
                str(r.memory.metadata.parent_id or r.memory.id) for r in recall_results
            }
            existing_ids = {r.memory.id for r in recall_results}

            # Filter to pids not already represented
            candidate_pids = {
                pid
                for pid in summary_parent_scores
                if pid not in existing_pids and pid not in existing_ids
            }

            if candidate_pids:
                self._inject_gist_results(
                    candidate_pids,
                    summary_parent_scores,
                    existing_ids,
                    base_where,
                    embedding,
                    recall_results,
                )

        # Neocortex deliberative re-ranking: cross-encoder evaluates
        # each (query, passage) pair jointly via cross-attention on
        # the full candidate pool BEFORE lateral inhibition.  Running
        # CE first lets it score all candidates; diversity enforcement
        # then operates on cross-attention-ranked results rather than
        # suppressing candidates the CE might have promoted.
        if (
            self._reranker is not None
            and self._settings.neocortex_rerank
            and recall_results
        ):
            passages = [r.memory.content for r in recall_results]
            ce_scores = self._reranker.rerank(query, passages)
            # Normalise CE scores to [0, 1] for blending
            ce_min = min(ce_scores)
            ce_max = max(ce_scores)
            ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
            # Normalise bi-encoder composite scores to [0, 1]
            bi_scores = [r.score for r in recall_results]
            bi_min = min(bi_scores)
            bi_max = max(bi_scores)
            bi_range = bi_max - bi_min if bi_max > bi_min else 1.0
            alpha = self._settings.neocortex_weight
            for r, ce_raw, bi_raw in zip(recall_results, ce_scores, bi_scores):
                ce_norm = (ce_raw - ce_min) / ce_range if ce_max > ce_min else 0.5
                bi_norm = (bi_raw - bi_min) / bi_range if bi_max > bi_min else 0.5
                r.score = alpha * ce_norm + (1.0 - alpha) * bi_norm
            recall_results.sort(key=lambda r: r.score, reverse=True)

        # Time-cell boost: temporal context matching (additive).
        # Applied AFTER CE blending so the temporal signal survives
        # cross-encoder normalization — even when CE disagrees about
        # relevance, the temporal match can override.
        if reference_date is not None:
            date_range = parse_temporal_reference(query, reference_date)
            if date_range:
                t_start, t_end = date_range
                for r in recall_results:
                    for tag in r.memory.metadata.tags:
                        session_dt = parse_session_date(tag)
                        if session_dt and t_start <= session_dt <= t_end:
                            r.score += self._settings.temporal_boost
                            break
                recall_results.sort(key=lambda r: r.score, reverse=True)

        # Lateral inhibition: enforce session diversity AFTER cross-encoder
        # scoring.  This ensures the CE evaluates the full candidate pool;
        # diversity constraints then prune the CE-ranked list so different
        # sessions get representation in the final results.
        # Group by scope when set; fall back to parent_id so that
        # production use without explicit scopes still gets diversity
        # across storage sessions instead of collapsing to None.
        lat_k = self._settings.lateral_k
        if lat_k > 0:
            scope_counts: dict[str | None, int] = {}
            inhibited: list[RecallResult] = []
            for r in recall_results:
                key = r.memory.metadata.scope or r.memory.metadata.parent_id
                count = scope_counts.get(key, 0)
                if count < lat_k:
                    inhibited.append(r)
                    scope_counts[key] = count + 1
            recall_results = inhibited

        return recall_results[:n]

    # ------------------------------------------------------------------
    # Forget
    # ------------------------------------------------------------------

    def _delete_memory_by_parent_id(self, parent_id: str) -> None:
        """Delete all chunks and gists belonging to parent_id.

        Caller must hold self._lock.  Used by the upsert path to purge the
        old memory before writing the replacement.
        """
        # Retrieve all chunk IDs that share this parent_id
        result = self._collection.get(
            where={"parent_id": parent_id},
            include=[],  # IDs only
        )
        chunk_ids: list[str] = result["ids"] if result["ids"] else []

        if chunk_ids:
            self._collection.delete(ids=chunk_ids)

        # Delete the session-level gist and any per-chunk gists
        gist_ids = [f"summary_{parent_id}"] + [
            f"summary_{cid}" for cid in chunk_ids if cid != parent_id
        ]
        try:
            self._summaries._collection.delete(ids=gist_ids)
        except Exception:
            pass  # Gist deletion is best-effort; chunks are already gone

    def forget(self, memory_id: str) -> bool:
        """Remove a memory by ID."""
        with self._lock:
            try:
                self._collection.delete(ids=[memory_id])
                return True
            except Exception:
                logger.warning("Failed to forget memory %s", memory_id, exc_info=True)
                return False

    def forget_batch(self, memory_ids: list[str]) -> int:
        """Remove multiple memories. Returns count deleted."""
        if not memory_ids:
            return 0
        with self._lock:
            self._collection.delete(ids=memory_ids)
            return len(memory_ids)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        with self._lock:
            return int(self._collection.count())

    def check_integrity(self) -> dict[str, Any]:
        """Verify data consistency between memories and summaries."""
        with self._lock:
            mem_count = int(self._collection.count())
            summary_count = int(self._summaries._collection.count())
        return {
            "memory_count": mem_count,
            "summary_count": summary_count,
            "consistent": mem_count >= summary_count,
        }

    def get_all_metadata(self) -> list[dict[str, Any]]:
        """Metadata for all memories (used by decay sweep)."""
        with self._lock:
            result = self._collection.get(include=["metadatas"])
            _metas = result["metadatas"]
            if _metas is None:
                return []
            return [{"id": id_, **meta} for id_, meta in zip(result["ids"], _metas)]

    def get_all_content(self) -> list[dict[str, str]]:
        """IDs and content for all memories (used by consolidation replay)."""
        with self._lock:
            result = self._collection.get(include=["documents"])
            _docs = result["documents"]
            if _docs is None:
                return []
            return [
                {"id": id_, "content": doc}
                for id_, doc in zip(result["ids"], _docs)
                if doc
            ]

    def get_by_ids(self, ids: list[str]) -> list[Memory]:
        """Retrieve specific memories by ID (used by thalamic buffer)."""
        if not ids:
            return []
        with self._lock:
            result = self._collection.get(ids=ids, include=["documents", "metadatas"])
            _docs = result["documents"]
            _metas = result["metadatas"]
            if _docs is None or _metas is None:
                return []
            memories: list[Memory] = []
            for i, mid in enumerate(result["ids"]):
                chroma_meta = dict(_metas[i])
                memories.append(
                    Memory(
                        id=mid,
                        content=_docs[i],
                        metadata=MemoryMetadata.from_chroma(chroma_meta),
                        access_count=int(chroma_meta.get("access_count") or 0),
                        created_at=datetime.fromisoformat(
                            str(chroma_meta["created_at"])
                        ),
                        last_accessed=datetime.fromisoformat(
                            str(chroma_meta["last_accessed"])
                        ),
                    )
                )
            return memories

    # ------------------------------------------------------------------
    # Gist retrieval (batched)
    # ------------------------------------------------------------------

    def _inject_gist_results(
        self,
        candidate_pids: set[str],
        gist_scores: dict[str, float],
        existing_ids: set[str],
        base_where: dict[str, Any] | None,
        embedding: list[float],
        recall_results: list[RecallResult],
    ) -> None:
        """Batch-fetch chunks for gist-matched sessions and inject best chunks.

        Replaces the old N-serial-lookup loop with at most 2 ChromaDB calls:
        one $or query by parent_id, one fallback ids=[...] for per-chunk gists.
        """
        pid_list = list(candidate_pids)
        includes: list[str] = ["documents", "metadatas", "embeddings"]

        # --- Batch 1: session-level gists (parent_id filter) ---
        or_clauses = [{"parent_id": pid} for pid in pid_list]
        parent_where: dict[str, Any] = (
            {"$or": or_clauses} if len(or_clauses) > 1 else or_clauses[0]
        )
        if base_where:
            parent_where = {"$and": [parent_where, base_where]}

        try:
            batch_results = self._collection.get(
                where=parent_where,
                include=includes,
            )
        except Exception:
            logger.debug(
                "Batch gist parent_id lookup failed",
                exc_info=True,
            )
            batch_results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": [],
            }

        # Group batch results by parent_id
        found_pids: set[str] = set()
        pid_chunks: dict[str, list[int]] = {}
        b_ids = batch_results["ids"]
        b_metas = batch_results["metadatas"] or []
        for i, meta in enumerate(b_metas):
            pid = str(meta.get("parent_id", ""))
            if pid in candidate_pids:
                found_pids.add(pid)
                pid_chunks.setdefault(pid, []).append(i)

        # --- Batch 2: per-chunk gist fallback for missing pids ---
        missing_pids = [p for p in pid_list if p not in found_pids]
        fallback_results: dict[str, Any] | None = None
        if missing_pids:
            try:
                fallback_results = self._collection.get(
                    ids=missing_pids,
                    include=includes,
                )
            except Exception:
                logger.debug(
                    "Batch gist ID fallback failed",
                    exc_info=True,
                )

        # --- Inject best chunk per pid ---
        b_docs = (
            batch_results["documents"] if batch_results["documents"] is not None else []
        )
        b_embs = (
            batch_results["embeddings"]
            if batch_results["embeddings"] is not None
            else []
        )

        for pid in pid_list:
            g_score = gist_scores.get(pid, 0.0)
            if pid in pid_chunks:
                # Session-level: pick closest chunk to query
                indices = pid_chunks[pid]
                best = self._pick_best_chunk(
                    indices,
                    b_ids,
                    b_docs,
                    b_embs,
                    b_metas,
                    embedding,
                    existing_ids,
                )
            elif fallback_results and pid in (fallback_results["ids"] or []):
                # Per-chunk gist: direct ID match
                fb_idx = fallback_results["ids"].index(pid)
                if base_where:
                    fb_meta = fallback_results["metadatas"]
                    assert fb_meta is not None
                    if not _matches_where(dict(fb_meta[fb_idx]), base_where):
                        continue
                fb_docs = (
                    fallback_results["documents"]
                    if fallback_results["documents"] is not None
                    else []
                )
                fb_embs = (
                    fallback_results["embeddings"]
                    if fallback_results["embeddings"] is not None
                    else []
                )
                fb_metas = (
                    fallback_results["metadatas"]
                    if fallback_results["metadatas"] is not None
                    else []
                )
                best = self._pick_best_chunk(
                    [fb_idx],
                    fallback_results["ids"],
                    fb_docs,
                    fb_embs,
                    fb_metas,
                    embedding,
                    existing_ids,
                )
            else:
                continue

            if best is None:
                continue
            idx, best_sim, source = best
            src_ids = source["ids"]
            src_docs = source["documents"]
            src_metas = source["metadatas"]
            chroma_meta = dict(src_metas[idx])
            mem = Memory(
                id=src_ids[idx],
                content=src_docs[idx],
                metadata=MemoryMetadata.from_chroma(chroma_meta),
                access_count=int(
                    chroma_meta.get("access_count") or 0,
                ),
                created_at=datetime.fromisoformat(
                    str(chroma_meta["created_at"]),
                ),
                last_accessed=datetime.fromisoformat(
                    str(chroma_meta["last_accessed"]),
                ),
            )
            recall_results.append(
                RecallResult(memory=mem, score=best_sim * (1.0 + g_score)),
            )

    @staticmethod
    def _pick_best_chunk(
        indices: list[int],
        ids: list[str],
        docs: list[str],
        embs: list[list[float]],
        metas: list[dict[str, Any]],
        query_emb: list[float],
        skip_ids: set[str],
    ) -> tuple[int, float, dict[str, Any]] | None:
        """Pick the chunk closest to query_emb from the given indices.

        Returns (best_index, similarity, source_dict) or None.
        """
        best_idx = -1
        best_sim = -1.0
        for i in indices:
            if ids[i] in skip_ids:
                continue
            sim = sum(a * b for a, b in zip(query_emb, embs[i]))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx < 0:
            return None
        source = {
            "ids": ids,
            "documents": docs,
            "metadatas": metas,
        }
        return best_idx, best_sim, source

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_ec_coords(chroma_meta: dict[str, str | int | float], text: str) -> None:
        """Encode entorhinal context coordinates onto chroma metadata.

        Coordinates: ec_topics (what), ec_region (where/domain),
        ec_speakers (who — source monitoring).
        """
        ec_topics = extract_keywords(text, top_n=5)
        if ec_topics:
            chroma_meta["ec_topics"] = ",".join(ec_topics)
        ec_region = assign_region(text)
        if ec_region:
            chroma_meta["ec_region"] = ec_region
        speakers = extract_speakers(text)
        if speakers:
            chroma_meta["ec_speakers"] = ",".join(speakers)

    @staticmethod
    def _attach_session_date(
        chroma_meta: dict[str, str | int | float],
        tags: list[str],
    ) -> None:
        """Extract a date ordinal from tags for time-cell filtering."""
        for tag in tags:
            dt = parse_session_date(tag)
            if dt is not None:
                chroma_meta["session_date"] = dt.toordinal()
                return

    @staticmethod
    def _build_chroma_meta(memory: Memory) -> dict[str, str | int | float]:
        meta: dict[str, str | int | float] = {
            "created_at": memory.created_at.isoformat(),
            "last_accessed": memory.last_accessed.isoformat(),
            "access_count": memory.access_count,
        }
        meta.update(memory.metadata.to_chroma())
        return meta

    @staticmethod
    def _merge_results(
        filtered: Any,
        unfiltered: Any,
        n: int,
    ) -> dict[str, Any]:
        """Merge filtered and unfiltered ChromaDB results, keeping top-n by distance."""
        seen: set[str] = set()
        merged: list[tuple[str, str, dict[str, Any], float]] = []

        for source in (filtered, unfiltered):
            ids = source["ids"][0] if source["ids"] and source["ids"][0] else []
            docs = (
                source["documents"][0]
                if source["documents"] and source["documents"][0]
                else []
            )
            metas = (
                source["metadatas"][0]
                if source["metadatas"] and source["metadatas"][0]
                else []
            )
            dists = (
                source["distances"][0]
                if source["distances"] and source["distances"][0]
                else []
            )

            for i, doc_id in enumerate(ids):
                if doc_id not in seen:
                    seen.add(doc_id)
                    merged.append((doc_id, docs[i], dict(metas[i]), dists[i]))

        # Sort by distance (lower = better) and take top n
        merged.sort(key=lambda x: x[3])
        merged = merged[:n]

        return {
            "ids": [[m[0] for m in merged]],
            "documents": [[m[1] for m in merged]],
            "metadatas": [[m[2] for m in merged]],
            "distances": [[m[3] for m in merged]],
        }

    @staticmethod
    def _build_where(**filters: str | None) -> dict[str, Any] | None:
        clauses = {k: v for k, v in filters.items() if v is not None}
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses
        return {"$and": [{k: v} for k, v in clauses.items()]}

    def _bump_access(self, memories: list[Memory]) -> None:
        """Increment access count and update timestamp for recalled memories."""
        if not memories:
            return
        now = datetime.now(UTC).isoformat()
        ids = [m.id for m in memories]
        # NOTE: called within _recall_impl which already holds self._lock
        current = self._collection.get(ids=ids, include=["metadatas"])
        _metas = current["metadatas"]
        if _metas is None:
            return
        updated_metas: list[dict[str, str | int | float | bool]] = []
        for meta in _metas:
            updated = dict(meta)
            updated["access_count"] = int(updated.get("access_count") or 0) + 1
            updated["last_accessed"] = now
            updated_metas.append(updated)
        self._collection.update(
            ids=current["ids"],
            metadatas=updated_metas,
        )
