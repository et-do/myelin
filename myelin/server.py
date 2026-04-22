"""MCP server — exposes store/recall/forget tools for AI agent integration."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import signal
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from .config import MyelinSettings, settings
from .ingest import IngestResult, ingest_directory, ingest_file
from .log import request_id, setup_logging, suppress_noisy_loggers
from .models import MemoryMetadata, RecallResult
from .recall import HebbianTracker, find_lru, find_stale
from .reranker import Neocortex
from .store import (
    Hippocampus,
    SemanticNetwork,
    ThalamicBuffer,
    replay,
)
from .timer import DecayTimer


@asynccontextmanager
async def _lifespan(_: FastMCP) -> AsyncIterator[None]:
    """Start model warm-up and decay timer; clean up on shutdown."""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, warm_up)
    _get_decay_timer().start()
    yield
    _get_decay_timer().stop()


mcp = FastMCP("myelin", lifespan=_lifespan)

logger = logging.getLogger(__name__)

# Lazy singletons — initialized on first tool call
_init_lock = threading.Lock()
_hippocampus: Hippocampus | None = None
_hebbian: HebbianTracker | None = None
_neocortex: SemanticNetwork | None = None
_thalamus: ThalamicBuffer | None = None
_decay_timer: DecayTimer | None = None
_cfg: MyelinSettings = settings
_store_count: int = 0


def configure(cfg: MyelinSettings) -> None:
    """Override settings for the server (used by tests and benchmarks)."""
    global _cfg, _hippocampus, _hebbian, _neocortex
    global _thalamus, _decay_timer, _worker, _store_count
    if _hebbian is not None:
        _hebbian.close()
    if _neocortex is not None:
        _neocortex.close()
    if _thalamus is not None:
        _thalamus.close()
    if _decay_timer is not None:
        _decay_timer.stop()
    _cfg = cfg
    _hippocampus = None
    _hebbian = None
    _neocortex = None
    _thalamus = None
    _decay_timer = None
    _store_count = 0


def _get_hippocampus() -> Hippocampus:
    global _hippocampus
    with _init_lock:
        if _hippocampus is None:
            reranker = Neocortex(model_name=_cfg.cross_encoder_model)
            _hippocampus = Hippocampus(
                cfg=_cfg,
                reranker=reranker,
                semantic_network=_get_neocortex(),
            )
    return _hippocampus


def _get_hebbian() -> HebbianTracker:
    global _hebbian
    if _hebbian is None:
        _cfg.ensure_dirs()
        _hebbian = HebbianTracker(cfg=_cfg)
    return _hebbian


def _get_neocortex() -> SemanticNetwork:
    global _neocortex
    if _neocortex is None:
        _cfg.ensure_dirs()
        _neocortex = SemanticNetwork(cfg=_cfg)
    return _neocortex


def _get_thalamus() -> ThalamicBuffer:
    global _thalamus
    if _thalamus is None:
        _cfg.ensure_dirs()
        _thalamus = ThalamicBuffer(cfg=_cfg)
    return _thalamus


def _get_decay_timer() -> DecayTimer:
    global _decay_timer
    if _decay_timer is None:
        _decay_timer = DecayTimer(
            fn=do_decay_sweep,
            interval_hours=_cfg.decay_interval_hours,
        )
    return _decay_timer


def warm_up() -> None:
    """Pre-warm models by running dummy inference (avoids cold-start lag)."""
    logger.info("loading models")
    _get_hippocampus().warm_up()
    logger.info("models ready")


def shutdown() -> None:
    """Close all initialized stores gracefully."""
    global _hippocampus, _hebbian, _neocortex, _thalamus, _decay_timer
    logger.info("shutting down")
    if _decay_timer is not None:
        _decay_timer.stop()
        _decay_timer = None
    if _hippocampus is not None:
        if _hippocampus._reranker is not None:
            _hippocampus._reranker.close()
        _hippocampus = None
    if _hebbian is not None:
        _hebbian.close()
        _hebbian = None
    if _neocortex is not None:
        _neocortex.close()
        _neocortex = None
    if _thalamus is not None:
        _thalamus.close()
        _thalamus = None


def _signal_handler(_signum: int, _frame: Any) -> None:
    raise SystemExit(0)


@contextmanager
def _track(operation: str) -> Iterator[None]:
    """Set a request ID and log operation duration."""
    rid = uuid4().hex[:12]
    token = request_id.set(rid)
    t0 = time.monotonic()
    logger.info("%s started", operation)
    try:
        yield
        dt = (time.monotonic() - t0) * 1000
        logger.info("%s completed in %.1fms", operation, dt)
    except Exception:
        dt = (time.monotonic() - t0) * 1000
        logger.exception("%s failed after %.1fms", operation, dt)
        raise
    finally:
        request_id.reset(token)


# ------------------------------------------------------------------
# Core logic (testable without MCP transport)
# ------------------------------------------------------------------

# Hard caps to reject absurdly large inputs at the API boundary.
_MAX_CONTENT_CHARS = 500_000  # ~125K tokens
_MAX_QUERY_CHARS = 10_000  # ~2.5K tokens


def _inject_relations(relations: str) -> int:
    """Parse a JSON-encoded list of [subject, predicate, object] triples and
    write each as a relationship edge in the semantic network.

    Silently skips malformed input so a bad ``relations`` value never blocks
    an otherwise-valid store call.  Returns the number of edges written.
    """
    try:
        raw = json.loads(relations)
    except (json.JSONDecodeError, ValueError):
        logger.debug("relations: invalid JSON, skipping")
        return 0
    if not isinstance(raw, list):
        return 0
    net = _get_neocortex()
    count = 0
    for item in raw:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 3
            and all(isinstance(v, str) and v.strip() for v in item)
        ):
            net.add_relationship(
                subject=item[0].strip(),
                object_=item[2].strip(),
                predicate=item[1].strip(),
            )
            count += 1
        else:
            logger.debug("relations: skipping malformed triple %r", item)
    return count


def do_store(
    content: str,
    project: str = "",
    language: str = "",
    scope: str = "",
    memory_type: str = "",
    tags: str = "",
    source: str = "",
    overwrite: bool = False,
    agent_id: str = "",
    relations: str = "",
) -> dict[str, Any]:
    """Store a memory.

    Always returns ``status``, ``id`` (``None`` when rejected), and
    ``reason`` (``None`` on success) so callers have a stable shape to
    pattern-match against without key-existence checks.
    """
    if len(content) > _MAX_CONTENT_CHARS:
        return {
            "status": "rejected",
            "id": None,
            "reason": f"content exceeds {_MAX_CONTENT_CHARS} char limit",
        }
    global _store_count
    metadata = MemoryMetadata(
        project=project or None,
        language=language or None,
        scope=scope or None,
        memory_type=memory_type or None,
        tags=[t.strip() for t in tags.split(",") if t.strip()] if tags else [],
        source=source or None,
        agent_id=agent_id or None,
    )
    hc = _get_hippocampus()
    count_before = hc.count()
    memory = hc.store(content, metadata, overwrite=overwrite)
    if memory is None:
        logger.info(
            "store rejected: too short or near-duplicate",
            extra={"project": project, "scope": scope},
        )
        return {
            "status": "rejected",
            "id": None,
            "reason": "too short or near-duplicate",
        }
    chunks_stored = hc.count() - count_before
    status = "updated" if memory.replaced_id else "stored"
    logger.info(
        "store %s memory %s (%d chars, %d chunks)",
        status,
        memory.id,
        len(content),
        max(chunks_stored, 1),
        extra={
            "memory_id": memory.id,
            "project": project,
            "scope": scope,
            "chunks": max(chunks_stored, 1),
            "replaced_id": memory.replaced_id,
        },
    )
    result: dict[str, Any] = {"id": memory.id, "status": status, "reason": None}
    if memory.replaced_id:
        result["replaced"] = memory.replaced_id
    if chunks_stored > 1:
        result["chunks"] = chunks_stored

    # Explicit relationship injection — parse before auto-consolidation so
    # edges are immediately queryable without waiting for the next replay.
    if relations:
        rel_count = _inject_relations(relations)
        if rel_count:
            result["relationships"] = rel_count

    # Storage cap — evict LRU memories if we're over the limit.
    # Pinned memories (thalamic relay) are never evicted.
    if _cfg.max_memories > 0:
        current = hc.count()
        if current > _cfg.max_memories:
            n_evict = current - _cfg.max_memories
            all_meta = hc.get_all_metadata()
            pinned_ids = {p["memory_id"] for p in _get_thalamus().get_pinned()}
            evict_ids = find_lru(all_meta, n_evict, exclude_ids=pinned_ids)
            if evict_ids:
                hc.forget_batch(evict_ids)
                valid_ids = {m["id"] for m in all_meta} - set(evict_ids)
                _get_hebbian().cleanup(valid_ids)
                _get_thalamus().cleanup(valid_ids)
                result["evicted"] = len(evict_ids)
                logger.info(
                    "cap eviction: %d memories removed (cap=%d)",
                    len(evict_ids),
                    _cfg.max_memories,
                    extra={"evicted": len(evict_ids), "cap": _cfg.max_memories},
                )

    # Auto-consolidation: replay into neocortex every N stores
    _store_count += 1
    if _cfg.consolidation_interval > 0 and (
        _store_count % _cfg.consolidation_interval == 0
    ):
        consolidation_result = do_consolidate()
        result["consolidation"] = consolidation_result

    return result


def do_recall(
    query: str,
    n_results: int = 5,
    project: str = "",
    language: str = "",
    scope: str = "",
    memory_type: str = "",
    reference_date: datetime | None = None,
    agent_id: str = "",
) -> list[dict[str, Any]]:
    """Recall memories. Returns list of result dicts."""
    if len(query) > _MAX_QUERY_CHARS:
        return []
    results = _get_hippocampus().recall(
        query,
        n_results=n_results,
        project=project or None,
        language=language or None,
        scope=scope or None,
        memory_type=memory_type or None,
        agent_id=agent_id or None,
        reference_date=reference_date or datetime.now(UTC),
    )

    logger.info(
        "recall found %d results for query (%d chars)",
        len(results),
        len(query),
        extra={"result_count": len(results), "project": project, "scope": scope},
    )

    # Hebbian boost — re-rank by co-access patterns
    if results:
        hebbian = _get_hebbian()
        results = hebbian.boost(results)
        hebbian.reinforce([r.memory.id for r in results])

    # Thalamus — prepend pinned memories, track recency
    thal = _get_thalamus()
    if results:
        thal.touch([r.memory.id for r in results])
    pinned = thal.get_pinned()
    if pinned:
        recalled_ids = {r.memory.id for r in results}
        missing_pin_ids = [
            p["memory_id"] for p in pinned if p["memory_id"] not in recalled_ids
        ]
        if missing_pin_ids:
            hc = _get_hippocampus()
            pinned_memories = hc.get_by_ids(missing_pin_ids)
            pinned_results = [
                RecallResult(memory=m, score=2.0) for m in pinned_memories
            ]
            results = pinned_results + results

    return [
        {
            "id": r.memory.id,
            "content": r.memory.content,
            "score": round(r.score, 4),
            "project": r.memory.metadata.project,
            "scope": r.memory.metadata.scope,
            "memory_type": r.memory.metadata.memory_type,
            "parent_id": r.memory.metadata.parent_id,
            "access_count": r.memory.access_count,
        }
        for r in results
    ]


def do_forget(memory_id: str) -> dict[str, Any]:
    """Forget a memory by ID."""
    ok = _get_hippocampus().forget(memory_id)
    _get_thalamus().unpin(memory_id)
    status = "forgotten" if ok else "not_found"
    logger.info(
        "forget %s: %s",
        memory_id,
        status,
        extra={"memory_id": memory_id, "status": status},
    )
    return {"id": memory_id, "status": status}


def do_decay_sweep() -> dict[str, Any]:
    """Prune stale memories."""
    hc = _get_hippocampus()
    metadata = hc.get_all_metadata()
    stale_ids = find_stale(metadata)
    if stale_ids:
        count = hc.forget_batch(stale_ids)
        valid_ids = {m["id"] for m in metadata} - set(stale_ids)
        _get_hebbian().cleanup(valid_ids)
        _get_thalamus().cleanup(valid_ids)
    else:
        count = 0
    remaining = hc.count()
    logger.info(
        "decay sweep: pruned %d, remaining %d",
        count,
        remaining,
        extra={"pruned": count, "remaining": remaining},
    )
    return {"pruned": count, "remaining": remaining}


def do_status() -> dict[str, Any]:
    """Memory system status."""
    hc = _get_hippocampus()
    net = _get_neocortex()
    return {
        "memory_count": hc.count(),
        "pinned_count": _get_thalamus().pinned_count(),
        "entity_count": net.entity_count(),
        "relationship_count": net.relationship_count(),
        "data_dir": str(_cfg.data_dir),
        "embedding_model": _cfg.embedding_model,
        "max_idle_days": _cfg.max_idle_days,
        "min_access_count": _cfg.min_access_count,
        "max_memories": _cfg.max_memories,
        "decay_timer_running": _get_decay_timer().is_running,
    }


def do_consolidate() -> dict[str, Any]:
    """Replay episodic memories into the semantic network (CLS)."""
    hc = _get_hippocampus()
    net = _get_neocortex()

    memories = hc.get_all_content()
    result = replay(memories, net)
    return {
        "memories_replayed": result.memories_replayed,
        "entities_found": result.entities_found,
        "relationships_created": result.relationships_created,
    }


def do_ingest(
    path: str,
    *,
    project: str = "",
    scope: str = "",
    source: str = "ingest",
    recursive: bool = True,
) -> dict[str, Any]:
    """Bulk-load content from a file or directory into memory.

    Supports ``.txt``, ``.md`` (with optional YAML frontmatter), and
    ``.json`` (array of objects with a ``"content"`` key, as produced by
    ``myelin export``).  When *path* is a directory every supported file
    under it is ingested.

    Frontmatter metadata in Markdown/text files overrides the *project*,
    *scope*, *language*, *memory_type*, *tags*, and *source* defaults.

    Returns a summary dict with ``stored``, ``skipped``, and ``errors``.
    """
    p = Path(path)
    if p.is_dir():
        result: IngestResult = ingest_directory(
            p,
            store_fn=do_store,
            default_project=project,
            default_scope=scope,
            default_source=source,
            recursive=recursive,
        )
    else:
        result = ingest_file(
            p,
            store_fn=do_store,
            default_project=project,
            default_scope=scope,
            default_source=source,
        )
    return {
        "stored": result.stored,
        "skipped": result.skipped,
        "errors": result.errors,
    }


def do_pin(memory_id: str, priority: int = 1, label: str = "") -> dict[str, Any]:
    """Pin a memory for always-available retrieval (thalamic relay)."""
    _get_thalamus().pin(memory_id, priority, label or None)
    return {"memory_id": memory_id, "status": "pinned", "priority": priority}


def do_unpin(memory_id: str) -> dict[str, Any]:
    """Remove a pinned memory from the thalamic buffer."""
    existed = _get_thalamus().unpin(memory_id)
    return {"memory_id": memory_id, "status": "unpinned" if existed else "not_found"}


def do_debug_recall(
    query: str,
    n_results: int = 5,
    project: str = "",
    language: str = "",
    scope: str = "",
    memory_type: str = "",
    agent_id: str = "",
) -> dict[str, Any]:
    """Recall with full pipeline transparency for debugging.

    Runs the normal recall pipeline then annotates each result with:
    - Raw bi-encoder cosine similarity from ChromaDB
    - Cross-encoder score (CE re-run on the result set)
    - Total Hebbian co-weight accumulated for this result in the set

    Also reports what the query planner inferred and which filters were
    applied automatically vs. explicitly.
    """
    from .recall.query_planner import plan as _plan_query
    from .store.amygdala import passes_gate

    hc = _get_hippocampus()
    memory_count = hc.count()

    # --- 1. Query plan (what PFC would infer) ---
    qplan = _plan_query(query)

    # --- 2. Amygdala gate check on the query text ---
    # Shows whether the query itself would be accepted if stored.
    # Useful for diagnosing why a very short or near-duplicate query
    # produces unexpected results.
    gate_ok, gate_reason = passes_gate(query, cfg=_cfg)

    # --- 3. Read-only recall (full 3-probe pipeline, no Hebbian reinforce) ---
    # We use hc.recall() + hebbian.boost() directly to avoid writing
    # Hebbian state or thalamus touch records during a diagnostic call.
    raw_results = hc.recall(
        query,
        n_results=n_results,
        project=project or None,
        language=language or None,
        scope=scope or None,
        memory_type=memory_type or None,
        agent_id=agent_id or None,
        reference_date=datetime.now(UTC),
    )
    if raw_results:
        raw_results = _get_hebbian().boost(raw_results)

    results: list[dict[str, Any]] = [
        {
            "id": r.memory.id,
            "content": r.memory.content,
            "score": round(r.score, 4),
            "project": r.memory.metadata.project,
            "scope": r.memory.metadata.scope,
            "memory_type": r.memory.metadata.memory_type,
            "access_count": r.memory.access_count,
        }
        for r in raw_results
    ]

    filters_applied: dict[str, Any] = {
        "project": project or None,
        "language": language or None,
        "scope": scope or None,
        "memory_type": memory_type or None,
        "agent_id": agent_id or None,
        "auto_memory_type": qplan.memory_type if not memory_type else None,
        "auto_scope": qplan.scope_hint if not scope else None,
    }

    if not results:
        return {
            "query": query,
            "query_plan": {
                "memory_type": qplan.memory_type,
                "scope_hint": qplan.scope_hint,
                "signals": qplan.signals,
            },
            "gate_check": {"would_store": gate_ok, "reason": gate_reason},
            "filters_applied": filters_applied,
            "memory_count": memory_count,
            "results": [],
        }

    # --- 4. Raw bi-encoder similarity (post-hoc ChromaDB query) ---
    bi_sim_map: dict[str, float] = {}
    with hc._lock:
        query_emb = hc._embedder.encode(query).tolist()
        n_fetch = min(max(n_results * 8, 50), memory_count)
        raw = hc._collection.query(
            query_embeddings=[query_emb],
            n_results=n_fetch,
            include=["distances"],
        )
    if raw.get("ids") and raw["ids"][0]:
        for rid, dist in zip(raw["ids"][0], raw["distances"][0]):
            bi_sim_map[rid] = round(1.0 - float(dist), 4)

    # --- 5. CE scores (re-run on result set against original query) ---
    ce_score_map: dict[str, float] = {}
    if hc._reranker is not None and _cfg.neocortex_rerank:
        passages = [r["content"] for r in results]
        raw_ce = hc._reranker.rerank(query, passages)
        for r, cs in zip(results, raw_ce):
            ce_score_map[r["id"]] = round(float(cs), 4)

    # --- 6. Hebbian co-weights for this result set ---
    result_ids = [r["id"] for r in results]
    hebbian_weights = _get_hebbian().lookup_weights(result_ids)

    # --- 7. Enrich results ---
    debug_results = []
    for i, r in enumerate(results):
        rid = r["id"]
        preview = r["content"]
        if len(preview) > 150:
            preview = preview[:147] + "..."
        debug_results.append(
            {
                "rank": i + 1,
                "id": rid,
                "final_score": r["score"],
                "bi_encoder_sim": bi_sim_map.get(rid),
                "ce_score": ce_score_map.get(rid),
                "hebbian_weight": hebbian_weights.get(rid, 0.0),
                "memory_type": r.get("memory_type"),
                "scope": r.get("scope"),
                "project": r.get("project"),
                "access_count": r.get("access_count", 0),
                "content_preview": preview,
            }
        )

    return {
        "query": query,
        "query_plan": {
            "memory_type": qplan.memory_type,
            "scope_hint": qplan.scope_hint,
            "signals": qplan.signals,
        },
        "gate_check": {"would_store": gate_ok, "reason": gate_reason},
        "filters_applied": filters_applied,
        "memory_count": memory_count,
        "results": debug_results,
    }


# ------------------------------------------------------------------
# MCP tool wrappers (thin JSON serialization layer)
# ------------------------------------------------------------------


@mcp.tool()
def store(
    content: str,
    project: str = "",
    language: str = "",
    scope: str = "",
    memory_type: str = "",
    tags: str = "",
    source: str = "",
    overwrite: bool = False,
    agent_id: str = "",
    relations: str = "",
) -> str:
    """Store a memory with optional context metadata.

    Args:
        content: The information to remember.
        project: Cortical region — project name for scoped retrieval.
        language: Programming language context.
        scope: Engram cluster — domain scope (e.g. "auth", "database").
        memory_type: Memory system (episodic/semantic/procedural/prospective).
        tags: Comma-separated tags for categorization.
        source: Which tool stored this ("copilot", "cursor", etc.).
        overwrite: If True and content is a near-duplicate of an existing
            memory, replace the old memory instead of rejecting.  The
            response will include ``"replaced": <old_parent_id>`` and
            ``"status": "updated"``.
        agent_id: Namespace identifier for the storing agent or bot.
            Memories tagged with an agent_id are only returned when the
            same agent_id is supplied at recall time.
        relations: JSON-encoded list of [subject, predicate, object] triples
            to assert as relationship edges immediately, e.g.
            ``'[["AuthService","depends_on","JWTHelper"]]'``.
            Malformed input is silently ignored so it never blocks the store.

    Returns:
        JSON with memory ID on success, or rejection reason.
    """
    with _track("store"):
        return json.dumps(
            do_store(
                content,
                project,
                language,
                scope,
                memory_type,
                tags,
                source,
                overwrite,
                agent_id,
                relations,
            )
        )


@mcp.tool()
def recall(
    query: str,
    n_results: int = 5,
    project: str = "",
    language: str = "",
    scope: str = "",
    memory_type: str = "",
    reference_date: str = "",
    agent_id: str = "",
) -> str:
    """Recall memories relevant to a query.

    Args:
        query: What to search for (semantic similarity).
        n_results: Maximum number of results to return.
        project: Filter to a specific cortical region (project).
        language: Filter to a specific language.
        scope: Filter to a specific engram cluster (domain scope).
        memory_type: Filter by memory system
            (episodic/semantic/procedural/prospective).
        reference_date: ISO-8601 date for temporal context
            (e.g. "2026-04-12"). Defaults to now.
        agent_id: Namespace identifier — only returns memories stored
            with the same agent_id.  Omit or leave empty to query the
            global shared namespace.

    Returns:
        JSON array of matching memories with scores.
    """
    with _track("recall"):
        ref: datetime | None = None
        if reference_date:
            try:
                ref = datetime.fromisoformat(reference_date)
            except ValueError:
                pass
        return json.dumps(
            do_recall(
                query, n_results, project, language, scope, memory_type, ref, agent_id
            ),
            indent=2,
        )


@mcp.tool()
def forget(memory_id: str) -> str:
    """Remove a specific memory by ID.

    Args:
        memory_id: The ID of the memory to remove.

    Returns:
        JSON confirmation or error.
    """
    with _track("forget"):
        return json.dumps(do_forget(memory_id))


@mcp.tool()
def pin_memory(memory_id: str, priority: int = 1, label: str = "") -> str:
    """Pin a memory so it is always included in recall results.

    Pinned memories model the thalamic relay — sustained activation that
    keeps critical facts always available to the cortex.

    Args:
        memory_id: The ID of the memory to pin.
        priority: 0 = identity/system context, 1 = critical facts.
        label: Optional human-readable label for this pin.

    Returns:
        JSON confirmation.
    """
    with _track("pin"):
        return json.dumps(do_pin(memory_id, priority, label))


@mcp.tool()
def unpin_memory(memory_id: str) -> str:
    """Remove a pinned memory from the thalamic buffer.

    Args:
        memory_id: The ID of the memory to unpin.

    Returns:
        JSON confirmation or not_found.
    """
    with _track("unpin"):
        return json.dumps(do_unpin(memory_id))


@mcp.tool()
def decay_sweep() -> str:
    """Prune stale memories that haven't been accessed recently.

    Removes memories exceeding the idle threshold with low access counts.

    Returns:
        JSON summary of pruned memories.
    """
    with _track("decay_sweep"):
        return json.dumps(do_decay_sweep())


@mcp.tool()
def status() -> str:
    """Show memory system status.

    Returns:
        JSON with memory count and configuration.
    """
    with _track("status"):
        return json.dumps(do_status())


@mcp.tool()
def consolidate() -> str:
    """Replay episodic memories into the semantic network (offline consolidation).

    Extracts entities from stored memories and builds co-occurrence
    relationships in the neocortex — like hippocampal replay during sleep.

    Returns:
        JSON summary of memories replayed, entities found, and relationships created.
    """
    with _track("consolidate"):
        return json.dumps(do_consolidate())


@mcp.tool()
def ingest(
    path: str,
    project: str = "",
    scope: str = "",
    source: str = "ingest",
    recursive: bool = True,
) -> str:
    """Bulk-load memories from a file or directory.

    Supported formats:

    - ``.txt`` / ``.md``: file body stored as one memory.  Optional YAML
      frontmatter (between ``---`` delimiters) sets project, scope, language,
      memory_type, tags, and source for that file.
    - ``.json``: array of objects with a ``"content"`` key (same shape as
      ``myelin export`` output).
    - **directory**: recurse and ingest every supported file found.

    Args:
        path: Absolute or relative path to a file or directory.
        project: Default project tag (overridden by per-file frontmatter).
        scope: Default scope tag (overridden by per-file frontmatter).
        source: Source label for all ingested memories.
        recursive: Descend into subdirectories (default: true).

    Returns:
        JSON summary with stored, skipped, and errors counts.
    """
    with _track("ingest"):
        return json.dumps(
            do_ingest(
                path, project=project, scope=scope, source=source, recursive=recursive
            )
        )


@mcp.tool()
def health() -> str:
    """Lightweight health check — returns ok without initializing stores.

    Returns:
        JSON with status and version.
    """
    from . import __version__

    return json.dumps({"status": "ok", "version": __version__})


def main() -> None:
    """Run the MCP server (stdio transport)."""
    suppress_noisy_loggers()
    setup_logging(level=getattr(logging, _cfg.log_level))
    _cfg.ensure_dirs()
    from . import __version__

    logger.info(
        "starting myelin server",
        extra={
            "version": __version__,
            "data_dir": str(_cfg.data_dir),
            "embedding_model": _cfg.embedding_model,
            "cross_encoder_model": _cfg.cross_encoder_model,
            "log_level": _cfg.log_level,
        },
    )
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(shutdown)
    mcp.run()


if __name__ == "__main__":
    main()
