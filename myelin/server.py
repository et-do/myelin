"""MCP server — exposes store/recall/forget tools for AI agent integration."""

from __future__ import annotations

import atexit
import json
import logging
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from .config import MyelinSettings, settings
from .lock import DataDirLock, DataDirLockedError
from .log import request_id, setup_logging
from .models import MemoryMetadata, RecallResult
from .recall import HebbianTracker, find_stale
from .reranker import Neocortex
from .store import (
    Hippocampus,
    SemanticNetwork,
    ThalamicBuffer,
    replay,
)

mcp = FastMCP("myelin")

logger = logging.getLogger(__name__)

# Lazy singletons — initialized on first tool call
_hippocampus: Hippocampus | None = None
_data_dir_lock: DataDirLock | None = None
_hebbian: HebbianTracker | None = None
_neocortex: SemanticNetwork | None = None
_thalamus: ThalamicBuffer | None = None
_cfg: MyelinSettings = settings
_store_count: int = 0


def configure(cfg: MyelinSettings) -> None:
    """Override settings for the server (used by tests and benchmarks)."""
    global _cfg, _hippocampus, _hebbian, _neocortex, _thalamus, _store_count
    if _hebbian is not None:
        _hebbian.close()
    if _neocortex is not None:
        _neocortex.close()
    if _thalamus is not None:
        _thalamus.close()
    _cfg = cfg
    _hippocampus = None
    _hebbian = None
    _neocortex = None
    _thalamus = None
    _store_count = 0


def _get_hippocampus() -> Hippocampus:
    global _hippocampus
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


def warm_up() -> None:
    """Pre-warm models by running dummy inference (avoids cold-start lag)."""
    _get_hippocampus().warm_up()


def shutdown() -> None:
    """Close all initialized stores gracefully."""
    global _hippocampus, _hebbian, _neocortex, _thalamus, _data_dir_lock
    logger.info("shutting down")
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
    if _data_dir_lock is not None:
        _data_dir_lock.release()
        _data_dir_lock = None


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


def do_store(
    content: str,
    project: str = "",
    language: str = "",
    scope: str = "",
    memory_type: str = "",
    tags: str = "",
    source: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Store a memory. Returns dict with status."""
    if len(content) > _MAX_CONTENT_CHARS:
        return {
            "status": "rejected",
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
    )
    hc = _get_hippocampus()
    count_before = hc.count()
    memory = hc.store(content, metadata, overwrite=overwrite)
    if memory is None:
        logger.info(
            "store rejected: too short or near-duplicate",
            extra={"project": project, "scope": scope},
        )
        return {"status": "rejected", "reason": "too short or near-duplicate"}
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
    result: dict[str, Any] = {"id": memory.id, "status": status}
    if memory.replaced_id:
        result["replaced"] = memory.replaced_id
    if chunks_stored > 1:
        result["chunks"] = chunks_stored

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


def do_pin(memory_id: str, priority: int = 1, label: str = "") -> dict[str, Any]:
    """Pin a memory for always-available retrieval (thalamic relay)."""
    _get_thalamus().pin(memory_id, priority, label or None)
    return {"memory_id": memory_id, "status": "pinned", "priority": priority}


def do_unpin(memory_id: str) -> dict[str, Any]:
    """Remove a pinned memory from the thalamic buffer."""
    existed = _get_thalamus().unpin(memory_id)
    return {"memory_id": memory_id, "status": "unpinned" if existed else "not_found"}


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

    Returns:
        JSON with memory ID on success, or rejection reason.
    """
    with _track("store"):
        return json.dumps(
            do_store(
                content, project, language, scope, memory_type, tags, source, overwrite
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
            do_recall(query, n_results, project, language, scope, memory_type, ref),
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
def health() -> str:
    """Lightweight health check — returns ok without initializing stores.

    Returns:
        JSON with status and version.
    """
    from . import __version__

    return json.dumps({"status": "ok", "version": __version__})


def main() -> None:
    """Run the MCP server (stdio transport)."""
    global _data_dir_lock
    setup_logging(level=getattr(logging, _cfg.log_level))

    # Acquire exclusive lock on the data directory before touching any store.
    # This prevents two MCP processes from corrupting shared state.
    _cfg.ensure_dirs()
    _data_dir_lock = DataDirLock(_cfg.data_dir)
    try:
        _data_dir_lock.acquire()
    except DataDirLockedError as exc:
        logger.error("data directory locked: %s", exc)
        raise SystemExit(1) from exc

    logger.info(
        "starting myelin server",
        extra={
            "data_dir": str(_cfg.data_dir),
            "embedding_model": _cfg.embedding_model,
            "cross_encoder_model": _cfg.cross_encoder_model,
            "log_level": _cfg.log_level,
        },
    )
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(shutdown)
    warm_up()
    mcp.run()


if __name__ == "__main__":
    main()
