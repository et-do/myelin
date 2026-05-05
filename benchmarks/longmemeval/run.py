"""Run Myelin against LongMemEval and produce a hypothesis file for evaluation.

Usage:
    uv run python -m benchmarks.longmemeval.run \
        benchmarks/longmemeval/data/longmemeval_oracle.json \
        benchmarks/longmemeval/output/myelin_oracle.jsonl

    # Parallel (4 workers -- ~4x faster):
    uv run python -m benchmarks.longmemeval.run \
        benchmarks/longmemeval/data/longmemeval_oracle.json \
        benchmarks/longmemeval/output/myelin_oracle.jsonl 5 4

The output is a JSONL file with {question_id, hypothesis} per line,
compatible with LongMemEval's evaluate_qa.py script.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any
from uuid import uuid4

# Load .env before any HF/torch imports
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

import chromadb  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from myelin.config import MyelinSettings  # noqa: E402
from myelin.models import Memory, MemoryMetadata  # noqa: E402
from myelin.recall.reranker import Neocortex  # noqa: E402
from myelin.store.chunking import chunk  # noqa: E402
from myelin.store.consolidation import extract_entities  # noqa: E402
from myelin.store.hippocampus import Hippocampus  # noqa: E402
from myelin.store.neocortex import SemanticNetwork  # noqa: E402
from myelin.store.perirhinal import summarise  # noqa: E402
from myelin.store.prefrontal import classify_memory_type  # noqa: E402


class _ThreadSafeEmbedder:
    """Wrap a SentenceTransformer so only one thread encodes at a time.

    PyTorch CPU inference spawns OpenMP threads internally;
    concurrent encode() calls from a ThreadPoolExecutor cause
    thread-explosion and deadlocks.
    """

    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model
        self._lock = threading.Lock()

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.encode(*args, **kwargs)


def flatten_session(session: list[dict[str, Any]]) -> str:
    """Flatten a list of chat turns into a single text block."""
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in session)


def _parse_question_date(date_str: str) -> datetime | None:
    """Parse a LongMemEval question_date like '2023/05/30 (Tue) 23:40'."""
    try:
        return datetime.strptime(date_str.split("(")[0].strip(), "%Y/%m/%d")
    except (ValueError, IndexError):
        return None


def _build_semantic_network(
    session_texts: list[str],
    cfg: MyelinSettings,
) -> SemanticNetwork:
    """Build a SemanticNetwork from entity co-occurrence across sessions."""
    neo_path = Path(tempfile.mktemp(suffix=".db"))
    neo = SemanticNetwork(db_path=neo_path, cfg=cfg)
    for text in session_texts:
        entities = extract_entities(text)
        for e in entities:
            neo.add_entity(e, entity_type="auto")
        for a, b in combinations(entities, 2):
            neo.add_relationship(a, b, predicate="co_occurs", weight=1.0)
    return neo


def run_instance(
    instance: dict[str, Any],
    data_dir: Path,
    n_results: int = 5,
    embedder: SentenceTransformer | None = None,
    client: Any | None = None,
    reranker: Neocortex | None = None,
) -> dict[str, Any]:
    """Store all sessions, recall for the question, return raw retrieval.

    Returns dict with question_id, hypothesis (top retrieved text),
    and retrieval metadata for analysis.
    """
    cfg = MyelinSettings(data_dir=data_dir)

    # Store each session
    session_ids = instance["haystack_session_ids"]
    dates = instance["haystack_dates"]
    sessions = instance["haystack_sessions"]

    all_texts: list[str] = []
    for i, session in enumerate(sessions):
        text = flatten_session(session)
        all_texts.append(text)

    # Build semantic network from entity co-occurrence
    neo = _build_semantic_network(all_texts, cfg)

    hc = Hippocampus(
        cfg=cfg,
        embedder=embedder,
        ephemeral=True,
        client=client,
        reranker=reranker,
        semantic_network=neo,
    )

    for i, text in enumerate(all_texts):
        metadata = MemoryMetadata(
            scope=f"session_{session_ids[i]}",
            tags=[dates[i]] if i < len(dates) else [],
        )
        hc.store(text, metadata)

    # Parse question_date for time-cell temporal boosting
    ref_date = _parse_question_date(instance.get("question_date", ""))

    # Recall
    results = hc.recall(
        instance["question"],
        n_results=n_results,
        reference_date=ref_date,
    )

    # Build hypothesis from top results
    if results:
        hypothesis = "\n---\n".join(r.memory.content for r in results)
    else:
        hypothesis = "I don't have information about that."

    # Extract per-result metadata for scoring.
    # scope is "session_<id>" — strip the prefix to get the raw session id.
    ranked: list[dict[str, Any]] = []
    for r in results:
        scope = r.memory.metadata.scope or ""
        sid = scope.removeprefix("session_") if scope.startswith("session_") else ""
        ranked.append({"session_id": sid, "score": round(r.score, 4)})

    return {
        "question_id": instance["question_id"],
        "hypothesis": hypothesis,
        "ranked": ranked,
        "n_stored": hc.count(),
        "n_retrieved": len(results),
        "top_score": round(results[0].score, 4) if results else 0.0,
    }


# ------------------------------------------------------------------
# Cached path — uses pre-computed embeddings from cache.py
# ------------------------------------------------------------------

CacheData = tuple["np.ndarray[Any, Any]", "np.ndarray[Any, Any]", dict[str, Any]]


def load_cache(data_file: str) -> CacheData | None:
    """Load pre-computed embedding cache for *data_file*, if available."""
    data_path = Path(data_file)
    cache_dir = data_path.parent.parent / "cache"
    stem = data_path.stem

    emb_path = cache_dir / f"{stem}_embeddings.npy"
    gist_path = cache_dir / f"{stem}_gists.npy"
    index_path = cache_dir / f"{stem}_index.json"

    if not (emb_path.exists() and gist_path.exists() and index_path.exists()):
        return None

    print(f"Loading embedding cache from {cache_dir}/ ...")
    embeddings: np.ndarray[Any, Any] = np.load(emb_path)
    gist_embeddings: np.ndarray[Any, Any] = np.load(gist_path)
    index = json.loads(index_path.read_text())

    # Validate cache was built with same config
    cfg = MyelinSettings()
    mismatches: list[str] = []
    if index.get("model") != cfg.embedding_model:
        mismatches.append(
            f"model: cache={index.get('model')} vs config={cfg.embedding_model}"
        )
    if index.get("chunk_max_chars") != cfg.chunk_max_chars:
        mismatches.append(
            f"chunk_max_chars: cache={index.get('chunk_max_chars')} "
            f"vs config={cfg.chunk_max_chars}"
        )
    if index.get("chunk_overlap_chars") != cfg.chunk_overlap_chars:
        mismatches.append(
            f"chunk_overlap_chars: cache={index.get('chunk_overlap_chars')} "
            f"vs config={cfg.chunk_overlap_chars}"
        )
    if mismatches:
        print("WARNING: Cache config mismatch — rebuild with cache.py!")
        for m in mismatches:
            print(f"  {m}")
        print("Falling back to live encoding.")
        return None

    print(
        f"  {embeddings.shape[0]} chunk embeddings, "
        f"{gist_embeddings.shape[0]} gist embeddings"
    )
    return embeddings, gist_embeddings, index


def run_instance_cached(
    instance: dict[str, Any],
    data_dir: Path,
    cache_embeddings: np.ndarray[Any, Any],
    cache_gist_embeddings: np.ndarray[Any, Any],
    cache_index: dict[str, Any],
    n_results: int = 5,
    embedder: SentenceTransformer | None = None,
    client: Any | None = None,
    reranker: Neocortex | None = None,
) -> dict[str, Any]:
    """Store sessions using pre-computed embeddings, then recall normally.

    Bypasses the gate check and encoder for the store phase —
    only the recall query embedding is computed live.
    """
    cfg = MyelinSettings(data_dir=data_dir)

    session_ids = instance["haystack_session_ids"]
    dates = instance["haystack_dates"]
    sessions = instance["haystack_sessions"]

    # Flatten all sessions first for semantic network building
    all_texts: list[str] = [flatten_session(s) for s in sessions]

    # Build semantic network from entity co-occurrence
    neo = _build_semantic_network(all_texts, cfg)

    hc = Hippocampus(
        cfg=cfg,
        embedder=embedder,
        ephemeral=True,
        client=client,
        reranker=reranker,
        semantic_network=neo,
    )

    # Collect all chunks + embeddings for a single bulk insert
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_embeddings: list[list[float]] = []
    all_metadatas: list[dict[str, Any]] = []
    gist_entries: list[tuple[str, str, list[float]]] = []

    for i, session in enumerate(sessions):
        sid = str(session_ids[i])
        text = flatten_session(session)

        session_cache = cache_index["sessions"].get(sid)
        if session_cache is None:
            # Not in cache — fall back to live store
            metadata = MemoryMetadata(
                scope=f"session_{sid}",
                tags=[dates[i]] if i < len(dates) else [],
            )
            hc.store(text, metadata)
            continue

        # Re-chunk (fast, deterministic)
        segments = chunk(
            text,
            max_chars=cfg.chunk_max_chars,
            overlap_chars=cfg.chunk_overlap_chars,
        )

        chunk_start, chunk_end = session_cache["chunk_range"]
        if len(segments) != (chunk_end - chunk_start):
            # Chunk config changed — fall back to live store
            metadata = MemoryMetadata(
                scope=f"session_{sid}",
                tags=[dates[i]] if i < len(dates) else [],
            )
            hc.store(text, metadata)
            continue

        # Classify (fast regex, no model)
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
            emb = cache_embeddings[chunk_start + j].tolist()

            chroma_meta = Hippocampus._build_chroma_meta(memory)
            Hippocampus._attach_ec_coords(chroma_meta, seg)
            Hippocampus._attach_session_date(chroma_meta, base_meta.tags)
            # Ensure parent_id is always set for gist-guided filtering
            if parent_id:
                chroma_meta["parent_id"] = parent_id
            else:
                chroma_meta["parent_id"] = memory.id

            all_ids.append(memory.id)
            all_docs.append(seg)
            all_embeddings.append(emb)
            all_metadatas.append(chroma_meta)

        # Perirhinal gist — same heuristic as store(), cached embedding
        gist = summarise(text)
        gist_key = parent_id if parent_id else all_ids[-1]
        if gist:
            gist_idx = session_cache["gist_idx"]
            gist_emb = cache_gist_embeddings[gist_idx].tolist()
            gist_entries.append((gist_key, gist, gist_emb))

    # Bulk insert chunks into ChromaDB (bypasses store() for speed)
    if all_ids:
        # ChromaDB limits batch size; split into groups of 5000
        batch = 5000
        for start in range(0, len(all_ids), batch):
            end = start + batch
            hc._collection.add(
                ids=all_ids[start:end],
                embeddings=all_embeddings[start:end],
                documents=all_docs[start:end],
                metadatas=all_metadatas[start:end],
            )

    # Bulk insert gists with pre-computed embeddings
    if gist_entries:
        hc._summaries._collection.upsert(
            ids=[f"summary_{e[0]}" for e in gist_entries],
            embeddings=[e[2] for e in gist_entries],
            documents=[e[1] for e in gist_entries],
            metadatas=[{"parent_id": e[0]} for e in gist_entries],
        )

    # Parse question_date for time-cell temporal boosting
    ref_date = _parse_question_date(instance.get("question_date", ""))

    # Recall as normal — only the query gets embedded here
    results = hc.recall(
        instance["question"],
        n_results=n_results,
        reference_date=ref_date,
    )

    # Build output (identical format to run_instance)
    if results:
        hypothesis = "\n---\n".join(r.memory.content for r in results)
    else:
        hypothesis = "I don't have information about that."

    ranked: list[dict[str, Any]] = []
    for r in results:
        scope = r.memory.metadata.scope or ""
        raw_sid = scope.removeprefix("session_") if scope.startswith("session_") else ""
        ranked.append({"session_id": raw_sid, "score": round(r.score, 4)})

    return {
        "question_id": instance["question_id"],
        "hypothesis": hypothesis,
        "ranked": ranked,
        "n_stored": hc.count(),
        "n_retrieved": len(results),
        "top_score": round(results[0].score, 4) if results else 0.0,
    }


def main(
    data_file: str, output_file: str, n_results: int = 5, workers: int = 1
) -> None:
    data = json.loads(Path(data_file).read_text())

    # Timestamp output so runs stack: myelin_oracle_20260410_153012.jsonl
    base = Path(output_file)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = base.parent / f"{base.stem}_{stamp}{base.suffix}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load embedder once, share across all instances.
    # Wrap in a lock for thread safety when workers > 1.
    raw_embedder = SentenceTransformer(MyelinSettings().embedding_model)
    embedder: Any = _ThreadSafeEmbedder(raw_embedder) if workers > 1 else raw_embedder

    total = len(data)

    # Dummy data_dir — EphemeralClient ignores it, but MyelinSettings needs one
    dummy_dir = Path("/tmp/myelin_bench")

    # Single shared chromadb client — avoids deadlocks from concurrent
    # EphemeralClient creation (chromadb Rust bindings aren't thread-safe).
    shared_client: Any = chromadb.EphemeralClient()

    # Load cross-encoder for neocortical re-ranking
    cfg = MyelinSettings()
    shared_reranker: Neocortex | None = None
    if cfg.neocortex_rerank:
        print("Loading cross-encoder for neocortical re-ranking...", flush=True)
        shared_reranker = Neocortex(model_name=cfg.cross_encoder_model)

    # Try loading pre-computed embedding cache (built by cache.py)
    cache = load_cache(data_file)
    cache_emb: np.ndarray[Any, Any] | None = None
    cache_gist: np.ndarray[Any, Any] | None = None
    cache_idx: dict[str, Any] | None = None
    if cache:
        cache_emb, cache_gist, cache_idx = cache
    else:
        print("No embedding cache found — using live encoding.", flush=True)

    def _run_one(inst: dict[str, Any]) -> dict[str, Any]:
        if cache_emb is not None:
            assert cache_gist is not None and cache_idx is not None
            return run_instance_cached(
                inst,
                dummy_dir,
                cache_emb,
                cache_gist,
                cache_idx,
                n_results,
                embedder=embedder,
                client=shared_client,
                reranker=shared_reranker,
            )
        return run_instance(
            inst,
            dummy_dir,
            n_results,
            embedder=embedder,
            client=shared_client,
            reranker=shared_reranker,
        )

    if workers <= 1:
        # Sequential — simpler, preserves order
        results_log: list[dict[str, Any]] = []
        with open(output_path, "w") as f:
            for i, instance in enumerate(data):
                qid = instance["question_id"]
                t0 = time.time()
                result = _run_one(instance)
                elapsed = time.time() - t0
                print(
                    f"[{i + 1}/{total}] {qid} — "
                    f"{result['n_stored']} stored, "
                    f"{result['n_retrieved']} retrieved, "
                    f"top_score={result['top_score']}, "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
                f.write(
                    json.dumps(
                        {
                            "question_id": qid,
                            "hypothesis": result["hypothesis"],
                            "ranked": result["ranked"],
                        }
                    )
                    + "\n"
                )
                f.flush()
                results_log.append(result)
    else:
        # Parallel — ThreadPoolExecutor (embedder releases GIL during forward)
        print(f"Running with {workers} workers...", flush=True)
        results_map: dict[str, dict[str, Any]] = {}

        def _run(idx: int, inst: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            return idx, _run_one(inst)

        t_start = time.time()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run, i, inst): inst["question_id"]
                for i, inst in enumerate(data)
            }
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                _idx, result = future.result()
                results_map[result["question_id"]] = result
                if done_count % 5 == 0 or done_count == total or done_count <= 3:
                    elapsed = time.time() - t_start
                    rate = done_count / elapsed
                    eta = (total - done_count) / rate if rate > 0 else 0
                    print(
                        f"  [{done_count}/{total}] "
                        f"{elapsed:.0f}s elapsed, "
                        f"{rate:.1f} q/s, "
                        f"ETA {eta:.0f}s",
                        flush=True,
                    )

        # Write in original order
        results_log = []
        with open(output_path, "w") as f:
            for instance in data:
                qid = instance["question_id"]
                result = results_map[qid]
                f.write(
                    json.dumps(
                        {
                            "question_id": qid,
                            "hypothesis": result["hypothesis"],
                            "ranked": result["ranked"],
                        }
                    )
                    + "\n"
                )
                results_log.append(result)

    # Summary
    avg_score = (
        sum(r["top_score"] for r in results_log) / len(results_log)
        if results_log
        else 0
    )
    print(f"\nDone. {total} instances processed.")
    print(f"Average top retrieval score: {avg_score:.4f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python -m benchmarks.longmemeval.run"
            " <data.json> <output.jsonl> [n_results] [workers]"
        )
        sys.exit(1)
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    w = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    main(sys.argv[1], sys.argv[2], n, w)
