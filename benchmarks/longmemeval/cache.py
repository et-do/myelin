"""Pre-compute embeddings for LongMemEval benchmark data.

Embeds all unique sessions once and saves to disk.  Subsequent
benchmark runs load cached embeddings instead of re-encoding,
cutting run time from hours to under an hour.

Usage:
    uv run python -m benchmarks.longmemeval.cache \
        benchmarks/longmemeval/data/longmemeval_s_cleaned.json

Output:
    benchmarks/longmemeval/cache/<stem>_embeddings.npy   (N_chunks, 384)
    benchmarks/longmemeval/cache/<stem>_gists.npy        (N_sessions, 384)
    benchmarks/longmemeval/cache/<stem>_index.json       session -> chunk range
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Load .env before any HF/torch imports
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from myelin.config import MyelinSettings  # noqa: E402
from myelin.store.chunking import chunk  # noqa: E402
from myelin.store.perirhinal import summarise  # noqa: E402


def flatten_session(session: list[dict[str, str]]) -> str:
    """Flatten a list of chat turns into a single text block."""
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in session)


def build_cache(data_file: str) -> Path:
    """Build an embedding cache for all unique sessions in *data_file*.

    Returns the path to the generated index JSON file.
    """
    data_path = Path(data_file)
    data = json.loads(data_path.read_text())
    cfg = MyelinSettings()

    # ------------------------------------------------------------------
    # 1. Collect unique sessions (preserving insertion order)
    # ------------------------------------------------------------------
    unique_sessions: dict[str, list[dict[str, str]]] = {}
    for inst in data:
        for sid, sess in zip(inst["haystack_session_ids"], inst["haystack_sessions"]):
            if sid not in unique_sessions:
                unique_sessions[sid] = sess

    print(f"Unique sessions: {len(unique_sessions)}")

    # ------------------------------------------------------------------
    # 2. Flatten + chunk every session
    # ------------------------------------------------------------------
    all_chunks: list[str] = []
    all_gist_texts: list[str] = []
    index: dict[str, dict[str, object]] = {}

    n_sess = len(unique_sessions)
    t_chunk = time.time()
    for i, (sid, sess) in enumerate(unique_sessions.items()):
        text = flatten_session(sess)
        segments = chunk(
            text,
            max_chars=cfg.chunk_max_chars,
            overlap_chars=cfg.chunk_overlap_chars,
        )
        start = len(all_chunks)
        all_chunks.extend(segments)
        gist = summarise(text) or ""
        all_gist_texts.append(gist)
        index[sid] = {
            "chunk_range": [start, len(all_chunks)],
            "gist_idx": len(all_gist_texts) - 1,
        }
        if (i + 1) % 1000 == 0 or i + 1 == n_sess:
            elapsed = time.time() - t_chunk
            rate = (i + 1) / elapsed
            eta = (n_sess - i - 1) / rate if rate > 0 else 0
            print(
                f"  Chunked {i + 1}/{n_sess} ({len(all_chunks)} chunks) "
                f"— {elapsed:.0f}s, ETA {eta:.0f}s",
                flush=True,
            )

    print(f"Total chunks: {len(all_chunks)}")
    print(f"Gists generated: {len(all_gist_texts)}")

    # ------------------------------------------------------------------
    # 3. Batch-encode all chunks + gists
    # ------------------------------------------------------------------
    model = SentenceTransformer(cfg.embedding_model)

    t0 = time.time()
    print("Encoding chunks...")
    chunk_embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=256)
    t_chunks = time.time() - t0
    print(f"  {len(all_chunks)} chunks in {t_chunks:.1f}s")

    # Filter to non-empty gists for encoding, map back
    non_empty = [(i, g) for i, g in enumerate(all_gist_texts) if g]
    gist_embeddings = np.zeros(
        (len(all_gist_texts), chunk_embeddings.shape[1]), dtype=np.float32
    )
    if non_empty:
        t0 = time.time()
        print("Encoding gists...")
        idxs, texts = zip(*non_empty)
        gist_emb = model.encode(list(texts), show_progress_bar=True, batch_size=256)
        for i, emb in zip(idxs, gist_emb):
            gist_embeddings[i] = emb
        t_gists = time.time() - t0
        print(f"  {len(non_empty)} gists in {t_gists:.1f}s")

    # ------------------------------------------------------------------
    # 4. Save cache files
    # ------------------------------------------------------------------
    cache_dir = data_path.parent.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    stem = data_path.stem

    emb_path = cache_dir / f"{stem}_embeddings.npy"
    gist_path = cache_dir / f"{stem}_gists.npy"
    index_path = cache_dir / f"{stem}_index.json"

    np.save(emb_path, chunk_embeddings)
    np.save(gist_path, gist_embeddings)
    index_path.write_text(
        json.dumps(
            {
                "model": cfg.embedding_model,
                "chunk_max_chars": cfg.chunk_max_chars,
                "chunk_overlap_chars": cfg.chunk_overlap_chars,
                "total_chunks": len(all_chunks),
                "total_sessions": len(unique_sessions),
                "sessions": index,
            },
            indent=2,
        )
    )

    emb_mb = emb_path.stat().st_size / 1024 / 1024
    gist_mb = gist_path.stat().st_size / 1024 / 1024
    print(
        f"\nCache saved to {cache_dir}/\n"
        f"  {emb_path.name}: {emb_mb:.1f} MB\n"
        f"  {gist_path.name}: {gist_mb:.1f} MB\n"
        f"  {index_path.name}"
    )
    return index_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python -m benchmarks.longmemeval.cache <data.json>\n"
            "Example: python -m benchmarks.longmemeval.cache "
            "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
        )
        sys.exit(1)
    build_cache(sys.argv[1])
