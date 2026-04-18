# Myelin — Production Roadmap

## P0 — Will break in production

### Error handling & logging
- [x] Replace 6 bare `except Exception: pass` with logged exceptions (debug/warning level)
- [x] Add module-level loggers to hippocampus.py, neocortex.py, activation.py
- [x] Input size validation at API boundary (500K char store limit, 10K char query limit)

### Concurrency safety
- [x] Add `threading.RLock` around all ChromaDB read/write operations in Hippocampus
- [x] Add lock around SummaryIndex operations (covered by Hippocampus lock — SummaryIndex is called within locked methods)
- [x] Add lock around SemanticNetwork SQLite operations
- [x] Audit Hebbian tracker for concurrent write safety
- [x] Enable SQLite WAL mode for concurrent reads during writes
- [x] `check_same_thread=False` on SQLite connections (serialized by RLock)
- [ ] Consider read-write lock (multiple concurrent reads, exclusive writes) — defer; single-client MCP, GIL limits parallelism, lock hold <300ms

### Recall latency (currently ~1.4s production estimate, target <500ms)
- [x] Batch gist retrieval pathway — collect all parent_ids, single `$or` query + ID fallback instead of N serial lookups (saves ~450ms)
- [ ] Add query embedding cache (LRU, ~1000 entries) — same question twice shouldn't re-encode (saves 50ms on hit, but MCP queries are rarely identical — low ROI, defer)
- [ ] Gate cross-encoder: skip if top bi-encoder score > high-confidence threshold (saves ~660ms on easy queries) — defer until profiling confirms CE is the bottleneck; risky to gate the key quality signal
- [ ] Make cross-encoder opt-in per MCP tool call (`thorough=true`) — fast path by default — defer; changes MCP API surface
- [x] Profile end-to-end recall to find actual bottleneck breakdown (embed / search / boost / rerank) — **Result: ~300ms avg on 5-session dataset** (embed 10-22ms, ChromaDB 3-9ms, gist 5-22ms, CE 40-80ms, multi-probe overhead ~150ms). Well under 500ms target.

### Store latency (~185ms simple, ~600ms long) — defer to P1 background worker
- [ ] Make gist embedding async — return from store() immediately, encode gist in background — needs P1 background worker (worker exists, gist not yet offloaded)
- [ ] Make Hebbian weight update async — fire-and-forget after store completes — needs P1 background worker (worker exists, Hebbian not yet offloaded)
- [ ] Make EC keyword extraction async — doesn't block the store's critical path — needs P1 background worker (worker exists, EC not yet offloaded)

### Data integrity
- [x] Wrap store() in try/except — gist indexing failures no longer crash store (chunks preserved)
- [x] Add startup integrity check — verify chunk count matches summary count per parent_id — `Hippocampus.check_integrity()` + `myelin status` shows counts
- [ ] Implement soft-delete (tombstone) before hard-delete in decay/forget — low risk since decay is manual CLI-only; move to P1

## P1 — Will degrade over weeks/months

### Cold start (~4s model load)
- [x] Model warm-up on startup — encode a dummy query + cross-encode a dummy pair
- [x] Lazy-load cross-encoder — only load on first recall that needs it, not at startup
- [ ] Evaluate keeping models in shared memory across server restarts (mmap weights)

### Background worker thread (async operations)
- [x] Create single background worker with task queue
- [x] Move consolidation to background worker (currently blocks every 50th store)
- [x] Move decay sweep to background worker (currently manual CLI only)
- [ ] Move gist embedding to background worker (fire after store returns)
- [ ] Move Hebbian weight updates to background worker
- [ ] Move periodic dedup sweep to background worker
- [ ] Add idle detection — run heavy tasks (consolidation, dedup) only when no active requests
- [x] Never run any background task inline with store() or recall() — falls back to inline only when worker not started (tests)

### Unbounded growth
- [ ] Add hard storage cap (configurable max memories) with LRU eviction
- [ ] Auto-decay on background timer (not manual CLI)
- [x] Fix immortal memory bug: access_count >= 2 should NOT exempt from all decay — added `max_idle_days_absolute=365` hard cap
- [ ] Periodic dedup sweep — find and merge near-duplicate memories that slipped past gate
- [ ] ChromaDB HNSW compaction (rebuild index after large deletes)

### Embedding model migration
- [ ] Plan for embedding model upgrades — all stored vectors become stale when model changes
- [ ] Options: lazy re-encode on access, bulk migration script, or dual-index during transition
- [x] Store model version in ChromaDB collection metadata for detection

## P2 — Missing neuroscience-grounded features

### Spreading activation
- [x] Wire SemanticNetwork.spread() into recall for multi-entity queries
- [x] Use entity graph expansion to generate additional query terms
- [x] Particularly valuable for multi-session questions ("What connections between X and Y?")
- [x] Integrated into hippocampus.recall() via multi-probe + post-retrieval boost (v8)

### Encoding context (Tulving's encoding specificity)
- [ ] Store lightweight "situation embedding" at encoding time (what was the agent doing?)
- [ ] During recall, boost memories where encoding context matches current context
- [ ] Could use the conversation's recent turns as encoding context

### Salience scoring (amygdala enhancement)
- [ ] Compute salience at store time: information density, specificity, belief-update magnitude
- [ ] Store salience as metadata, use as retrieval boost
- [ ] High-salience memories should resist decay longer

### Fast familiarity reject gate
- [ ] If all gist scores are below threshold, skip expensive cross-encoder AND gist pathway
- [ ] Return "no relevant memories" fast instead of wasting compute on bad candidates
- [ ] Mirrors brain's fast "nothing here" signal from perirhinal cortex

### Sleep replay / offline consolidation
- [ ] Background thread that periodically replays recent memories during idle
- [ ] Strengthen cross-references between related memories
- [ ] Extract and store higher-level patterns ("user generally prefers X")
- [ ] Promote frequently-accessed episodic memories to semantic facts

## P2 — Production operability

### Observability
- [x] Structured logging (JSON) with request IDs
- [ ] Latency histograms per operation (store, recall, rerank, gist, embed)
- [ ] Memory count / storage size metrics exposed via MCP resource
- [ ] Silent rejection logging (amygdala gate rejects with reason)
- [ ] Recall audit trail — which signals contributed to ranking, what was considered and rejected

### Startup / shutdown
- [x] Graceful shutdown handler — flush pending background tasks, close ChromaDB + SQLite
- [x] Health check endpoint (model loaded? ChromaDB responsive? background worker alive? disk space OK?) — lightweight `health` MCP tool
- [x] Configuration validation on startup (reject invalid combos, warn on aggressive thresholds) — pydantic field validators for ranges, overlap < max, positive ints

## P2 — Benchmark / production parity

### Cached store path — private API coupling
- [ ] Add `store_precomputed()` public method to Hippocampus (accepts pre-computed embeddings, still runs gate + metadata)
- [ ] Migrate `run_instance_cached()` to use `store_precomputed()` instead of `hc._collection.add()` / `hc._summaries._collection.upsert()`
- [ ] Remove direct calls to `Hippocampus._build_chroma_meta()`, `_attach_ec_coords()`, `_attach_session_date()`
- [ ] Remove hardcoded `batch = 5000` — use config or Hippocampus method

### Server private API usage
- [x] `do_consolidate()` uses `hc._collection.get()` — exposed `Hippocampus.get_all_content()` public method

### Gate check gap
- [ ] Cached benchmark path skips `passes_gate()` — near-duplicates and short content stored that production would reject
- [ ] Once `store_precomputed()` exists, gate runs automatically

### Classification granularity
- [ ] Cached path classifies full session text; production classifies per-chunk after segmentation
- [ ] Minor impact (regex-based classifier is coarse), but should be consistent

### User isolation
- [ ] Enforce user-level isolation at API boundary (not just scope field)
- [ ] Separate ChromaDB collections per user, or strict where-clause enforcement
- [ ] Prevent cross-user memory leakage

### Data management
- [x] Export/import memories (JSON) for portability and backup — `myelin export`/`myelin import` CLI commands
- [ ] Point-in-time backup strategy (ChromaDB snapshot + SQLite backup)
- [ ] Memory versioning — when a preference updates, keep history (not just overwrite)
- [x] CLI command: `myelin debug-recall "query"` — show full ranking breakdown for debugging

## P3 — Performance tuning (after production basics)

### ChromaDB tuning
- [ ] Tune HNSW parameters: ef_construction (default 100), M (default 16) — benchmark with real data
- [ ] Consider HNSW ef_search parameter for recall-speed tradeoff
- [ ] Evaluate IVF-PQ index for 1M+ vector scale

### Embedding optimization
- [ ] Evaluate ONNX runtime for bi-encoder (2-3x faster than PyTorch on CPU)
- [ ] Evaluate quantized cross-encoder (INT8) for latency reduction
- [ ] Consider caching chunk embeddings to disk (avoid re-encoding on restart with PersistentClient)

### Async recall pipeline
- [ ] Run bi-encoder search + gist search in parallel (both need query embedding, independent after that)
- [ ] Return fast bi-encoder top-k immediately as "draft" results via MCP streaming
- [ ] Async refine with cross-encoder only if time budget allows
- [ ] Run temporal dual-pathway search concurrently with main search
