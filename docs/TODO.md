# Myelin — Production Roadmap

> Items without a checkbox are resolved. Open items are `[ ]`. Deferred items note the reason.

---

## P1 — Active work

### Background worker (async hot path)
Worker infrastructure exists; these operations still block store/recall:
- [ ] Move gist embedding to background worker (fire after store returns)
- [ ] Move Hebbian weight updates to background worker
- [ ] Move periodic dedup sweep to background worker
- [ ] Add idle detection — run consolidation/dedup only when no active requests

### Unbounded growth
- [ ] Periodic dedup sweep — find and merge near-duplicates that slipped past the gate
- [ ] ChromaDB HNSW compaction after large-scale deletes

### Embedding model migration
- [ ] Build migration tooling — lazy re-encode on access, bulk script, or dual-index — defer until model actually changes (detection + upgrade path already documented)

### Data integrity
- [ ] Soft-delete (tombstone) before hard delete in decay/forget — low risk, deferred from P0

---

## P2 — Queued

### MCP protocol completeness
- [ ] Add `list_memories(project, scope, memory_type, limit)` MCP tool — lets agents browse without a semantic query ("what do you know about me?")
- [ ] Tool annotations (MCP 0.9+): `readOnlyHint` on `recall`/`status`/`stats`/`health`; `destructiveHint` on `forget`/`decay_sweep`; `idempotentHint` on `pin_memory`/`unpin_memory`
- [ ] Structured error responses — `store` rejections should return `isError: true` in the MCP tool result, not `{"status": "rejected"}` with 200
- [ ] User/tenant isolation — `agent_id` is a soft filter today, not enforced at the collection boundary; a missing filter can leak across agents
- [ ] Streaming recall — defer until MCP SDK streaming is mainstream

### Recall quality (neuroscience-grounded)
- [ ] Fast familiarity reject gate — if all gist scores are below threshold, skip cross-encoder entirely (mirrors perirhinal "nothing here" signal)
- [ ] Salience scoring at store time — information density/specificity as a retrieval boost and decay resistance signal
- [ ] Encoding context (Tulving specificity) — store a lightweight situation embedding at encode time, boost matching context on recall
- [ ] Sleep replay — idle background thread replays recent episodes, strengthens cross-references, promotes recurring facts to semantic memory

### Recall latency (deferred optimizations)
- [ ] Query embedding cache (LRU ~1000) — low ROI since MCP queries are rarely identical, defer
- [ ] Cross-encoder gating — skip CE when top bi-encoder score > high-confidence threshold; defer until profiling confirms CE is bottleneck
- [ ] Cross-encoder opt-in per call (`thorough=true`) — changes API surface, defer
- [ ] Parallel bi-encoder + gist search, async CE refinement

### Observability
- [ ] Latency histograms per operation (store, recall, rerank, gist, embed)
- [ ] Silent rejection logging — amygdala gate rejects with reason code
- [ ] Recall audit trail — which signals (semantic, gist, Hebbian, temporal) contributed to final ranking
- [ ] Memory metrics as MCP resource (subscribable, not just tool-polled)

### Data management
- [ ] Export/import round-trip fidelity — verify numeric/timestamp fields (access_count, created_at, tags, parent_id) survive exact serialization
- [ ] Point-in-time backup (ChromaDB snapshot + SQLite `.backup()`)
- [ ] Memory versioning — keep history when a preference is overwritten

### Benchmark parity (benchmark ↔ production code paths)
- [ ] `store_precomputed()` public API on Hippocampus — benchmark currently bypasses `passes_gate()` and uses private `_collection.add()` directly
- [ ] Once `store_precomputed()` exists: gate check and classification granularity gaps close automatically
- [ ] Multi-agent isolation benchmark — store memories across N agent_ids, measure cross-contamination rate on recall
- [ ] p50/p95/p99 latency under concurrent load (simultaneous store + recall)
- [ ] LoCoMo temporal score wired into regression baseline

### Test coverage
- [ ] `test_amygdala.py` — test rejection *reasons* explicitly (too short, near-duplicate), not just pass/fail outcome
- [ ] Chunking boundary integration — store long doc, recall by mid-section content
- [ ] Pin immortality invariant — pinned memory meeting decay criteria must never be pruned
- [ ] Export/import field fidelity — numeric/timestamp fields survive round-trip
- [ ] `test_mcp_integration.py` tool list assertion — extend to cover `pin_memory`, `unpin_memory`, `decay_sweep`, `consolidate`, `stats`, `ingest`

---

## P3 — Research / future

### Vector index tuning
- [ ] Tune HNSW ef_construction (default 100) and M (default 16) against real data
- [ ] Evaluate IVF-PQ for 1M+ vector scale

### Model optimization
- [ ] ONNX runtime for bi-encoder (2-3× faster on CPU)
- [ ] INT8 quantized cross-encoder
- [ ] Shared memory model weights across server restarts (mmap)
