# Architecture

This document describes Myelin's internal pipeline in detail — how memories are stored, recalled, and maintained. For setup and usage, see the [README](../README.md).

## Table of Contents

- [Store Walkthrough](#store-walkthrough)
- [Recall Walkthrough](#recall-walkthrough)
- [Inspecting Your Data](#inspecting-your-data)
- [Advanced Configuration](#advanced-configuration)
- [Neuroscience Mapping](#neuroscience-mapping)
- [References](#references)

---

## Store Walkthrough

### Example: `"We decided to use JWT with RS256 for the auth service"`

**1. Amygdala** — `store/amygdala.py` (input gate)
- Content is 54 chars — passes `min_content_length` (20)
- Embeds content and queries ChromaDB for nearest neighbors
- Max similarity < `dedup_similarity_threshold` (0.95) — not a duplicate
- → **Accepted**

**2. Prefrontal Cortex** — `store/prefrontal.py` (schema classification)
- Matches content against **5 schema templates**, each with 4–5 regex marker patterns:
  - `decision` → semantic (markers: "decided", "chose", "went with", "agreed on", …)
  - `preference` → procedural (markers: "always", "prefer", "convention", "style guide", …)
  - `procedure` → procedural (markers: "step 1", "how to", "deploy/build/test", …)
  - `plan` → prospective (markers: "TODO", "next steps", "roadmap", "going to", …)
  - `event` → episodic (markers: "yesterday", "debugged", "incident", "happened", …)
- `"decided"` fires the **decision** schema → `memory_type = "semantic"`
- Confidence = fraction of markers that fire (1/5 = 0.2 — one match is enough)
- No match → defaults to `"episodic"`

**3. Chunking** — `store/chunking.py` (pattern separation)
- Content is 54 chars — well under `chunk_max_chars` (1000) → stored as a single memory
- For longer content:
  - **Conversation detection**: looks for role markers (`user:`/`assistant:`) or named speakers (`Caroline:`, `Dr. Smith:`)
  - **Exchange-pair splitting**: keeps user + assistant turns together
  - **Topic-shift detection**: computes keyword overlap (Jaccard) between adjacent turns — when overlap drops below **15%**, forces a new chunk boundary
  - **Text fallback**: overlapping segments (**200-char overlap**) split at paragraph boundaries
- Embedding model has a 256-token window (~1000 chars) — chunking ensures every segment fits

**4. Entorhinal Cortex** — `store/entorhinal.py` (context coordinates)
- **LEC pathway** — topic keywords:
  - Term-frequency extraction on non-stop-words → top 5 keywords
  - → `ec_topics`: `["jwt", "rs256", "auth", "service"]`
- **MEC pathway** — domain region:
  - Matches against **6 region classifiers** (each a regex pattern set):
    - `technology` — code, api, database, docker, python, react, sql, …
    - `security` — auth, jwt, oauth, encrypt, token, password, rbac, …
    - `health` — doctor, diagnosis, fitness, prescription, therapy, …
    - `finance` — budget, invest, mortgage, tax, billing, invoice, …
    - `personal` — birthday, family, vacation, recipe, pet, wedding, …
    - `work` — meeting, sprint, deadline, roadmap, onboarding, …
  - Requires **≥2 pattern hits** to assign (avoids false positives)
  - `"auth"` + `"jwt"` → `ec_region`: `"security"`
- **Speaker detection** — "who" pathway:
  - Extracts named speakers from `"Name:"` patterns at line start
  - Filters generic roles (`user`, `assistant`, `human`, `ai`, `system`, `bot`)
  - → `ec_speakers`: e.g., `["Caroline", "Dr. Smith"]`

**5. Perirhinal Cortex** — `store/perirhinal.py` (gist extraction)
- **Extractive summarisation** (no LLM) — scores each sentence by:
  - Signal regex hits (decisions, state changes, personal facts, life events, activities)
  - Named entity rarity — names/places appearing in only 1–2 sentences score higher
- Selects top sentences up to ~200 chars
- → Gist embedding stored in a **separate ChromaDB summary collection**, linked to parent session via `parent_id`

**6. Hippocampus** — `store/hippocampus.py` (episodic store)
- Encodes content → **384-dim vector** via `all-MiniLM-L6-v2`
- Stores in ChromaDB with full metadata:
  - `memory_type`, `project`, `scope`, `tags`
  - `ec_topics`, `ec_region`, `ec_speakers`
  - `session_date`, `parent_id`
- → Returns memory ID: `"mem_a1b2c3"`

---

## Recall Walkthrough

### Example: `"What auth approach did we pick?"`

**1. Query Planner** — `recall/query_planner.py` (auto-filter inference)
- Matches query against regex patterns to infer **memory_type**:
  - `semantic` — "what did we decide/choose", "what is/was", "definition of"
  - `procedural` — "how do/does", "prefer/convention/style"
  - `episodic` — "when did/what happened", "yesterday/last week", "who said"
  - `prospective` — "what should", "plan/next/todo", "roadmap/deadline"
- Matches query against **10 scope patterns**: `auth`, `database`, `deploy`, `security`, `testing`, `api`, `frontend`, `backend`, `billing`, `monitoring`
- `"What … did we pick"` → `memory_type = "semantic"`, `scope = "auth"`

**2. Multi-Probe** — `store/hippocampus.py` (3 query variants)
- **Probe 1**: original query — `"What auth approach did we pick?"`
- **Probe 2**: keyword-focused — top 8 extracted keywords joined as text
- **Probe 3**: entity-expanded — spreads seed keywords through the neocortex entity graph, appends related entities discovered via co-occurrence

**3. Per-Probe Retrieval** (×3, each runs the full sub-pipeline)

Each probe passes through these stages:

- **a. Perirhinal gist search** (`store/perirhinal.py`) — queries the summary collection for familiar sessions, returns top-k `parent_id`s ranked by gist similarity
- **b. Dual-pathway ChromaDB search** (`store/hippocampus.py`):
  - *Filtered path*: applies auto-inferred `memory_type` + `scope` as ChromaDB `where` clause
  - *Unfiltered path*: applies only explicit params (`project`, `language`)
  - Merges by lowest distance per ID — prevents auto-filters from excluding relevant results
- **c. Entorhinal re-rank** (`store/entorhinal.py`) — keyword overlap boost:
  - `score *= 1.0 + entorhinal_boost (0.3) × Jaccard overlap`
  - If query mentions a known speaker name: `score *= 1.0 + speaker_boost (0.2)`
- **d. Perirhinal gist boost** — sessions matching gist search:
  - `score *= 1.0 + perirhinal_boost (0.5) × gist similarity`
- **e. Gist retrieval pathway** — injects best chunks from high-scoring gist sessions (batch ChromaDB lookup by `parent_id`)
- **f. Cross-encoder re-ranking** (`store/hippocampus.py`) — blends CE and bi-encoder scores:
  - `score = α × CE_normalized + (1-α) × bi_normalized` where `α = neocortex_weight (0.6)`
- **g. Time-cell boost** (`recall/time_cells.py`) — detects temporal expressions (`"3 days ago"`, `"last Tuesday"`, `"in March"`) with ± buffer windows (±1 day for days, ±3 for weeks, ±7 for months):
  - Recency formula: `boost = 2^(-age_days / half_life_days)`
  - Additive: `score += temporal_boost (0.6)` for date-range matches
- **h. Lateral inhibition** (`store/hippocampus.py`) — max `lateral_k` (1) result per session/scope, keeps highest-scoring per group

**4. Multiprobe Merge** — `store/hippocampus.py`
- Pools all candidates from 3 probes, keeps best score per memory ID
- Re-scores merged pool with cross-encoder against original query (at `α/2` blending weight to preserve per-probe rankings)
- Soft recency gradient: `score *= 1.0 + 0.1 × 2^(-age / half_life)`

**5. Spreading Activation** — `store/neocortex.py` (entity graph boost)
- Extracts entities from top results, walks the neocortex entity graph (co-occurrence edges, max 2 hops, distance-decayed propagation)
- `score *= 1.0 + spreading_boost (0.15) × activation`

**6. Session Evidence Aggregation**
- Groups results by session — sessions with multiple retrieved chunks get a logarithmic boost:
  - `score *= 1.0 + log1p(chunk_count - 1) × agg_boost`
- Applied to top chunk per session — rewards sessions with broad evidence coverage

**7. Lateral Inhibition** (final pass)
- Enforces session diversity one more time after merge + boosts
- Max `lateral_k` (1) result per session

**8. Return top-5**

| Result | Score |
|--------|-------|
| "We decided to use JWT with RS256 for the auth service" | 0.94 |
| "RS256 key rotation runs every 90 days" | 0.71 |
| … | … |

**Post-recall:**
- **Hebbian LTP** (`recall/activation.py`) — co-retrieved memories strengthen mutual links: `weight += hebbian_delta` per pair, future boost = `hebbian_scale × log1p(weight)`
- **Thalamus overlay** (`store/thalamus.py`) — prepends any pinned memories (L0 identity/system context, L1 critical facts) to every result set

---

## Inspecting Your Data

You can inspect and validate the contents of your Myelin databases directly.

### Quick health check

```bash
myelin status
```

Returns memory count, summary count, consistency status, data directory, and model info.

### Export all memories to readable JSON

```bash
myelin export memories.json
```

This dumps every memory with its full metadata (content, timestamps, access counts, memory type, project, scope, tags, etc.) into a single JSON file you can open in any editor.

### Browse SQLite databases directly

The `.db` files are standard SQLite databases. You can open them with any SQLite tool:

```bash
# Hebbian co-access links (which memories are associated)
sqlite3 ~/.myelin/hebbian.db ".tables"
sqlite3 ~/.myelin/hebbian.db "SELECT * FROM hebbian_links ORDER BY weight DESC LIMIT 10;"

# Semantic network (entities and relationships built by consolidation)
sqlite3 ~/.myelin/neocortex.db ".tables"
sqlite3 ~/.myelin/neocortex.db "SELECT * FROM entities LIMIT 10;"
sqlite3 ~/.myelin/neocortex.db "SELECT * FROM edges ORDER BY weight DESC LIMIT 10;"

# Pinned memories and recency tracking
sqlite3 ~/.myelin/thalamus.db ".tables"
sqlite3 ~/.myelin/thalamus.db "SELECT * FROM pinned;"
sqlite3 ~/.myelin/thalamus.db "SELECT * FROM recency ORDER BY last_accessed DESC LIMIT 10;"
```

> [!TIP]
> On Windows, install `sqlite3` via `winget install SQLite.SQLite` or use [DB Browser for SQLite](https://sqlitebrowser.org/) for a GUI. The `sqlite-utils` package (included with Myelin) also works: `sqlite-utils tables ~/.myelin/hebbian.db` and `sqlite-utils rows ~/.myelin/hebbian.db hebbian_links --limit 10`.

### Browse the ChromaDB vector store

ChromaDB stores embeddings and metadata in `~/.myelin/chroma/`. You can query it programmatically:

```python
import chromadb
client = chromadb.PersistentClient(path="~/.myelin/chroma")
collection = client.get_collection("memories")

# Count all memories
print(collection.count())

# Peek at the first 5 memories
results = collection.peek(limit=5)
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(meta.get("memory_type"), "-", doc[:100])
```

### What to look for

| Check | What It Tells You |
|---|---|
| `myelin status` shows `consistent: true` | Embedding count matches metadata count — no orphans |
| `myelin export` has entries with your project name | Memories are being tagged correctly |
| `hebbian.db` has rows | Co-recall learning is active (fires after multiple recalls) |
| `neocortex.db` has entities/edges | Consolidation has run and built the semantic network |
| `thalamus.db` pinned table has rows | You have pinned memories that prepend to every recall |

---

## Advanced Configuration

All parameters use environment variables with a `MYELIN_` prefix. For common settings, see the [Configuration section in the README](../README.md#configuration). Below are the advanced tuning parameters.

### Boosting Parameters

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `entorhinal_boost` | `0.3` | Keyword overlap multiplier |
| `speaker_boost` | `0.2` | Speaker mention multiplier |
| `perirhinal_boost` | `0.5` | Gist-match multiplier |
| `perirhinal_top_k` | `10` | Gist summaries to search |
| `temporal_boost` | `0.6` | Temporal reference boost (additive, after CE) |
| `recency_half_life_days` | `180` | Soft recency gradient half-life |

### Spreading Activation Parameters

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `spreading_activation` | `true` | Entity-graph post-retrieval boost |
| `spreading_boost` | `0.15` | Per-entity activation multiplier |
| `spreading_max_depth` | `2` | Max hops in entity graph |
| `spreading_top_k` | `10` | Max related entities to activate |

### Maintenance Parameters

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `max_idle_days` | `90` | Days of inactivity before pruning eligibility |
| `min_access_count` | `2` | Accesses needed to survive pruning |
| `max_memories` | `0` | Hard memory cap; 0 disables. When exceeded, LRU memories are evicted (pinned memories are never evicted) |
| `decay_interval_hours` | `0` | Auto-decay sweep interval in hours; 0 disables background timer |
| `hebbian_delta` | `0.1` | Co-access weight increment |
| `hebbian_scale` | `0.1` | Logarithmic boost scale |
| `thalamus_recency_limit` | `20` | Recency buffer size |
| `consolidation_interval` | `50` | Queue a background consolidation every N stores (0 = disabled) |
| `log_level` | `INFO` | Logging verbosity (structured JSON to stderr) |

### Background Worker Parameters

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `worker_decay_interval_hours` | `24.0` | How often the background worker runs a decay sweep (0 = disabled) |
| `worker_queue_maxsize` | `10` | Max pending consolidation tasks in the queue; extras are dropped safely |

The background worker runs in a daemon thread alongside the MCP server. It handles two types of work:

- **Consolidation** — when `do_store` reaches every `consolidation_interval` stores, it queues a task instead of blocking the store call. The response includes `"consolidation": "scheduled"` so your agent knows work is in flight.
- **Periodic decay** — every `worker_decay_interval_hours`, the worker automatically prunes stale memories. No need to remember to run `myelin decay` manually.

Worker status is visible in `myelin status` output under the `"worker"` key: last consolidation time, last decay time, and current queue depth.

---

## Neuroscience Mapping

| Myelin Component | Brain Region | Principle | Fidelity |
|-----------------|-------------|-----------|----------|
| `store/hippocampus.py` | Hippocampus | Rapid one-shot encoding, pattern completion | **High** |
| `store/chunking.py` | Dentate Gyrus | Sparse coding / pattern separation | **High** |
| `store/prefrontal.py` | Prefrontal Cortex | Schema-consistent encoding | **High** |
| `recall/query_planner.py` | Prefrontal Cortex | Inhibitory gating | **High** |
| `store/neocortex.py` | Temporal Neocortex | Spreading activation | **High** |
| `store/consolidation.py` | Sleep replay | CLS theory | **High** |
| `store/amygdala.py` | Amygdala | Significance gating | **Medium** |
| `store/entorhinal.py` | Entorhinal Cortex | Context coordinates | **Medium** |
| `store/perirhinal.py` | Perirhinal Cortex | Familiarity signaling | **Medium** |
| `store/thalamus.py` | Thalamus | Sensory relay + attention | **Medium** |
| `recall/activation.py` | Synapses | Hebbian LTP | **Medium** |
| `recall/time_cells.py` | Hippocampal time cells | Temporal context | **Medium** |
| `recall/decay.py` | Synapse pruning | Ebbinghaus forgetting curve | **Low** |

**Faithfully modeled:** CLS (fast hippocampal encode + slow neocortical consolidation), pattern separation, spreading activation, encoding specificity, retrieval-induced inhibition, Hebbian learning.

**More metaphorical:** No neural dynamics (spiking, LTP/LTD). Amygdala is an importance scorer, not emotional valence. Consolidation is triggered, not sleep-driven.

---

## References

| Principle | Paper | Used In |
|-----------|-------|---------|
| Schema-consistent encoding | [Tse et al. (2007). *Science*](https://doi.org/10.1126/science.1135935) | `store/prefrontal.py` |
| Spreading activation | [Collins & Loftus (1975). *Psych. Review*](https://doi.org/10.1037/0033-295X.82.6.407) | `store/neocortex.py` |
| Complementary learning systems | [McClelland et al. (1995). *Psych. Review*](https://doi.org/10.1037/0033-295X.102.3.419) | `store/consolidation.py` |
| Retrieval-induced inhibition | [Anderson & Green (2001). *Nature*](https://doi.org/10.1038/35066572) | `recall/query_planner.py` |
| Encoding specificity | [Tulving & Thomson (1973). *Psych. Review*](https://doi.org/10.1037/h0036255) | Metadata filters |
| Schema augmented memory | [van Kesteren et al. (2012). *Trends in Neurosciences*](https://doi.org/10.1016/j.tins.2012.02.001) | Schema detection |
| Hebbian learning | [Hebb (1949). *The Organization of Behavior*](https://en.wikipedia.org/wiki/Hebbian_theory) | `recall/activation.py` |
| Sleep and memory | [Rasch & Born (2013). *Physiological Reviews*](https://doi.org/10.1152/physrev.00032.2012) | `store/consolidation.py` |
