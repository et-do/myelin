<div align="center">
  <img src="img/myelin.png" alt="Myelin" width="200"/>

  <h1>Myelin</h1>

  <strong>Neuromorphic long-term AI memory</strong> — brain-inspired persistent context for AI agents.

  <p>
    <a href="#quick-start">Quick Start</a> ·
    <a href="#how-it-works">How It Works</a> ·
    <a href="#bencmark-results">Benchmarks</a> ·
    <a href="#cli--mcp-tools">CLI & Tools</a> ·
    <a href="CONTRIBUTING.md">Contributing</a>
  </p>

  <p>
    <a href="https://github.com/et-do/myelin/blob/main/LICENSE"><img src="https://img.shields.io/github/license/et-do/myelin?colorA=363a4f&colorB=b7bdf8" alt="License"></a>
    <a href="https://pypi.org/project/myelin-mcp/"><img src="https://img.shields.io/pypi/v/myelin-mcp?colorA=363a4f&colorB=b7bdf8&label=pypi" alt="PyPI"></a>
    <a href="https://github.com/et-do/myelin/actions/workflows/ci.yml"><img src="https://github.com/et-do/myelin/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://github.com/et-do/myelin/stargazers"><img src="https://img.shields.io/github/stars/et-do/myelin?colorA=363a4f&colorB=b7bdf8&style=flat" alt="Stars"></a>
    <a href="https://github.com/et-do/myelin/issues"><img src="https://img.shields.io/github/issues/et-do/myelin?colorA=363a4f&colorB=b7bdf8" alt="Issues"></a>
    <a href="https://github.com/et-do/myelin/commits/main"><img src="https://img.shields.io/github/last-commit/et-do/myelin?colorA=363a4f&colorB=b7bdf8" alt="Last Commit"></a>
    <img src="https://img.shields.io/badge/python-≥3.11-b7bdf8?colorA=363a4f" alt="Python 3.11+">
  </p>

  <br />

  <p>
    Myelin gives AI tools (GitHub Copilot, Claude, Cursor) a local, private memory system<br />
    modeled after how the human brain stores and retrieves information.<br />
    It encodes context, strengthens with use, separates patterns, and prunes what fades —<br />
    no LLM calls, no API keys, fully offline.
  </p>
</div>

<br />

## Table of Contents

- [Why Myelin](#why-myelin)
- [Quick Start](#quick-start)
- [Setup Guides](#setup-guides)
- [Teaching Your Agent](#teaching-your-agent)
- [Benchmarks](#benchmark-results)
- [How It Works](#how-it-works)
- [CLI & MCP Tools](#cli--mcp-tools)
- [Configuration](#configuration)
- [Upgrading](#upgrading)
- [Development](#development)
- [Contributing](CONTRIBUTING.md)

---

## Why Myelin

**The problem:** Every AI agent conversation starts from scratch. Context is lost between sessions, across tools, across projects. Agents repeat mistakes, forget decisions, and can't build on prior work.

**What Myelin does:**

- **Persistent memory across sessions** — decisions, patterns, and debugging insights survive after the chat window closes
- **Cross-agent context** — Copilot, Claude, and Cursor share the same memory. What one agent learns, all agents can recall.
- **Cross-project knowledge** — architectural patterns from project A inform decisions in project B
- **Self-organizing** — auto-classifies memory types (decisions, procedures, events), auto-infers recall filters, auto-prunes stale knowledge
- **Private and local** — all data stays on your machine (or your team's server). No API keys. No cloud dependency. No data leaving your network.
- **Gets better with use** — frequently co-recalled memories strengthen their association ([Hebbian learning](#hebbian-boost)). The more you use it, the better recall gets.
- **98.2% Recall@5 on LongMemEval** — beats LLM-based systems using only local 22M-parameter models

---

## Quick Start

**Requirements:** Python 3.11+ · ~500 MB disk (models download on first run) · No GPU · No API keys

### Install

**Option A — uv (recommended, no admin required):**

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install myelin-mcp
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv tool install myelin-mcp
```

**Option B — pip:**

```bash
# --user installs to ~/.local/bin (Linux/macOS) or %APPDATA%\Python\...\Scripts (Windows)
# No admin rights required
pip install --user myelin-mcp
```

> [!NOTE]
> **After installing, run `myelin status` once before opening VS Code.** This downloads ~500 MB of embedding models and pre-warms them so your first tool call is instant. The MCP server starts without waiting for models, but the first `store` or `recall` will be slower if models haven't loaded yet.
> If `myelin` isn't found, your user bin dir isn't on `$PATH`. Add `~/.local/bin` to your shell profile (Linux/macOS), or find the correct path with `python -m site --user-scripts` (Windows).

### Configure your AI tool

Add Myelin as an MCP server.

**VS Code** — open `mcp.json` (`Ctrl+Shift+P` → `MCP: Open User Configuration`):

```json
{
  "servers": {
    "myelin": {
      "command": "myelin",
      "args": ["serve"]
    }
  }
}
```

> [!IMPORTANT]
> **`myelin` not found by VS Code?** MCP hosts launch in a clean environment and may not inherit your shell `$PATH`. Use the full path (e.g. `~/.local/bin/myelin` on Linux/macOS) or switch to `"command": "uvx"` with `"args": ["myelin-mcp", "serve"]` — `uvx` resolves the tool location automatically.

**Claude Desktop** — add the same block to `claude_desktop_config.json`.

### Verify

1. **Restart VS Code** (`Ctrl+Shift+P` → `Developer: Reload Window`).
2. Open **Output** panel (`Ctrl+Shift+U`) → **MCP: myelin** — you should see the server start and discover tools:

<p align="center">
  <img src="img/vscode-mcp-output.png" alt="MCP output showing Myelin server starting and discovering 9 tools" width="700"/>
</p>

3. Click **Configure Tools** (filter icon in Chat input bar) to verify Myelin's tools are listed:

<p align="center">
  <img src="img/vscode-configure-tools.png" alt="Configure Tools button in VS Code Chat" width="400"/>
</p>

<p align="center">
  <img src="img/vscode-myelin-tools.png" alt="Myelin tools listed in VS Code Configure Tools" width="600"/>
</p>

4. Ask your agent: *"Check myelin status"* — it should call the `status` tool and return memory counts.

---

## Setup Guides

Myelin stores all data in a single directory (`~/.myelin` by default). How you deploy that directory determines the scope of memory.

### Personal (Cross-Project)

**Best for:** Solo developers who want one memory across all projects and agents.

This is the default setup. Follow the [Quick Start Install](#install) steps above — they set up user-level config by default.

All projects and agents share `~/.myelin/`. Use `project` metadata when storing to keep work organized — just tell your agent:

> *"Store this as project=backend, scope=auth"*

Recall can filter by project or search across everything.

### Per-Repository

**Best for:** Teams who want memory scoped to a single repo, committed alongside the code.

**1. Set `MYELIN_DATA_DIR` to a path inside the repo:**

Create a `.vscode/mcp.json` in the repo:

```json
{
  "servers": {
    "myelin": {
      "command": "myelin",
      "args": ["serve"],
      "env": {
        "MYELIN_DATA_DIR": "${workspaceFolder}/.myelin"
      }
    }
  }
}
```

> [!TIP]
> Use `"command": "uvx"` with `"args": ["myelin-mcp", "serve"]` if `myelin` isn't on the PATH in your MCP host's environment.

**2. Decide whether to commit the data:**

- **Commit `.myelin/`** — the team shares accumulated knowledge (architectural decisions, conventions, debugging history). New contributors inherit project memory. Good for stable, curated knowledge.
- **Gitignore `.myelin/`** — each developer builds their own memory. Add `.myelin/` to `.gitignore`. Good for personal workflow memory you don't want to share.

**3. Add agent instructions** (see [Teaching Your Agent](#teaching-your-agent) below).

### Multi-Agent (Shared Instance, Isolated Namespaces)

**Best for:** Multiple trusted agents sharing one Myelin data directory, each
working from its own memory pool.

Myelin filters recalls by `agent_id` at the database level — a recall with
`agent_id="copilot"` will never return a memory stored with `agent_id="ci-bot"`.
The filter is unconditional once applied.

**However, `agent_id` is not authenticated.** The server accepts whatever
value the caller supplies. Any agent — or user — that knows another agent's
`agent_id` can read and write to that namespace. There is no credential,
token, or OS-level enforcement preventing this.

This means `agent_id` is a **namespace convention for cooperating agents**,
not an access-control boundary. It is appropriate when:
- Agents are trusted (same team, same deployment)
- The goal is preventing accidental cross-contamination between agent contexts
- You are not trying to hide memories from a potentially adversarial caller

If hard isolation is a requirement, run separate Myelin instances pointing at
separate data directories.

**To keep agents in their own namespace**, add a line to each agent's
instructions file (`.github/copilot-instructions.md` or equivalent):

```markdown
Always pass agent_id="copilot" on every myelin store and recall call.
```

Without this instruction, an agent may omit `agent_id` and write to the global
namespace — visible to all callers regardless of `agent_id`.

The global namespace (no `agent_id`) is intentional for shared project context
that every agent should see, such as architectural decisions and conventions.

```bash
# Scope a debug-recall diagnostic to one namespace
myelin debug-recall "auth approach" --agent-id copilot
```

---

### Team / Cloud

**Best for:** Organizations that want shared memory across team members and CI environments.

Myelin itself is a local process — it reads/writes to a data directory. For team sharing, you point that directory at shared storage. This does not require deploying Myelin as a hosted service.

**Option A: Shared network drive or mounted volume**

Point `MYELIN_DATA_DIR` to a shared filesystem (NFS, SMB, EFS, GCS FUSE, etc.):

```bash
export MYELIN_DATA_DIR=/mnt/team-memory/myelin
```

SQLite uses WAL mode and file-level locking, which works on most network filesystems for light concurrency. For heavy concurrent writes, consider Option B.

**Option B: Sync via export/import**

Use the CLI to periodically export and import memory between environments:

```bash
# On one machine — export
myelin export team-memory.json

# On another machine — import
myelin import team-memory.json
```

This can be automated in CI (e.g., export after each deploy, import at dev environment setup).

**Seeding from existing docs**

Use `myelin ingest` to bulk-load existing documentation, notes, or wiki exports into memory:

```bash
myelin ingest ./docs/ --project myapp --source wiki
myelin ingest ./decisions/ --scope architecture --source adr
```

Markdown files with YAML frontmatter get their metadata (`project`, `scope`, `tags`, `memory_type`) automatically applied per file. JSON exports from other Myelin instances can also be ingested directly.

**Option C: Shared server (future)**

A dedicated Myelin server with HTTP transport is on the roadmap. For now, the export/import workflow covers most team use cases.

---

## Teaching Your Agent

Connecting the MCP server gives your agent the *ability* to store and recall — but it won't use memory automatically unless you tell it to.

### Agent Instructions

Add a `.github/copilot-instructions.md` (VS Code / Copilot) or equivalent instructions file to your project:

```markdown
## Memory

You have access to a long-term memory system (Myelin) via MCP tools.

### When to Recall
- At the START of every task, recall relevant context about the current project,
  file, or problem domain.
- Before making architectural decisions, recall past decisions and their rationale.
- When debugging, recall similar past issues and their resolutions.

### When to Store
- After making significant decisions — record WHAT was decided and WHY.
- After resolving non-trivial bugs — record the symptoms, root cause, and fix.
- When discovering project conventions, patterns, or gotchas.
- After completing a meaningful feature — summarize the approach and trade-offs.

### How to Store Effectively
- Always include `project` metadata (e.g., project="myapp").
- Use `scope` to organize by domain (e.g., scope="auth", scope="database").
- Use `tags` for cross-cutting concerns (e.g., tags="performance,optimization").
- Use `memory_type` when it's clear: "semantic" for decisions/facts,
  "procedural" for how-to, "episodic" for events, "prospective" for plans.
- Be specific. "We use JWT RS256 because asymmetric keys let the API gateway
  verify without the signing secret" is better than "We use JWT."

### Maintenance
- After extended sessions (10+ stores), run `consolidate` to build the
  [semantic network](#consolidation-offline) — it improves recall by linking related entities.
- [Consolidation](#consolidation-offline) auto-triggers every 50 stores, but running it manually
  after a burst of activity gives immediate benefit.
- Periodically run `decay_sweep` to prune stale memories (90+ days idle,
  <2 accesses).

### What NOT to Store
- Trivial or ephemeral information (typo fixes, one-off commands).
- Exact code blocks — store the reasoning, not the implementation.
- Anything sensitive (secrets, credentials, PII).
```

### Tips for Effective Memory

- **Use `project` consistently.** It's the primary organizational axis. An agent working on "myapp" should always store with `project="myapp"` so recall can filter by project.
- **Pin critical context.** Use `pin_memory` for things every session should know (system architecture, active conventions, team preferences). Pinned memories are prepended to every recall result via the [Thalamus overlay](#thalamus-overlay).
- **Run consolidation periodically.** `myelin consolidate` (or it auto-runs every 50 stores) builds the [semantic network](#consolidation-offline) — entity relationships that improve recall quality over time.
- **Run decay periodically.** `myelin decay` prunes memories that haven't been accessed in 90+ days with fewer than 2 accesses. Keeps the memory clean without manual curation.
- **Export before major changes.** `myelin export backup.json` creates a full backup you can restore with `myelin import`.

---

## Benchmark Results

### LongMemEval_S — 500 questions, zero LLM calls

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) tests long-term conversational memory: can the system find the right conversation session given a natural-language question? **R@k** measures whether *any* ground-truth session appears in the top-k results (binary hit).

| Metric | Myelin | MemPalace (GPT-4o) |
|--------|--------|---------------------|
| **R@1** | **91.2%** | — |
| **R@3** | **98.0%** | — |
| **R@5** | **98.2%** | 96.6% |
| **R@10** | **98.2%** | — |
| **NDCG@5** | **95.2%** | — |
| LLM calls | 0 | requires GPT-4o |

98.2% R@5 (491/500 questions) using only local models — no LLM calls. Exceeds MemPalace's 96.6% R@5 which relies on GPT-4o.

#### Per-Category Breakdown

| Category | Questions | R@1 | R@5 |
|----------|-----------|-----|-----|
| knowledge-update | 78 | 97.4% | **100.0%** |
| single-session-assistant | 56 | 100.0% | **100.0%** |
| single-session-user | 70 | 88.6% | **100.0%** |
| multi-session | 133 | 91.0% | 98.5% |
| temporal-reasoning | 133 | 90.2% | 96.2% |
| single-session-preference | 30 | 70.0% | 93.3% |

### LoCoMo — 1,986 questions, 10 conversations

[LoCoMo](https://github.com/snap-research/locomo) (Snap Research) tests memory over long conversations. Stricter metric: **R@k** = fraction of *all* evidence sessions found in top-k (not binary hit). Multi-evidence questions require retrieving multiple sessions simultaneously.

| Metric | Myelin | MemPalace hybrid v5 |
|--------|--------|---------------------|
| **R@5** | **88.9%** | — |
| **R@10** | **95.1%** | 88.9% |
| **R@20** | **95.1%** | — |

### Latency — 8-core CPU, no GPU

| Operation | n | Mean | Min | Max | Notes |
|-----------|---|------|-----|-----|-------|
| **store** | 100 | **94ms** | 47ms | 149ms | embed (15ms) + dedup check + gist + ChromaDB write |
| **store** | 500 | **67ms** | 43ms | 130ms | flat: HNSW dedup-query adds <5ms over 500 items |
| **recall** | 100 | **142ms** | 94ms | 171ms | 3-probe pipeline: embed + HNSW + CE rerank |
| **recall** | 500 | **130ms** | 116ms | 153ms | flat retrieval scaling confirmed |
| **recall** | 1000 | **134ms** | 104ms | 173ms | still flat at 10× scale |
| recall + project filter | 100 | **162ms** | 120ms | 224ms | similar to unfiltered at n=100; benefits grow with n |
| recall + scope filter | 100 | **149ms** | 76ms | 188ms | similar to unfiltered at n=100; benefits grow with n |
| [Hebbian](#hebbian-boost) + [Thalamus](#thalamus-overlay) overhead | 100 | **+~10ms** | — | — | seeded Hebbian (~125 pairs); SQLite WAL reads/writes |

**Retrieval scales flat with collection size.** Recall averages 142ms at n=100, 130ms at n=500, and 134ms at n=1000 — all within the variance of each other. The bottleneck is fixed model inference: embedding the query (~15ms) and cross-encoder scoring the candidate pool (~60ms across 3 probes). HNSW index search adds <5ms at these scales. This means retrieval stays fast as memory grows — a user with 1000 memories pays the same latency as one with 100.

**Store variance is high.** The 47ms–149ms range reflects gist extraction cost, which varies with content length and semantic density. Short, single-topic memories hit the low end; anything requiring multi-chunk gists hits higher. Mean of ~80ms is well within the 500ms agent tool-call budget.

**Filters add negligible overhead at small n.** At n=100, project and scope filters show ~10–20ms higher means than unfiltered recall, but this is within measurement noise (stddev 24–38ms). Filter benefits appear at larger collection sizes where a filter can meaningfully reduce the cross-encoder candidate pool.

### Methodology

- **LongMemEval**: [LongMemEval_S cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) — 500 questions, 6 categories (ICLR 2025). Oracle mode, chunks deduplicated to sessions.
- **LoCoMo**: 10 conversations, 1,986 QA pairs. R@k = fraction of evidence sessions found in top-k.
- **Latency**: `pytest-benchmark` micro-timings, ephemeral ChromaDB, warm models, 8-core CPU, no GPU, n=100–1000 memories. Run `uv run pytest tests/benchmarks/test_latency.py -p no:xdist --override-ini="addopts="` to reproduce.
- **Models**: `all-MiniLM-L6-v2` (22M params) + `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params)
- **Hardware**: 8-core CPU, no GPU
- **LLM calls**: Zero in retrieval

---

## How It Works

### Core Concepts

| Concept | Neuroscience | Myelin Equivalent |
|---------|-------------|-------------------|
| **Cortical Region** | Specialized brain areas for different domains | `project` — each project is a distinct neural territory |
| **Engram Cluster** | Co-active neurons forming a memory trace | `scope` — related memories (auth, billing) share a cluster |
| **Memory System** | Distinct encoding/retrieval strategies | `memory_type` — episodic, semantic, procedural, prospective |
| **Association Fiber** | White matter connecting co-active regions | Hebbian links — built from co-retrieval patterns |
| **Gist Trace** | Meaning and detail stored in parallel | Vector embedding (gist) + raw content (verbatim) |
| **Sparse Code** | Only 1-5% of neurons fire per stimulus | Chunking — each segment is a focused representation |

### Memory Systems

| System | `memory_type` | What It Stores | Example |
|--------|---------------|----------------|---------|
| **Episodic** | `episodic` | Events with temporal context | "What happened when we deployed?" |
| **Semantic** | `semantic` | Decisions, facts, knowledge | "What did we decide for auth?" |
| **Procedural** | `procedural` | Habits, preferences, how-to | "How do we run migrations?" |
| **Prospective** | `prospective` | Future plans, recommendations | "What are the next steps?" |

### Pipeline Overview

```
STORE (fast, zero-LLM)              RECALL (multi-probe)

  content                              query
    │                                    │
    ▼                                    ▼
  Amygdala ─── reject noise          Query Planner ─── auto-infer filters
    │                                    │
    ▼                                    ▼
  Prefrontal ── auto-classify         Multi-probe (3 query variants)
    │                                    │
    ▼                                    ▼
  Chunking ──── pattern separation    Per-probe retrieval
    │                                    │ dual-path search + re-rank
    ▼                                    ▼
  Entorhinal ── context coordinates   Pool merge + cross-encoder re-score
    │                                    │
    ▼                                    ▼
  Perirhinal ── gist extraction       Spreading activation + lateral inhibition
    │                                    │
    ▼                                    ▼
  Hippocampus ─ embed + store         Return top-k
```

### Post-Recall

| Component | What It Does |
|-----------|-------------|
| **Hebbian Boost** | Co-retrieved memories strengthen mutual links |
| **Thalamus Overlay** | Prepends pinned memories, tracks recency |
| **Decay Sweep** | TTL pruning of unrehearsed, low-access memories |
| **Consolidation** | Entity extraction → semantic network (auto-triggers every N stores) |

> For the full step-by-step pipeline walkthrough, neuroscience mapping, and advanced configuration, see [docs/architecture.md](docs/architecture.md).

---

## CLI & MCP Tools

### CLI

```bash
myelin status       # Health + integrity check
myelin stats        # KPI dashboard: counts, types, age, Hebbian links
myelin serve        # Start MCP server (stdio)
myelin decay        # Prune stale memories
myelin consolidate  # Replay episodes into semantic network
myelin export out.json      # Export all memories to JSON
myelin import out.json      # Import memories from JSON
myelin export-md ./exports/ # Export memories as Markdown files with YAML frontmatter
myelin import-md ./exports/ # Import memories from a directory of Markdown files
myelin obsidian-export ~/vaults/work  # Export to an Obsidian vault (graph view)
myelin obsidian-import ~/vaults/work  # Import from an Obsidian vault
myelin ingest ./docs/       # Bulk-load .txt/.md/.json files into memory
myelin debug-recall "your query"  # Full pipeline breakdown for debugging
```

`myelin stats` accepts `--project`, `--agent-id`, and `--json` flags for filtering and machine-readable output.

The `ingest` command bulk-loads content from files or directories:

```bash
myelin ingest ./docs/                        # Recurse into directory
myelin ingest ./notes/arch.md                # Single file
myelin ingest ./docs/ --project myapp --scope architecture
myelin ingest ./data/ --source wiki --no-recursive  # Top-level only
```

Supported formats:
- **`.txt` / `.md`** — file body becomes one memory; YAML frontmatter between `---` delimiters is parsed for `project`, `scope`, `tags`, `memory_type`, `language`, `source` fields
- **`.json`** — list of objects in the same shape as `myelin export`; each must have a `"content"` key

The `export-md` / `import-md` commands round-trip memories as individual Markdown files with YAML frontmatter — useful for version-controlling memories in a git repo or editing them manually:

```bash
myelin export-md ./memory-backup/
# edit files...
myelin import-md ./memory-backup/
myelin import-md ./memory-backup/ --source restored
```

The `obsidian-export` command exports memories into an [Obsidian](https://obsidian.md) vault for graph-view visualisation. Each memory becomes a note with YAML frontmatter and `[[entity]]` wikilinks — Obsidian's graph view then clusters memories by shared entities, projects, and scopes:

```bash
myelin obsidian-export ~/vaults/work           # export all memories
myelin obsidian-export ~/vaults/work --type semantic   # decisions only
myelin obsidian-export ~/vaults/work --project myapp   # one project
```

The resulting vault structure:

```
vault/
├── Memory Index.md
└── Memories/
    ├── semantic/   ← decisions, facts
    ├── episodic/   ← events, bug fixes
    ├── procedural/ ← preferences, how-tos
    └── prospective/← plans, advice
```

To bring hand-edited or new notes back into myelin:

```bash
myelin obsidian-import ~/vaults/work
myelin obsidian-import ~/vaults/work --source obsidian
```

The `debug-recall` command runs a recall query and shows exactly what happened at each stage of the pipeline:

```
myelin debug-recall "what auth approach did we pick?" [-n N] [--project P] [--scope S] [--memory-type T] [--json]
```

Output includes:
- **Query plan** — what the PFC query planner inferred (memory type, scope, signals)
- **Amygdala gate** — whether the query would be accepted if stored
- **Results** with per-result score breakdown:
  - `bi` — raw bi-encoder cosine similarity from ChromaDB
  - `ce` — cross-encoder re-rank score
  - `hebbian` — co-access weight accumulated from prior co-recalls
  - `final_score` — after Hebbian boost (the score used for ranking)

> [!NOTE]
> If running from a dev checkout instead of an installed package, prefix with `uv run`: `uv run myelin status`

### MCP Tools

| Tool | Description |
|---|---|
| `store` | Encode a memory with context metadata (auto-classifies type, auto-chunks, 500K char limit). Pass `overwrite=true` to replace a near-duplicate instead of rejecting. Pass `agent_id` to store in an isolated namespace. |
| `recall` | Retrieve by semantic similarity (auto-inferred filters, multi-probe, [Hebbian boost](#hebbian-boost), 10K char limit). Pass `agent_id` to restrict results to that namespace. |
| `forget` | Remove a specific memory by ID |
| `pin_memory` | Pin a memory — always included in recall results |
| `unpin_memory` | Remove a pin |
| `decay_sweep` | Prune stale memories (access-based TTL) |
| `consolidate` | Replay episodes into the [semantic network](#consolidation-offline) |
| `status` | Memory system health check (counts, configuration) |
| `stats` | KPI dashboard: counts by type/project/scope/region, access health, age distribution, Hebbian links, decay candidates. Accepts `project` and `agent_id` filters. |
| `health` | Lightweight liveness probe (ok + version, no store initialization) |
| `ingest` | Bulk-load a file or directory into memory (`.txt`, `.md`, `.json`; supports YAML frontmatter metadata) |

### Data Storage

All data lives in `~/.myelin/` (configurable via `MYELIN_DATA_DIR`):

| File | Purpose |
|---|---|
| `chroma/` | Vector database (ChromaDB) — embeddings and metadata |
| `hebbian.db` | Co-access patterns between memories |
| `neocortex.db` | Semantic network — entities and relationships |
| `thalamus.db` | Pinned memories and recency tracking |

SQLite files use WAL mode for concurrent read performance. For more on inspecting these databases, see [docs/architecture.md](docs/architecture.md#inspecting-your-data).

---

## Configuration

All parameters use environment variables with a `MYELIN_` prefix. Defaults work out of the box — most users won't need to change anything.

### Common Parameters

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `data_dir` | `~/.myelin` | Where all data lives |
| `default_n_results` | `5` | Results returned per recall |
| `max_memories` | `0` | Hard memory cap; 0 = unlimited. LRU eviction when exceeded |
| `consolidation_interval` | `50` | Auto-consolidate every N stores (0 = disabled) |
| `log_level` | `INFO` | Logging verbosity (structured JSON to stderr) |

### Storage Parameters

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `embedding_model` | `all-MiniLM-L6-v2` | Bi-encoder model (384-dim, 22M params) |
| `chunk_max_chars` | `1000` | Max characters per chunk |
| `chunk_overlap_chars` | `200` | Overlap between text chunks |
| `min_content_length` | `20` | Minimum chars to pass the input gate |
| `dedup_similarity_threshold` | `0.95` | Above this = near-duplicate, rejected |

### Recall Parameters

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `recall_over_factor` | `8` | Over-retrieval multiplier for re-ranking headroom |
| `multiprobe` | `true` | 3-probe retrieval (original + keywords + entity-expanded) |
| `neocortex_rerank` | `true` | Cross-encoder re-ranking |
| `neocortex_weight` | `0.6` | CE/bi-encoder blend (0.0–1.0) |
| `cross_encoder_model` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder model (22M params) |
| `lateral_k` | `1` | Max results per session/scope (0 = off) |

For advanced tuning parameters (boosting weights, spreading activation, maintenance thresholds, background worker), see [docs/architecture.md](docs/architecture.md#advanced-configuration).

---

## Upgrading

### Patch and minor updates (0.x.y)

```bash
# uv
uv tool upgrade myelin-mcp

# pip
pip install --upgrade myelin-mcp
```

Patch and minor updates are backward-compatible — your existing data directory works without changes. Run `myelin status` after upgrading to verify.

### Backup before major updates

Before upgrading across major versions, export your data:

```bash
myelin export backup.json
# upgrade
myelin import backup.json
```

### Embedding model changes

Myelin records the embedding model version in ChromaDB metadata. If a future release changes the default embedding model, existing vectors would need re-encoding to maintain recall quality. This has not happened yet — `all-MiniLM-L6-v2` has been the model since v0.1.0.

When a model change does ship, the release notes will include migration instructions. The safe path is always: `myelin export` → upgrade → `myelin import` (re-encodes all content with the new model).

---

## Development

```bash
git clone https://github.com/et-do/myelin.git
cd myelin
uv sync --extra dev
uv run pre-commit install
uv run pytest -v --cov=myelin
```

A Dev Container config is included — open in VS Code and "Reopen in Container" for a zero-setup environment.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow: branching, conventional commits, automated releases, benchmarking, and project structure.

---

## Further Reading

- [Architecture & internals](docs/architecture.md) — pipeline walkthroughs, neuroscience mapping, data inspection, advanced configuration
- [Contributing](CONTRIBUTING.md) — branching, conventional commits, releases
- [Changelog](CHANGELOG.md) — release history

## License

MIT
