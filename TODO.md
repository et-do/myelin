# Myelin TODO

## In Progress

- [ ] **`feat/obsidian-export`** — Export memories to Obsidian vault (`myelin obsidian-export <vault>`)
  - [x] `myelin/integrations/` package with `Exporter` / `Importer` ABCs
  - [x] `ObsidianExporter` — YAML frontmatter, entity wikilinks, index note, type subfolders
  - [x] `ObsidianImporter` — scan `Memories/**/*.md`, strip Related section, return content+meta pairs
  - [x] CLI: `myelin obsidian-export <vault> [--project] [--type] [--scope]`
  - [x] CLI: `myelin obsidian-import <vault> [--source]`
  - [x] Tests: `tests/integrations/test_obsidian.py`
  - [ ] Open PR

## Planned

- [ ] **`feat/graph-ui`** — Local graph visualisation (`myelin graph`)
  - [ ] Serve a local HTML page with D3 force-directed graph
  - [ ] Nodes: entities from `neocortex.db`, sized by connection count
  - [ ] Edges: co-occurrence relationships with weight
  - [ ] Filter panel: by project, memory_type, min edge weight
  - [ ] Click a node → show linked memory snippets
  - [ ] No external dependencies (single self-contained HTML file served by stdlib `http.server`)

- [ ] **Obsidian vault import** (follow-on to `feat/obsidian-export`)
  - `ObsidianImporter` stub is implemented; needs end-to-end CLI test with real vault
  - Consider: deduplication strategy when re-importing an already-exported vault
  - Consider: mapping Obsidian `[[wikilink]]` backlinks as `parent_id` or `scope`

- [ ] **`feat/stats-command`** — `myelin stats` KPI dashboard + MCP tool
  - Branch exists with 20 tests and mypy fixes; needs PR

- [ ] **Pending PRs to open**
  - `fix/suppress-noisy-logs` (3 follow-up commits after #58 merged)
  - `feat/stats-command`
  - `docs/update-todo`
  - `docs/readme-restructure`
  - `docs/fix-contributing-release-types`

## Future Ideas

- [ ] Logseq integration (same markdown format, slightly different graph conventions)
- [ ] GraphML / DOT export from neocortex for Gephi / yEd
- [ ] `get_graph` MCP tool — return entity graph as JSON for agent-side visualisation
- [ ] Obsidian plugin (Electron) — live sync instead of periodic export
