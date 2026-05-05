# Contributing to Myelin

## Development Environment

### Dev Container (recommended)

The repo includes a Dev Container that sets up everything automatically:

1. Open the repo in VS Code
2. "Reopen in Container" when prompted (or `Ctrl+Shift+P` → "Dev Containers: Reopen in Container")
3. The container installs dependencies, pre-commit hooks, and VS Code extensions automatically

### Manual Setup

```bash
git clone https://github.com/et-do/myelin.git
cd myelin
uv sync --extra dev
uv run pre-commit install
```

## Workflow

We use a trunk-based workflow. All work goes through PRs into `main`.

1. **Branch off main** using a prefix that matches the conventional commit type you'll use (see table below)
2. **Use conventional commits** (see below)
3. **Open a PR** — the PR title becomes the squash commit message on `main`, so write it as a conventional commit
4. CI runs lint, type checks, tests, build verification, and a dependency vulnerability audit
5. **Squash and merge** when all checks pass (keeps `main` history clean and avoids duplicate changelog entries)

### Branch naming

Branch names follow the same type prefixes as conventional commits. Pick the prefix that matches what will land on `main`:

| Branch prefix | Use when | Triggers release? |
|---------------|----------|-------------------|
| `feat/description` | Adding new functionality | yes — minor bump |
| `fix/description` | Fixing a bug in shipped code | yes — patch bump |
| `ci/description` | CI/CD workflow changes | no |
| `chore/description` | Maintenance, deps, tooling | no |
| `docs/description` | Documentation only | no |
| `refactor/description` | Code restructuring, no behaviour change | no |
| `test/description` | Adding or updating tests | no |
| `perf/description` | Performance improvement | no |

> The branch name itself has no effect on tooling — only the squash commit message matters. Consistent naming is purely for human readability.

### Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/) to drive automated releases. Every commit message should follow this format:

```
<type>: <description>

[optional body]
```

| Type | When to use | Triggers release? | Version bump |
|------|-------------|-------------------|--------------|
| `feat:` | New user-facing feature | **yes** | minor (`0.1.0 → 0.2.0`) |
| `fix:` | Bug fix in shipped code | **yes** | patch (`0.1.0 → 0.1.1`) |
| `feat!:` or `fix!:` | Breaking change | **yes** | major (`0.1.0 → 1.0.0`) |
| `perf:` | Performance improvement | no (hidden in changelog) | — |
| `deps:` | Dependency updates | no (hidden in changelog) | — |
| `revert:` | Reverting a commit | no (hidden in changelog) | — |
| `docs:` | Documentation only | no (hidden in changelog) | — |
| `chore:` | Maintenance, tooling | no | — |
| `ci:` | CI/CD changes | no | — |
| `refactor:` | Code restructuring, no behaviour change | no | — |
| `test:` | Adding or updating tests | no | — |
| `build:` | Build system changes | no | — |
| `style:` | Formatting, whitespace | no | — |

Only `feat:`, `fix:`, and breaking changes (`!`) trigger a release and version bump. All other types (including `docs:`, `perf:`, `deps:`, `revert:`) appear hidden in the CHANGELOG but do **not** publish to PyPI. `chore`, `ci`, `refactor`, `test`, `build`, and `style` are never shown and never trigger a release.

Examples:
```
feat: add TTL-based memory expiration
fix: prevent duplicate chunks on rapid store calls
docs: update setup guide for cloud deployment
feat!: change recall API to return scored results
```

### Releases

Releases are fully automated via [release-please](https://github.com/googleapis/release-please):

1. Merge PRs with conventional commits into `main`
2. Release-please accumulates changes and opens a **Release PR** with a version bump + CHANGELOG
3. When you're ready to cut a release, merge the Release PR
4. A GitHub Release + version tag is created automatically
5. The tag triggers `publish.yml`: full regression gate (~60 min) → PyPI publish if it passes

You never need to manually edit the version number.

#### Version bumping while pre-1.0

While the project is pre-1.0 (`0.X.Y`), standard semver applies with one adjustment: breaking changes do **not** bump to `1.0.0` automatically — they stay in the `0.X.Y` range until you decide the API is stable enough for 1.0. Use `feat!:` for breaking changes; release-please will bump the major when you're ready if you configure it to do so, otherwise treat the `0.X.0` bump as the signal that something significant changed.

In practice: use `feat:` freely for new features — each one bumps `0.X → 0.X+1`. Use `fix:` for patches. Save `feat!:` for genuine API breaks.

## Quality Checks

### CI jobs

| Job | Runs on | Blocks merge? | Purpose |
|-----|---------|---------------|---------|
| **Lint & format** | PR + push to main | yes | `ruff` lint + format, lockfile freshness. All other jobs wait for this. |
| **Type check** | PR + push to main | yes | `mypy --strict` across the full package. |
| **Unit & integration tests** | PR + push to main | yes | `pytest` with coverage report. |
| **Build distribution** | PR + push to main | yes | Builds wheel + sdist; uploads as artifact. |
| **Dependency vulnerability audit** | PR only | no | OSV database scan of new dependencies. |
| **Regression gate** (smoke) | Push to main (`myelin/**` changes) | no (post-merge) | ~12 LME + 1 LoCoMo; fails if recall drops >2pp from baseline. |
| **Latency benchmarks** | Weekly (Monday 06:00 UTC) + manual | no | `pytest-benchmark` micro-timings for store/recall. |
| **Full regression + publish** | `v*` tag push (every release) | — | ~54 LME + 2 LoCoMo (~110 min); blocks PyPI publish on failure. |

### Running checks locally

```bash
# Verify uv.lock is in sync with pyproject.toml (run after editing pyproject.toml)
uv lock --check

# Tests with coverage
uv run pytest -q --tb=short --cov=myelin

# Skip slow MCP integration tests (spawns a real server subprocess) during local dev
uv run pytest -q --tb=short --cov=myelin -m "not integration"

# Run integration tests explicitly
uv run pytest -m integration -v

# Linting
uv run ruff check .
uv run ruff format --diff .

# Type checking (strict)
uv run mypy myelin/ --strict

# Pre-commit (runs ruff, trailing whitespace, etc.)
uv run pre-commit run --all-files
```

## Repository Settings

Branch and tag protection rulesets are stored as JSON in `.github/` and applied via the `gh` CLI — **not** managed through the GitHub UI.

After editing a `*-protection.json` file, apply the changes:

```bash
# Requires: gh CLI authenticated + jq installed
.github/apply-rulesets.sh
```

The script PUTs each ruleset by its numeric `id` field. Running it multiple times is safe (idempotent).

> **Required status check names** must match the raw job `name:` field from the workflow YAML — not the decorated `CI / Job name (pull_request)` string shown in the GitHub UI.

## Administration

This section is for repo admins only. Contributors do not need to read this.

### RELEASE_TOKEN (fine-grained PAT)

Several workflows require a fine-grained PAT stored as the `RELEASE_TOKEN` repo secret:

| Workflow | Why GITHUB_TOKEN is insufficient |
|----------|----------------------------------|
| `release-please.yml` | Tags created by GITHUB_TOKEN don't trigger downstream `push:tags` events — `publish.yml` would never fire |
| `sync-rulesets.yml` | The GitHub Rulesets API requires `administration: write`, which is not a valid GITHUB_TOKEN scope |

**Required PAT permissions** (fine-grained, scoped to `et-do/myelin` only):
- Contents: Read and write
- Pull requests: Read and write
- Administration: Read and write

**Current token expiry: 2027-04-19.** Regenerate at https://github.com/settings/tokens?type=beta before expiry and update the `RELEASE_TOKEN` secret at https://github.com/et-do/myelin/settings/secrets/actions.

If the token expires, release-please will silently fail to create tags and PyPI publish will stop firing.

### PyPI trusted publisher

Publishing uses OIDC (no API token). The trusted publisher must be configured at https://pypi.org/manage/project/myelin-mcp/settings/publishing/ to allow workflow `publish.yml` from repo `et-do/myelin`, environment `pypi`.

## Project Structure

```
myelin/               # Core library
  store/              # Store path modules
    hippocampus.py    # Vector store (ChromaDB) — embed + store + dual-path search
    amygdala.py       # Input gate — rejects noise, near-duplicates
    prefrontal.py     # Auto-classifies memory_type via schema matching
    chunking.py       # Pattern separation — splits long content into chunks
    entorhinal.py     # Context coordinates — keywords, region, speakers
    perirhinal.py     # Gist extraction — extractive summaries (no LLM)
    neocortex.py      # Semantic network — entity graph + spreading activation
    consolidation.py  # Offline replay — entity extraction + graph building
    thalamus.py       # Pinned memories + recency tracking
  recall/             # Recall path modules
    query_planner.py  # Auto-infers filters from query
    activation.py     # Hebbian co-access learning
    decay.py          # TTL pruning of stale memories
    time_cells.py     # Temporal reference detection + boost
    reranker.py       # Cross-encoder re-ranking (Neocortex class)
  integrations/       # File-format importers / exporters
    ingest.py         # Bulk-load .txt/.md/.json files into memory
    obsidian.py       # Obsidian vault import/export
  ui/                 # Browser-based admin dashboard
    serve.py          # Graph data builder + HTTP server
    graph.html        # 3-tab SPA (Graph | Overview | Memories)
  mcp.py              # MCP server exposing tools to AI agents
  background.py       # Background worker + decay timer threads
  cli.py              # CLI interface
  config.py           # Configuration (pydantic-settings)
  models.py           # Pydantic data models
tests/                # Unit tests (pytest)
benchmarks/           # LongMemEval, LoCoMo, regression gate
.github/              # CI workflows, Copilot instructions
```

## Benchmarking

### LongMemEval

```bash
bash benchmarks/longmemeval/download_data.sh

uv run python -m benchmarks.longmemeval.run \
    benchmarks/longmemeval/data/longmemeval_oracle.json \
    benchmarks/longmemeval/output/myelin_oracle.jsonl 5 4

uv run python -m benchmarks.longmemeval.score \
    benchmarks/longmemeval/data/longmemeval_oracle.json \
    benchmarks/longmemeval/output/myelin_oracle_YYYYMMDD_HHMMSS.jsonl
```

### LoCoMo

```bash
uv run python -m benchmarks.locomo.run
uv run python -m benchmarks.locomo.score benchmarks/locomo/output/myelin_locomo.json
```

### Regression Gate

Two modes — both fail if any metric drops >2pp below the committed baseline files in
`benchmarks/regression/baseline/`. The baseline is **not** managed by the CI cache;
the committed JSON files are the authoritative reference.

**Updating the baseline** — run this locally after a genuine improvement (e.g. a new
algorithm that scores higher) and commit the resulting JSON files:
```bash
# Smoke baseline (fast, ~15 min):
uv run python -m benchmarks.regression.run --create-baseline

# Full baseline (~110 min — only needed before a release if full scores changed):
uv run python -m benchmarks.regression.run --full --create-baseline
```

**Smoke** (default, ~15 min — runs on every push to main that touches `myelin/`):
```bash
uv run python -m benchmarks.regression.run
```

**Full** (~110 min — runs as a gate before every PyPI release):
```bash
uv run python -m benchmarks.regression.run --full
```

### Latency Benchmarks

`pytest-benchmark` micro-timings for `store()` and `recall()` at corpus sizes of 100 and 500 memories.

```bash
# Run with timing output (disable xdist so each test runs serially)
uv run pytest tests/benchmarks/test_latency.py -p no:xdist -v

# Save results for future comparison
uv run pytest tests/benchmarks/test_latency.py -p no:xdist \
    --benchmark-json=benchmarks/perf/results.json

# Compare against a saved baseline
uv run pytest tests/benchmarks/test_latency.py -p no:xdist --benchmark-compare
```

These tests run as correctness checks in the normal `pytest` suite; timing collection is disabled automatically when xdist is active.
