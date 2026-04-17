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

1. **Branch off main**: `git checkout -b feat/my-feature`
2. **Use conventional commits** (see below)
3. **Open a PR** into `main`
4. CI runs lint, tests, type checking, and dependency review
5. **Squash and merge** when all checks pass (keeps `main` history clean and avoids duplicate changelog entries)

### Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/) to drive automated releases. Every commit message should follow this format:

```
<type>: <description>

[optional body]
```

| Type | When to use | Version bump |
|------|-------------|--------------|
| `feat:` | New feature | minor (0.1.0 → 0.2.0) |
| `fix:` | Bug fix | patch (0.1.0 → 0.1.1) |
| `feat!:` or `fix!:` | Breaking change | minor while pre-1.0 |
| `docs:` | Documentation only | none |
| `chore:` | Maintenance, CI, deps | none |
| `refactor:` | Code change, no new feature or fix | none |
| `test:` | Adding or updating tests | none |
| `perf:` | Performance improvement | none |

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
4. A GitHub Release + tag is created automatically, and `pyproject.toml` version is bumped

You never need to manually edit the version number.

## Quality Checks

All of these run in CI and should pass before merging:

```bash
# Verify uv.lock is in sync with pyproject.toml (run after editing pyproject.toml)
uv lock --check

# Tests with coverage
uv run pytest -v --cov=myelin

# Skip slow MCP integration tests (spawns a real server subprocess) during local dev
uv run pytest -v --cov=myelin -m "not integration"

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
  server.py           # MCP server exposing tools to AI agents
  cli.py              # CLI interface
  config.py           # Configuration (pydantic-settings)
  models.py           # Pydantic data models
  reranker.py         # Cross-encoder re-ranking
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

Fast subset (54 LME + 304 LoCoMo questions). Fails if any metric drops >2pp below baseline.

```bash
uv run python -m benchmarks.regression.run --create-baseline
uv run python -m benchmarks.regression.run
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
