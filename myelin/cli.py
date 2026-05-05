"""CLI entrypoint for myelin."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC
from typing import Any

from .config import settings
from .log import setup_logging, suppress_noisy_loggers

suppress_noisy_loggers()


def cmd_status(_args: argparse.Namespace) -> None:
    from .mcp import configure, do_status

    configure(settings)
    result = do_status()
    print(json.dumps(result, indent=2))


# ANSI colour helpers — no external dependencies
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_WHITE = "\033[97m"

_BAR_CHARS = "█▓▒░"
_BAR_WIDTH = 20


def _bar(value: int, total: int, width: int = _BAR_WIDTH) -> str:
    """Return a filled progress bar string."""
    if total == 0:
        return "░" * width
    filled = round(width * value / total)
    empty = width - filled
    return _GREEN + "█" * filled + _DIM + "░" * empty + _RESET


def _pct(value: int, total: int) -> str:
    if total == 0:
        return "  0.0%"
    return f"{100 * value / total:5.1f}%"


def _header(title: str, width: int = 60) -> None:
    print()
    print(_BOLD + _CYAN + "  " + title + _RESET)
    print(_DIM + "  " + "─" * (width - 2) + _RESET)


def _row(label: str, value: int, total: int, label_width: int = 18) -> None:
    bar = _bar(value, total)
    pct = _pct(value, total)
    print(
        f"  {_WHITE}{label:<{label_width}}{_RESET} "
        f"{bar} {_YELLOW}{value:>6}{_RESET}  {_DIM}{pct}{_RESET}"
    )


def cmd_stats(args: argparse.Namespace) -> None:
    import sqlite3
    from collections import Counter
    from datetime import datetime

    import chromadb

    # All reads go directly to the underlying stores — no model loading.
    # Avoids the sentence-transformers cold-start that Hippocampus init triggers
    # (store/__init__.py eagerly imports Hippocampus → SentenceTransformer).
    settings.ensure_dirs()
    data_dir = settings.data_dir

    # ── ChromaDB: memory metadata ─────────────────────────────────────
    # embedding_function=None prevents ChromaDB loading its default ONNX model.
    try:
        chroma = chromadb.PersistentClient(path=str(data_dir / "chroma"))
        col = chroma.get_collection("memories", embedding_function=None)
        result = col.get(include=["metadatas"])
        raw_ids = result["ids"] or []
        raw_metas = result["metadatas"] or []
        all_meta = [{"id": id_, **meta} for id_, meta in zip(raw_ids, raw_metas)]
    except Exception:
        all_meta = []

    # ── SQLite: neocortex entity/relationship counts ──────────────────
    neo_path = data_dir / "neocortex.db"
    entity_count = relationship_count = 0
    if neo_path.exists():
        try:
            with sqlite3.connect(str(neo_path)) as con:
                row = con.execute(
                    "SELECT (SELECT COUNT(*) FROM entities), "
                    "(SELECT COUNT(*) FROM relationships)"
                ).fetchone()
                if row:
                    entity_count, relationship_count = int(row[0]), int(row[1])
        except sqlite3.OperationalError:
            pass

    # ── SQLite: Hebbian link stats ────────────────────────────────────
    hebb_path = data_dir / "hebbian.db"
    hebbian: dict[str, object] = {"link_count": 0, "avg_weight": 0.0, "max_weight": 0.0}
    if hebb_path.exists():
        with sqlite3.connect(str(hebb_path)) as con:
            row = con.execute(
                "SELECT COUNT(*), AVG(weight), MAX(weight) FROM co_access"
            ).fetchone()
            if row and row[0]:
                hebbian = {
                    "link_count": int(row[0]),
                    "avg_weight": round(float(row[1] or 0), 4),
                    "max_weight": round(float(row[2] or 0), 4),
                }

    # ── SQLite: thalamus pinned count ─────────────────────────────────
    thal_path = data_dir / "thalamus.db"
    pinned_count = 0
    if thal_path.exists():
        with sqlite3.connect(str(thal_path)) as con:
            row = con.execute("SELECT COUNT(*) FROM pinned").fetchone()
            if row:
                pinned_count = int(row[0])

    project = args.project or ""
    agent_id = args.agent_id or ""
    if project:
        all_meta = [m for m in all_meta if m.get("project") == project]
    if agent_id:
        all_meta = [m for m in all_meta if m.get("agent_id") == agent_id]

    total = len(all_meta)
    now = datetime.now(tz=UTC)

    by_type: Counter[str] = Counter()
    by_project: Counter[str] = Counter()
    by_scope: Counter[str] = Counter()
    by_region: Counter[str] = Counter()
    cold = hot = lukewarm = 0
    age_lt7 = age_7_30 = age_30_90 = age_90_365 = age_over365 = 0
    decay_candidates = 0

    for m in all_meta:
        by_type[str(m.get("memory_type") or "unknown")] += 1
        by_project[str(m.get("project") or "(none)")] += 1
        by_scope[str(m.get("scope") or "(none)")] += 1
        by_region[str(m.get("ec_region") or "(none)")] += 1

        acc = int(m.get("access_count") or 0)  # type: ignore[arg-type]
        if acc == 0:
            cold += 1
        elif acc == 1:
            lukewarm += 1
        else:
            hot += 1

        try:
            created = datetime.fromisoformat(str(m["created_at"]))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            age_days = (now - created).days
        except (KeyError, ValueError):
            age_days = 0

        if age_days < 7:
            age_lt7 += 1
        elif age_days < 30:
            age_7_30 += 1
        elif age_days < 90:
            age_30_90 += 1
        elif age_days < 365:
            age_90_365 += 1
        else:
            age_over365 += 1

        try:
            last = datetime.fromisoformat(str(m["last_accessed"]))
            if last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            idle_days = (now - last).days
        except (KeyError, ValueError):
            idle_days = 0
        if idle_days >= settings.max_idle_days and acc < settings.min_access_count:
            decay_candidates += 1

    d: dict[str, Any] = {
        "total": total,
        "entity_count": entity_count,
        "relationship_count": relationship_count,
        "pinned_count": pinned_count,
        "hebbian": hebbian,
        "by_type": dict(by_type.most_common()),
        "by_project": dict(by_project.most_common(10)),
        "by_scope": dict(by_scope.most_common(10)),
        "by_region": dict(by_region.most_common()),
        "access": {"hot": hot, "lukewarm": lukewarm, "cold": cold},
        "age": {
            "<7d": age_lt7,
            "7-30d": age_7_30,
            "30-90d": age_30_90,
            "90-365d": age_90_365,
            ">1yr": age_over365,
        },
        "decay_candidates": decay_candidates,
        "filter": {"project": project or None, "agent_id": agent_id or None},
    }

    if args.json:
        print(json.dumps(d, indent=2))
        return

    def sep(char: str = "═") -> None:
        print(_DIM + char * 60 + _RESET)

    print()
    sep()
    title = "  MYELIN MEMORY STATS"
    if d["filter"]["project"]:
        title += f"  {_DIM}[project={d['filter']['project']}]{_RESET}"
    if d["filter"]["agent_id"]:
        title += f"  {_DIM}[agent={d['filter']['agent_id']}]{_RESET}"
    print(_BOLD + _CYAN + title + _RESET)
    sep()

    # ── Summary ──────────────────────────────────────────────────────
    total = d["total"]
    hebb = d["hebbian"]
    print()
    cols = [
        ("Memories", str(total), _GREEN),
        ("Entities", str(d["entity_count"]), _CYAN),
        ("Relationships", str(d["relationship_count"]), _CYAN),
        ("Pinned", str(d["pinned_count"]), _YELLOW),
        ("Hebbian links", str(hebb["link_count"]), _MAGENTA),
    ]
    for label, val, colour in cols:
        print(f"  {_DIM}{label:<18}{_RESET}{colour}{_BOLD}{val}{_RESET}")

    if hebb["link_count"] > 0:
        print(
            f"  {_DIM}{'Avg link weight':<18}{_RESET}"
            f"{_DIM}{hebb['avg_weight']:.4f}  max {hebb['max_weight']:.4f}{_RESET}"
        )

    if total == 0:
        print()
        print(f"  {_DIM}No memories found.{_RESET}")
        print()
        sep()
        return

    # ── By Memory Type ───────────────────────────────────────────────
    _header("Memory Type", 60)
    for mt, count in d["by_type"].items():
        _row(mt, count, total)

    # ── By Project ───────────────────────────────────────────────────
    if not args.project and len(d["by_project"]) > 1:
        _header("Top Projects", 60)
        for proj, count in d["by_project"].items():
            _row(proj, count, total)

    # ── By Scope ─────────────────────────────────────────────────────
    if len(d["by_scope"]) > 1:
        _header("Top Scopes", 60)
        for sc, count in d["by_scope"].items():
            _row(sc, count, total)

    # ── By Region (entorhinal context) ───────────────────────────────
    non_none_regions = {k: v for k, v in d["by_region"].items() if k != "(none)"}
    if non_none_regions:
        _header("Knowledge Domains", 60)
        for region, count in d["by_region"].items():
            _row(region, count, total)

    # ── Access Health ────────────────────────────────────────────────
    acc = d["access"]
    _header("Access Health", 60)
    _row("hot  (≥2 recalls)", acc["hot"], total)
    _row("warm (1 recall)", acc["lukewarm"], total)
    _row("cold (never)", acc["cold"], total)

    # ── Age Distribution ─────────────────────────────────────────────
    age = d["age"]
    _header("Memory Age", 60)
    for bucket, count in age.items():
        _row(bucket, count, total)

    # ── Decay ────────────────────────────────────────────────────────
    print()
    cands = d["decay_candidates"]
    if cands == 0:
        dc_str = _GREEN + "0 decay candidates" + _RESET
    else:
        dc_str = (
            _RED
            + _BOLD
            + str(cands)
            + _RESET
            + f" decay candidates  {_DIM}(run `myelin decay` to prune){_RESET}"
        )
    print(f"  {dc_str}")
    print()
    sep()
    print()


def cmd_decay(_args: argparse.Namespace) -> None:
    from .recall import find_stale
    from .store import Hippocampus

    hc = Hippocampus()
    metadata = hc.get_all_metadata()
    stale_ids = find_stale(metadata)
    if stale_ids:
        count = hc.forget_batch(stale_ids)
        print(f"Pruned {count} stale memories.")
    else:
        print("No stale memories found.")


def cmd_serve(_args: argparse.Namespace) -> None:
    from .mcp import main as serve_main

    serve_main()


def cmd_consolidate(_args: argparse.Namespace) -> None:
    from .mcp import do_consolidate

    result = do_consolidate()
    print(json.dumps(result, indent=2))


def cmd_graph(args: argparse.Namespace) -> None:
    """Serve the interactive admin dashboard."""
    from .ui.serve import serve_graph

    serve_graph(args)


def cmd_debug_recall(args: argparse.Namespace) -> None:
    from .mcp import configure, do_debug_recall

    configure(settings)
    data = do_debug_recall(
        query=args.query,
        n_results=args.n,
        project=args.project or "",
        language=args.language or "",
        scope=args.scope or "",
        memory_type=args.memory_type or "",
        agent_id=args.agent_id or "",
    )

    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return

    # --- Human-readable terminal output ---
    col_width = 72  # column width

    def _sep(char: str = "─") -> None:
        print(char * col_width)

    def _kv(label: str, value: object, width: int = 18) -> None:
        print(f"  {label:<{width}}{value}")

    print()
    _sep("═")
    print("  MYELIN DEBUG RECALL")
    _sep("═")
    print(f'  Query          "{data["query"]}"')
    print(f"  Memory count   {data['memory_count']}")
    print()

    # Query plan
    print("  QUERY PLAN  (PFC inference)")
    _sep()
    qp = data["query_plan"]
    _kv("memory_type →", qp["memory_type"] or "none inferred")
    _kv("scope_hint  →", qp["scope_hint"] or "none inferred")
    if qp["signals"]:
        signals_str = "  ".join(f"{k}={v}" for k, v in qp["signals"].items())
        _kv("signals", signals_str)
    fa = data["filters_applied"]
    explicit = {k: v for k, v in fa.items() if v and not k.startswith("auto_")}
    auto = {k[5:]: v for k, v in fa.items() if k.startswith("auto_") and v}
    if explicit:
        _kv("explicit →", "  ".join(f"{k}={v}" for k, v in explicit.items()))
    if auto:
        _kv("auto-applied →", "  ".join(f"{k}={v}" for k, v in auto.items()))
    elif not explicit:
        _kv("filters →", "none")
    print()

    # Gate check
    gc = data["gate_check"]
    gate_label = "✓ ok" if gc["would_store"] else f"✗ {gc['reason']}"
    print("  AMYGDALA GATE  (would this query be stored as a memory?)")
    _sep()
    _kv("verdict →", gate_label)
    print()

    # Results
    results = data["results"]
    if not results:
        print("  No results found.")
        print()
        _sep("═")
        return

    print(f"  RESULTS  (top {len(results)})")
    for r in results:
        _sep()
        rank = r["rank"]
        score = r["final_score"]
        bi = r["bi_encoder_sim"]
        ce = r["ce_score"]
        hebb = r["hebbian_weight"]
        mt = r.get("memory_type") or "—"
        sc = r.get("scope") or "—"
        proj = r.get("project") or "—"
        acc = r.get("access_count", 0)

        bi_str = f"{bi:.4f}" if bi is not None else "n/a"
        ce_str = f"{ce:.4f}" if ce is not None else "n/a (CE disabled)"

        print(
            f"  #{rank}  score={score:.4f}  "
            f"bi={bi_str}  ce={ce_str}  hebbian={hebb:.3f}"
        )
        print(f"  {r['id']}  [{proj} / {sc} / {mt}]  access={acc}")
        # Wrap content preview at col_width-4 chars
        preview = r["content_preview"]
        chunk = col_width - 4
        lines = [preview[i : i + chunk] for i in range(0, len(preview), chunk)]
        for line in lines:
            print(f"    {line}")
    print()
    _sep("═")
    print()


def cmd_export(args: argparse.Namespace) -> None:
    from .store import Hippocampus

    hc = Hippocampus()
    memories = hc.get_all_content()
    metadata = hc.get_all_metadata()

    # Merge content + metadata by ID
    meta_by_id = {m["id"]: m for m in metadata}
    export_data = []
    for mem in memories:
        mid = mem["id"]
        meta = meta_by_id.get(mid, {})
        export_data.append({**meta, "content": mem["content"]})

    output = args.output
    if output == "-":
        print(json.dumps(export_data, indent=2, default=str))
    else:
        with open(output, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"Exported {len(export_data)} memories to {output}")


def cmd_export_md(args: argparse.Namespace) -> None:
    """Export memories as individual Markdown files with YAML frontmatter."""
    import os
    from pathlib import Path

    from .store import Hippocampus

    hc = Hippocampus()
    memories = hc.get_all_content()
    metadata = hc.get_all_metadata()
    meta_by_id = {m["id"]: m for m in metadata}

    out_dir = args.output_dir
    out_path = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Incremental: build set of IDs that need (re)writing
    skip_ids: set[str] = set()
    if args.incremental:
        from .integrations.sync import SyncRegistry

        registry = SyncRegistry(settings.data_dir / "sync.db")
        merged_for_filter = [
            {**meta_by_id.get(m["id"], {}), "id": m["id"], "content": m["content"]}
            for m in memories
        ]
        to_write = {
            m["id"]
            for m in registry.filter_for_export(
                "export-md", out_path, merged_for_filter
            )
        }
        skip_ids = {m["id"] for m in merged_for_filter} - to_write

    meta_fields = ("project", "scope", "language", "memory_type", "source", "tags")
    count = unchanged = 0
    written_mems: list[dict[str, Any]] = []
    for mem in memories:
        mid = mem["id"]
        if mid in skip_ids:
            unchanged += 1
            continue
        meta = meta_by_id.get(mid, {})
        content = mem["content"]

        # Build YAML frontmatter — only non-empty fields
        fm_lines = [f"id: {mid}"]
        for key in meta_fields:
            val = meta.get(key)
            if val:
                fm_lines.append(f"{key}: {val}")
        if meta.get("created_at"):
            fm_lines.append(f"created_at: {meta['created_at']}")

        frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n\n"
        filename = os.path.join(out_dir, f"{mid}.md")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(frontmatter + content)
        count += 1
        written_mems.append({**meta, "id": mid, "content": content})

    if args.incremental:
        registry.record_exports("export-md", out_path, written_mems)
        suffix = f" ({count} written, {unchanged} unchanged)"
    else:
        suffix = ""
    print(f"Exported {count + unchanged} memories to {out_dir}/{suffix}")


def cmd_import_md(args: argparse.Namespace) -> None:
    """Import memories from a directory of Markdown files with YAML frontmatter."""
    import os
    import re
    from pathlib import Path

    from .mcp import configure, do_store

    configure(settings)

    fm_re = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
    kv_re = re.compile(r"^(\w+)\s*:\s*(.+)$", re.MULTILINE)
    meta_fields = {"project", "scope", "language", "memory_type", "source", "tags"}

    md_dir = args.input_dir
    md_path = Path(md_dir)
    files = sorted(f for f in os.listdir(md_dir) if f.endswith(".md"))
    if not files:
        print(f"No .md files found in {md_dir}")
        return

    # Incremental: filter to only new/changed files
    registry = None
    skip_paths: frozenset[Path] = frozenset()
    new_files_for_record: list[Path] = []
    if args.incremental:
        from .integrations.sync import SyncRegistry

        registry = SyncRegistry(settings.data_dir / "sync.db")
        all_paths = [md_path / f for f in files]
        new_paths = registry.filter_for_import("import-md", md_path, all_paths)
        skip_paths = frozenset(all_paths) - frozenset(new_paths)
        new_files_for_record = new_paths

    stored = skipped = unchanged = 0
    for fname in files:
        path = md_path / fname
        if path in skip_paths:
            unchanged += 1
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()

        meta: dict[str, str] = {}
        body = text
        m = fm_re.match(text)
        if m:
            for kv in kv_re.finditer(m.group(1)):
                key = kv.group(1).strip().lower()
                val = kv.group(2).strip()
                if key in meta_fields:
                    meta[key] = val
            body = text[m.end() :].strip()

        if not body:
            skipped += 1
            continue

        result = do_store(
            body,
            project=meta.get("project", ""),
            scope=meta.get("scope", ""),
            memory_type=meta.get("memory_type", ""),
            language=meta.get("language", ""),
            tags=meta.get("tags", ""),
            source=meta.get("source", args.source),
        )
        if result.get("status") in {"stored", "updated"}:
            stored += 1
        else:
            skipped += 1

    if registry is not None:
        registry.record_imports("import-md", md_path, new_files_for_record)
        suffix = f" ({unchanged} files unchanged)"
    else:
        suffix = ""
    print(f"Imported {stored} memories ({skipped} skipped) from {md_dir}/{suffix}")


def cmd_import(args: argparse.Namespace) -> None:
    from .mcp import configure, do_store

    configure(settings)

    with open(args.input) as f:
        data = json.load(f)

    stored = 0
    skipped = 0
    for mem in data:
        content = mem.get("content", "")
        if not content:
            skipped += 1
            continue
        result = do_store(
            content,
            project=mem.get("project", ""),
            scope=mem.get("scope", ""),
            memory_type=mem.get("memory_type", ""),
            language=mem.get("language", ""),
            tags=mem.get("tags", ""),
            source=mem.get("source", "import"),
        )
        if result.get("status") == "stored":
            stored += 1
        else:
            skipped += 1

    print(f"Imported {stored} memories ({skipped} skipped).")


def cmd_obsidian_export(args: argparse.Namespace) -> None:
    """Export memories to an Obsidian vault."""
    from pathlib import Path

    from .integrations.obsidian import ObsidianExporter
    from .store import Hippocampus

    hc = Hippocampus()
    memories = hc.get_all_content()
    metadata = hc.get_all_metadata()
    meta_by_id = {m["id"]: m for m in metadata}

    merged = [
        {**meta_by_id.get(m["id"], {}), "content": m["content"]} for m in memories
    ]

    if args.project:
        merged = [m for m in merged if m.get("project") == args.project]
    if args.memory_type:
        merged = [m for m in merged if m.get("memory_type") == args.memory_type]
    if args.scope:
        merged = [m for m in merged if m.get("scope") == args.scope]

    vault = Path(args.vault)
    exporter = ObsidianExporter()

    if args.incremental:
        from .integrations.sync import SyncRegistry

        registry = SyncRegistry(settings.data_dir / "sync.db")
        new_memories = registry.filter_for_export("obsidian", vault, merged)
        skip_ids = frozenset(m["id"] for m in merged) - frozenset(
            m["id"] for m in new_memories
        )
        count = exporter.export(merged, vault, skip_ids=skip_ids)
        registry.record_exports("obsidian", vault, new_memories)
        print(
            f"Exported {len(merged)} memories to {vault}/"
            f" ({count} written, {len(skip_ids)} unchanged)"
        )
    else:
        count = exporter.export(merged, vault)
        print(f"Exported {count} memories to {vault}/")


def cmd_obsidian_import(args: argparse.Namespace) -> None:
    """Import memories from an Obsidian vault."""
    from pathlib import Path

    from .integrations.obsidian import ObsidianImporter
    from .mcp import configure, do_store

    configure(settings)

    vault = Path(args.vault)
    importer = ObsidianImporter()

    registry = None
    new_files: list[Path] = []
    if args.incremental:
        from .integrations.sync import SyncRegistry

        registry = SyncRegistry(settings.data_dir / "sync.db")
        memories_root = vault / "Memories"
        all_files = (
            sorted(memories_root.rglob("*.md")) if memories_root.exists() else []
        )
        new_files = registry.filter_for_import("obsidian", vault, all_files)
        unchanged = len(all_files) - len(new_files)
        pairs = importer.import_(
            vault, only_files=frozenset(new_files), default_source=args.source
        )
    else:
        pairs = importer.import_(vault, default_source=args.source)

    stored = skipped = 0
    for content, meta in pairs:
        result = do_store(
            content,
            project=meta.get("project", ""),
            scope=meta.get("scope", ""),
            memory_type=meta.get("memory_type", ""),
            language=meta.get("language", ""),
            tags=meta.get("tags", ""),
            source=meta.get("source", args.source),
        )
        if result.get("status") in {"stored", "updated"}:
            stored += 1
        else:
            skipped += 1

    if registry is not None:
        registry.record_imports("obsidian", vault, new_files)
        print(
            f"Imported {stored} memories ({skipped} skipped) from {vault}/"
            f" ({unchanged} files unchanged)"
        )
    else:
        print(f"Imported {stored} memories ({skipped} skipped) from {vault}/")


def cmd_ingest(args: argparse.Namespace) -> None:
    from .mcp import configure, do_ingest

    configure(settings)
    result = do_ingest(
        args.path,
        project=args.project or "",
        scope=args.scope or "",
        source=args.source or "ingest",
        recursive=not args.no_recursive,
    )
    print(
        f"Ingested: {result['stored']} stored, "
        f"{result['skipped']} skipped"
        + (f", {len(result['errors'])} errors" if result["errors"] else "")
    )
    for err in result["errors"]:
        print(f"  ERROR: {err}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="myelin", description="Neuromorphic AI memory"
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status", help="Show memory system status")
    sub.add_parser("decay", help="Prune stale memories")
    sub.add_parser("serve", help="Run MCP server")
    sub.add_parser("consolidate", help="Replay memories into semantic network")

    p_graph = sub.add_parser(
        "graph", help="Serve interactive entity graph in the browser"
    )
    p_graph.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to serve on (default: random free port)",
    )
    p_graph.add_argument(
        "--min-weight",
        dest="min_weight",
        type=float,
        default=0.0,
        help="Minimum edge weight to include (default: 0 = all)",
    )
    p_graph.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Max nodes to display, ordered by degree (default: 300)",
    )
    p_graph.add_argument(
        "--no-open",
        dest="no_open",
        action="store_true",
        help="Do not open browser automatically",
    )

    p_stats = sub.add_parser("stats", help="Show memory KPI dashboard")
    p_stats.add_argument("--project", default="", help="Filter to a specific project")
    p_stats.add_argument(
        "--agent-id", dest="agent_id", default="", help="Filter to an agent namespace"
    )
    p_stats.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    p_export = sub.add_parser("export", help="Export memories to JSON")
    p_export.add_argument(
        "output", nargs="?", default="-", help="Output file (default: stdout)"
    )

    p_import = sub.add_parser("import", help="Import memories from JSON")
    p_import.add_argument("input", help="JSON file to import")

    p_export_md = sub.add_parser(
        "export-md", help="Export memories as Markdown files with YAML frontmatter"
    )
    p_export_md.add_argument("output_dir", help="Directory to write .md files into")
    p_export_md.add_argument(
        "--incremental",
        action="store_true",
        help="Only write memories that are new or changed since last export",
    )

    p_import_md = sub.add_parser(
        "import-md", help="Import memories from a directory of Markdown files"
    )
    p_import_md.add_argument("input_dir", help="Directory containing .md files")
    p_import_md.add_argument(
        "--source",
        default="import-md",
        help="Source label for imported memories (default: import-md)",
    )
    p_import_md.add_argument(
        "--incremental",
        action="store_true",
        help="Only import files that are new or changed since last import",
    )

    p_ingest = sub.add_parser(
        "ingest", help="Bulk-load memories from a file or directory"
    )
    p_ingest.add_argument("path", help="File or directory to ingest")
    p_ingest.add_argument("--project", default="", help="Default project tag")
    p_ingest.add_argument("--scope", default="", help="Default scope tag")
    p_ingest.add_argument(
        "--source", default="ingest", help="Source label (default: ingest)"
    )
    p_ingest.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subdirectories",
    )

    p_debug = sub.add_parser(
        "debug-recall",
        help="Run a recall query and show full pipeline breakdown",
    )
    p_debug.add_argument("query", help="Query string to recall against")
    p_debug.add_argument(
        "-n",
        type=int,
        default=5,
        metavar="N",
        help="Number of results (default: 5)",
    )
    p_debug.add_argument("--project", default="", help="Filter by project")
    p_debug.add_argument("--language", default="", help="Filter by language")
    p_debug.add_argument("--scope", default="", help="Filter by scope")
    p_debug.add_argument(
        "--memory-type", dest="memory_type", default="", help="Filter by memory_type"
    )
    p_debug.add_argument(
        "--agent-id", dest="agent_id", default="", help="Filter by agent namespace"
    )
    p_debug.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    p_obs_export = sub.add_parser(
        "obsidian-export",
        help="Export memories to an Obsidian vault directory",
    )
    p_obs_export.add_argument("vault", help="Path to the Obsidian vault root")
    p_obs_export.add_argument("--project", default="", help="Filter by project")
    p_obs_export.add_argument(
        "--type",
        dest="memory_type",
        default="",
        help="Filter by memory_type (episodic, semantic, procedural, prospective)",
    )
    p_obs_export.add_argument("--scope", default="", help="Filter by scope")
    p_obs_export.add_argument(
        "--incremental",
        action="store_true",
        help="Only write memories that are new or changed since last export",
    )

    p_obs_import = sub.add_parser(
        "obsidian-import",
        help="Import memories from an Obsidian vault directory",
    )
    p_obs_import.add_argument("vault", help="Path to the Obsidian vault root")
    p_obs_import.add_argument(
        "--source",
        default="obsidian-import",
        help="Source label for imported memories (default: obsidian-import)",
    )
    p_obs_import.add_argument(
        "--incremental",
        action="store_true",
        help="Only import files that are new or changed since last import",
    )

    p_gh_import = sub.add_parser(
        "github-import",
        help="Import git commit / PR / issue history as memories",
    )
    p_gh_import.add_argument("repo", help="Path to the local git repository")
    p_gh_import.add_argument(
        "--since",
        default="",
        metavar="DATE",
        help="Only import items after this date (e.g. '2025-01-01' or '6 months ago')",
    )
    p_gh_import.add_argument(
        "--branch",
        default="",
        help="Branch to walk for commits (default: current branch)",
    )
    p_gh_import.add_argument(
        "--include",
        default="commits",
        metavar="TYPES",
        help="Comma-separated list of item types: commits,prs,issues (default: commits)",  # noqa: E501
    )
    p_gh_import.add_argument(
        "--project", default="", help="Project label for imported memories"
    )
    p_gh_import.add_argument(
        "--scope", default="", help="Scope label for imported memories"
    )
    p_gh_import.add_argument(
        "--incremental",
        action="store_true",
        help="Only import commits/items not yet seen since last import",
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    setup_logging(level=getattr(logging, settings.log_level))

    commands: dict[str, Any] = {
        "status": cmd_status,
        "stats": cmd_stats,
        "decay": cmd_decay,
        "serve": cmd_serve,
        "consolidate": cmd_consolidate,
        "graph": cmd_graph,
        "export": cmd_export,
        "import": cmd_import,
        "export-md": cmd_export_md,
        "import-md": cmd_import_md,
        "obsidian-export": cmd_obsidian_export,
        "obsidian-import": cmd_obsidian_import,
        "github-import": cmd_github_import,
        "ingest": cmd_ingest,
        "debug-recall": cmd_debug_recall,
    }
    commands[args.command](args)


def cmd_github_import(args: argparse.Namespace) -> None:
    """Import git commit / PR / issue history from a repository as memories."""
    from pathlib import Path

    from .integrations.github import GitHubImporter
    from .mcp import configure, do_store

    configure(settings)

    repo = Path(args.repo).resolve()
    include = [s.strip() for s in args.include.split(",")]
    importer = GitHubImporter()

    # Incremental: filter to only new items via SyncRegistry
    registry = None
    only_shas: frozenset[str] | None = None
    only_ids: frozenset[str] | None = None
    src_key = str(repo)

    if args.incremental:
        from .integrations.sync import SyncRegistry

        registry = SyncRegistry(settings.data_dir / "sync.db")

        if "commits" in include:
            import subprocess

            sha_cmd = [
                "git",
                "-C",
                str(repo),
                "log",
                "--no-merges",
                "--format=%H",
            ]
            if args.since:
                sha_cmd.extend(["--since", args.since])
            if args.branch:
                sha_cmd.append(args.branch)
            result = subprocess.run(sha_cmd, capture_output=True, text=True, check=True)
            all_shas = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
            new_shas = registry.filter_new_items("github-commits", src_key, all_shas)
            only_shas = frozenset(new_shas)
            unchanged_commits = len(all_shas) - len(new_shas)
        else:
            unchanged_commits = 0

    try:
        pairs = importer.import_(
            repo,
            since=args.since if not args.incremental else None,
            branch=args.branch,
            include=include,
            project=args.project,
            scope=args.scope,
            only_shas=only_shas,
            only_ids=only_ids,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    stored = skipped = 0
    imported_shas: list[str] = []
    for content, meta in pairs:
        store_result = do_store(
            content,
            project=meta.get("project", ""),
            scope=meta.get("scope", ""),
            memory_type=meta.get("memory_type", ""),
            tags=meta.get("tags", ""),
            source=meta.get("source", ""),
            created_at=meta.get("created_at", ""),
        )
        if store_result.get("status") in {"stored", "updated"}:
            stored += 1
            src = meta.get("source", "")
            if src.startswith("git:"):
                imported_shas.append(src[4:])
        else:
            skipped += 1

    if registry is not None:
        if imported_shas:
            registry.record_items("github-commits", src_key, imported_shas)
        suffix = f" ({unchanged_commits} commits already up to date)"
    else:
        suffix = ""

    print(f"Imported {stored} memories ({skipped} skipped) from {repo}{suffix}")


if __name__ == "__main__":
    main()
