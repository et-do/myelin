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
    from .server import configure, do_status

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
    from .server import main as serve_main

    serve_main()


def cmd_consolidate(_args: argparse.Namespace) -> None:
    from .server import do_consolidate

    result = do_consolidate()
    print(json.dumps(result, indent=2))


def cmd_graph(args: argparse.Namespace) -> None:
    """Serve an interactive D3 graph of the semantic entity network."""
    import http.server
    import importlib.resources
    import socket
    import threading
    import webbrowser

    from .store.neocortex import SemanticNetwork

    net = SemanticNetwork()
    graph_data = net.get_graph(min_weight=args.min_weight, limit_nodes=args.limit)
    graph_json = json.dumps(graph_data)

    # Load the HTML template and inject graph data
    try:
        # Python 3.9+ importlib.resources API
        ref = importlib.resources.files("myelin").joinpath("graph.html")
        html_template = ref.read_text(encoding="utf-8")
    except Exception:
        import os

        html_template = open(
            os.path.join(os.path.dirname(__file__), "graph.html"), encoding="utf-8"
        ).read()

    html = html_template.replace(
        '/*GRAPH_DATA*/{"nodes":[],"edges":[]}', f"/*GRAPH_DATA*/{graph_json}"
    )

    # Find a free port
    with socket.socket() as s:
        s.bind(("127.0.0.1", args.port))
        port = s.getsockname()[1]

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *a: object) -> None:  # noqa: ANN002
            pass  # silence request logs

    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{port}"

    n_nodes = len(graph_data["nodes"])
    n_edges = len(graph_data["edges"])
    print(f"Graph: {n_nodes} nodes, {n_edges} edges")
    print(f"Serving at {url}  (Ctrl-C to stop)")

    if not args.no_open:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def cmd_debug_recall(args: argparse.Namespace) -> None:
    from .server import configure, do_debug_recall

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

    from .store import Hippocampus

    hc = Hippocampus()
    memories = hc.get_all_content()
    metadata = hc.get_all_metadata()
    meta_by_id = {m["id"]: m for m in metadata}

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    meta_fields = ("project", "scope", "language", "memory_type", "source", "tags")
    count = 0
    for mem in memories:
        mid = mem["id"]
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

    print(f"Exported {count} memories to {out_dir}/")


def cmd_import_md(args: argparse.Namespace) -> None:
    """Import memories from a directory of Markdown files with YAML frontmatter."""
    import os
    import re

    from .server import configure, do_store

    configure(settings)

    fm_re = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
    kv_re = re.compile(r"^(\w+)\s*:\s*(.+)$", re.MULTILINE)
    meta_fields = {"project", "scope", "language", "memory_type", "source", "tags"}

    md_dir = args.input_dir
    files = sorted(f for f in os.listdir(md_dir) if f.endswith(".md"))
    if not files:
        print(f"No .md files found in {md_dir}")
        return

    stored = skipped = 0
    for fname in files:
        path = os.path.join(md_dir, fname)
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

    print(f"Imported {stored} memories ({skipped} skipped) from {md_dir}/")


def cmd_import(args: argparse.Namespace) -> None:
    from .server import configure, do_store

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
    count = exporter.export(merged, vault)
    print(f"Exported {count} memories to {vault}/")


def cmd_obsidian_import(args: argparse.Namespace) -> None:
    """Import memories from an Obsidian vault."""
    from pathlib import Path

    from .integrations.obsidian import ObsidianImporter
    from .server import configure, do_store

    configure(settings)

    vault = Path(args.vault)
    importer = ObsidianImporter()
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

    print(f"Imported {stored} memories ({skipped} skipped) from {vault}/")


def cmd_ingest(args: argparse.Namespace) -> None:
    from .server import configure, do_ingest

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

    p_import_md = sub.add_parser(
        "import-md", help="Import memories from a directory of Markdown files"
    )
    p_import_md.add_argument("input_dir", help="Directory containing .md files")
    p_import_md.add_argument(
        "--source",
        default="import-md",
        help="Source label for imported memories (default: import-md)",
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
        "ingest": cmd_ingest,
        "debug-recall": cmd_debug_recall,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
