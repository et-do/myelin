"""CLI entrypoint for myelin."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from .config import settings
from .log import setup_logging, suppress_noisy_loggers

suppress_noisy_loggers()


def cmd_status(_args: argparse.Namespace) -> None:
    from .server import configure, do_status

    configure(settings)
    result = do_status()
    print(json.dumps(result, indent=2))


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
        "--json", action="store_true", help="Output raw JSON instead of formatted text"
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    setup_logging(level=getattr(logging, settings.log_level))

    commands: dict[str, Any] = {
        "status": cmd_status,
        "decay": cmd_decay,
        "serve": cmd_serve,
        "consolidate": cmd_consolidate,
        "export": cmd_export,
        "import": cmd_import,
        "export-md": cmd_export_md,
        "import-md": cmd_import_md,
        "ingest": cmd_ingest,
        "debug-recall": cmd_debug_recall,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
