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
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
