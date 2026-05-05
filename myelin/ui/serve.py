"""Graph UI server — builds the data payload and serves the HTML dashboard.

Extracted from cli.py so that ``cmd_graph`` is a thin dispatch call and
this module can be tested independently of argparse.
"""

from __future__ import annotations

import argparse
import http.server
import importlib.resources
import json
import socket
import threading
import webbrowser
from typing import Any

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_chromadb_raw(
    data_dir: Any,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Return (ids, docs, metas) from ChromaDB without loading the embedding model."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
        collection = client.get_collection("memories")
        result = collection.get(include=["documents", "metadatas"])
        return (
            result.get("ids") or [],
            result.get("documents") or [],
            [dict(m) for m in (result.get("metadatas") or [])],
        )
    except Exception:
        return [], [], []


def enrich_graph_with_memories(
    graph_data: dict[str, Any],
    ids: list[str],
    docs: list[str],
    metas: list[dict[str, Any]],
) -> None:
    """Enrich graph nodes with memory snippets and project/type metadata."""
    node_set = {n["id"].lower() for n in graph_data["nodes"]}
    entity_mems: dict[str, list[dict[str, str]]] = {}
    all_projects: set[str] = set()
    all_memory_types: set[str] = set()

    for mid, doc, meta in zip(ids, docs, metas):
        if not doc or meta is None:
            continue
        doc_lower = doc.lower()
        project = str(meta.get("project") or "")
        scope = str(meta.get("scope") or "")
        mtype = str(meta.get("memory_type") or "")
        tags = str(meta.get("tags") or "")
        if project:
            all_projects.add(project)
        if mtype:
            all_memory_types.add(mtype)

        snippet = doc[:140].replace("\n", " ").strip()
        for entity in node_set:
            if entity in doc_lower:
                bucket = entity_mems.setdefault(entity, [])
                if len(bucket) < 5:
                    bucket.append(
                        {
                            "id": mid[:8],
                            "snippet": snippet,
                            "project": project,
                            "scope": scope,
                            "memory_type": mtype,
                            "tags": tags,
                        }
                    )

    for node in graph_data["nodes"]:
        mems = entity_mems.get(node["id"].lower(), [])
        node["memory_count"] = len(mems)
        projects = [m["project"] for m in mems if m["project"]]
        node["dominant_project"] = (
            max(set(projects), key=projects.count) if projects else ""
        )
        node["projects"] = sorted(set(projects))

    graph_data["memories"] = {k: v for k, v in entity_mems.items()}
    graph_data["projects"] = sorted(all_projects)
    graph_data["memory_types"] = sorted(all_memory_types)


def build_memories_and_stats(
    ids: list[str],
    docs: list[str],
    metas: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the full memory list and aggregate stats dict for the dashboard."""
    all_memories: list[dict[str, Any]] = []
    by_type: dict[str, int] = {}
    by_project: dict[str, int] = {}
    by_scope: dict[str, int] = {}

    for mid, doc, meta in zip(ids, docs, metas):
        if not doc or meta is None:
            continue
        project = str(meta.get("project") or "")
        scope = str(meta.get("scope") or "")
        mtype = str(meta.get("memory_type") or "general")
        tags = str(meta.get("tags") or "")
        access_count = int(meta.get("access_count") or 0)
        created_at = str(meta.get("created_at") or "")
        last_accessed = str(meta.get("last_accessed") or "")

        all_memories.append(
            {
                "id": mid[:8],
                "content": doc[:300].replace("\n", " ").strip(),
                "project": project,
                "scope": scope,
                "memory_type": mtype,
                "tags": tags,
                "access_count": access_count,
                "created_at": created_at[:10],
                "last_accessed": last_accessed[:10],
            }
        )
        by_type[mtype] = by_type.get(mtype, 0) + 1
        if project:
            by_project[project] = by_project.get(project, 0) + 1
        if scope:
            by_scope[scope] = by_scope.get(scope, 0) + 1

    all_memories.sort(key=lambda m: m["last_accessed"], reverse=True)

    stats: dict[str, Any] = {
        "total": len(all_memories),
        "by_type": dict(sorted(by_type.items(), key=lambda x: x[1], reverse=True)),
        "by_project": dict(
            sorted(by_project.items(), key=lambda x: x[1], reverse=True)
        ),
        "by_scope": dict(sorted(by_scope.items(), key=lambda x: x[1], reverse=True)),
    }
    return all_memories, stats


# ---------------------------------------------------------------------------
# HTTP serving
# ---------------------------------------------------------------------------

_PLACEHOLDER = '/*APP_DATA*/{"graph":{},"all_memories":[],"stats":{}}'


def _load_template() -> str:
    """Load graph.html from the ui subpackage."""
    try:
        ref = importlib.resources.files("myelin.ui").joinpath("graph.html")
        return ref.read_text(encoding="utf-8")
    except Exception:
        import os

        return open(
            os.path.join(os.path.dirname(__file__), "graph.html"),
            encoding="utf-8",
        ).read()


def serve_graph(args: argparse.Namespace) -> None:
    """Build the app payload and start the local HTTP server."""
    from ..config import settings
    from ..store.neocortex import SemanticNetwork

    net = SemanticNetwork()
    graph_data = net.get_graph(min_weight=args.min_weight, limit_nodes=args.limit)

    ids, docs, metas = load_chromadb_raw(settings.data_dir)
    enrich_graph_with_memories(graph_data, ids, docs, metas)
    all_memories, stats = build_memories_and_stats(ids, docs, metas)

    payload_json = json.dumps(
        {
            "graph": graph_data,
            "all_memories": all_memories,
            "stats": stats,
        }
    )

    html = _load_template().replace(_PLACEHOLDER, f"/*APP_DATA*/{payload_json}")

    with socket.socket() as s:
        s.bind(("127.0.0.1", args.port))
        port = s.getsockname()[1]

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *a: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{port}"

    n_nodes = len(graph_data["nodes"])
    n_edges = len(graph_data["edges"])
    n_mems = stats.get("total", 0)
    print(f"Graph: {n_nodes} nodes, {n_edges} edges · {n_mems} memories")
    print(f"Serving at {url}  (Ctrl-C to stop)")

    if not args.no_open:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
