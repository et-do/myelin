"""Tests for myelin graph — get_graph data function and cmd_graph CLI."""

from __future__ import annotations

import argparse
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from myelin.config import MyelinSettings
from myelin.store.neocortex import SemanticNetwork


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def net(tmp_path: Path) -> SemanticNetwork:
    """SemanticNetwork backed by a temp SQLite DB, pre-populated."""
    sn = SemanticNetwork(db_path=tmp_path / "neocortex.db")

    # Build a small graph:
    #   jwt ─── auth ─── oauth   (weight 2 for jwt-auth)
    #   auth ─── security
    sn.add_relationship("jwt", "auth", weight=2.0)
    sn.add_relationship("jwt", "auth", weight=2.0)  # Hebbian bump → 4.0
    sn.add_relationship("oauth", "auth", weight=1.5)
    sn.add_relationship("auth", "security", weight=1.0)

    return sn


# ---------------------------------------------------------------------------
# SemanticNetwork.get_graph
# ---------------------------------------------------------------------------


class TestGetGraph:
    def test_returns_nodes_and_edges_keys(self, net: SemanticNetwork) -> None:
        g = net.get_graph()
        assert "nodes" in g
        assert "edges" in g

    def test_nodes_have_required_fields(self, net: SemanticNetwork) -> None:
        g = net.get_graph()
        for node in g["nodes"]:
            assert "id" in node
            assert "entity_type" in node
            assert "degree" in node

    def test_edges_have_required_fields(self, net: SemanticNetwork) -> None:
        g = net.get_graph()
        for edge in g["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert "weight" in edge

    def test_all_fixture_nodes_present(self, net: SemanticNetwork) -> None:
        g = net.get_graph()
        ids = {n["id"] for n in g["nodes"]}
        assert {"jwt", "auth", "oauth", "security"} <= ids

    def test_auth_has_highest_degree(self, net: SemanticNetwork) -> None:
        g = net.get_graph()
        by_degree = {n["id"]: n["degree"] for n in g["nodes"]}
        # auth is connected to jwt, oauth, security → degree 3
        assert by_degree["auth"] == 3

    def test_min_weight_filters_edges(self, net: SemanticNetwork) -> None:
        # Only jwt-auth edge has weight >= 4.0 after two adds
        g = net.get_graph(min_weight=4.0)
        assert all(e["weight"] >= 4.0 for e in g["edges"])

    def test_min_weight_filters_isolated_nodes(self, net: SemanticNetwork) -> None:
        # With high min_weight, nodes only connected by low-weight edges disappear
        g = net.get_graph(min_weight=4.0)
        ids = {n["id"] for n in g["nodes"]}
        # security and oauth have weight < 4 → not in filtered graph
        assert "security" not in ids
        assert "oauth" not in ids

    def test_limit_nodes_caps_result(self, net: SemanticNetwork) -> None:
        g = net.get_graph(limit_nodes=2)
        assert len(g["nodes"]) <= 2

    def test_limit_nodes_prefers_high_degree(self, net: SemanticNetwork) -> None:
        g = net.get_graph(limit_nodes=1)
        # auth has the highest degree, should be kept
        assert g["nodes"][0]["id"] == "auth"

    def test_empty_graph_returns_empty(self, tmp_path: Path) -> None:
        sn = SemanticNetwork(db_path=tmp_path / "empty.db")
        g = sn.get_graph()
        assert g == {"nodes": [], "edges": []}

    def test_edges_only_between_selected_nodes(self, net: SemanticNetwork) -> None:
        g = net.get_graph(limit_nodes=2)
        node_ids = {n["id"] for n in g["nodes"]}
        for e in g["edges"]:
            assert e["source"] in node_ids
            assert e["target"] in node_ids


# ---------------------------------------------------------------------------
# cmd_graph — HTTP serving
# ---------------------------------------------------------------------------


class TestCmdGraph:
    def _run_graph_server(
        self, net: SemanticNetwork, tmp_settings: MyelinSettings
    ) -> tuple[str, threading.Thread]:
        """Start cmd_graph in a background thread, return (url, thread)."""
        import myelin.cli as cli_mod

        # Patch SemanticNetwork so cmd_graph uses our fixture net
        original = cli_mod.__dict__.get("_get_net")

        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        args = argparse.Namespace(
            port=port,
            min_weight=0.0,
            limit=300,
            no_open=True,
        )

        # Monkeypatch SemanticNetwork inside cmd_graph's local scope
        import myelin.store.neocortex as nc_mod

        _orig_cls = nc_mod.SemanticNetwork

        class _PatchedNet(SemanticNetwork):
            def __init__(self, **kwargs: object) -> None:  # type: ignore[override]
                pass  # skip real init

            def get_graph(self, **kwargs: object) -> dict:  # type: ignore[override]
                return net.get_graph()

        nc_mod.SemanticNetwork = _PatchedNet  # type: ignore[assignment]
        ready = threading.Event()

        def _serve() -> None:
            ready.set()
            try:
                cli_mod.cmd_graph(args)
            except Exception:
                pass

        t = threading.Thread(target=_serve, daemon=True)
        t.start()
        ready.wait(timeout=2)
        nc_mod.SemanticNetwork = _orig_cls  # type: ignore[assignment]
        return f"http://127.0.0.1:{port}", t

    def test_graph_html_served(
        self, net: SemanticNetwork, tmp_settings: MyelinSettings
    ) -> None:
        import socket
        import time

        import myelin.cli as cli_mod
        import myelin.store.neocortex as nc_mod

        _orig_cls = nc_mod.SemanticNetwork

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        captured_net = net

        class _PatchedNet(SemanticNetwork):  # type: ignore[override]
            def __init__(self, **_: object) -> None:
                pass

            def get_graph(self, **_: object) -> dict:  # type: ignore[override]
                return captured_net.get_graph()

        nc_mod.SemanticNetwork = _PatchedNet  # type: ignore[assignment]

        args = argparse.Namespace(port=port, min_weight=0.0, limit=300, no_open=True)

        t = threading.Thread(target=cli_mod.cmd_graph, args=(args,), daemon=True)
        t.start()
        time.sleep(0.5)  # let server start

        nc_mod.SemanticNetwork = _orig_cls  # type: ignore[assignment]

        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=3)
            body = resp.read().decode("utf-8")
            assert "myelin graph" in body
            assert "GRAPH_DATA" in body
            # Graph data injected
            data = json.loads(body.split("/*GRAPH_DATA*/")[1].split(";")[0].rstrip())
            assert len(data["nodes"]) > 0
        finally:
            # Daemon thread dies with test process
            pass

    def test_graph_html_contains_d3_script(
        self, net: SemanticNetwork, tmp_settings: MyelinSettings
    ) -> None:
        """Smoke test: HTML template loads D3 from CDN."""
        import importlib.resources

        ref = importlib.resources.files("myelin").joinpath("graph.html")
        html = ref.read_text(encoding="utf-8")
        assert "d3" in html.lower()
        assert "forceSimulation" in html
