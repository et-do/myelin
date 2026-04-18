"""MCP server integration tests — spawn the real server process and talk JSON-RPC.

These tests use the MCP Python SDK's own stdio_client to connect to a live
``myelin serve`` subprocess.  They validate that the server starts up, responds
to initialize, and that the core MCP tools are callable end-to-end.

Marked with ``@pytest.mark.integration`` so they can be skipped in fast unit
runs with ``pytest -m "not integration"``.  They run in the normal CI suite
because the server starts quickly (warm-up is backgrounded via lifespan).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters, stdio_client


def _server_params(data_dir: Path) -> StdioServerParameters:
    """Build stdio server params pointing at a tmp data directory."""
    env = {**os.environ, "MYELIN_DATA_DIR": str(data_dir)}
    return StdioServerParameters(
        command="uv",
        args=["run", "myelin", "serve"],
        env=env,
    )


@pytest.mark.integration
class TestMcpServerStartup:
    def test_initialize_succeeds(self, tmp_path: Path) -> None:
        """Server responds to MCP initialize with protocol version and name."""

        async def _run() -> tuple[str, str]:
            async with stdio_client(_server_params(tmp_path)) as (r, w):
                async with ClientSession(r, w) as session:
                    result = await session.initialize()
                    return result.protocolVersion, result.serverInfo.name

        proto, name = anyio.run(_run)
        assert proto, "protocolVersion should be non-empty"
        assert name == "myelin"

    def test_tools_list_includes_core_tools(self, tmp_path: Path) -> None:
        """Server exposes the expected set of MCP tools after initialize."""

        async def _run() -> set[str]:
            async with stdio_client(_server_params(tmp_path)) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    return {t.name for t in resp.tools}

        tools = anyio.run(_run)
        expected = {"store", "recall", "forget", "status", "health"}
        assert expected <= tools, f"Missing tools: {expected - tools}"

    def test_health_tool_returns_ok(self, tmp_path: Path) -> None:
        """health tool returns ok without initializing stores."""

        async def _run() -> dict:  # type: ignore[type-arg]
            async with stdio_client(_server_params(tmp_path)) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    result = await session.call_tool("health", {})
                    return json.loads(result.content[0].text)  # type: ignore[index]

        data = anyio.run(_run)
        assert data["status"] == "ok"
        assert "version" in data

    def test_store_then_recall_round_trip(self, tmp_path: Path) -> None:
        """store followed by recall finds the stored memory via MCP transport."""

        async def _run() -> tuple[str, list[dict]]:  # type: ignore[type-arg]
            async with stdio_client(_server_params(tmp_path)) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()

                    content = (
                        "The deployment pipeline uses GitHub Actions"
                        " with canary releases"
                    )
                    stored = await session.call_tool(
                        "store",
                        {"content": content, "project": "devops"},
                    )
                    store_data = json.loads(stored.content[0].text)  # type: ignore[index]

                    recalled = await session.call_tool(
                        "recall",
                        {"query": "deployment pipeline", "project": "devops"},
                    )
                    recall_data = json.loads(recalled.content[0].text)  # type: ignore[index]
                    return store_data["status"], recall_data

        status, results = anyio.run(_run)
        assert status == "stored"
        assert len(results) >= 1
        assert any("canary" in r["content"] for r in results)

    def test_status_includes_worker_info(self, tmp_path: Path) -> None:
        """status tool response includes background worker health info."""

        async def _run() -> dict:  # type: ignore[type-arg]
            async with stdio_client(_server_params(tmp_path)) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    result = await session.call_tool("status", {})
                    return json.loads(result.content[0].text)  # type: ignore[index]

        data = anyio.run(_run)
        assert "worker" in data, "status response is missing 'worker' key"
        worker = data["worker"]
        assert "running" in worker
        assert "queue_depth" in worker
        assert "last_consolidation_at" in worker
        assert "last_decay_at" in worker
        # Worker should be running because the MCP lifespan starts it
        assert worker["running"] is True
