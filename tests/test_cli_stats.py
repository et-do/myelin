"""Tests for the `myelin stats` CLI command (cmd_stats in myelin.cli)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from myelin.cli import main
from myelin.config import MyelinSettings
from myelin.mcp import configure, do_store


@pytest.fixture(autouse=True)
def _isolate(tmp_settings: MyelinSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point CLI + server at a temp directory, ensure dirs exist."""
    configure(tmp_settings)
    monkeypatch.setattr("myelin.cli.settings", tmp_settings)
    monkeypatch.setattr("myelin.store.hippocampus.settings", tmp_settings)
    tmp_settings.ensure_dirs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_neocortex(data_dir: Path, entities: int, relationships: int) -> None:
    db_path = data_dir / "neocortex.db"
    with sqlite3.connect(str(db_path)) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL DEFAULT 'concept',
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL
            )"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS relationships (
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL DEFAULT 'co_occurs',
                object TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                valid_from TEXT,
                valid_to TEXT,
                PRIMARY KEY (subject, predicate, object)
            )"""
        )
        for i in range(entities):
            con.execute(
                "INSERT OR IGNORE INTO entities "
                "VALUES (?, 'concept', '2024-01-01', '2024-01-01')",
                [f"entity_{i}"],
            )
        for i in range(relationships):
            con.execute(
                "INSERT OR IGNORE INTO relationships "
                "VALUES (?, 'co_occurs', ?, 1.0, NULL, NULL)",
                [f"entity_{i}", f"entity_{i + 1}"],
            )
        con.commit()


def _seed_hebbian(data_dir: Path, links: int, weight: float = 1.0) -> None:
    db_path = data_dir / "hebbian.db"
    with sqlite3.connect(str(db_path)) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS co_access (
                id_a TEXT NOT NULL,
                id_b TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (id_a, id_b)
            )"""
        )
        for i in range(links):
            con.execute(
                "INSERT OR IGNORE INTO co_access VALUES (?, ?, ?)",
                [f"mem_{i}", f"mem_{i + 1}", weight],
            )
        con.commit()


def _seed_thalamus(data_dir: Path, pinned: int) -> None:
    db_path = data_dir / "thalamus.db"
    with sqlite3.connect(str(db_path)) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS pinned (
                memory_id TEXT PRIMARY KEY,
                priority INTEGER NOT NULL DEFAULT 1,
                label TEXT,
                pinned_at TEXT NOT NULL
            )"""
        )
        for i in range(pinned):
            con.execute(
                "INSERT OR IGNORE INTO pinned "
                "VALUES (?, 1, NULL, '2024-01-01T00:00:00+00:00')",
                [f"pinned_{i}"],
            )
        con.commit()


def _run_stats(
    *extra_args: str,
    capsys: pytest.CaptureFixture[str],
) -> tuple[str, str]:
    argv = ["myelin", "stats", *extra_args]
    with patch("sys.argv", argv):
        main()
    cap = capsys.readouterr()
    return cap.out, cap.err


# ---------------------------------------------------------------------------
# Empty DB
# ---------------------------------------------------------------------------


class TestCmdStatsEmpty:
    def test_json_empty_db(self, capsys: pytest.CaptureFixture[str]) -> None:
        out, _ = _run_stats("--json", capsys=capsys)
        data = json.loads(out)
        assert data["total"] == 0
        assert data["entity_count"] == 0
        assert data["relationship_count"] == 0
        assert data["pinned_count"] == 0
        assert data["hebbian"]["link_count"] == 0

    def test_terminal_empty_db_shows_no_memories(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out, _ = _run_stats(capsys=capsys)
        assert "No memories found" in out

    def test_json_output_has_expected_top_level_keys(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out, _ = _run_stats("--json", capsys=capsys)
        data = json.loads(out)
        required_keys = {
            "total",
            "entity_count",
            "relationship_count",
            "pinned_count",
            "hebbian",
            "by_type",
            "by_project",
            "by_scope",
            "by_region",
            "access",
            "age",
            "decay_candidates",
            "filter",
        }
        assert required_keys <= data.keys()


# ---------------------------------------------------------------------------
# Memory counting
# ---------------------------------------------------------------------------


class TestCmdStatsMemoryCounts:
    def test_total_reflects_stored_memories(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        do_store("First memory for stats test", memory_type="semantic")
        do_store("Second memory for stats test", memory_type="episodic")
        do_store("Third memory for stats test", memory_type="semantic")

        out, _ = _run_stats("--json", capsys=capsys)
        data = json.loads(out)
        assert data["total"] == 3

    def test_by_type_groups_correctly(self, capsys: pytest.CaptureFixture[str]) -> None:
        do_store(
            "Python uses indentation to define code blocks", memory_type="semantic"
        )
        do_store(
            "The sun rises in the east and sets in the west", memory_type="semantic"
        )
        do_store(
            "Yesterday I attended a team meeting about the project",
            memory_type="episodic",
        )

        out, _ = _run_stats("--json", capsys=capsys)
        data = json.loads(out)
        assert data["by_type"].get("semantic") == 2
        assert data["by_type"].get("episodic") == 1

    def test_by_project_groups_correctly(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        do_store(
            "Python uses indentation to delimit code blocks cleanly", project="alpha"
        )
        do_store("The capital of France is Paris, a city on the Seine", project="alpha")
        do_store(
            "Rust prevents memory bugs via ownership and borrowing rules",
            project="beta",
        )

        out, _ = _run_stats("--json", capsys=capsys)
        data = json.loads(out)
        assert data["by_project"].get("alpha") == 2
        assert data["by_project"].get("beta") == 1


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestCmdStatsFiltering:
    def test_filter_by_project_limits_total(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        do_store("Alpha project memory for filtering test", project="alpha")
        do_store("Beta project memory one for filtering", project="beta")
        do_store("Beta project memory two for filtering", project="beta")

        out, _ = _run_stats("--project", "alpha", "--json", capsys=capsys)
        data = json.loads(out)
        assert data["total"] == 1
        assert data["filter"]["project"] == "alpha"

    def test_filter_by_agent_id(self, capsys: pytest.CaptureFixture[str]) -> None:
        do_store("Agent A memory for filter test", agent_id="agent-a")
        do_store("Agent B memory for filter test", agent_id="agent-b")

        out, _ = _run_stats("--agent-id", "agent-a", "--json", capsys=capsys)
        data = json.loads(out)
        assert data["total"] == 1
        assert data["filter"]["agent_id"] == "agent-a"

    def test_filter_project_populates_filter_field(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out, _ = _run_stats("--project", "myproj", "--json", capsys=capsys)
        data = json.loads(out)
        assert data["filter"]["project"] == "myproj"

    def test_no_filter_leaves_filter_fields_null(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out, _ = _run_stats("--json", capsys=capsys)
        data = json.loads(out)
        assert data["filter"]["project"] is None
        assert data["filter"]["agent_id"] is None


# ---------------------------------------------------------------------------
# Auxiliary store counts
# ---------------------------------------------------------------------------


class TestCmdStatsAuxiliaryStores:
    def test_neocortex_entity_and_relationship_counts(
        self,
        tmp_settings: MyelinSettings,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _seed_neocortex(tmp_settings.data_dir, entities=4, relationships=3)

        out, _ = _run_stats("--json", capsys=capsys)
        data = json.loads(out)
        assert data["entity_count"] == 4
        assert data["relationship_count"] == 3

    def test_hebbian_link_count_and_stats(
        self,
        tmp_settings: MyelinSettings,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _seed_hebbian(tmp_settings.data_dir, links=5, weight=2.0)

        out, _ = _run_stats("--json", capsys=capsys)
        data = json.loads(out)
        hebb = data["hebbian"]
        assert hebb["link_count"] == 5
        assert hebb["avg_weight"] == pytest.approx(2.0, abs=0.01)
        assert hebb["max_weight"] == pytest.approx(2.0, abs=0.01)

    def test_hebbian_zero_when_db_absent(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # No seeding — hebbian.db does not exist
        out, _ = _run_stats("--json", capsys=capsys)
        data = json.loads(out)
        assert data["hebbian"]["link_count"] == 0

    def test_pinned_count(
        self,
        tmp_settings: MyelinSettings,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _seed_thalamus(tmp_settings.data_dir, pinned=3)

        out, _ = _run_stats("--json", capsys=capsys)
        data = json.loads(out)
        assert data["pinned_count"] == 3

    def test_neocortex_zero_when_db_absent(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out, _ = _run_stats("--json", capsys=capsys)
        data = json.loads(out)
        assert data["entity_count"] == 0
        assert data["relationship_count"] == 0


# ---------------------------------------------------------------------------
# Terminal output (non-JSON)
# ---------------------------------------------------------------------------


class TestCmdStatsTerminalOutput:
    def test_terminal_output_includes_header_banner(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out, _ = _run_stats(capsys=capsys)
        assert "MYELIN MEMORY STATS" in out

    def test_terminal_output_shows_memory_count(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        do_store("A single memory for terminal output test")
        out, _ = _run_stats(capsys=capsys)
        assert "Memories" in out

    def test_terminal_output_shows_access_health_section(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        do_store("Memory to test access health terminal section")
        out, _ = _run_stats(capsys=capsys)
        assert "Access Health" in out

    def test_terminal_output_shows_memory_age_section(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        do_store("Memory to test age distribution section")
        out, _ = _run_stats(capsys=capsys)
        assert "Memory Age" in out

    def test_project_filter_shown_in_header(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out, _ = _run_stats("--project", "testproject", capsys=capsys)
        assert "testproject" in out
