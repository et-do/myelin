"""Tests for CLI commands."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from myelin.cli import main
from myelin.config import MyelinSettings
from myelin.server import configure


@pytest.fixture(autouse=True)
def _isolate(tmp_settings: MyelinSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point CLI + server + hippocampus at temp directory for each test."""
    configure(tmp_settings)
    monkeypatch.setattr("myelin.cli.settings", tmp_settings)
    monkeypatch.setattr("myelin.store.hippocampus.settings", tmp_settings)


class TestCmdStatus:
    def test_prints_json_status(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("sys.argv", ["myelin", "status"]):
            main()
        out = capsys.readouterr().out
        assert "memory_count" in out
        assert "summary_count" in out
        assert "consistent" in out
        assert "data_dir" in out
        assert "embedding_model" in out
        assert "worker" in out

    def test_status_shows_zero_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("sys.argv", ["myelin", "status"]):
            main()
        out = capsys.readouterr().out
        assert '"memory_count": 0' in out


class TestCmdDecay:
    def test_decay_no_stale(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("sys.argv", ["myelin", "decay"]):
            main()
        out = capsys.readouterr().out
        assert "No stale memories" in out


class TestCmdServe:
    def test_serve_calls_server_main(self) -> None:
        with (
            patch("sys.argv", ["myelin", "serve"]),
            patch("myelin.cli.cmd_serve") as mock_serve,
        ):
            main()
        mock_serve.assert_called_once()


class TestCmdConsolidate:
    def test_consolidate_prints_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("sys.argv", ["myelin", "consolidate"]):
            main()
        out = capsys.readouterr().out
        assert "memories_replayed" in out
        assert "entities_found" in out


class TestMainNoArgs:
    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("sys.argv", ["myelin"]), pytest.raises(SystemExit, match="0"):
            main()
        out = capsys.readouterr().out
        assert "usage" in out.lower() or "myelin" in out.lower()
