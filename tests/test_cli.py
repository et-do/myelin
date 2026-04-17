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
        assert "data_dir" in out
        assert "embedding_model" in out

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


class TestCmdDecayWithStale:
    def test_decay_prunes_stale(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """decay command reports pruned count when stale memories exist."""
        from unittest.mock import MagicMock

        import myelin.cli as cli_mod

        fake_hc = MagicMock()
        fake_hc.get_all_metadata.return_value = [{"id": "abc"}]
        fake_hc.forget_batch.return_value = 1
        monkeypatch.setattr(cli_mod, "Hippocampus", lambda: fake_hc, raising=False)

        # Patch find_stale to return a stale ID
        def _make_stale_hc():
            return fake_hc

        with (
            patch("sys.argv", ["myelin", "decay"]),
            patch("myelin.cli.Hippocampus", return_value=fake_hc),
            patch("myelin.recall.find_stale", return_value=["abc"]),
        ):
            main()
        out = capsys.readouterr().out
        assert "Pruned" in out


class TestCmdExport:
    def test_export_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        """export with no file arg prints JSON to stdout."""
        with patch("sys.argv", ["myelin", "export"]):
            main()
        out = capsys.readouterr().out
        import json

        data = json.loads(out)
        assert isinstance(data, list)

    def test_export_to_file(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """export to a file writes JSON and prints confirmation."""
        import json

        out_file = tmp_path / "export.json"
        with patch("sys.argv", ["myelin", "export", str(out_file)]):
            main()
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert isinstance(data, list)
        out = capsys.readouterr().out
        assert "Exported" in out


class TestCmdImport:
    def test_import_valid_memories(
        self,
        tmp_path: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """import stores valid memories and reports counts."""
        import json

        memories = [
            {"content": "The API gateway routes requests to microservices"},
            {"content": "Redis is used for session caching across nodes"},
        ]
        in_file = tmp_path / "import.json"
        in_file.write_text(json.dumps(memories))

        with patch("sys.argv", ["myelin", "import", str(in_file)]):
            main()
        out = capsys.readouterr().out
        assert "Imported" in out
        assert "memories" in out

    def test_import_skips_empty_content(
        self,
        tmp_path: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """import skips entries with no content field."""
        import json

        memories = [
            {"content": ""},
            {"project": "only-metadata-no-content"},
        ]
        in_file = tmp_path / "import.json"
        in_file.write_text(json.dumps(memories))

        with patch("sys.argv", ["myelin", "import", str(in_file)]):
            main()
        out = capsys.readouterr().out
        assert "0 memories" in out or "skipped" in out.lower()
