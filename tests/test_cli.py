"""Tests for CLI commands."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from myelin.cli import main
from myelin.config import MyelinSettings
from myelin.mcp import configure


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
        assert "decay_timer_running" in out

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


class TestCmdExportMd:
    """Tests for the export-md subcommand."""

    def test_empty_store_creates_no_files(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out_dir = tmp_path / "exports"
        with patch("sys.argv", ["myelin", "export-md", str(out_dir)]):
            main()
        assert out_dir.exists()
        assert list(out_dir.iterdir()) == []
        assert "Exported 0 memories" in capsys.readouterr().out

    def test_exported_file_has_frontmatter_and_content(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from myelin.mcp import do_store

        do_store(
            "This is a test memory",
            project="myproj",
            scope="tests",
            memory_type="semantic",
        )
        out_dir = tmp_path / "exports"
        with patch("sys.argv", ["myelin", "export-md", str(out_dir)]):
            main()
        files = list(out_dir.glob("*.md"))
        assert len(files) == 1
        text = files[0].read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "project: myproj" in text
        assert "scope: tests" in text
        assert "memory_type: semantic" in text
        assert "This is a test memory" in text
        assert "Exported 1 memories" in capsys.readouterr().out

    def test_file_named_by_memory_id(self, tmp_path: pytest.TempPathFactory) -> None:
        from myelin.mcp import do_store

        result = do_store("Another memory for naming test")
        mid = result["id"]
        out_dir = tmp_path / "named"
        with patch("sys.argv", ["myelin", "export-md", str(out_dir)]):
            main()
        assert (out_dir / f"{mid}.md").exists()

    def test_empty_metadata_fields_omitted(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from myelin.mcp import do_store

        do_store("Minimal memory to test that empty frontmatter fields are omitted")
        out_dir = tmp_path / "min"
        with patch("sys.argv", ["myelin", "export-md", str(out_dir)]):
            main()
        text = next(iter(out_dir.glob("*.md"))).read_text(encoding="utf-8")
        # Fields with empty values should not appear in frontmatter
        assert "project: \n" not in text
        assert "scope: \n" not in text


class TestCmdImportMd:
    """Tests for the import-md subcommand."""

    def test_import_roundtrip(
        self,
        tmp_path: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """export-md produces files that import-md can ingest without error."""
        from myelin.mcp import do_store

        do_store(
            "Roundtrip test memory for markdown export and import validation",
            project="demo",
            scope="roundtrip",
        )
        out_dir = tmp_path / "round"
        with patch("sys.argv", ["myelin", "export-md", str(out_dir)]):
            main()
        capsys.readouterr()  # discard export output

        # Files should exist and be importable (import-md counts stored or skipped)
        assert len(list(out_dir.glob("*.md"))) == 1
        with patch("sys.argv", ["myelin", "import-md", str(out_dir)]):
            main()
        out = capsys.readouterr().out
        # The memory is a duplicate — it will be updated or skipped either way
        assert "memories" in out

    def test_import_parses_frontmatter_metadata(
        self,
        tmp_path: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        md_dir = tmp_path / "md_in"
        md_dir.mkdir()
        (md_dir / "mem1.md").write_text(
            "---\nproject: proj1\nscope: sc1\nmemory_type: episodic\n---\n\n"
            "Content body long enough to pass minimum length validation",
            encoding="utf-8",
        )
        with patch("sys.argv", ["myelin", "import-md", str(md_dir)]):
            main()
        out = capsys.readouterr().out
        assert "Imported 1 memories" in out

    def test_import_skips_empty_body(
        self,
        tmp_path: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        md_dir = tmp_path / "empty_body"
        md_dir.mkdir()
        (md_dir / "empty.md").write_text(
            "---\nproject: p\n---\n\n",
            encoding="utf-8",
        )
        with patch("sys.argv", ["myelin", "import-md", str(md_dir)]):
            main()
        out = capsys.readouterr().out
        assert "0 memories" in out

    def test_import_no_md_files(
        self,
        tmp_path: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        md_dir = tmp_path / "empty_dir"
        md_dir.mkdir()
        with patch("sys.argv", ["myelin", "import-md", str(md_dir)]):
            main()
        out = capsys.readouterr().out
        assert "No .md files found" in out

    def test_import_without_frontmatter(
        self,
        tmp_path: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Files with no frontmatter should still be imported as raw content."""
        md_dir = tmp_path / "raw"
        md_dir.mkdir()
        (md_dir / "raw.md").write_text(
            "Just plain content with no frontmatter block",
            encoding="utf-8",
        )
        with patch("sys.argv", ["myelin", "import-md", str(md_dir)]):
            main()
        out = capsys.readouterr().out
        assert "Imported 1 memories" in out

    def test_import_custom_source_label(
        self,
        tmp_path: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        md_dir = tmp_path / "custom_src"
        md_dir.mkdir()
        (md_dir / "m.md").write_text(
            "Content for source label test",
            encoding="utf-8",
        )
        with patch(
            "sys.argv", ["myelin", "import-md", str(md_dir), "--source", "test-run"]
        ):
            main()
        out = capsys.readouterr().out
        assert "Imported 1 memories" in out
