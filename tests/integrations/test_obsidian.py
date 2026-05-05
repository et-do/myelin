"""Tests for the Obsidian integration — exporter and importer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from myelin.integrations.obsidian import ObsidianExporter, ObsidianImporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEMANTIC_MEMORY: dict[str, Any] = {
    "id": "abc12345def67890",
    "content": (
        "We use JWT RS256 for API auth because asymmetric keys let the gateway"
        " verify without the signing secret."
    ),
    "memory_type": "semantic",
    "project": "api",
    "scope": "auth",
    "source": "copilot",
    "created_at": "2025-01-01T00:00:00+00:00",
    "tags": "",
}

_EPISODIC_MEMORY: dict[str, Any] = {
    "id": "bbbbbbbbcccccccc",
    "content": (
        "Fixed a 500 error in the login endpoint caused by missing null-check"
        " on user.email."
    ),
    "memory_type": "episodic",
    "project": "api",
    "scope": "auth",
    "source": "copilot",
    "created_at": "2025-02-01T00:00:00+00:00",
    "tags": "bug,login",
}

_MEMORIES = [_SEMANTIC_MEMORY, _EPISODIC_MEMORY]


# ---------------------------------------------------------------------------
# ObsidianExporter
# ---------------------------------------------------------------------------


class TestObsidianExporter:
    def test_returns_count(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        count = exporter.export(_MEMORIES, tmp_path)
        assert count == 2

    def test_creates_memories_subfolders(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        exporter.export(_MEMORIES, tmp_path)
        assert (tmp_path / "Memories" / "semantic").is_dir()
        assert (tmp_path / "Memories" / "episodic").is_dir()

    def test_creates_index_note(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        exporter.export(_MEMORIES, tmp_path)
        index = tmp_path / "Memory Index.md"
        assert index.exists()
        text = index.read_text()
        assert "Myelin Memory Index" in text
        assert "2** memories" in text

    def test_note_contains_frontmatter(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        exporter.export([_SEMANTIC_MEMORY], tmp_path)
        notes = list((tmp_path / "Memories" / "semantic").glob("*.md"))
        assert len(notes) == 1
        text = notes[0].read_text()
        assert "---" in text
        assert "id: abc12345def67890" in text
        assert "memory_type: semantic" in text
        assert "project: api" in text
        assert "scope: auth" in text

    def test_note_contains_content(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        exporter.export([_SEMANTIC_MEMORY], tmp_path)
        notes = list((tmp_path / "Memories" / "semantic").glob("*.md"))
        text = notes[0].read_text()
        assert "JWT RS256" in text

    def test_note_contains_entity_links(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        exporter.export([_SEMANTIC_MEMORY], tmp_path)
        notes = list((tmp_path / "Memories" / "semantic").glob("*.md"))
        text = notes[0].read_text()
        # JWT RS256 or api auth should appear as wikilinks
        assert "[[" in text
        assert "## Related" in text

    def test_entity_links_suppressed_by_opt(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        exporter.export([_SEMANTIC_MEMORY], tmp_path, include_entity_links=False)
        notes = list((tmp_path / "Memories" / "semantic").glob("*.md"))
        text = notes[0].read_text()
        assert "## Related" not in text

    def test_obsidian_tags_include_type_and_project(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        exporter.export([_SEMANTIC_MEMORY], tmp_path)
        notes = list((tmp_path / "Memories" / "semantic").glob("*.md"))
        text = notes[0].read_text()
        assert "type/semantic" in text
        assert "project/api" in text

    def test_tags_from_metadata_included(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        exporter.export([_EPISODIC_MEMORY], tmp_path)
        notes = list((tmp_path / "Memories" / "episodic").glob("*.md"))
        text = notes[0].read_text()
        assert "bug" in text
        assert "login" in text

    def test_empty_memories_writes_zero(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        count = exporter.export([], tmp_path)
        assert count == 0

    def test_vault_created_if_not_exists(self, tmp_path: Path) -> None:
        vault = tmp_path / "new_vault"
        exporter = ObsidianExporter()
        exporter.export(_MEMORIES, vault)
        assert vault.is_dir()

    def test_filename_starts_with_id_prefix(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        exporter.export([_SEMANTIC_MEMORY], tmp_path)
        notes = list((tmp_path / "Memories" / "semantic").glob("*.md"))
        assert notes[0].name.startswith("abc12345")

    def test_memories_without_type_go_to_general(self, tmp_path: Path) -> None:
        mem = {**_SEMANTIC_MEMORY, "memory_type": None}
        exporter = ObsidianExporter()
        count = exporter.export([mem], tmp_path)
        assert count == 1
        assert (tmp_path / "Memories" / "general").is_dir()

    def test_index_links_to_notes(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        exporter.export(_MEMORIES, tmp_path)
        index = (tmp_path / "Memory Index.md").read_text()
        assert "[[Memories/" in index


# ---------------------------------------------------------------------------
# ObsidianImporter
# ---------------------------------------------------------------------------


class TestObsidianImporter:
    def _export_then_import(self, tmp_path: Path) -> list[tuple[str, dict[str, str]]]:
        exporter = ObsidianExporter()
        exporter.export(_MEMORIES, tmp_path)
        importer = ObsidianImporter()
        return importer.import_(tmp_path)

    def test_returns_same_count(self, tmp_path: Path) -> None:
        pairs = self._export_then_import(tmp_path)
        assert len(pairs) == 2

    def test_content_is_preserved(self, tmp_path: Path) -> None:
        pairs = self._export_then_import(tmp_path)
        contents = [p[0] for p in pairs]
        assert any("JWT RS256" in c for c in contents)
        assert any("500 error" in c for c in contents)

    def test_related_section_stripped(self, tmp_path: Path) -> None:
        pairs = self._export_then_import(tmp_path)
        for content, _ in pairs:
            assert "## Related" not in content
            assert "[[" not in content

    def test_metadata_roundtrip(self, tmp_path: Path) -> None:
        pairs = self._export_then_import(tmp_path)
        metas = [p[1] for p in pairs]
        semantic_meta = next(m for m in metas if m.get("memory_type") == "semantic")
        assert semantic_meta["project"] == "api"
        assert semantic_meta["scope"] == "auth"

    def test_default_source_applied(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        exporter.export(_MEMORIES, tmp_path)
        # Remove source from one note to simulate hand-written note
        notes = list((tmp_path / "Memories" / "semantic").glob("*.md"))
        text = notes[0].read_text()
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("source:")
        )
        notes[0].write_text(text)

        importer = ObsidianImporter()
        pairs = importer.import_(tmp_path, default_source="hand-written")
        metas = [p[1] for p in pairs]
        sources = {m.get("source") for m in metas}
        assert "hand-written" in sources

    def test_raises_if_no_memories_dir(self, tmp_path: Path) -> None:
        importer = ObsidianImporter()
        with pytest.raises(FileNotFoundError, match="Memories"):
            importer.import_(tmp_path)

    def test_skips_empty_notes(self, tmp_path: Path) -> None:
        exporter = ObsidianExporter()
        exporter.export(_MEMORIES, tmp_path)
        # Write an empty note
        empty = tmp_path / "Memories" / "semantic" / "empty.md"
        empty.write_text("---\nmemory_type: semantic\n---\n\n")
        importer = ObsidianImporter()
        pairs = importer.import_(tmp_path)
        # Still only 2 real memories (the empty one is skipped)
        assert len(pairs) == 2

    def test_hand_written_note_without_frontmatter(self, tmp_path: Path) -> None:
        """Notes with no frontmatter are imported with empty metadata."""
        exporter = ObsidianExporter()
        exporter.export([], tmp_path)  # creates Memories/ dir + index
        hand = tmp_path / "Memories" / "general" / "hand.md"
        hand.parent.mkdir(parents=True, exist_ok=True)
        hand.write_text("This is a manually written note about Redis caching.\n")
        importer = ObsidianImporter()
        pairs = importer.import_(tmp_path)
        assert len(pairs) == 1
        content, meta = pairs[0]
        assert "Redis" in content
        assert meta["source"] == "obsidian-import"
