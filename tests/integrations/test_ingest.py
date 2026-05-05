"""Tests for myelin.integrations.ingest — bulk memory loading from files."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

from myelin.integrations.ingest import (
    IngestResult,
    _parse_frontmatter,
    ingest_directory,
    ingest_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store_fn(accept: bool = True) -> MagicMock:
    """Return a mock store function that records calls."""
    fn = MagicMock()
    status = "stored" if accept else "rejected"
    fn.return_value = {
        "status": status,
        "id": "abc123" if accept else None,
        "reason": None if accept else "too short",
    }
    return fn


# ---------------------------------------------------------------------------
# IngestResult
# ---------------------------------------------------------------------------


class TestIngestResult:
    def test_total(self) -> None:
        r = IngestResult(stored=3, skipped=1, errors=["e1"])
        assert r.total == 5

    def test_merge(self) -> None:
        a = IngestResult(stored=2, skipped=1, errors=["e1"])
        b = IngestResult(stored=1, skipped=0, errors=["e2", "e3"])
        a.merge(b)
        assert a.stored == 3
        assert a.skipped == 1
        assert a.errors == ["e1", "e2", "e3"]


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_no_frontmatter(self) -> None:
        meta, body = _parse_frontmatter("Hello world")
        assert meta == {}
        assert body == "Hello world"

    def test_basic_frontmatter(self) -> None:
        text = textwrap.dedent("""\
            ---
            project: myproject
            scope: auth
            ---
            Body content here.
        """)
        meta, body = _parse_frontmatter(text)
        assert meta["project"] == "myproject"
        assert meta["scope"] == "auth"
        assert "Body content here." in body

    def test_unknown_keys_ignored(self) -> None:
        text = "---\nfoo: bar\nproject: p\n---\nbody"
        meta, _ = _parse_frontmatter(text)
        assert "foo" not in meta
        assert meta["project"] == "p"

    def test_all_metadata_keys(self) -> None:
        text = textwrap.dedent("""\
            ---
            project: proj
            scope: scp
            language: python
            memory_type: semantic
            source: test
            tags: a,b,c
            ---
            body
        """)
        meta, _ = _parse_frontmatter(text)
        assert meta == {
            "project": "proj",
            "scope": "scp",
            "language": "python",
            "memory_type": "semantic",
            "source": "test",
            "tags": "a,b,c",
        }

    def test_empty_body_after_frontmatter(self) -> None:
        text = "---\nproject: p\n---\n"
        meta, body = _parse_frontmatter(text)
        assert meta["project"] == "p"
        assert body == ""


# ---------------------------------------------------------------------------
# ingest_file — text / markdown
# ---------------------------------------------------------------------------


class TestIngestFileText:
    def test_stores_plain_text(self, tmp_path: Path) -> None:
        f = tmp_path / "note.txt"
        f.write_text("The auth service uses JWT tokens for all endpoints.")
        store = _make_store_fn()
        result = ingest_file(f, store_fn=store)
        assert result.stored == 1
        assert result.skipped == 0
        store.assert_called_once()

    def test_stores_markdown_no_frontmatter(self, tmp_path: Path) -> None:
        f = tmp_path / "note.md"
        f.write_text("# Architecture\n\nWe use PostgreSQL backed by pgvector.")
        store = _make_store_fn()
        result = ingest_file(f, store_fn=store)
        assert result.stored == 1

    def test_frontmatter_overrides_defaults(self, tmp_path: Path) -> None:
        f = tmp_path / "note.md"
        f.write_text(
            "---\nproject: myproj\nscope: db\n---\nPostgres is our primary store."
        )
        store = _make_store_fn()
        ingest_file(f, store_fn=store, default_project="other", default_scope="other")
        assert store.call_args[1]["project"] == "myproj"
        assert store.call_args[1]["scope"] == "db"

    def test_defaults_used_when_no_frontmatter(self, tmp_path: Path) -> None:
        f = tmp_path / "note.txt"
        f.write_text("Some content worth remembering for the project.")
        store = _make_store_fn()
        ingest_file(f, store_fn=store, default_project="proj", default_scope="scp")
        assert store.call_args[1]["project"] == "proj"
        assert store.call_args[1]["scope"] == "scp"

    def test_empty_file_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("   \n  ")
        store = _make_store_fn()
        result = ingest_file(f, store_fn=store)
        assert result.skipped == 1
        store.assert_not_called()

    def test_rejected_by_store_counts_as_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "note.txt"
        f.write_text("hi")
        store = _make_store_fn(accept=False)
        result = ingest_file(f, store_fn=store)
        assert result.stored == 0
        assert result.skipped == 1

    def test_unsupported_extension_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "note.pdf"
        f.write_text("content")
        store = _make_store_fn()
        result = ingest_file(f, store_fn=store)
        assert result.skipped == 1
        store.assert_not_called()


# ---------------------------------------------------------------------------
# ingest_file — JSON
# ---------------------------------------------------------------------------


class TestIngestFileJson:
    def test_ingests_json_array(self, tmp_path: Path) -> None:
        data = [
            {"content": "We use Postgres", "project": "backend"},
            {"content": "Auth via JWT tokens"},
        ]
        f = tmp_path / "memories.json"
        f.write_text(json.dumps(data))
        store = _make_store_fn()
        result = ingest_file(f, store_fn=store)
        assert result.stored == 2
        assert store.call_count == 2

    def test_skips_items_without_content(self, tmp_path: Path) -> None:
        data = [{"content": ""}, {"content": "Valid memory content here."}]
        f = tmp_path / "m.json"
        f.write_text(json.dumps(data))
        store = _make_store_fn()
        result = ingest_file(f, store_fn=store)
        assert result.stored == 1
        assert result.skipped == 1

    def test_invalid_json_logged_as_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json at all {{{{")
        store = _make_store_fn()
        result = ingest_file(f, store_fn=store)
        assert len(result.errors) == 1
        store.assert_not_called()

    def test_non_array_json_logged_as_error(self, tmp_path: Path) -> None:
        f = tmp_path / "obj.json"
        f.write_text('{"content": "single object not array"}')
        store = _make_store_fn()
        result = ingest_file(f, store_fn=store)
        assert len(result.errors) == 1

    def test_tags_list_joined(self, tmp_path: Path) -> None:
        data = [{"content": "Auth content is meaningful here.", "tags": ["a", "b"]}]
        f = tmp_path / "m.json"
        f.write_text(json.dumps(data))
        store = _make_store_fn()
        ingest_file(f, store_fn=store)
        assert store.call_args[1]["tags"] == "a,b"

    def test_default_source_applied(self, tmp_path: Path) -> None:
        data = [{"content": "JWT is used for authentication in this system."}]
        f = tmp_path / "m.json"
        f.write_text(json.dumps(data))
        store = _make_store_fn()
        ingest_file(f, store_fn=store, default_source="obsidian")
        assert store.call_args[1]["source"] == "obsidian"


# ---------------------------------------------------------------------------
# ingest_directory
# ---------------------------------------------------------------------------


class TestIngestDirectory:
    def test_ingests_mixed_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("Markdown content about authentication system.")
        (tmp_path / "b.txt").write_text("Plain text content about the database layer.")
        (tmp_path / "c.json").write_text(
            '[{"content": "JSON memory for the project infrastructure."}]'
        )
        (tmp_path / "d.pdf").write_text("ignored")
        store = _make_store_fn()
        result = ingest_directory(tmp_path, store_fn=store)
        # .pdf is silently skipped at directory level (not a supported extension)
        assert result.stored == 3
        assert store.call_count == 3

    def test_skips_hidden_directories(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".git"
        hidden.mkdir()
        (hidden / "note.md").write_text("Should be skipped by ingest tool.")
        (tmp_path / "visible.md").write_text("Visible markdown content for ingestion.")
        store = _make_store_fn()
        result = ingest_directory(tmp_path, store_fn=store)
        assert result.stored == 1

    def test_non_recursive(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.md").write_text("Deep content that should not be ingested here.")
        (tmp_path / "top.md").write_text("Top level content for the memory system.")
        store = _make_store_fn()
        result = ingest_directory(tmp_path, store_fn=store, recursive=False)
        assert result.stored == 1

    def test_non_existent_directory(self) -> None:
        store = _make_store_fn()
        result = ingest_directory(Path("/no/such/dir"), store_fn=store)
        assert len(result.errors) == 1
        store.assert_not_called()
