"""Tests for myelin.integrations.sync.SyncRegistry."""

from __future__ import annotations

from pathlib import Path

from myelin.integrations.sync import SyncRegistry, _hash_file, _hash_memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry(tmp_path: Path) -> SyncRegistry:
    return SyncRegistry(tmp_path / "sync.db")


def _mem(id: str, content: str = "some content") -> dict:
    return {"id": id, "content": content}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# _hash_memory / _hash_file
# ---------------------------------------------------------------------------


def test_hash_memory_consistent():
    m = _mem("abc", "hello")
    assert _hash_memory(m) == _hash_memory(m)


def test_hash_memory_changes_on_content_change():
    m1 = _mem("abc", "hello")
    m2 = _mem("abc", "goodbye")
    assert _hash_memory(m1) != _hash_memory(m2)


def test_hash_memory_changes_on_id_change():
    m1 = _mem("abc", "hello")
    m2 = _mem("xyz", "hello")
    assert _hash_memory(m1) != _hash_memory(m2)


def test_hash_memory_length():
    assert len(_hash_memory(_mem("a", "b"))) == 16


def test_hash_file_consistent(tmp_path: Path):
    f = tmp_path / "file.md"
    f.write_bytes(b"hello world")
    assert _hash_file(f) == _hash_file(f)


def test_hash_file_changes_on_content_change(tmp_path: Path):
    f = tmp_path / "file.md"
    f.write_bytes(b"original")
    h1 = _hash_file(f)
    f.write_bytes(b"changed")
    h2 = _hash_file(f)
    assert h1 != h2


def test_hash_file_length(tmp_path: Path):
    f = tmp_path / "file.md"
    f.write_bytes(b"x")
    assert len(_hash_file(f)) == 16


# ---------------------------------------------------------------------------
# SyncRegistry — export side
# ---------------------------------------------------------------------------


def test_filter_for_export_all_new(tmp_path: Path):
    reg = _registry(tmp_path)
    dest = tmp_path / "vault"
    mems = [_mem("1"), _mem("2"), _mem("3")]
    result = reg.filter_for_export("obsidian", dest, mems)
    assert result == mems


def test_filter_for_export_none_after_record(tmp_path: Path):
    reg = _registry(tmp_path)
    dest = tmp_path / "vault"
    mems = [_mem("1"), _mem("2")]
    reg.record_exports("obsidian", dest, mems)
    result = reg.filter_for_export("obsidian", dest, mems)
    assert result == []


def test_filter_for_export_changed_memory(tmp_path: Path):
    reg = _registry(tmp_path)
    dest = tmp_path / "vault"
    original = _mem("1", "original content")
    reg.record_exports("obsidian", dest, [original])

    updated = _mem("1", "updated content")
    result = reg.filter_for_export("obsidian", dest, [updated])
    assert result == [updated]


def test_filter_for_export_unchanged_returns_subset(tmp_path: Path):
    reg = _registry(tmp_path)
    dest = tmp_path / "vault"
    m1 = _mem("1", "content a")
    m2 = _mem("2", "content b")
    reg.record_exports("obsidian", dest, [m1, m2])

    m2_updated = _mem("2", "content b changed")
    result = reg.filter_for_export("obsidian", dest, [m1, m2_updated])
    assert result == [m2_updated]


def test_record_exports_upserts(tmp_path: Path):
    reg = _registry(tmp_path)
    dest = tmp_path / "vault"
    m = _mem("1", "v1")
    reg.record_exports("obsidian", dest, [m])
    m2 = _mem("1", "v2")
    reg.record_exports("obsidian", dest, [m2])  # should not error (upsert)
    result = reg.filter_for_export("obsidian", dest, [m2])
    assert result == []


def test_remove_export_records(tmp_path: Path):
    reg = _registry(tmp_path)
    dest = tmp_path / "vault"
    mems = [_mem("1"), _mem("2")]
    reg.record_exports("obsidian", dest, mems)
    reg.remove_export_records("obsidian", dest, ["1"])
    result = reg.filter_for_export("obsidian", dest, mems)
    # Only m1 is new again (was removed); m2 is still known
    assert len(result) == 1
    assert result[0]["id"] == "1"


def test_export_summary_empty(tmp_path: Path):
    reg = _registry(tmp_path)
    dest = tmp_path / "vault"
    summary = reg.export_summary("obsidian", dest)
    assert summary["total_tracked"] == 0
    assert summary["last_exported"] is None


def test_export_summary_after_records(tmp_path: Path):
    reg = _registry(tmp_path)
    dest = tmp_path / "vault"
    mems = [_mem("1"), _mem("2"), _mem("3")]
    reg.record_exports("obsidian", dest, mems)
    summary = reg.export_summary("obsidian", dest)
    assert summary["total_tracked"] == 3
    assert summary["last_exported"] is not None


def test_export_isolated_by_integration(tmp_path: Path):
    reg = _registry(tmp_path)
    dest = tmp_path / "vault"
    mems = [_mem("1")]
    reg.record_exports("obsidian", dest, mems)
    # Different integration should not see the records
    result = reg.filter_for_export("export-md", dest, mems)
    assert result == mems


def test_export_isolated_by_dest(tmp_path: Path):
    reg = _registry(tmp_path)
    dest1 = tmp_path / "vault1"
    dest2 = tmp_path / "vault2"
    mems = [_mem("1")]
    reg.record_exports("obsidian", dest1, mems)
    result = reg.filter_for_export("obsidian", dest2, mems)
    assert result == mems


# ---------------------------------------------------------------------------
# SyncRegistry — import side
# ---------------------------------------------------------------------------


def test_filter_for_import_all_new(tmp_path: Path):
    reg = _registry(tmp_path)
    src = tmp_path / "vault"
    files_dir = src / "Memories"
    files_dir.mkdir(parents=True)
    f1 = files_dir / "a.md"
    f2 = files_dir / "b.md"
    f1.write_text("content a")
    f2.write_text("content b")
    result = reg.filter_for_import("obsidian", src, [f1, f2])
    assert set(result) == {f1, f2}


def test_filter_for_import_none_after_record(tmp_path: Path):
    reg = _registry(tmp_path)
    src = tmp_path / "vault"
    fd = src / "Memories"
    fd.mkdir(parents=True)
    f = fd / "a.md"
    f.write_text("content a")
    reg.record_imports("obsidian", src, [f])
    result = reg.filter_for_import("obsidian", src, [f])
    assert result == []


def test_filter_for_import_changed_file(tmp_path: Path):
    reg = _registry(tmp_path)
    src = tmp_path / "vault"
    fd = src / "Memories"
    fd.mkdir(parents=True)
    f = fd / "a.md"
    f.write_text("original")
    reg.record_imports("obsidian", src, [f])
    f.write_text("changed")
    result = reg.filter_for_import("obsidian", src, [f])
    assert result == [f]


def test_filter_for_import_unchanged_returns_subset(tmp_path: Path):
    reg = _registry(tmp_path)
    src = tmp_path / "vault"
    fd = src / "Memories"
    fd.mkdir(parents=True)
    f1 = fd / "a.md"
    f2 = fd / "b.md"
    f1.write_text("a")
    f2.write_text("b")
    reg.record_imports("obsidian", src, [f1, f2])
    f2.write_text("b changed")
    result = reg.filter_for_import("obsidian", src, [f1, f2])
    assert result == [f2]


def test_record_imports_upserts(tmp_path: Path):
    reg = _registry(tmp_path)
    src = tmp_path / "vault"
    fd = src / "Memories"
    fd.mkdir(parents=True)
    f = fd / "a.md"
    f.write_text("v1")
    reg.record_imports("obsidian", src, [f])
    f.write_text("v2")
    reg.record_imports("obsidian", src, [f])  # should not error (upsert)
    result = reg.filter_for_import("obsidian", src, [f])
    assert result == []


def test_import_summary_empty(tmp_path: Path):
    reg = _registry(tmp_path)
    src = tmp_path / "vault"
    summary = reg.import_summary("obsidian", src)
    assert summary["total_tracked"] == 0
    assert summary["last_imported"] is None


def test_import_summary_after_records(tmp_path: Path):
    reg = _registry(tmp_path)
    src = tmp_path / "vault"
    fd = src / "Memories"
    fd.mkdir(parents=True)
    f1 = fd / "a.md"
    f2 = fd / "b.md"
    f1.write_text("a")
    f2.write_text("b")
    reg.record_imports("obsidian", src, [f1, f2])
    summary = reg.import_summary("obsidian", src)
    assert summary["total_tracked"] == 2
    assert summary["last_imported"] is not None


def test_import_isolated_by_integration(tmp_path: Path):
    reg = _registry(tmp_path)
    src = tmp_path / "vault"
    fd = src / "Memories"
    fd.mkdir(parents=True)
    f = fd / "a.md"
    f.write_text("x")
    reg.record_imports("obsidian", src, [f])
    result = reg.filter_for_import("import-md", src, [f])
    assert result == [f]


def test_import_isolated_by_src(tmp_path: Path):
    reg = _registry(tmp_path)
    src1 = tmp_path / "vault1"
    src2 = tmp_path / "vault2"
    fd1 = src1 / "Memories"
    fd1.mkdir(parents=True)
    f = fd1 / "a.md"
    f.write_text("x")
    reg.record_imports("obsidian", src1, [f])
    # src2 has no records, even if the path happens to be the same relative
    # (won't match as src2 resolves differently)
    fd2 = src2 / "Memories"
    fd2.mkdir(parents=True)
    f2 = fd2 / "a.md"
    f2.write_text("x")
    result = reg.filter_for_import("obsidian", src2, [f2])
    assert result == [f2]


# ---------------------------------------------------------------------------
# Schema persistence (separate connection)
# ---------------------------------------------------------------------------


def test_data_persists_across_instances(tmp_path: Path):
    dest = tmp_path / "vault"
    db = tmp_path / "sync.db"
    mems = [_mem("1"), _mem("2")]

    reg1 = SyncRegistry(db)
    reg1.record_exports("obsidian", dest, mems)

    reg2 = SyncRegistry(db)
    result = reg2.filter_for_export("obsidian", dest, mems)
    assert result == []
