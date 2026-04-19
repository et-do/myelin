"""Ingest — bulk-load content into memory from files and directories.

Supports three source formats:

- **Plain text / Markdown** (``.txt``, ``.md``): the file body becomes a
  single store call; frontmatter metadata (YAML between ``---`` delimiters)
  is parsed and mapped to project/scope/tags/source fields.

- **JSON export** (``.json``): expects a list of objects in the same shape
  produced by ``myelin export``.  Each object must have a ``"content"`` key;
  all other keys map to metadata fields.

- **Directory**: recurse into the tree and ingest every ``.txt``, ``.md``,
  and ``.json`` file found.  Hidden directories (names starting with ``.``)
  are skipped.

The ingest functions return an :class:`IngestResult` summary so callers can
report progress without coupling to the store layer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Summary of a bulk-ingest operation."""

    stored: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.stored + self.skipped + len(self.errors)

    def merge(self, other: IngestResult) -> None:
        """Merge another result into this one in-place (for directory walks)."""
        self.stored += other.stored
        self.skipped += other.skipped
        self.errors.extend(other.errors)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
_KV_RE = re.compile(r"^(\w+)\s*:\s*(.+)$", re.MULTILINE)

_METADATA_KEYS = {"project", "scope", "language", "memory_type", "source", "tags"}


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split YAML frontmatter from body.

    Returns ``(metadata_dict, body_text)``.  If no frontmatter block is
    found, metadata is empty and body is the original text unchanged.
    Only the subset of keys in ``_METADATA_KEYS`` is extracted; unknown
    keys are ignored so arbitrary frontmatter doesn't cause errors.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    meta: dict[str, str] = {}
    for kv in _KV_RE.finditer(m.group(1)):
        key = kv.group(1).strip().lower()
        val = kv.group(2).strip().strip('"').strip("'")
        if key in _METADATA_KEYS:
            meta[key] = val

    body = text[m.end() :]
    return meta, body


# ---------------------------------------------------------------------------
# Single-file ingest
# ---------------------------------------------------------------------------


def ingest_file(
    path: Path,
    *,
    store_fn: Any,
    default_project: str = "",
    default_scope: str = "",
    default_source: str = "ingest",
) -> IngestResult:
    """Ingest a single file into memory.

    Args:
        path: Path to the file to ingest (.txt, .md, or .json).
        store_fn: Callable matching the ``do_store`` signature.
        default_project: Project to use when the file has no frontmatter.
        default_scope: Scope to use when the file has no frontmatter.
        default_source: ``source`` field value for all stored memories.

    Returns:
        :class:`IngestResult` with counts for this file.
    """
    result = IngestResult()
    suffix = path.suffix.lower()

    try:
        if suffix == ".json":
            _ingest_json_file(
                path,
                store_fn=store_fn,
                default_project=default_project,
                default_scope=default_scope,
                default_source=default_source,
                result=result,
            )
        elif suffix in {".md", ".txt", ""}:
            _ingest_text_file(
                path,
                store_fn=store_fn,
                default_project=default_project,
                default_scope=default_scope,
                default_source=default_source,
                result=result,
            )
        else:
            result.skipped += 1
    except OSError as exc:
        result.errors.append(f"{path}: {exc}")

    return result


def _ingest_text_file(
    path: Path,
    *,
    store_fn: Any,
    default_project: str,
    default_scope: str,
    default_source: str,
    result: IngestResult,
) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta, body = _parse_frontmatter(text)
    body = body.strip()
    if not body:
        result.skipped += 1
        return

    r = store_fn(
        body,
        project=meta.get("project", default_project),
        scope=meta.get("scope", default_scope),
        language=meta.get("language", ""),
        memory_type=meta.get("memory_type", ""),
        tags=meta.get("tags", ""),
        source=meta.get("source", default_source),
    )
    if r.get("status") in {"stored", "updated"}:
        result.stored += 1
    else:
        result.skipped += 1


def _ingest_json_file(
    path: Path,
    *,
    store_fn: Any,
    default_project: str,
    default_scope: str,
    default_source: str,
    result: IngestResult,
) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        result.errors.append(f"{path}: {exc}")
        return

    if not isinstance(data, list):
        result.errors.append(f"{path}: expected JSON array, got {type(data).__name__}")
        return

    for item in data:
        if not isinstance(item, dict):
            result.skipped += 1
            continue
        content = item.get("content", "")
        if not content:
            result.skipped += 1
            continue
        tags_raw = item.get("tags", "")
        tags = tags_raw if isinstance(tags_raw, str) else ",".join(tags_raw)
        r = store_fn(
            content,
            project=item.get("project", default_project) or default_project,
            scope=item.get("scope", default_scope) or default_scope,
            language=item.get("language", ""),
            memory_type=item.get("memory_type", ""),
            tags=tags,
            source=item.get("source", default_source) or default_source,
        )
        if r.get("status") in {"stored", "updated"}:
            result.stored += 1
        else:
            result.skipped += 1


# ---------------------------------------------------------------------------
# Directory ingest
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {".md", ".txt", ".json", ""}


def ingest_directory(
    directory: Path,
    *,
    store_fn: Any,
    default_project: str = "",
    default_scope: str = "",
    default_source: str = "ingest",
    recursive: bool = True,
) -> IngestResult:
    """Ingest all supported files under a directory.

    Hidden directories (names starting with ``.``) are always skipped.

    Args:
        directory: Root directory to scan.
        store_fn: Callable matching the ``do_store`` signature.
        default_project: Project tag for files without frontmatter.
        default_scope: Scope tag for files without frontmatter.
        default_source: Source tag for all stored memories.
        recursive: Whether to descend into subdirectories.

    Returns:
        :class:`IngestResult` aggregated across all files.
    """
    result = IngestResult()
    if not directory.is_dir():
        result.errors.append(f"{directory}: not a directory")
        return result

    pattern = "**/*" if recursive else "*"
    for path in sorted(directory.glob(pattern)):
        # Skip hidden directories anywhere in the path
        if any(part.startswith(".") for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        file_result = ingest_file(
            path,
            store_fn=store_fn,
            default_project=default_project,
            default_scope=default_scope,
            default_source=default_source,
        )
        result.merge(file_result)

    return result
