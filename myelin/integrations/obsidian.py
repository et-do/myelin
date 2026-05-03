"""Obsidian vault integration — export and import myelin memories.

Export layout
-------------
Given ``--vault /path/to/vault``, the exporter writes::

    vault/
    ├── Memory Index.md          # overview note linking all memories
    └── Memories/
        ├── episodic/
        │   └── <id8>-<slug>.md
        ├── semantic/
        │   └── <id8>-<slug>.md
        └── …

Each note contains YAML frontmatter (``id``, ``tags``, ``project``, …)
followed by the memory content and, when consolidation entities are
available, a ``## Related`` section with ``[[entity]]`` wikilinks.  These
wikilinks are what drive Obsidian's graph-view clustering.

Import layout
-------------
The importer scans ``vault/Memories/**/*.md`` (any depth), parses YAML
frontmatter, strips the ``## Related`` section, and returns
``(content, metadata)`` pairs ready for ``do_store``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..store.consolidation import extract_entities
from .base import Exporter, Importer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_SPACE_RE = re.compile(r"\s+")


def _slugify(text: str, max_len: int = 48) -> str:
    """Return a filesystem-safe slug from *text*."""
    slug = _UNSAFE_RE.sub("", text.strip())
    slug = _SPACE_RE.sub("-", slug).strip("-")
    return slug[:max_len]


def _first_line(content: str) -> str:
    """Return the first non-empty line of *content*."""
    for line in content.splitlines():
        line = line.strip()
        if line:
            return line
    return "Untitled"


def _yaml_str(value: str) -> str:
    """Quote *value* for YAML if it contains special characters."""
    if any(c in value for c in ('"', "'", ":", "#", "\n")):
        escaped = value.replace('"', "'")
        return f'"{escaped}"'
    return value


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


class ObsidianExporter(Exporter):
    """Export myelin memories to an Obsidian vault directory.

    Each memory becomes an individual Markdown note with YAML frontmatter
    and optional entity wikilinks.  An index note is also written at the
    vault root.
    """

    def export(
        self,
        memories: list[dict[str, Any]],
        dest: Path,
        **opts: Any,
    ) -> int:
        """Write *memories* into the Obsidian vault at *dest*.

        Parameters
        ----------
        memories:
            Merged memory dicts (``id``, ``content``, plus metadata).
        dest:
            Vault root directory.  Created if it does not exist.
        **opts:
            Reserved for future options (e.g. ``include_entity_links=False``).

        Returns
        -------
        int
            Number of notes written.
        """
        include_entity_links: bool = opts.get("include_entity_links", True)

        dest.mkdir(parents=True, exist_ok=True)
        memories_root = dest / "Memories"
        memories_root.mkdir(exist_ok=True)

        # Group by memory_type for subfolder organisation
        by_type: dict[str, list[dict[str, Any]]] = {}
        for m in memories:
            mtype = str(m.get("memory_type") or "general")
            by_type.setdefault(mtype, []).append(m)

        written = 0
        index_sections: list[str] = []

        for mtype in sorted(by_type):
            mems = by_type[mtype]
            type_dir = memories_root / mtype
            type_dir.mkdir(exist_ok=True)
            index_sections.append(f"\n## {mtype.title()}\n")

            for m in mems:
                mid: str = str(m.get("id", ""))
                content: str = str(m.get("content", ""))
                if not content:
                    continue

                title = _first_line(content)
                slug = _slugify(title)
                fname = f"{mid[:8]}-{slug}.md" if slug else f"{mid[:8]}.md"

                # --- YAML frontmatter ---
                tags = ["myelin", f"type/{mtype}"]
                project = str(m.get("project") or "")
                scope = str(m.get("scope") or "")
                source = str(m.get("source") or "")
                created_at = str(m.get("created_at") or "")
                raw_tags = str(m.get("tags") or "")

                if project:
                    tags.append(f"project/{project}")
                if scope:
                    tags.append(f"scope/{scope}")
                for t in raw_tags.split(","):
                    t = t.strip()
                    if t:
                        tags.append(t)

                fm: list[str] = ["---", f"id: {mid}", f"title: {_yaml_str(title)}"]
                fm.append("tags:")
                for tag in tags:
                    fm.append(f"  - {tag}")
                if project:
                    fm.append(f"project: {project}")
                if scope:
                    fm.append(f"scope: {scope}")
                fm.append(f"memory_type: {mtype}")
                if source:
                    fm.append(f"source: {source}")
                if created_at:
                    fm.append(f"created_at: {_yaml_str(created_at)}")
                fm.append("---")

                # --- Body ---
                body_parts = [content]
                if include_entity_links:
                    entities = extract_entities(content)
                    if entities:
                        body_parts.append("\n## Related\n")
                        body_parts.append(
                            " ".join(f"[[{e}]]" for e in sorted(set(entities)))
                        )

                note = "\n".join(fm) + "\n\n" + "\n".join(body_parts) + "\n"
                (type_dir / fname).write_text(note, encoding="utf-8")

                rel_link = f"Memories/{mtype}/{fname[:-3]}"
                index_sections.append(f"- [[{rel_link}|{title[:60]}]]")
                written += 1

        # --- Index note ---
        index_body = (
            "---\ntags:\n  - myelin\n  - index\n---\n\n"
            "# Myelin Memory Index\n\n"
            f"Exported {written} memories.\n" + "\n".join(index_sections) + "\n"
        )
        (dest / "Memory Index.md").write_text(index_body, encoding="utf-8")

        return written


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

_FM_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
_KV_RE = re.compile(r"^(\w+)\s*:\s*(.+)$", re.MULTILINE)
_META_FIELDS = {"project", "scope", "language", "memory_type", "source", "tags"}


class ObsidianImporter(Importer):
    """Import myelin memories from an Obsidian vault directory.

    Scans ``vault/Memories/**/*.md``, parses YAML frontmatter produced by
    :class:`ObsidianExporter`, strips the ``## Related`` entity-link section,
    and returns ``(content, metadata)`` pairs.

    Notes written by hand (not by the exporter) are also supported as long as
    they reside under ``vault/Memories/`` and optionally carry frontmatter
    with any of the recognised metadata fields.
    """

    def import_(
        self,
        src: Path,
        **opts: Any,
    ) -> list[tuple[str, dict[str, str]]]:
        """Import from the Obsidian vault at *src*.

        Parameters
        ----------
        src:
            Vault root directory.
        **opts:
            ``default_source`` (str, default ``"obsidian-import"``) — value
            used for the ``source`` metadata field when the note has none.

        Returns
        -------
        list[tuple[str, dict[str, str]]]
            ``(content, metadata_kwargs)`` pairs.
        """
        default_source: str = str(opts.get("default_source", "obsidian-import"))

        memories_root = src / "Memories"
        if not memories_root.exists():
            raise FileNotFoundError(f"No 'Memories/' directory found in vault: {src}")

        results: list[tuple[str, dict[str, str]]] = []
        for md_file in sorted(memories_root.rglob("*.md")):
            text = md_file.read_text(encoding="utf-8", errors="replace")
            meta: dict[str, str] = {}
            body = text

            m = _FM_RE.match(text)
            if m:
                for kv in _KV_RE.finditer(m.group(1)):
                    key = kv.group(1).strip().lower()
                    val = kv.group(2).strip().strip("\"'")
                    if key in _META_FIELDS:
                        meta[key] = val
                body = text[m.end() :].strip()

            # Strip the ## Related section — entity wikilinks aren't content
            related_idx = body.find("\n## Related")
            if related_idx != -1:
                body = body[:related_idx].strip()

            if not body:
                continue

            if "source" not in meta:
                meta["source"] = default_source

            results.append((body, meta))

        return results
