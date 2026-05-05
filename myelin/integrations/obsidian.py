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

from .base import Exporter, Importer
from ..store.consolidation import extract_entities

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
        skip_ids: frozenset[str] = frozenset(opts.get("skip_ids", ()))

        dest.mkdir(parents=True, exist_ok=True)
        memories_root = dest / "Memories"
        memories_root.mkdir(exist_ok=True)

        # Build lookup structures for index pages
        by_type: dict[str, list[dict[str, Any]]] = {}
        by_project: dict[str, list[dict[str, Any]]] = {}
        by_scope: dict[str, list[dict[str, Any]]] = {}

        for m in memories:
            mtype = str(m.get("memory_type") or "general")
            project = str(m.get("project") or "")
            scope = str(m.get("scope") or "")
            by_type.setdefault(mtype, []).append(m)
            if project:
                by_project.setdefault(project, []).append(m)
            if scope:
                by_scope.setdefault(scope, []).append(m)

        written = 0
        # note_refs[id] = (rel_link, title) for use in index pages
        note_refs: dict[str, tuple[str, str]] = {}

        for mtype in sorted(by_type):
            mems = by_type[mtype]
            type_dir = memories_root / mtype
            type_dir.mkdir(exist_ok=True)

            for m in mems:
                mid: str = str(m.get("id", ""))
                content: str = str(m.get("content", ""))
                if not content:
                    continue

                title = _first_line(content)
                slug = _slugify(title)
                fname = f"{mid[:8]}-{slug}.md" if slug else f"{mid[:8]}.md"

                rel_link = f"Memories/{mtype}/{fname[:-3]}"
                note_refs[mid] = (rel_link, title)  # always register for index

                if mid in skip_ids:
                    continue  # file already up to date on disk

                # --- YAML frontmatter ---
                project = str(m.get("project") or "")
                scope = str(m.get("scope") or "")
                source = str(m.get("source") or "")
                created_at = str(m.get("created_at") or "")
                last_accessed = str(m.get("last_accessed") or "")
                access_count = m.get("access_count", 0)
                raw_tags = str(m.get("tags") or "")

                tags = ["myelin", f"type/{mtype}"]
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
                    fm.append(f"project: {_yaml_str(project)}")
                if scope:
                    fm.append(f"scope: {_yaml_str(scope)}")
                fm.append(f"memory_type: {mtype}")
                if source:
                    fm.append(f"source: {source}")
                if created_at:
                    fm.append(f"created_at: {_yaml_str(created_at)}")
                if last_accessed:
                    fm.append(f"last_accessed: {_yaml_str(last_accessed)}")
                if access_count:
                    fm.append(f"access_count: {access_count}")
                fm.append("---")

                # --- Body ---
                body_parts = [content]

                # Context callout (project + scope header)
                if project or scope:
                    ctx_parts = []
                    if project:
                        ctx_parts.append(f"**Project:** {project}")
                    if scope:
                        ctx_parts.append(f"**Scope:** {scope}")
                    if raw_tags:
                        ctx_parts.append(f"**Tags:** {raw_tags}")
                    body_parts.append(
                        "\n> [!info] Context\n> " + "  \n> ".join(ctx_parts)
                    )

                if include_entity_links:
                    entities = extract_entities(content)
                    if entities:
                        body_parts.append("\n## Related\n")
                        body_parts.append(
                            " ".join(f"[[{e}]]" for e in sorted(set(entities)))
                        )

                note = "\n".join(fm) + "\n\n" + "\n".join(body_parts) + "\n"
                (type_dir / fname).write_text(note, encoding="utf-8")
                written += 1

        # --- Project index pages ---
        if by_project:
            proj_dir = dest / "Projects"
            proj_dir.mkdir(exist_ok=True)
            for proj_name in sorted(by_project):
                proj_mems = by_project[proj_name]
                # Group by scope within project
                by_scope_within: dict[str, list[dict[str, Any]]] = {}
                for m in proj_mems:
                    sc = str(m.get("scope") or "_general")
                    by_scope_within.setdefault(sc, []).append(m)

                # Collect memory type distribution
                type_counts: dict[str, int] = {}
                for m in proj_mems:
                    mt = str(m.get("memory_type") or "general")
                    type_counts[mt] = type_counts.get(mt, 0) + 1
                type_summary = ", ".join(
                    f"{cnt} {mt}" for mt, cnt in sorted(type_counts.items())
                )

                lines = [
                    "---",
                    f"project: {_yaml_str(proj_name)}",
                    "tags:",
                    "  - myelin",
                    f"  - project/{_slugify(proj_name)}",
                    "---",
                    "",
                    f"# {proj_name}",
                    "",
                    f"> {len(proj_mems)} memories — {type_summary}",
                    "",
                ]
                for sc in sorted(by_scope_within):
                    display_sc = sc if sc != "_general" else "General"
                    lines.append(f"\n## {display_sc}\n")
                    for m in by_scope_within[sc]:
                        mid = str(m.get("id", ""))
                        ref = note_refs.get(mid)
                        if ref:
                            rel_link, title = ref
                            mt = str(m.get("memory_type") or "")
                            lines.append(
                                f"- [[{rel_link}|{title[:60]}]]"
                                + (f" `{mt}`" if mt else "")
                            )

                (proj_dir / f"{_slugify(proj_name)}.md").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )

        # --- Scope index pages ---
        if by_scope:
            scope_dir = dest / "Scopes"
            scope_dir.mkdir(exist_ok=True)
            for scope_name in sorted(by_scope):
                scope_mems = by_scope[scope_name]
                # Group by project within scope
                by_proj_within: dict[str, list[dict[str, Any]]] = {}
                for m in scope_mems:
                    pj = str(m.get("project") or "_none")
                    by_proj_within.setdefault(pj, []).append(m)

                lines = [
                    "---",
                    f"scope: {_yaml_str(scope_name)}",
                    "tags:",
                    "  - myelin",
                    f"  - scope/{_slugify(scope_name)}",
                    "---",
                    "",
                    f"# Scope: {scope_name}",
                    "",
                    f"> {len(scope_mems)} memories",
                    "",
                ]
                for pj in sorted(by_proj_within):
                    display_pj = pj if pj != "_none" else "No project"
                    lines.append(f"\n## {display_pj}\n")
                    for m in by_proj_within[pj]:
                        mid = str(m.get("id", ""))
                        ref = note_refs.get(mid)
                        if ref:
                            rel_link, title = ref
                            lines.append(f"- [[{rel_link}|{title[:60]}]]")

                (scope_dir / f"{_slugify(scope_name)}.md").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )

        # --- Memory Index (project-grouped) ---
        index_lines = [
            "---",
            "tags:",
            "  - myelin",
            "  - index",
            "---",
            "",
            "# Myelin Memory Index",
            "",
            f"Exported **{written}** memories across **{len(by_project)}** projects "
            f"and **{len(by_type)}** memory types.",
            "",
        ]

        # Stats table
        if by_type:
            index_lines += ["## By Type", "", "| Type | Count |", "| --- | --- |"]
            for mt in sorted(by_type):
                index_lines.append(f"| {mt} | {len(by_type[mt])} |")
            index_lines.append("")

        if by_project:
            index_lines += ["## By Project", ""]
            for proj_name in sorted(by_project):
                proj_slug = _slugify(proj_name)
                index_lines.append(
                    f"- [[Projects/{proj_slug}|{proj_name}]] "
                    f"({len(by_project[proj_name])} memories)"
                )
            index_lines.append("")

        if by_scope:
            index_lines += ["## By Scope", ""]
            for sc in sorted(by_scope):
                sc_slug = _slugify(sc)
                index_lines.append(
                    f"- [[Scopes/{sc_slug}|{sc}]] ({len(by_scope[sc])} memories)"
                )
            index_lines.append("")

        # Full listing grouped by project → memory_type
        index_lines += ["## All Memories", ""]
        if by_project:
            for proj_name in sorted(by_project):
                index_lines.append(f"\n### {proj_name}\n")
                proj_by_type: dict[str, list[dict[str, Any]]] = {}
                for m in by_project[proj_name]:
                    mt = str(m.get("memory_type") or "general")
                    proj_by_type.setdefault(mt, []).append(m)
                for mt in sorted(proj_by_type):
                    index_lines.append(f"\n#### {mt.title()}\n")
                    for m in proj_by_type[mt]:
                        mid = str(m.get("id", ""))
                        ref = note_refs.get(mid)
                        if ref:
                            rel_link, title = ref
                            sc = str(m.get("scope") or "")
                            index_lines.append(
                                f"- [[{rel_link}|{title[:60]}]]"
                                + (f" _{sc}_" if sc else "")
                            )
        else:
            # No projects — fall back to type grouping
            for mtype in sorted(by_type):
                index_lines.append(f"\n### {mtype.title()}\n")
                for m in by_type[mtype]:
                    mid = str(m.get("id", ""))
                    ref = note_refs.get(mid)
                    if ref:
                        rel_link, title = ref
                        index_lines.append(f"- [[{rel_link}|{title[:60]}]]")

        (dest / "Memory Index.md").write_text(
            "\n".join(index_lines) + "\n", encoding="utf-8"
        )

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
        only_files: frozenset[Path] | None = (
            frozenset(opts["only_files"]) if "only_files" in opts else None
        )

        memories_root = src / "Memories"
        if not memories_root.exists():
            raise FileNotFoundError(f"No 'Memories/' directory found in vault: {src}")

        results: list[tuple[str, dict[str, str]]] = []
        for md_file in sorted(memories_root.rglob("*.md")):
            if only_files is not None and md_file not in only_files:
                continue  # incremental: skip unchanged files
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
