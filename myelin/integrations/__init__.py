"""Myelin integration adapters — export/import bridges to external tools.

Adding a new integration
------------------------
1. Create ``myelin/integrations/<name>.py`` with a class that subclasses
   :class:`Exporter` and/or :class:`Importer`.
2. Register the class in :data:`EXPORTERS` / :data:`IMPORTERS` below.
3. Add ``myelin <name>-export`` / ``myelin <name>-import`` subcommands in
   ``myelin/cli.py``.
4. Add tests under ``tests/integrations/``.
"""

from __future__ import annotations

from .base import Exporter, Importer
from .obsidian import ObsidianExporter, ObsidianImporter

__all__ = [
    "EXPORTERS",
    "IMPORTERS",
    "Exporter",
    "Importer",
    "ObsidianExporter",
    "ObsidianImporter",
]

#: Registry of available exporters keyed by integration name.
EXPORTERS: dict[str, type[Exporter]] = {
    "obsidian": ObsidianExporter,
}

#: Registry of available importers keyed by integration name.
IMPORTERS: dict[str, type[Importer]] = {
    "obsidian": ObsidianImporter,
}
