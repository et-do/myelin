"""Abstract base classes for myelin integration exporters and importers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Exporter(ABC):
    """Write myelin memories to an external format or destination.

    Each integration that supports export should subclass this and implement
    :meth:`export`.  The integration is registered in
    :data:`myelin.integrations.EXPORTERS` so it can be discovered by the CLI.
    """

    @abstractmethod
    def export(
        self,
        memories: list[dict[str, Any]],
        dest: Path,
        **opts: Any,
    ) -> int:
        """Export *memories* to *dest*.

        Parameters
        ----------
        memories:
            List of merged memory dicts.  Each dict contains at minimum
            ``"id"`` and ``"content"`` keys, plus any stored metadata fields
            (``project``, ``scope``, ``memory_type``, ``tags``, ``source``,
            ``created_at``, …).
        dest:
            Destination path — a directory or file, depending on the
            integration.
        **opts:
            Integration-specific keyword options.

        Returns
        -------
        int
            Number of memories successfully written.
        """
        ...


class Importer(ABC):
    """Read memories from an external format or source.

    Each integration that supports import should subclass this and implement
    :meth:`import_`.  The integration is registered in
    :data:`myelin.integrations.IMPORTERS`.
    """

    @abstractmethod
    def import_(
        self,
        src: Path,
        **opts: Any,
    ) -> list[tuple[str, dict[str, str]]]:
        """Import memories from *src*.

        Parameters
        ----------
        src:
            Source path — a directory or file, depending on the integration.
        **opts:
            Integration-specific keyword options.

        Returns
        -------
        list[tuple[str, dict[str, str]]]
            List of ``(content, metadata_kwargs)`` pairs.  ``metadata_kwargs``
            is a flat dict of **string** values suitable for passing directly
            to ``do_store`` keyword arguments.
        """
        ...
