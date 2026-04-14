"""Synaptic decay — access-based TTL pruning.

A memory is pruned when EITHER condition is met:
  - days since last access > max_idle_days AND access_count < min_access_count
  - days since last access > max_idle_days_absolute
    (hard cap, regardless of access count)

The absolute threshold prevents "immortal" memories that were accessed a
few times years ago from accumulating forever.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..config import settings


def find_stale(
    metadata_list: list[dict[str, Any]],
    *,
    max_idle_days: int | None = None,
    min_access_count: int | None = None,
    max_idle_days_absolute: int | None = None,
) -> list[str]:
    """Return IDs of memories that should be pruned."""
    max_days = max_idle_days if max_idle_days is not None else settings.max_idle_days
    min_count = (
        min_access_count if min_access_count is not None else settings.min_access_count
    )
    abs_max = (
        max_idle_days_absolute
        if max_idle_days_absolute is not None
        else settings.max_idle_days_absolute
    )
    now = datetime.now(UTC)
    stale_ids: list[str] = []

    for entry in metadata_list:
        last = datetime.fromisoformat(entry["last_accessed"])
        idle_days = (now - last).days
        access_count = int(entry.get("access_count", 0))

        # Low-access memories decay at the normal threshold
        if idle_days > max_days and access_count < min_count:
            stale_ids.append(entry["id"])
        # Hard cap: even frequently accessed memories expire eventually
        elif idle_days > abs_max:
            stale_ids.append(entry["id"])

    return stale_ids
