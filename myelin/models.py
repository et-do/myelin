"""Data models for the Myelin memory system."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# Memory system types — mapped from neuroscience, not filing labels.
# episodic:    specific events with temporal context (hippocampus)
# semantic:    general knowledge, facts, decisions (temporal neocortex)
# procedural:  habits, preferences, know-how (basal ganglia)
# prospective: future-oriented plans, advice (prefrontal cortex)
MemoryType = Literal["episodic", "semantic", "procedural", "prospective"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MemoryMetadata(BaseModel):
    """Context-dependent encoding metadata.

    Attached at store time so retrieval can filter by context —
    like how the brain encodes situation alongside content.
    """

    project: str | None = None  # cortical region — domain territory
    language: str | None = None
    scope: str | None = None  # engram cluster — e.g. "auth", "database"
    memory_type: MemoryType | None = None  # memory system classification
    tags: list[str] = Field(default_factory=list)
    source: str | None = None  # e.g. "copilot", "cursor", "claude"
    parent_id: str | None = None  # group ID linking chunks from the same source

    def to_chroma(self) -> dict[str, str | int | float]:
        """Flatten to ChromaDB-compatible metadata (scalars only)."""
        d: dict[str, str | int | float] = {}
        if self.project:
            d["project"] = self.project
        if self.language:
            d["language"] = self.language
        if self.scope:
            d["scope"] = self.scope
        if self.memory_type:
            d["memory_type"] = self.memory_type
        if self.tags:
            d["tags"] = ",".join(self.tags)
        if self.source:
            d["source"] = self.source
        if self.parent_id:
            d["parent_id"] = self.parent_id
        return d

    @classmethod
    def from_chroma(cls, meta: dict[str, Any]) -> MemoryMetadata:
        """Reconstruct from ChromaDB metadata dict."""
        tags_raw = meta.get("tags", "")
        return cls(
            project=meta.get("project"),
            language=meta.get("language"),
            scope=meta.get("scope"),
            memory_type=meta.get("memory_type"),
            tags=tags_raw.split(",") if tags_raw else [],
            source=meta.get("source"),
            parent_id=meta.get("parent_id"),
        )


class Memory(BaseModel):
    """A single stored memory unit."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    content: str
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    created_at: datetime = Field(default_factory=_utcnow)
    last_accessed: datetime = Field(default_factory=_utcnow)
    access_count: int = 0
    # Set by store(overwrite=True) when an existing memory was replaced.
    # Not persisted to storage; only present on the return value of store().
    replaced_id: str | None = Field(default=None, exclude=True)


class RecallResult(BaseModel):
    """A memory returned from recall, ranked by relevance."""

    memory: Memory
    score: float
