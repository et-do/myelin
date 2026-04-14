"""Prefrontal cortex — schema-consistent encoding.

Neuroscience basis: The prefrontal cortex maintains abstract *schemas* —
templates for recurring information patterns (Tse et al., 2007; van Kesteren
et al., 2012). When new input arrives, the PFC checks whether it matches an
existing schema and tags the hippocampal trace accordingly. Content that
doesn't match any schema is stored as raw episodic memory.

This differs from MemPalace's `general_extractor.py` filing approach in one
key way: schemas can be *learned* from the corpus during consolidation, not
just hardcoded. The initial set below bootstraps the system; consolidation
can discover new patterns over time.

References:
    Tse, D. et al. (2007). Schemas and memory consolidation. Science, 316.
    van Kesteren, M. T. R. et al. (2012). How schema and novelty augment
        memory formation. Trends in Neurosciences, 35(4).
    Preston, A. R. & Eichenbaum, H. (2013). Interplay of hippocampus and
        prefrontal cortex in memory. Current Biology, 23(17).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from ..models import MemoryType

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

_SchemaID = Literal["decision", "preference", "procedure", "event", "plan"]


@dataclass(frozen=True, slots=True)
class _Schema:
    """A PFC schema template — pattern + classification output."""

    id: _SchemaID
    memory_type: MemoryType
    # Regex patterns that indicate this schema (case-insensitive).
    # Any match is sufficient.
    markers: tuple[re.Pattern[str], ...] = field(default_factory=tuple)


# Each schema's markers are compiled once at import time.
_SCHEMAS: tuple[_Schema, ...] = (
    _Schema(
        id="decision",
        memory_type="semantic",
        markers=tuple(
            re.compile(p, re.IGNORECASE)
            for p in (
                r"\b(?:we |I )?(?:decided|chose|picked"
                r"|selected|went with|switched to)\b",
                r"\b(?:the |our )?(?:decision|trade-?off|conclusion) (?:is|was)\b",
                r"\blet'?s (?:go with|use|stick with|switch to)\b",
                r"\bagreed (?:to|on|that)\b",
                r"\bafter (?:discussion|debate|considering)\b",
            )
        ),
    ),
    _Schema(
        id="preference",
        memory_type="procedural",
        markers=tuple(
            re.compile(p, re.IGNORECASE)
            for p in (
                r"\b(?:always|never|prefer|avoid|don'?t like"
                r"|hate|love)\b.*\b(?:use|do|write|format|style)\b",
                r"\b(?:I |we |team )?prefer(?:s)?\b",
                r"\b(?:convention|standard|style.?guide|best practice|rule of thumb)\b",
                r"\bshould (?:always|never)\b",
                r"\b(?:naming convention|coding style|formatting rule)\b",
            )
        ),
    ),
    _Schema(
        id="procedure",
        memory_type="procedural",
        markers=tuple(
            re.compile(p, re.IGNORECASE)
            for p in (
                r"\b(?:step \d|first,? |then,? |next,? |finally,? |how to)\b",
                r"\b(?:procedure|workflow|process|recipe|runbook|playbook)\b",
                r"\b(?:to (?:deploy|build|test|release|set ?up|configure|install))\b",
                r"\b(?:run |execute |invoke |call )\b.+\b(?:command|script|tool)\b",
            )
        ),
    ),
    _Schema(
        id="plan",
        memory_type="prospective",
        markers=tuple(
            re.compile(p, re.IGNORECASE)
            for p in (
                r"\b(?:TODO|FIXME|HACK|next ?steps?|action ?items?|follow[- ]?up)\b",
                r"\b(?:we |I )?(?:should|need to|plan to|will|going to|want to)\b",
                r"\b(?:roadmap|milestone|deadline|sprint|backlog|goal)\b",
                r"\b(?:schedule|timeline|by (?:end of|next) (?:week|month|quarter))\b",
            )
        ),
    ),
    _Schema(
        id="event",
        memory_type="episodic",
        markers=tuple(
            re.compile(p, re.IGNORECASE)
            for p in (
                r"\b(?:yesterday|today|last (?:week|month|time)|just now)\b",
                r"\b(?:we |I )?(?:debugged|fixed|broke|shipped"
                r"|deployed|launched|discovered|found|hit)\b",
                r"\b(?:incident|outage|bug|crash|error|failure|breakthrough)\b",
                r"\b(?:happened|occurred|noticed|realized|saw that)\b",
            )
        ),
    ),
)


# ---------------------------------------------------------------------------
# Schema matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SchemaMatch:
    """Result of schema classification."""

    schema_id: _SchemaID
    memory_type: MemoryType
    confidence: float  # 0.0-1.0, based on marker match density


def classify(text: str) -> SchemaMatch | None:
    """Match text against known PFC schemas.

    Returns the best-matching schema, or None if no schema exceeds the
    confidence threshold (content is stored as raw episodic).

    The confidence score is the fraction of a schema's markers that fire.
    A single marker match is enough to classify; confidence indicates
    how strongly the text fits the schema.
    """
    best: SchemaMatch | None = None

    for schema in _SCHEMAS:
        hits = sum(1 for m in schema.markers if m.search(text))
        if hits == 0:
            continue
        confidence = hits / len(schema.markers)
        if best is None or confidence > best.confidence:
            best = SchemaMatch(
                schema_id=schema.id,
                memory_type=schema.memory_type,
                confidence=confidence,
            )

    return best


def classify_memory_type(text: str) -> MemoryType:
    """Classify text into a memory type, defaulting to episodic."""
    match = classify(text)
    return match.memory_type if match else "episodic"
