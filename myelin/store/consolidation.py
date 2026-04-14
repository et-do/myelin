"""Consolidation — offline hippocampal replay and episodic→semantic promotion.

Neuroscience basis: Complementary Learning Systems theory (McClelland,
McNaughton & O'Reilly, 1995). The hippocampus encodes episodes rapidly
(one-shot). During offline periods, the hippocampus "replays" stored
episodes to the neocortex, which extracts statistical regularities —
entities, co-occurrence patterns, recurring themes — and builds a
slow-learning semantic network.

This means:
- The **store path stays fast** — no entity extraction overhead at ingest
- The **consolidation path runs asynchronously** — like sleep replay
- Episodic memories that have been consolidated into the semantic
  network can be downscaled (accessed less, eventual decay)

Implementation: scan recent episodic memories, extract named entities
via lightweight regex (no LLM), build co-occurrence edges in the
semantic network (neocortex), and optionally promote high-confidence
facts from episodic → semantic memory type.

References:
    McClelland, J. L., McNaughton, B. L. & O'Reilly, R. C. (1995).
        Why there are complementary learning systems in the hippocampus
        and neocortex. Psychological Review, 102(3).
    Kumaran, D., Hassabis, D. & McClelland, J. L. (2016).
        What learning systems do intelligent agents need?
        Trends in Cognitive Sciences, 20(7).
    Rasch, B. & Born, J. (2013). About sleep's role in memory.
        Physiological Reviews, 93(2).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .neocortex import SemanticNetwork

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity extraction — lightweight regex, no LLM
# ---------------------------------------------------------------------------

# Capitalized multi-word names (e.g., "Kai Tanaka", "Project Alpha")
_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# Common tech/project identifiers (CamelCase or UPPER_CASE)
_TECH_PATTERN = re.compile(r"\b([A-Z][a-z]+[A-Z]\w+|[A-Z]{2,}(?:_[A-Z]{2,})*)\b")

# Simple quoted terms — "JWT", "OAuth 2.0"
_QUOTED_PATTERN = re.compile(r'"([^"]{2,40})"')

# Filter out common English words that match name patterns
_STOPWORDS = frozenset(
    {
        "the",
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "been",
        "were",
        "they",
        "their",
        "about",
        "would",
        "could",
        "should",
        "which",
        "there",
        "where",
        "after",
        "before",
        "since",
        "while",
        "other",
        "what",
        "when",
        "your",
        "some",
        "each",
        "also",
        "than",
        "more",
        "into",
        "over",
        "just",
        "like",
        "back",
        "only",
        "well",
        "then",
        "here",
        "sure",
        "yeah",
        "okay",
        "right",
        "great",
        "good",
        "nice",
        # Common start-of-sentence words
        "However",
        "Meanwhile",
        "Therefore",
        "Furthermore",
        "Additionally",
    }
)


def extract_entities(text: str) -> list[str]:
    """Extract candidate entity names from text.

    Returns lowercase, deduplicated entity strings.
    """
    entities: set[str] = set()

    for m in _NAME_PATTERN.finditer(text):
        candidate = m.group(1)
        words = candidate.split()
        if not any(w in _STOPWORDS for w in words):
            entities.add(candidate.lower())

    for m in _TECH_PATTERN.finditer(text):
        candidate = m.group(1)
        if candidate not in _STOPWORDS and len(candidate) >= 3:
            entities.add(candidate.lower())

    for m in _QUOTED_PATTERN.finditer(text):
        candidate = m.group(1).strip()
        if len(candidate) >= 2:
            entities.add(candidate.lower())

    return sorted(entities)


# ---------------------------------------------------------------------------
# Consolidation engine
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationResult:
    """Summary of a consolidation sweep."""

    memories_replayed: int = 0
    entities_found: int = 0
    relationships_created: int = 0


def replay(
    memories: list[dict[str, Any]],
    network: SemanticNetwork,
) -> ConsolidationResult:
    """Replay episodic memories into the semantic network.

    For each memory:
    1. Extract entities (names, tech terms, quoted terms)
    2. Register entities as nodes in the semantic network
    3. Create co-occurrence edges between entities found in the same memory

    This is the hippocampus→neocortex transfer: raw episodes become
    structured associations in the semantic network.
    """
    result = ConsolidationResult()
    all_new_entities: set[str] = set()
    rel_count_before = network.relationship_count()

    for mem in memories:
        content = mem.get("content", "")
        if not content:
            continue

        entities = extract_entities(content)
        if not entities:
            continue

        result.memories_replayed += 1

        # Register entities
        for entity in entities:
            network.add_entity(entity, entity_type="auto")
            all_new_entities.add(entity)

        # Co-occurrence edges: every pair of entities in the same memory
        for a, b in combinations(entities, 2):
            network.add_relationship(a, b, predicate="co_occurs", weight=1.0)

    result.entities_found = len(all_new_entities)
    result.relationships_created = network.relationship_count() - rel_count_before

    logger.info(
        "consolidation complete",
        extra={
            "memories_replayed": result.memories_replayed,
            "entities_found": result.entities_found,
            "relationships_created": result.relationships_created,
        },
    )

    return result
