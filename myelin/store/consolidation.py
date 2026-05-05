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
using spaCy NER (with regex fallback when spaCy is unavailable), build
co-occurrence edges in the semantic network (neocortex), and optionally
promote high-confidence facts from episodic → semantic memory type.

Entity typing strategy (hybrid):
- spaCy NER handles PERSON, ORG, GPE/LOC, PRODUCT reliably on prose
- Regex supplements with CamelCase/UPPER_CASE tech identifiers and
  quoted terms that NER models systematically miss on technical text
- Both pipelines are batched for performance; spaCy processes all
  memory texts in a single ``nlp.pipe()`` call

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

from .entorhinal import _STOP_WORDS as _STOPWORDS
from .neocortex import SemanticNetwork

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy — optional; loaded lazily as a module-level singleton
# ---------------------------------------------------------------------------

_NLP: Any = None
_SPACY_AVAILABLE = False
try:
    import spacy as _spacy  # type: ignore

    _NLP = _spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    _SPACY_AVAILABLE = True
    logger.debug("spaCy NER loaded (en_core_web_sm)")
except Exception:
    logger.debug("spaCy not available — using regex entity extraction")

# spaCy label → our entity_type (unmapped labels fall through to regex)
_SPACY_TYPE_MAP: dict[str, str] = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "PRODUCT": "technology",
    "LANGUAGE": "technology",
    "EVENT": "event",
    "NORP": "organization",
}

# ---------------------------------------------------------------------------
# Regex patterns — used as supplement to spaCy, or sole extractor as fallback
# ---------------------------------------------------------------------------

# Capitalized multi-word names (e.g., "Kai Tanaka", "Project Alpha")
_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# CamelCase identifiers (e.g., "FastMCP", "ChromaDB", "PostgreSQL")
_CAMEL_PATTERN = re.compile(r"\b([A-Z][a-z]+[A-Z]\w*)\b")

# ALL_CAPS acronyms (e.g., "JWT", "OAuth", "CDC")
_UPPER_PATTERN = re.compile(r"\b([A-Z]{2,}(?:_[A-Z]{2,})*)\b")

# Simple quoted terms — "JWT", "OAuth 2.0"
_QUOTED_PATTERN = re.compile(r'"([^"]{2,40})"')


def _regex_typed_entities(text: str) -> list[tuple[str, str]]:
    """Extract entities using regex patterns with inferred types."""
    results: dict[str, str] = {}

    for m in _NAME_PATTERN.finditer(text):
        candidate = m.group(1)
        words = candidate.split()
        if not any(w.lower() in _STOPWORDS for w in words):
            results[candidate.lower()] = "auto"  # could be person or org

    for m in _CAMEL_PATTERN.finditer(text):
        candidate = m.group(1)
        if candidate.lower() not in _STOPWORDS and len(candidate) >= 3:
            results.setdefault(candidate.lower(), "technology")

    for m in _UPPER_PATTERN.finditer(text):
        candidate = m.group(1)
        if candidate.lower() not in _STOPWORDS and len(candidate) >= 2:
            results.setdefault(candidate.lower(), "technology")

    for m in _QUOTED_PATTERN.finditer(text):
        candidate = m.group(1).strip()
        if len(candidate) >= 2:
            results.setdefault(candidate.lower(), "concept")

    return list(results.items())


def extract_entities_typed(
    texts: list[str],
) -> list[list[tuple[str, str]]]:
    """Extract (name, entity_type) pairs from a batch of texts.

    Uses spaCy NER when available, supplemented by regex for technical
    identifiers that NER models miss. Falls back to pure regex if spaCy
    is not installed.

    Returns one list of ``(name, type)`` tuples per input text, in the
    same order as *texts*.
    """
    per_text: list[list[tuple[str, str]]] = [[] for _ in texts]

    if _SPACY_AVAILABLE and _NLP is not None:
        for i, doc in enumerate(_NLP.pipe(texts, batch_size=32)):
            typed: dict[str, str] = {}

            # 1. spaCy NER — high confidence for PERSON/ORG/GPE
            for ent in doc.ents:
                etype = _SPACY_TYPE_MAP.get(ent.label_)
                if etype is not None:
                    name = ent.text.lower().strip()
                    if name and name not in _STOPWORDS:
                        typed[name] = etype

            # 2. Regex supplement — catches CamelCase/UPPER tech identifiers
            #    that spaCy frequently misclassifies or ignores
            for name, etype in _regex_typed_entities(texts[i]):
                typed.setdefault(name, etype)  # NER wins on conflict

            per_text[i] = list(typed.items())
    else:
        for i, text in enumerate(texts):
            per_text[i] = _regex_typed_entities(text)

    return per_text


def extract_entities(text: str) -> list[str]:
    """Extract candidate entity names from text (names only, no types).

    Backward-compatible wrapper around :func:`extract_entities_typed`.
    Prefer the typed variant for new callers.
    """
    return sorted({name for name, _ in _regex_typed_entities(text)})


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
    1. Extract typed entities using spaCy NER + regex supplement
    2. Register entities as typed nodes in the semantic network
    3. Create co-occurrence edges between entities in the same memory

    This is the hippocampus→neocortex transfer: raw episodes become
    structured associations in the semantic network.
    """
    result = ConsolidationResult()
    all_new_entities: set[str] = set()
    rel_count_before = network.relationship_count()

    # Filter to non-empty content upfront
    valid_memories = [m for m in memories if m.get("content")]
    texts = [m["content"] for m in valid_memories]

    if not texts:
        return result

    # Batch NER extraction — single nlp.pipe() call for all memories
    per_text_entities = extract_entities_typed(texts)

    for typed_entities in per_text_entities:
        if not typed_entities:
            continue

        result.memories_replayed += 1

        names = [name for name, _ in typed_entities]

        # Register entities with inferred types
        for name, etype in typed_entities:
            network.add_entity(name, entity_type=etype)
            all_new_entities.add(name)

        # Co-occurrence edges: every pair of entities in the same memory
        for a, b in combinations(names, 2):
            network.add_relationship(a, b, predicate="co_occurs", weight=1.0)

    result.entities_found = len(all_new_entities)
    result.relationships_created = network.relationship_count() - rel_count_before

    logger.info(
        "consolidation complete",
        extra={
            "memories_replayed": result.memories_replayed,
            "entities_found": result.entities_found,
            "relationships_created": result.relationships_created,
            "spacy_available": _SPACY_AVAILABLE,
        },
    )

    return result
