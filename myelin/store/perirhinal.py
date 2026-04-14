"""Perirhinal cortex — gist extraction for summary-guided retrieval.

Neuroscience basis: The perirhinal cortex (PRC) sits between the
entorhinal cortex and neocortex in the medial temporal lobe.  It creates
compressed "gist" representations of complex objects and scenes —
familiarity signals that guide whether full episodic retrieval is needed.

In Myelin, the perirhinal cortex generates short text summaries of
stored sessions/conversations.  During recall the query is matched
against summaries first (cheap, high-signal), then verbatim chunks
from matching sessions are retrieved (focused, precise).

This mirrors MemPalace's summary-index architecture:
    MemPalace summary → PRC gist
    MemPalace room ID → parent_id (shared by chunks from one session)

The two-stage recall flow:
    1. **Gist match** — embed query, search summary collection, get
       parent_ids of top-matching sessions.
    2. **Verbatim retrieval** — search the main chunk collection
       filtered to those parent_ids.

This narrows the search space *before* dense retrieval rather than
re-ranking *after* — the key architectural difference from flat search.

References:
    Murray, E. A. & Richmond, B. J. (2001). Role of perirhinal cortex
        in object perception, memory, and associations. Current Opinion
        in Neurobiology, 11(2).
    Bussey, T. J. & Saksida, L. M. (2007). Memory, perception, and the
        ventral visual-perirhinal-hippocampal stream. Hippocampus, 17(9).
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Extractive summarisation — no LLM, pure heuristic
# ---------------------------------------------------------------------------

_ROLE_RE = re.compile(
    r"^(user|assistant|human|ai|system)\s*:\s*",
    re.MULTILINE | re.IGNORECASE,
)

# Sentences that are likely high-signal (questions, decisions, facts)
_SIGNAL_RE = re.compile(
    r"[^.!?\n]*(?:"
    # Preferences and decisions
    r"\b(?:should|must|always|never|prefer|recommend|decided|configured"
    # Identity and contact
    r"|password|key|token|issue|problem|error|fix|solution|use|name"
    r"|address|email|phone|birthday|favorite|allergic|born|live|work"
    # State changes (tech)
    r"|created|set up|installed|moved|started|changed|updated|switched"
    # Life events and personal facts
    r"|married|divorced|engaged|pregnant|adopted|retired|promoted|hired"
    r"|fired|graduated|enrolled|traveled|visited|bought|sold|rented"
    r"|met|joined|quit|broke up|passed away|diagnosed|recovered|surgery"
    # Activities and hobbies
    r"|hobby|hobbies|enjoy|played|practice|train|running|cooking|paint"
    r"|volunteer|attend|trip|vacation|celebrate|concert|marathon|tattoo)"
    r")[^.!?\n]*[.!?]",
    re.IGNORECASE,
)

# Capitalized names / proper nouns — cheap entity detection for gist scoring
_ENTITY_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)\b")

# Common sentence starters that aren't entities
_ENTITY_STOP = frozenset(
    "The This That What When Where Which Who How And But Also Just Well "
    "After Before Since While Because Although However Let Yes Please".split()
)


def _count_entities(sentences: list[str]) -> dict[str, int]:
    """Count named entity occurrences across all sentences."""
    counts: dict[str, int] = {}
    for sent in sentences:
        for match in _ENTITY_RE.finditer(sent):
            name = match.group(1)
            if name.split()[0] not in _ENTITY_STOP:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _entity_signal(sent: str, entity_counts: dict[str, int]) -> float:
    """Score boost for sentences containing rare named entities.

    Entities appearing in only 1-2 sentences are likely specific facts
    worth preserving in the gist (a person, place, or event mentioned
    briefly).  Common entities (the speakers' names) appear everywhere
    and get no bonus.
    """
    bonus = 0.0
    for match in _ENTITY_RE.finditer(sent):
        name = match.group(1)
        if name.split()[0] in _ENTITY_STOP:
            continue
        count = entity_counts.get(name, 0)
        if 1 <= count <= 2:
            bonus += 0.4
    return min(bonus, 1.0)  # cap at 1.0


def summarise(text: str, max_sentences: int = 6) -> str:
    """Extract a short gist from text using heuristic sentence selection.

    Strategy:
    1. Strip role prefixes to get clean sentences.
    2. Score sentences by signal keywords (facts, decisions, preferences).
    3. Take top-N highest-signal sentences in original order.
    4. Fallback: first + last meaningful sentences if no signals found.

    Keeps summaries short (~200-400 chars) so embedding stays focused.
    """
    # Strip role prefixes
    clean = _ROLE_RE.sub("", text).strip()
    if not clean:
        return ""

    # Split into sentences
    sentences = _split_sentences(clean)
    if not sentences:
        return ""

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # Score by signal keywords
    scored: list[tuple[int, float, str]] = []
    # Pre-compute named entity frequency for rarity-based scoring
    entity_counts = _count_entities(sentences)
    for idx, sent in enumerate(sentences):
        hits = len(_SIGNAL_RE.findall(sent))
        # Bonus for early/late position (intro + conclusion)
        pos_bonus = 0.5 if idx < 2 or idx >= len(sentences) - 2 else 0.0
        # Bonus for length (longer = more informative, up to a point)
        len_bonus = min(len(sent) / 200, 0.5)
        # Bonus for sentences containing rare named entities (specific facts)
        entity_bonus = _entity_signal(sent, entity_counts)
        scored.append((idx, hits + pos_bonus + len_bonus + entity_bonus, sent))

    # Take top sentences by score, preserve original order
    scored.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted(s[0] for s in scored[:max_sentences])
    selected = [sentences[i] for i in top_indices]

    return " ".join(selected)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering noise."""
    # Split on sentence-ending punctuation or newlines
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences: list[str] = []
    for s in raw:
        s = s.strip()
        # Filter very short fragments and filler
        if len(s) < 15:
            continue
        # Filter pure greetings/filler
        if re.match(
            r"^(?:hi|hello|hey|thanks|thank you|sure|okay|ok|yes|no|"
            r"great|good|got it|sounds good|absolutely|of course)[.!?,\s]*$",
            s,
            re.IGNORECASE,
        ):
            continue
        sentences.append(s)
    return sentences


# ---------------------------------------------------------------------------
# Summary collection management
# ---------------------------------------------------------------------------


class SummaryIndex:
    """Manages a separate ChromaDB collection of session summaries.

    Each summary is keyed by parent_id (shared by all chunks from one
    store() call).  During recall, query the summary collection first
    to identify relevant parent_ids, then filter the main collection.
    """

    def __init__(
        self,
        client: Any,
        embedder: Any,
        prefix: str = "",
        collection_name: str = "",
    ) -> None:
        coll_name = collection_name or (f"{prefix}summaries" if prefix else "summaries")
        self._collection = client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder

    def add(
        self,
        parent_id: str,
        summary_text: str,
        metadata: dict[str, str | int | float] | None = None,
    ) -> None:
        """Store a summary for a parent_id."""
        if not summary_text.strip():
            return
        embedding = self._embedder.encode(summary_text).tolist()
        meta: dict[str, str | int | float] = {"parent_id": parent_id}
        if metadata:
            meta.update(metadata)
        self._collection.upsert(
            ids=[f"summary_{parent_id}"],
            embeddings=[embedding],
            documents=[summary_text],
            metadatas=[meta],
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 3,
    ) -> list[tuple[str, float]]:
        """Find parent_ids whose summaries best match the query.

        Returns list of (parent_id, score) tuples.
        """
        if self._collection.count() == 0:
            return []
        n = min(n_results, self._collection.count())
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["metadatas", "distances"],
        )
        _metas = results["metadatas"]
        _dists = results["distances"]
        assert _metas is not None
        assert _dists is not None

        out: list[tuple[str, float]] = []
        for meta, dist in zip(_metas[0], _dists[0]):
            pid = str(meta.get("parent_id", ""))
            if pid:
                out.append((pid, 1.0 - dist))
        return out

    def count(self) -> int:
        return int(self._collection.count())
