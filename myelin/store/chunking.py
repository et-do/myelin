"""Chunking — split long content into embeddable segments.

Neuroscience analogy: pattern separation in the dentate gyrus.
Distinct, focused representations improve recall specificity
over averaged, diffuse ones.

Two strategies:
- Conversation-aware: split at exchange pairs (user + assistant turns)
  with topic-shift detection.  When keyword overlap between adjacent
  turns drops below a threshold, a new chunk is started — even if the
  combined size would fit.  This creates semantically coherent chunks
  rather than arbitrary size-based splits.
- Text fallback: overlapping segments at paragraph boundaries.

The embedding model (all-MiniLM-L6-v2) has a 256-token limit (~1000 chars).
Content beyond that gets truncated silently, destroying recall quality.
Chunking ensures every segment fits within the model's attention window.
"""

from __future__ import annotations

import re
from collections import Counter

_ROLE_PATTERN = re.compile(
    r"^(user|assistant|human|ai|system)\s*:",
    re.MULTILINE | re.IGNORECASE,
)

# Named speakers: "Caroline:", "Dr. Smith:", "Bob:", etc.
# Matches a capitalized word (optionally followed by more words) at line
# start.  Used as a fallback when _ROLE_PATTERN doesn't fire, so that
# real-world conversations with named participants get chunked properly.
_NAMED_SPEAKER_PATTERN = re.compile(
    r"^([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s*:",
    re.MULTILINE,
)

_DEFAULT_MAX_CHARS = 1000
_DEFAULT_OVERLAP_CHARS = 200
_TOPIC_SHIFT_THRESHOLD = 0.15  # Jaccard overlap below this → topic shift

# Lightweight keyword extraction for topic-shift detection
# Avoids importing entorhinal to keep the module self-contained
_CHUNK_STOP = frozenset(
    "a about above after again against all am an and any are as at be "
    "because been before being below between both but by can could did do "
    "does doing down during each few for from further get got had has have "
    "having he her here hers herself him himself his how i if in into is "
    "it its itself just let me more most my myself no nor not of off on "
    "once only or other our ours ourselves out over own same she should "
    "so some such than that the their theirs them themselves then there "
    "these they this those through to too under until up us very was we "
    "were what when where which while who whom why will with would you "
    "your yours yourself yourselves also like really right well yeah yes "
    "know think going want need help make sure thing things way good great "
    "user assistant human system ai thanks thank hello hi hey ok okay".split()
)
_CHUNK_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}")


def _turn_keywords(text: str) -> set[str]:
    """Extract keyword set from a turn for topic-shift detection."""
    words = _CHUNK_WORD_RE.findall(text.lower())
    counts = Counter(w for w in words if w not in _CHUNK_STOP and len(w) > 2)
    return set(w for w, _ in counts.most_common(8))


def _topic_shifted(prev_text: str, next_text: str) -> bool:
    """Check if topic shifted between two text segments."""
    prev_kw = _turn_keywords(prev_text)
    next_kw = _turn_keywords(next_text)
    if not prev_kw or not next_kw:
        return False
    intersection = prev_kw & next_kw
    union = prev_kw | next_kw
    overlap = len(intersection) / len(union) if union else 0.0
    return overlap < _TOPIC_SHIFT_THRESHOLD


def is_conversation(text: str) -> bool:
    """Detect whether text contains conversation role markers."""
    if len(_ROLE_PATTERN.findall(text)) >= 2:
        return True
    # Fallback: named speakers (Caroline:, Melanie:, …).  Require ≥ 3
    # matches to avoid false positives on key-value or Markdown text.
    return len(_NAMED_SPEAKER_PATTERN.findall(text)) >= 3


def _conversation_pattern(text: str) -> re.Pattern[str]:
    """Return the split pattern appropriate for this conversation."""
    if len(_ROLE_PATTERN.findall(text)) >= 2:
        return _ROLE_PATTERN
    return _NAMED_SPEAKER_PATTERN


def chunk_conversation(
    text: str,
    max_chars: int = _DEFAULT_MAX_CHARS,
    detect_topic_shift: bool = True,
) -> list[str]:
    """Split conversation text into exchange-pair segments.

    Keeps user + assistant turns together when possible.
    If a single turn exceeds max_chars, it becomes its own chunk.
    When *detect_topic_shift* is True, forces a chunk boundary when
    keyword overlap between the current chunk and the next turn drops
    below the threshold — creating semantically coherent chunks.
    """
    pattern = _conversation_pattern(text)
    parts = pattern.split(text)
    # Named speakers produce short turns (50-200 chars) where keyword
    # overlap between turns is unreliable.  Skip topic-shift detection
    # and rely on the size gate alone to produce coherent chunks.
    if pattern is _NAMED_SPEAKER_PATTERN:
        detect_topic_shift = False

    # Reconstruct turns: parts = [pre, role1, content1, role2, content2, ...]
    turns: list[str] = []
    i = 1
    while i < len(parts) - 1:
        role = parts[i]
        content = parts[i + 1].rstrip()
        turns.append(f"{role}:{content}")
        i += 2

    if not turns:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks: list[str] = []
    current = ""

    for turn in turns:
        if not current:
            current = turn
            continue

        candidate = current + "\n" + turn

        # Size gate
        size_overflow = len(candidate) > max_chars
        # Topic-shift gate: only check once the chunk has enough content
        shift = (
            detect_topic_shift
            and len(current) > 60  # need enough text for keywords
            and _topic_shifted(current, turn)
        )

        if size_overflow or shift:
            if current.strip():
                chunks.append(current.strip())
            current = turn
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_text(
    text: str,
    max_chars: int = _DEFAULT_MAX_CHARS,
    overlap_chars: int = _DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split text into overlapping segments at paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", text)

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if not current:
            current = para
            continue

        candidate = current + "\n\n" + para
        if len(candidate) > max_chars:
            chunks.append(current.strip())
            # Overlap: carry the tail of the previous chunk
            if overlap_chars > 0 and len(current) > overlap_chars:
                current = current[-overlap_chars:].lstrip() + "\n\n" + para
            else:
                current = para
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk(
    text: str,
    max_chars: int = _DEFAULT_MAX_CHARS,
    overlap_chars: int = _DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Auto-detect content type and chunk accordingly.

    Returns a single-element list if content fits within max_chars.
    """
    if len(text.strip()) <= max_chars:
        stripped = text.strip()
        return [stripped] if stripped else []

    if is_conversation(text):
        return chunk_conversation(text, max_chars)

    return chunk_text(text, max_chars, overlap_chars)
