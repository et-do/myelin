"""Entorhinal cortex — context coordinate system for memory organisation.

Neuroscience basis: The entorhinal cortex (EC) is the gateway between
neocortex and hippocampus. It creates a coordinate system for memories
via two pathways:

- **Lateral EC** (LEC): "what" pathway — content/topic identity.
  Grid cells in LEC encode object-level features. Here we extract
  topic keywords that identify the semantic content of a memory.

- **Medial EC** (MEC): "where/when" pathway — spatiotemporal context.
  Grid cells in MEC encode spatial and temporal coordinates. Here we
  assign a broad *region* label (domain classification) that partitions
  memory space into coarse areas.

During encoding (store), the EC generates context coordinates stored
alongside the hippocampal trace. During retrieval (recall), the EC
infers coordinates from the query and uses them to re-rank results —
memories whose coordinates overlap the query's coordinates get boosted.

This is analogous to MemPalace's wing/hall/room hierarchy, but
derived from neuroscience rather than architectural metaphor:

    MemPalace Wing  → EC Region (cortical territory / broad domain)
    MemPalace Hall  → Memory Type (already handled by prefrontal.py)
    MemPalace Room  → EC Topics (grid-cell activation pattern / keyword set)

The +36% retrieval boost MemPalace achieves from hierarchical filtering
comes from this same principle: encoding specificity — memories retrieved
in context matching their encoding context have higher fidelity.

References:
    Hafting, T. et al. (2005). Microstructure of a spatial map in the
        entorhinal cortex. Nature, 436.
    Moser, E. I. et al. (2008). Place cells, grid cells, and the brain's
        spatial representation system. Annual Review of Neuroscience, 31.
    Eichenbaum, H. et al. (2007). The medial temporal lobe and recognition
        memory. Annual Review of Neuroscience, 30.
"""

from __future__ import annotations

import re
from collections import Counter

# ---------------------------------------------------------------------------
# Stop words — filtered out during keyword extraction (LEC pathway)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    "a about above after again against all am an and any are aren't as at "
    "be because been before being below between both but by can't cannot "
    "could couldn't did didn't do does doesn't doing don't down during each "
    "few for from further get got had hadn't has hasn't have haven't having "
    "he he'd he'll he's her here here's hers herself him himself his how "
    "how's i i'd i'll i'm i've if in into is isn't it it's its itself "
    "let's me more most mustn't my myself no nor not of off on once only "
    "or other ought our ours ourselves out over own same shan't she she'd "
    "she'll she's should shouldn't so some such than that that's the their "
    "theirs them themselves then there there's these they they'd they'll "
    "they're they've this those through to too under until up us very was "
    "wasn't we we'd we'll we're we've were weren't what what's when when's "
    "where where's which while who who's whom why why's with won't would "
    "wouldn't you you'd you'll you're you've your yours yourself yourselves "
    "also just like really right well yeah yes know think going "
    "want need help make sure thing things way good great "
    "user assistant human system ai "
    "thanks thank hello hi hey ok okay "
    "however meanwhile therefore furthermore additionally".split()
)

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}")

# ---------------------------------------------------------------------------
# Region classification (MEC pathway — broad domain categories)
# ---------------------------------------------------------------------------

_REGION_PATTERNS: dict[str, re.Pattern[str]] = {
    "technology": re.compile(
        r"\b(?:code|software|api|database|deploy|server|bug|debug|git"
        r"|docker|cloud|aws|gcp|azure|kubernetes|k8s|ci/?cd|pipeline"
        r"|python|javascript|typescript|rust|java|golang|react|vue"
        r"|angular|node|express|django|flask|sql|nosql|postgres|mongo"
        r"|redis|graphql|rest|grpc|microservice|frontend|backend)\b",
        re.IGNORECASE,
    ),
    "security": re.compile(
        r"\b(?:auth(?:entication)?|oauth|jwt|sso|login|credential"
        r"|encrypt|decrypt|tls|ssl|cert|firewall|vulnerability|cve"
        r"|permission|rbac|token|secret|password|2fa|mfa)\b",
        re.IGNORECASE,
    ),
    "health": re.compile(
        r"\b(?:doctor|hospital|medicine|symptom|diagnos|health|fitness"
        r"|exercise|workout|diet|nutrition|vitamin|weight|pain|therapy"
        r"|appointment|prescription|blood|heart|mental|sleep)\b",
        re.IGNORECASE,
    ),
    "finance": re.compile(
        r"\b(?:money|budget|invest|stock|savings?|bank|credit|debit"
        r"|mortgage|loan|tax|salary|income|expense|payment|billing"
        r"|invoice|subscription|pricing|insurance|retirement|401k)\b",
        re.IGNORECASE,
    ),
    "personal": re.compile(
        r"\b(?:birthday|family|friend|hobby|vacation|travel|movie|book"
        r"|music|restaurant|recipe|cooking|pet|dog|cat|garden|home"
        r"|wedding|anniversary|party|gift|holiday|weekend)\b",
        re.IGNORECASE,
    ),
    "work": re.compile(
        r"\b(?:meeting|project|deadline|sprint|standup|review|manager"
        r"|team|colleague|presentation|report|roadmap|milestone|hire"
        r"|onboard|promotion|performance|remote|office)\b",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract the top-N most distinctive keywords from text (LEC pathway).

    Uses term frequency on non-stop-words. Simple but effective for
    creating topic coordinates without any external dependencies.
    """
    words = _WORD_RE.findall(text.lower())
    filtered = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def assign_region(text: str) -> str | None:
    """Classify text into a broad domain region (MEC pathway).

    Returns the region with the most pattern matches, or None if
    no clear region is detected.
    """
    best_region: str | None = None
    best_hits = 0

    for region, pattern in _REGION_PATTERNS.items():
        hits = len(pattern.findall(text))
        if hits > best_hits:
            best_hits = hits
            best_region = region

    # Require at least 2 hits to assign a region (avoid false positives)
    return best_region if best_hits >= 2 else None


def topic_overlap(query_keywords: list[str], memory_keywords: list[str]) -> float:
    """Compute Jaccard-like overlap between keyword sets.

    Returns 0.0 (no overlap) to 1.0 (perfect overlap).
    """
    if not query_keywords or not memory_keywords:
        return 0.0
    q_set = set(query_keywords)
    m_set = set(memory_keywords)
    intersection = q_set & m_set
    union = q_set | m_set
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Speaker detection (source monitoring — "who" pathway)
# ---------------------------------------------------------------------------

# Pattern: "Speaker:" at start of line (conversational memory)
_SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z .'-]{1,40}):", re.MULTILINE)

# Generic role labels that aren't real speaker names
_GENERIC_ROLES = frozenset(
    "user assistant human ai system bot agent moderator admin".split()
)


def extract_speakers(text: str) -> list[str]:
    """Detect speaker names from conversational text.

    Returns a list of unique speaker names (non-generic), ordered by
    frequency (most turns first).  Generic role labels (user, assistant)
    are excluded — they appear in all conversations and carry no
    discriminative information.
    """
    raw = _SPEAKER_RE.findall(text)
    counts: Counter[str] = Counter()
    for name in raw:
        name = name.strip()
        if name.lower() not in _GENERIC_ROLES and len(name) > 1:
            counts[name] += 1
    return [name for name, _ in counts.most_common()]


def detect_query_speakers(query: str, known_speakers: list[str]) -> list[str]:
    """Find known speaker names mentioned in a query.

    Case-insensitive substring match against the speaker list.
    """
    query_lower = query.lower()
    return [s for s in known_speakers if s.lower() in query_lower]
