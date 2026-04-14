"""Query planner — prefrontal inhibitory gating for recall.

Neuroscience basis: During retrieval, the prefrontal cortex performs
*selective inhibition* — activating relevant memory traces while
suppressing irrelevant ones (Anderson & Green, 2001; Levy & Anderson,
2002). This is not passive filtering; the PFC actively infers which
context dimensions to gate on from the query itself.

When someone asks "what did we decide about auth?", the PFC:
1. Detects "decide" → activates the decision schema → gate on semantic
2. Detects "auth" → activates auth engram cluster → gate on scope
3. Laterally inhibits unrelated clusters (billing, deploy, personal)

This differs from MemPalace's explicit wing/room/hall filtering:
there, the *caller* specifies filters. Here, the *system* infers them
from query content — like how the brain resolves retrieval cues
automatically.

References:
    Anderson, M. C. & Green, C. (2001). Suppressing unwanted memories
        by executive control. Nature, 410.
    Levy, B. J. & Anderson, M. C. (2002). Inhibitory processes and the
        control of memory retrieval. Trends in Cognitive Sciences, 6(7).
    Badre, D. & Wagner, A. D. (2007). Left ventrolateral prefrontal
        cortex and the cognitive control of memory. Neuropsychologia, 45(13).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import MemoryType

# ---------------------------------------------------------------------------
# Query context signals
# ---------------------------------------------------------------------------

# Memory type signals — what kind of memory is the query searching for?
_TYPE_SIGNALS: dict[MemoryType, tuple[re.Pattern[str], ...]] = {
    "semantic": tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\b(?:what (?:is|are|was|were)"
            r"|what did (?:we|I|you) (?:decide|choose|conclude))\b",
            r"\b(?:decision|fact|definition|conclusion|the (?:answer|result))\b",
            r"\b(?:explain|describe|tell me about|what do (?:we|you) know)\b",
        )
    ),
    "procedural": tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\b(?:how (?:do|does|did|to|should)"
            r"|what(?:'s| is) (?:the |our )?(?:process|workflow|procedure))\b",
            r"\b(?:prefer|convention|standard|style|rule|habit|always|never)\b",
            r"\b(?:what (?:do|does) .+ (?:prefer|like|want|use))\b",
        )
    ),
    "prospective": tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\b(?:what should|what(?:'s| is) (?:the )?(?:plan|next|todo|action))\b",
            r"\b(?:going to|will we|schedule|roadmap|deadline|upcoming)\b",
            r"\b(?:recommend|suggest|advise|proposed)\b",
        )
    ),
    "episodic": tuple(
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\b(?:when did|what happened|last time|remember when|that time)\b",
            r"\b(?:yesterday|last (?:week|month|session)|recently|the other day)\b",
            r"\b(?:who (?:did|was|said)|where (?:did|was))\b",
        )
    ),
}

# Scope detection — named domain mentions in the query
_SCOPE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # Common software domains
        r"\b(auth(?:entication)?|oauth|jwt|sso|login|credentials)\b",
        r"\b(database|db|sql|postgres|mongo|redis|migration)\b",
        r"\b(deploy(?:ment)?|ci/?cd|pipeline|docker|k8s|kubernetes|infra)\b",
        r"\b(security|encryption|tls|ssl|certs?|firewall|vulnerability)\b",
        r"\b(testing|test|unit.?test|e2e|integration.?test|coverage)\b",
        r"\b(api|rest|graphql|grpc|endpoint|route|middleware)\b",
        r"\b(frontend|ui|ux|css|react|vue|angular|component)\b",
        r"\b(backend|server|service|microservice|worker|queue)\b",
        r"\b(billing|payment|stripe|invoice|subscription|pricing)\b",
        r"\b(monitoring|logging|metrics|alerting|observability|trace)\b",
    )
)


@dataclass(slots=True)
class QueryPlan:
    """Inferred retrieval context — filters the PFC decides to apply.

    Fields set to None mean "don't filter on this dimension" (no inhibition).
    """

    memory_type: MemoryType | None = None
    scope_hint: str | None = None  # detected domain, not an exact scope match
    signals: dict[str, float] = field(default_factory=dict)


def plan(query: str) -> QueryPlan:
    """Analyze a query and infer retrieval filters.

    Returns a QueryPlan with auto-detected memory_type and scope hints.
    The caller decides whether to apply them (hard filter vs. soft preference).
    """
    result = QueryPlan()

    # --- Memory type inference ---
    best_type: MemoryType | None = None
    best_score = 0.0

    for mtype, patterns in _TYPE_SIGNALS.items():
        hits = sum(1 for p in patterns if p.search(query))
        if hits > 0:
            score = hits / len(patterns)
            result.signals[f"type_{mtype}"] = round(score, 3)
            if score > best_score:
                best_score = score
                best_type = mtype

    if best_type is not None:
        result.memory_type = best_type

    # --- Scope inference ---
    for pattern in _SCOPE_PATTERNS:
        m = pattern.search(query)
        if m:
            result.scope_hint = m.group(1).lower()
            result.signals["scope_detected"] = 1.0
            break  # take first match (most specific)

    return result
