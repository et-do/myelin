"""Hippocampal time cells — temporal context encoding and retrieval bias.

Neuroscience basis: Time cells in the hippocampus fire at specific
temporal offsets, giving memories a position on a mental timeline.
During recall, temporal context cues activate time cells that bias
retrieval toward memories encoded at the matching time point.

Here we parse temporal expressions from queries ("5 days ago",
"last Tuesday", "two weeks ago") and resolve them to absolute date
ranges.  Memories whose timestamps fall in the range receive a boost
during re-ranking.

References:
    MacDonald, C. J. et al. (2011). Hippocampal "time cells" bridge
        the gap in memory for discontiguous events. Neuron, 71(4).
    Eichenbaum, H. (2014). Time cells in the hippocampus: a new
        dimension for mapping memories. Nature Reviews Neuroscience, 15.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

# Word-to-number mapping for temporal expressions
_WORD_NUMBERS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_WEEKDAYS: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_MONTHS: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

# Build regex component for number-or-word
_NUM_OR_WORD = r"(\d+|" + "|".join(_WORD_NUMBERS.keys()) + r")"

# Compile patterns — order matters: first match wins
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"{_NUM_OR_WORD}\s+days?\s+ago", re.IGNORECASE), "days"),
    (re.compile(rf"{_NUM_OR_WORD}\s+weeks?\s+ago", re.IGNORECASE), "weeks"),
    (re.compile(rf"{_NUM_OR_WORD}\s+months?\s+ago", re.IGNORECASE), "months"),
    (
        re.compile(
            r"last\s+(" + "|".join(_WEEKDAYS.keys()) + r")",
            re.IGNORECASE,
        ),
        "last_weekday",
    ),
    (
        re.compile(
            r"\b(?:in|during|last)\s+(" + "|".join(_MONTHS.keys()) + r")\b",
            re.IGNORECASE,
        ),
        "named_month",
    ),
]

# Subset: only relative expressions that can be resolved from "now"
# (X ago, last weekday).  Named months require the correct year context
# and should NOT trigger auto-inference of a reference date.
_RELATIVE_PATTERNS = [p for p, u in _PATTERNS if u != "named_month"]


def has_relative_temporal_expression(query: str) -> bool:
    """Check if *query* contains a relative temporal expression.

    Only matches expressions that can be resolved from the current date
    (e.g. "3 days ago", "last Tuesday").  Named months ("in June") are
    excluded because they require calendar-year context.
    """
    return any(p.search(query) for p in _RELATIVE_PATTERNS)


def _parse_number(s: str) -> int:
    """Convert a numeric string or word to int."""
    s_lower = s.lower().strip()
    if s_lower in _WORD_NUMBERS:
        return _WORD_NUMBERS[s_lower]
    return int(s)


def parse_temporal_reference(
    query: str,
    reference_date: datetime,
) -> tuple[datetime, datetime] | None:
    """Extract a temporal expression from *query* and resolve to a date range.

    Returns (start, end) inclusive datetime range, or None if no temporal
    expression is found.  The range includes a buffer for imprecision.
    """
    for pattern, unit in _PATTERNS:
        m = pattern.search(query)
        if not m:
            continue

        if unit == "last_weekday":
            day_name = m.group(1).lower()
            target_weekday = _WEEKDAYS[day_name]
            ref_weekday = reference_date.weekday()
            days_back = (ref_weekday - target_weekday) % 7
            if days_back == 0:
                days_back = 7  # "last Tuesday" when today is Tuesday → 7 days ago
            target = reference_date - timedelta(days=days_back)
            return (target - timedelta(days=1), target + timedelta(days=1))

        if unit == "named_month":
            month_name = m.group(1).lower()
            month_num = _MONTHS[month_name]
            # Use the reference year; if the month is after the reference
            # month, assume the previous year
            year = reference_date.year
            if month_num >= reference_date.month:
                year -= 1
            start = datetime(year, month_num, 1)
            # Last day of month
            if month_num == 12:
                end = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end = datetime(year, month_num + 1, 1) - timedelta(days=1)
            return (start, end)

        n = _parse_number(m.group(1))

        if unit == "days":
            center = reference_date - timedelta(days=n)
            return (center - timedelta(days=1), center + timedelta(days=1))
        if unit == "weeks":
            center = reference_date - timedelta(weeks=n)
            return (center - timedelta(days=3), center + timedelta(days=3))
        if unit == "months":
            center = reference_date - timedelta(days=30 * n)
            return (center - timedelta(days=7), center + timedelta(days=7))

    return None


def parse_session_date(date_str: str) -> datetime | None:
    """Parse a session date string like ``'2023/05/20 (Sat) 02:21'``.

    Only the date part is extracted — time-of-day is discarded for
    day-level temporal matching.
    """
    try:
        return datetime.strptime(date_str.split("(")[0].strip(), "%Y/%m/%d")
    except (ValueError, IndexError):
        return None


def recency_boost(session_ordinal: int, ref_ordinal: int, half_life_days: int) -> float:
    """Compute a soft recency multiplier based on age relative to reference.

    Uses an exponential decay: newer memories get a larger boost.
    The half-life controls how quickly the boost decays with age.

    Returns a value in (0, 1] where 1.0 = same day as reference,
    0.5 = half_life_days old, 0.25 = 2x half_life_days old, etc.

    This mirrors the temporal context model (Howard & Kahana, 2002):
    recent memories have higher baseline accessibility, fading
    continuously rather than as a step function.
    """
    if half_life_days <= 0:
        return 0.0
    age_days = max(ref_ordinal - session_ordinal, 0)
    # Exponential decay: 2^(-age/half_life)
    return float(2.0 ** (-age_days / half_life_days))
