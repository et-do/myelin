"""Amygdala — lightweight input gate for memory storage.

No LLM in the hot path. The calling agent already decided this is
worth storing; we just filter obvious noise and near-duplicates.
"""

from __future__ import annotations

from ..config import MyelinSettings, settings


def passes_gate(
    content: str,
    existing_similarities: list[float] | None = None,
    cfg: MyelinSettings | None = None,
) -> tuple[bool, str]:
    """Check whether content should be stored.

    Returns (should_store, reason).
    """
    _cfg = cfg or settings
    stripped = content.strip()
    if len(stripped) < _cfg.min_content_length:
        min_len = _cfg.min_content_length
        return False, f"too short ({len(stripped)} chars, min {min_len})"

    if existing_similarities:
        max_sim = max(existing_similarities)
        if max_sim >= _cfg.dedup_similarity_threshold:
            return False, f"near-duplicate (similarity {max_sim:.2f})"

    return True, "ok"
