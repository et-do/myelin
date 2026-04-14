"""Tests for the amygdala input gate."""

from myelin.config import MyelinSettings
from myelin.store.amygdala import passes_gate


class TestPassesGate:
    def test_rejects_short_content(self) -> None:
        ok, reason = passes_gate("hi")
        assert not ok
        assert "too short" in reason

    def test_accepts_valid_content(self) -> None:
        ok, reason = passes_gate(
            "The auth service uses JWT tokens for session management"
        )
        assert ok
        assert reason == "ok"

    def test_rejects_near_duplicate(self) -> None:
        ok, reason = passes_gate(
            "Some valid content that is long enough",
            existing_similarities=[0.97],
        )
        assert not ok
        assert "near-duplicate" in reason

    def test_accepts_when_no_duplicates(self) -> None:
        ok, _reason = passes_gate(
            "Some valid content that is long enough",
            existing_similarities=[0.5, 0.3],
        )
        assert ok

    def test_respects_custom_min_length(self) -> None:
        cfg = MyelinSettings(min_content_length=5)
        ok, _reason = passes_gate("short", cfg=cfg)
        assert ok

    def test_respects_custom_dedup_threshold(self) -> None:
        cfg = MyelinSettings(dedup_similarity_threshold=0.80)
        ok, reason = passes_gate(
            "Some valid content that is long enough",
            existing_similarities=[0.85],
            cfg=cfg,
        )
        assert not ok
        assert "near-duplicate" in reason
