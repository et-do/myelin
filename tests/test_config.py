"""Tests for MyelinSettings configuration validation."""

from __future__ import annotations

import pytest

from myelin.config import MyelinSettings


class TestConfigValidators:
    def test_default_settings_valid(self) -> None:
        cfg = MyelinSettings()
        assert cfg.neocortex_weight == 0.6
        assert cfg.chunk_overlap_chars < cfg.chunk_max_chars

    def test_neocortex_weight_rejects_above_one(self) -> None:
        with pytest.raises(ValueError, match=r"between 0.0 and 1.0"):
            MyelinSettings(neocortex_weight=1.5)

    def test_neocortex_weight_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match=r"between 0.0 and 1.0"):
            MyelinSettings(neocortex_weight=-0.1)

    def test_neocortex_weight_accepts_boundaries(self) -> None:
        assert MyelinSettings(neocortex_weight=0.0).neocortex_weight == 0.0
        assert MyelinSettings(neocortex_weight=1.0).neocortex_weight == 1.0

    def test_dedup_threshold_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match=r"between 0.0 and 1.0"):
            MyelinSettings(dedup_similarity_threshold=2.0)

    def test_overlap_rejects_gte_max(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap_chars"):
            MyelinSettings(chunk_max_chars=200, chunk_overlap_chars=200)

    def test_overlap_rejects_greater_than_max(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap_chars"):
            MyelinSettings(chunk_max_chars=200, chunk_overlap_chars=300)

    def test_overlap_accepts_less_than_max(self) -> None:
        cfg = MyelinSettings(chunk_max_chars=500, chunk_overlap_chars=100)
        assert cfg.chunk_overlap_chars == 100

    def test_positive_int_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            MyelinSettings(default_n_results=0)

    def test_positive_int_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            MyelinSettings(max_idle_days=-1)

    def test_consolidation_interval_allows_zero(self) -> None:
        """consolidation_interval=0 disables auto-consolidation."""
        cfg = MyelinSettings(consolidation_interval=0)
        assert cfg.consolidation_interval == 0

    def test_consolidation_interval_rejects_negative(self) -> None:
        with pytest.raises(Exception):
            MyelinSettings(consolidation_interval=-1)

    def test_decay_interval_allows_zero(self) -> None:
        """Zero disables the auto-decay timer."""
        cfg = MyelinSettings(decay_interval_hours=0)
        assert cfg.decay_interval_hours == 0.0

    def test_decay_interval_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            MyelinSettings(decay_interval_hours=-1.0)
