"""Tests for neocortex.py — cross-encoder re-ranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from myelin.reranker import Neocortex


class TestNeocortexRerank:
    def test_rerank_returns_float_scores(self) -> None:
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([2.5, -0.3, 1.1])
        neo = Neocortex(cross_encoder=mock_ce)

        scores = neo.rerank("test query", ["a", "b", "c"])
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert scores[0] == 2.5
        assert scores[1] == -0.3

    def test_rerank_empty_passages(self) -> None:
        neo = Neocortex(cross_encoder=MagicMock())
        assert neo.rerank("query", []) == []

    def test_rerank_single_passage(self) -> None:
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([1.0])
        neo = Neocortex(cross_encoder=mock_ce)

        scores = neo.rerank("query", ["only one"])
        assert len(scores) == 1
        assert scores[0] == 1.0

    def test_predict_called_with_pairs(self) -> None:
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.5, 0.8])
        neo = Neocortex(cross_encoder=mock_ce)

        neo.rerank("what color?", ["red", "blue"])
        call_args = mock_ce.predict.call_args[0][0]
        assert call_args == [["what color?", "red"], ["what color?", "blue"]]

    @patch("myelin.reranker.CrossEncoder")
    def test_default_model_loaded(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = MagicMock()
        mock_cls.return_value.predict.return_value = [0.5]
        neo = Neocortex()
        # Model loads lazily on first rerank call
        mock_cls.assert_not_called()
        neo.rerank("test", ["passage"])
        mock_cls.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")
