"""Neocortex re-ranker — slow deliberative re-ranking via cross-encoder.

Neuroscience basis: The neocortex performs slow, deliberative evaluation
of memory candidates surfaced by the fast hippocampal retrieval pathway.
While hippocampal pattern-matching (bi-encoder cosine similarity) is
fast but shallow, neocortical processing evaluates query-memory pairs
together through cross-attention — catching semantic relationships that
independent embeddings miss.

This two-stage architecture mirrors the brain's dual-process retrieval:

    Hippocampus (fast, parallel)  →  top-N candidates by embedding similarity
    Neocortex (slow, sequential)  →  re-rank candidates by cross-attention

The cross-encoder model processes (query, passage) pairs jointly, enabling
it to capture token-level interactions between the query and each candidate.
This is much more accurate than cosine similarity of independent embeddings,
but ~100x slower — hence the two-stage design.

Model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (22M params, free, local)
    - Trained on MS MARCO passage ranking
    - ~33 ms per pair on CPU
    - Apache 2.0 license

References:
    Norman, K. A. & O'Reilly, R. C. (2003). Modeling hippocampal and
        neocortical contributions to recognition memory: A
        complementary-learning-systems approach. Psychological Review, 110.
    McClelland, J. L. et al. (1995). Why there are complementary learning
        systems in the hippocampus and neocortex. Psychological Review, 102.
"""

from __future__ import annotations

from typing import Any

from ..log import suppress_noisy_loggers

suppress_noisy_loggers()

from sentence_transformers import CrossEncoder  # noqa: E402

# Default model — small, fast, and effective for passage re-ranking
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Neocortex:
    """Cross-encoder re-ranker — slow deliberative evaluation."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        cross_encoder: CrossEncoder | None = None,
    ) -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = cross_encoder

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def close(self) -> None:
        """Release the loaded model (frees memory)."""
        self._model = None

    def rerank(
        self,
        query: str,
        passages: list[str],
    ) -> list[float]:
        """Score each passage against the query. Returns a list of scores.

        Scores are raw logits from the cross-encoder (not normalised).
        Higher = more relevant.
        """
        if not passages:
            return []
        pairs: list[list[str]] = [[query, p] for p in passages]
        scores: Any = self._get_model().predict(pairs)  # type: ignore[arg-type]
        return [float(s) for s in scores]


__all__ = ["Neocortex", "DEFAULT_CROSS_ENCODER"]
