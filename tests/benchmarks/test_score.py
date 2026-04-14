"""Tests for the LongMemEval scoring module."""

import json

from benchmarks.longmemeval.score import (
    _unique_session_ids,
    keyword_containment,
    ndcg_at_k,
    recall_at_k,
    score,
)


class TestRecallAtK:
    def test_hit_at_rank_1(self) -> None:
        assert recall_at_k(["s1", "s2", "s3"], {"s1"}, k=5) == 1.0

    def test_hit_at_rank_5(self) -> None:
        assert recall_at_k(["s1", "s2", "s3", "s4", "s5"], {"s5"}, k=5) == 1.0

    def test_miss_beyond_k(self) -> None:
        assert recall_at_k(["s1", "s2", "s3", "s4", "s5", "s6"], {"s6"}, k=5) == 0.0

    def test_empty_ranking(self) -> None:
        assert recall_at_k([], {"s1"}, k=5) == 0.0

    def test_multiple_answer_sessions_any(self) -> None:
        assert recall_at_k(["s1", "s2"], {"s2", "s3"}, k=5) == 1.0

    def test_no_answer_sessions_in_top_k(self) -> None:
        assert recall_at_k(["s1", "s2"], {"s3", "s4"}, k=5) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        assert ndcg_at_k(["s1", "s2", "s3"], {"s1"}, k=3) == 1.0

    def test_zero_when_no_relevant(self) -> None:
        assert ndcg_at_k(["s1", "s2"], {"s3"}, k=2) == 0.0

    def test_lower_for_deeper_rank(self) -> None:
        ndcg_rank1 = ndcg_at_k(["s1", "s2"], {"s1"}, k=2)
        ndcg_rank2 = ndcg_at_k(["s2", "s1"], {"s1"}, k=2)
        assert ndcg_rank1 > ndcg_rank2

    def test_empty_ranking(self) -> None:
        assert ndcg_at_k([], {"s1"}, k=5) == 0.0


class TestKeywordContainment:
    def test_match(self) -> None:
        assert keyword_containment("The answer is 42.", "42") == 1.0

    def test_case_insensitive(self) -> None:
        assert keyword_containment("Hello World", "hello world") == 1.0

    def test_no_match(self) -> None:
        assert keyword_containment("The answer is 42.", "43") == 0.0

    def test_empty_answer(self) -> None:
        assert keyword_containment("anything", "") == 1.0


class TestUniqueSessionIds:
    def test_deduplicates(self) -> None:
        ranked = [
            {"session_id": "s1", "score": 0.9},
            {"session_id": "s1", "score": 0.8},
            {"session_id": "s2", "score": 0.7},
        ]
        assert _unique_session_ids(ranked) == ["s1", "s2"]

    def test_preserves_order(self) -> None:
        ranked = [
            {"session_id": "s3", "score": 0.9},
            {"session_id": "s1", "score": 0.8},
            {"session_id": "s2", "score": 0.7},
        ]
        assert _unique_session_ids(ranked) == ["s3", "s1", "s2"]

    def test_skips_empty(self) -> None:
        ranked = [
            {"session_id": "", "score": 0.9},
            {"session_id": "s1", "score": 0.8},
        ]
        assert _unique_session_ids(ranked) == ["s1"]

    def test_empty_list(self) -> None:
        assert _unique_session_ids([]) == []


class TestScoreIntegration:
    """End-to-end test using tiny fixture data."""

    def test_score_hit(self, tmp_path: object) -> None:
        gt = [
            {
                "question_id": "q1",
                "question": "What color is the sky?",
                "question_type": "single-session-user",
                "answer": "blue",
                "answer_session_ids": ["s2"],
                "haystack_session_ids": ["s1", "s2", "s3"],
                "haystack_dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "haystack_sessions": [
                    [{"role": "user", "content": "hello"}],
                    [{"role": "user", "content": "the sky is blue today"}],
                    [{"role": "user", "content": "nice weather"}],
                ],
            }
        ]
        p = tmp_path / "gt.json"  # type: ignore[union-attr]
        p.write_text(json.dumps(gt))

        result = {
            "question_id": "q1",
            "hypothesis": "the sky is blue today",
            "ranked": [
                {"session_id": "s2", "score": 0.95},
                {"session_id": "s1", "score": 0.60},
            ],
        }
        rp = tmp_path / "results.jsonl"  # type: ignore[union-attr]
        rp.write_text(json.dumps(result) + "\n")

        report = score(str(p), str(rp))
        overall = report["overall"]
        assert overall["R@1"] == 1.0  # type: ignore[index]
        assert overall["R@5"] == 1.0  # type: ignore[index]
        assert overall["keyword"] == 1.0  # type: ignore[index]
        assert report["n"] == 1

    def test_score_miss(self, tmp_path: object) -> None:
        gt = [
            {
                "question_id": "q1",
                "question": "What color?",
                "question_type": "knowledge-update",
                "answer": "blue",
                "answer_session_ids": ["s3"],
                "haystack_session_ids": ["s1", "s2", "s3"],
                "haystack_dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "haystack_sessions": [
                    [{"role": "user", "content": "hello"}],
                    [{"role": "user", "content": "how are you"}],
                    [{"role": "user", "content": "the sky is blue"}],
                ],
            }
        ]
        p = tmp_path / "gt.json"  # type: ignore[union-attr]
        p.write_text(json.dumps(gt))

        result = {
            "question_id": "q1",
            "hypothesis": "hello, how are you",
            "ranked": [
                {"session_id": "s1", "score": 0.9},
                {"session_id": "s2", "score": 0.8},
            ],
        }
        rp = tmp_path / "results.jsonl"  # type: ignore[union-attr]
        rp.write_text(json.dumps(result) + "\n")

        report = score(str(p), str(rp))
        overall = report["overall"]
        assert overall["R@5"] == 0.0  # type: ignore[index]
        assert overall["keyword"] == 0.0  # type: ignore[index]

    def test_backward_compat_no_ranked(self, tmp_path: object) -> None:
        """Old result files without ``ranked`` still score keyword."""
        gt = [
            {
                "question_id": "q1",
                "question": "What?",
                "question_type": "single-session-user",
                "answer": "hello",
                "answer_session_ids": ["s1"],
                "haystack_session_ids": ["s1"],
                "haystack_dates": ["2025-01-01"],
                "haystack_sessions": [
                    [{"role": "user", "content": "hello world"}],
                ],
            }
        ]
        p = tmp_path / "gt.json"  # type: ignore[union-attr]
        p.write_text(json.dumps(gt))

        result = {"question_id": "q1", "hypothesis": "hello world"}
        rp = tmp_path / "results.jsonl"  # type: ignore[union-attr]
        rp.write_text(json.dumps(result) + "\n")

        report = score(str(p), str(rp))
        overall = report["overall"]
        assert overall["keyword"] == 1.0  # type: ignore[index]
        # R@k should be 0 because no ranked data
        assert overall["R@5"] == 0.0  # type: ignore[index]
