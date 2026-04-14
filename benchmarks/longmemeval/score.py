"""Score Myelin LongMemEval results with multiple metrics.

Computes the same metrics MemPalace uses (R@k, NDCG@k) plus keyword
containment, all broken down by question category.

Usage:
    uv run python -m benchmarks.longmemeval.score \
        benchmarks/longmemeval/data/longmemeval_oracle.json \
        benchmarks/longmemeval/output/myelin_oracle_20260410_233726.jsonl

Metrics:
    R@k  — Recall at k.  Binary: is *any* answer session in the top-k
           ranked results?  This is the metric MemPalace reports
           (96.6% R@5 headline).
    NDCG@k — Normalized Discounted Cumulative Gain.  Accounts for
             *rank position* of correct sessions.
    Keyword — Answer string appears verbatim in retrieved text.
              Strict but easy to compute; ceiling is <50% because
              many answers are LLM inferences, not direct quotes.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------
# Metric functions
# ------------------------------------------------------------------


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(ranked_session_ids: list[str], answer_ids: set[str], k: int) -> float:
    """NDCG@k — normalised DCG for ranked session list."""
    rels = [1.0 if sid in answer_ids else 0.0 for sid in ranked_session_ids[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return _dcg(rels, k) / idcg


def recall_at_k(ranked_session_ids: list[str], answer_ids: set[str], k: int) -> float:
    """R@k — is *any* answer session present in the top k?"""
    top_k = set(ranked_session_ids[:k])
    return 1.0 if top_k & answer_ids else 0.0


def keyword_containment(hypothesis: str, answer: str) -> float:
    """1.0 if the answer string appears in the hypothesis text, else 0.0."""
    return 1.0 if answer.lower() in hypothesis.lower() else 0.0


# ------------------------------------------------------------------
# Deduplication helper
# ------------------------------------------------------------------


def _unique_session_ids(ranked: list[dict[str, object]]) -> list[str]:
    """Deduplicate ranked chunk results to a session-level ranking.

    Multiple chunks may belong to the same session. The first
    occurrence (highest ranked) determines the session's rank.
    """
    seen: set[str] = set()
    out: list[str] = []
    for entry in ranked:
        sid = str(entry.get("session_id", ""))
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


# ------------------------------------------------------------------
# Scoring pipeline
# ------------------------------------------------------------------

KS = (1, 3, 5, 10)


def score(
    ground_truth_path: str | Path,
    results_path: str | Path,
) -> dict[str, object]:
    """Score a results JSONL against the ground-truth JSON.

    Returns a nested dict::

        {
            "overall": {"R@5": float, "R@10": float, ...},
            "by_category": {
                "multi-session": {"R@5": float, ...},
                ...
            },
            "n": int,
        }
    """
    gt_data = json.loads(Path(ground_truth_path).read_text())
    gt_by_id = {e["question_id"]: e for e in gt_data}

    with open(results_path) as f:
        results = {json.loads(line)["question_id"]: json.loads(line) for line in f}

    # Accumulators: metric_name → list[float]
    overall: dict[str, list[float]] = defaultdict(list)
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for qid, entry in gt_by_id.items():
        result = results.get(qid)
        if result is None:
            continue

        qtype: str = entry["question_type"]
        answer_sids = set(entry["answer_session_ids"])
        answer_text: str = str(entry.get("answer", ""))
        hypothesis: str = str(result.get("hypothesis", ""))

        # Session-level ranking (deduplicated from chunk results)
        ranked_raw: list[dict[str, object]] = result.get("ranked", [])
        ranked_sids = _unique_session_ids(ranked_raw)

        # Compute metrics
        kw = keyword_containment(hypothesis, answer_text)
        overall["keyword"].append(kw)
        by_cat[qtype]["keyword"].append(kw)

        for k in KS:
            r = recall_at_k(ranked_sids, answer_sids, k)
            n = ndcg_at_k(ranked_sids, answer_sids, k)
            overall[f"R@{k}"].append(r)
            overall[f"NDCG@{k}"].append(n)
            by_cat[qtype][f"R@{k}"].append(r)
            by_cat[qtype][f"NDCG@{k}"].append(n)

    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _summarise(acc: dict[str, list[float]]) -> dict[str, float]:
        return {name: round(_avg(vals), 4) for name, vals in sorted(acc.items())}

    return {
        "overall": _summarise(overall),
        "by_category": {
            cat: {**_summarise(metrics), "n": len(next(iter(metrics.values())))}
            for cat, metrics in sorted(by_cat.items())
        },
        "n": len(results),
    }


# ------------------------------------------------------------------
# Pretty printer
# ------------------------------------------------------------------

_METRIC_ORDER = ["R@1", "R@3", "R@5", "R@10", "NDCG@5", "NDCG@10", "keyword"]


def print_report(report: dict[str, object]) -> None:
    """Print a human-readable scoring report to stdout."""
    overall: Any = report["overall"]
    by_cat: Any = report["by_category"]
    n = report["n"]

    # Choose which metrics are available
    metrics = [m for m in _METRIC_ORDER if m in overall]

    print(f"\n{'=' * 72}")
    print(f"  Myelin x LongMemEval  ({n} questions)")
    print(f"{'=' * 72}")

    # Overall
    print(f"\n  {'OVERALL':30}", end="")
    for m in metrics:
        print(f"  {m:>8}", end="")
    print()
    print(f"  {'':30}", end="")
    for m in metrics:
        print(f"  {overall[m]:8.4f}", end="")
    print()

    # Per category
    print(f"\n  {'CATEGORY':30}", end="")
    for m in metrics:
        print(f"  {m:>8}", end="")
    print(f"  {'n':>5}")
    print(f"  {'-' * 30}", end="")
    for _ in metrics:
        print(f"  {'--------':>8}", end="")
    print(f"  {'-----':>5}")

    for cat, cat_metrics in sorted(by_cat.items()):
        print(f"  {cat:30}", end="")
        for m in metrics:
            print(f"  {cat_metrics.get(m, 0):8.4f}", end="")
        print(f"  {cat_metrics.get('n', 0):5}")

    print(f"\n{'=' * 72}\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python -m benchmarks.longmemeval.score"
            " <ground_truth.json> <results.jsonl>"
        )
        sys.exit(1)

    report = score(sys.argv[1], sys.argv[2])
    print_report(report)
