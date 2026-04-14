"""Score Myelin LoCoMo results.

LoCoMo retrieval recall = fraction of *evidence sessions* found in top-k.
This differs from LongMemEval (binary: any hit in top-k).

Usage:
    uv run python -m benchmarks.locomo.score \
        benchmarks/locomo/output/myelin_locomo.json

Categories:
    1 = Single-hop       (282 questions)
    2 = Temporal          (321 questions)
    3 = Temporal-inference (96 questions)
    4 = Open-domain       (841 questions)
    5 = Adversarial       (446 questions)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.locomo.run import CATEGORIES, evidence_to_session_ids

# ------------------------------------------------------------------
# Metric functions
# ------------------------------------------------------------------


def retrieval_recall(
    retrieved_ids: list[str],
    evidence_session_ids: set[str],
    k: int,
) -> float:
    """Fraction of evidence sessions found in the top-k retrieved results.

    If evidence is empty, return 1.0 (vacuously true — no evidence needed).
    """
    if not evidence_session_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    found = len(evidence_session_ids & top_k)
    return found / len(evidence_session_ids)


def binary_recall_at_k(
    retrieved_ids: list[str],
    evidence_session_ids: set[str],
    k: int,
) -> float:
    """Binary recall: 1 if ALL evidence sessions are in top-k, else 0."""
    if not evidence_session_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return 1.0 if evidence_session_ids <= top_k else 0.0


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

KS = (5, 10, 20)


def score(results_path: str | Path) -> dict[str, Any]:
    """Score LoCoMo results.

    Returns a nested dict with overall and per-category metrics.
    """
    results: list[dict[str, Any]] = json.loads(Path(results_path).read_text())

    overall: dict[str, list[float]] = defaultdict(list)
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for entry in results:
        category = entry["category"]
        cat_name = CATEGORIES.get(category, f"cat_{category}")
        evidence = entry.get("evidence", [])
        retrieved = entry.get("retrieved_ids", [])

        ev_sessions = evidence_to_session_ids(evidence)

        for k in KS:
            r = retrieval_recall(retrieved, ev_sessions, k)
            br = binary_recall_at_k(retrieved, ev_sessions, k)
            overall[f"R@{k}"].append(r)
            overall[f"BR@{k}"].append(br)
            by_cat[cat_name][f"R@{k}"].append(r)
            by_cat[cat_name][f"BR@{k}"].append(br)

        # Track perfect retrieval
        k_max = max(KS)
        r_max = retrieval_recall(retrieved, ev_sessions, k_max)
        overall["perfect"].append(1.0 if r_max >= 1.0 else 0.0)
        by_cat[cat_name]["perfect"].append(1.0 if r_max >= 1.0 else 0.0)

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

_METRIC_ORDER = ["R@5", "R@10", "R@20", "BR@5", "BR@10", "BR@20", "perfect"]


def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable scoring report to stdout."""
    overall: dict[str, float] = report["overall"]  # type: ignore[assignment]
    by_cat: dict[str, dict[str, Any]] = report["by_category"]  # type: ignore[assignment]
    n = report["n"]

    metrics = [m for m in _METRIC_ORDER if m in overall]

    print(f"\n{'=' * 80}")
    print(f"  Myelin x LoCoMo  ({n} questions)")
    print(f"{'=' * 80}")

    # Metric descriptions
    print("\n  R@k   = fraction of evidence sessions found in top-k (higher = better)")
    print("  BR@k  = binary: ALL evidence sessions in top-k? (strict)")
    print("  perfect = all evidence found at max k")

    # Overall
    print(f"\n  {'OVERALL':28}", end="")
    for m in metrics:
        print(f"  {m:>8}", end="")
    print()
    print(f"  {'':28}", end="")
    for m in metrics:
        print(f"  {overall[m]:8.1%}", end="")
    print()

    # Per category
    print(f"\n  {'CATEGORY':28}", end="")
    for m in metrics:
        print(f"  {m:>8}", end="")
    print(f"  {'n':>5}")
    print(f"  {'-' * 28}", end="")
    for _ in metrics:
        print(f"  {'--------':>8}", end="")
    print(f"  {'-----':>5}")

    for cat, cat_metrics in sorted(by_cat.items()):
        print(f"  {cat:28}", end="")
        for m in metrics:
            print(f"  {cat_metrics.get(m, 0):8.1%}", end="")
        print(f"  {cat_metrics.get('n', 0):5}")

    # MemPalace comparison
    print(f"\n  {'─' * 80}")
    print("  MemPalace comparison (session-level, R@10, no LLM rerank):")
    print("    Raw baseline:     60.3%")
    print("    Hybrid v5:        88.9%  ← honest number to beat")
    print("    bge-large hybrid: 92.4%")
    print(f"  {'=' * 80}\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m benchmarks.locomo.score <results.json>")
        sys.exit(1)

    report = score(sys.argv[1])
    print_report(report)
