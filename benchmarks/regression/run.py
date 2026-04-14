"""Regression gate — fast dual-benchmark check for pipeline changes.

Runs a stratified subset of LongMemEval_S + LoCoMo questions and diffs
results against a frozen baseline.  Catches regressions before a full
benchmark run.

Subset sizes:
    LongMemEval_S:  ~50 questions (8-9 per category)
    LoCoMo:         2 conversations (~300 questions)
    Total runtime:  ~10 minutes vs ~3 hours for full runs

Usage:
    # Create baseline from current best results:
    uv run python -m benchmarks.regression.run --create-baseline

    # Run regression check against baseline:
    uv run python -m benchmarks.regression.run
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------

# LongMemEval: 8-9 questions per type (stratified), ~50 total
_LME_PER_TYPE = 9
# LoCoMo: 2 conversations (pick smallest + medium for speed)
_LOCOMO_CONV_IDS = ["conv-30", "conv-26"]  # 105 + 199 = 304 QA

_DATA_DIR = Path(__file__).resolve().parent.parent
_BASELINE_DIR = _DATA_DIR / "regression" / "baseline"


def _select_lme_subset(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick a stratified subset of LongMemEval questions."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in data:
        by_type[q.get("question_type", "unknown")].append(q)

    subset: list[dict[str, Any]] = []
    for qtype in sorted(by_type):
        questions = by_type[qtype]
        # Deterministic selection: take evenly-spaced indices
        step = max(1, len(questions) // _LME_PER_TYPE)
        for i in range(0, len(questions), step):
            subset.append(questions[i])
            if (
                len([q for q in subset if q.get("question_type") == qtype])
                >= _LME_PER_TYPE
            ):
                break
    return subset


def _select_locomo_subset(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the predefined LoCoMo conversations."""
    return [s for s in data if s.get("sample_id") in _LOCOMO_CONV_IDS]


# ---------------------------------------------------------------------------
# Runners — reuse existing benchmark infrastructure
# ---------------------------------------------------------------------------


def _run_lme(subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run LongMemEval subset through Myelin recall pipeline."""
    from sentence_transformers import SentenceTransformer

    from benchmarks.longmemeval.run import run_instance
    from myelin.config import MyelinSettings
    from myelin.reranker import Neocortex

    cfg = MyelinSettings(data_dir=Path(tempfile.mkdtemp()))
    embedder = SentenceTransformer(cfg.embedding_model)
    reranker = Neocortex(model_name=cfg.cross_encoder_model)

    results: list[dict[str, Any]] = []
    for i, q in enumerate(subset):
        print(f"  LME [{i + 1}/{len(subset)}] {q['question_type']}", end="", flush=True)
        t0 = time.time()
        r = run_instance(
            q,
            data_dir=Path(tempfile.mkdtemp()),
            n_results=5,
            embedder=embedder,
            reranker=reranker,
        )
        elapsed = time.time() - t0
        # Add answer_session_ids for scoring
        r["answer_session_ids"] = q.get("answer_session_ids", [])
        r["question_type"] = q.get("question_type", "unknown")
        results.append(r)
        print(f" [{elapsed:.1f}s]", flush=True)
    return results


def _run_locomo(subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run LoCoMo subset through Myelin recall pipeline."""
    from sentence_transformers import SentenceTransformer

    from benchmarks.locomo.run import run_conversation
    from myelin.config import MyelinSettings
    from myelin.reranker import Neocortex

    cfg = MyelinSettings(data_dir=Path(tempfile.mkdtemp()))
    embedder = SentenceTransformer(cfg.embedding_model)
    reranker = Neocortex(model_name=cfg.cross_encoder_model)

    all_results: list[dict[str, Any]] = []
    for sample in subset:
        sid = sample.get("sample_id", "unknown")
        n_qa = len(sample["qa"])
        print(f"  LoCoMo {sid}: {n_qa} QA", end="", flush=True)
        t0 = time.time()
        results = run_conversation(
            sample, n_results=10, embedder=embedder, reranker=reranker
        )
        elapsed = time.time() - t0
        all_results.extend(results)
        print(f" [{elapsed:.1f}s]", flush=True)
    return all_results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_lme(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Score LongMemEval results: R@1, R@3, R@5 overall + per type."""
    from benchmarks.longmemeval.score import _unique_session_ids, recall_at_k

    overall: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        qtype = r.get("question_type", "unknown")
        answer_ids = set(r.get("answer_session_ids", []))
        ranked = _unique_session_ids(r.get("ranked", []))
        for k in (1, 3, 5):
            val = recall_at_k(ranked, answer_ids, k)
            overall[f"R@{k}"].append(val)
            by_type[qtype][f"R@{k}"].append(val)

    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "n": len(results),
        "overall": {m: round(_avg(v), 4) for m, v in sorted(overall.items())},
        "by_type": {
            t: {m: round(_avg(v), 4) for m, v in sorted(metrics.items())}
            for t, metrics in sorted(by_type.items())
        },
    }


def _score_locomo(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Score LoCoMo results: R@5, R@10 overall + per category."""
    from benchmarks.locomo.run import CATEGORIES, evidence_to_session_ids

    overall: dict[str, list[float]] = defaultdict(list)
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        cat_name = CATEGORIES.get(r.get("category", 0), "unknown")
        ev = evidence_to_session_ids(r.get("evidence", []))
        retrieved = r.get("retrieved_ids", [])
        for k in (5, 10):
            if not ev:
                val = 1.0
            else:
                found = len(ev & set(retrieved[:k]))
                val = found / len(ev)
            overall[f"R@{k}"].append(val)
            by_cat[cat_name][f"R@{k}"].append(val)

    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "n": len(results),
        "overall": {m: round(_avg(v), 4) for m, v in sorted(overall.items())},
        "by_category": {
            c: {m: round(_avg(v), 4) for m, v in sorted(metrics.items())}
            for c, metrics in sorted(by_cat.items())
        },
    }


# ---------------------------------------------------------------------------
# Per-question diff
# ---------------------------------------------------------------------------

_REGRESSION_THRESHOLD = 0.02  # 2pp drop at category level triggers warning


def _diff_scores(
    label: str,
    baseline: dict[str, Any],
    current: dict[str, Any],
    breakdown_key: str,
) -> tuple[bool, list[str]]:
    """Compare current vs baseline scores, return (passed, messages)."""
    messages: list[str] = []
    passed = True

    # Overall comparison
    for metric in sorted(
        set(baseline.get("overall", {})) | set(current.get("overall", {}))
    ):
        b_val = baseline.get("overall", {}).get(metric, 0.0)
        c_val = current.get("overall", {}).get(metric, 0.0)
        delta = c_val - b_val
        marker = "+" if delta >= 0 else ""
        status = "OK" if delta >= -_REGRESSION_THRESHOLD else "REGRESSED"
        if status == "REGRESSED":
            passed = False
        messages.append(
            f"  {label} {metric}: {b_val:.1%} -> {c_val:.1%}"
            f" ({marker}{delta:.1%}) [{status}]"
        )

    # Per-category comparison
    b_cats = baseline.get(breakdown_key, {})
    c_cats = current.get(breakdown_key, {})
    for cat in sorted(set(b_cats) | set(c_cats)):
        for metric in sorted(set(b_cats.get(cat, {})) | set(c_cats.get(cat, {}))):
            b_val = b_cats.get(cat, {}).get(metric, 0.0)
            c_val = c_cats.get(cat, {}).get(metric, 0.0)
            delta = c_val - b_val
            if delta < -_REGRESSION_THRESHOLD:
                passed = False
                messages.append(
                    f"  {label} {cat}/{metric}:"
                    f" {b_val:.1%} -> {c_val:.1%} ({delta:+.1%}) [REGRESSED]"
                )
            elif delta > _REGRESSION_THRESHOLD:
                messages.append(
                    f"  {label} {cat}/{metric}:"
                    f" {b_val:.1%} -> {c_val:.1%} ({delta:+.1%}) [IMPROVED]"
                )

    return passed, messages


# ---------------------------------------------------------------------------
# Per-question flip tracking
# ---------------------------------------------------------------------------


def _track_lme_flips(
    baseline_results: list[dict[str, Any]],
    current_results: list[dict[str, Any]],
) -> list[str]:
    """Track individual questions that flipped hit↔miss at R@5."""
    from benchmarks.longmemeval.score import _unique_session_ids

    baseline_by_qid = {r["question_id"]: r for r in baseline_results}
    messages: list[str] = []
    gained = 0
    lost = 0

    for r in current_results:
        qid = r["question_id"]
        b = baseline_by_qid.get(qid)
        if b is None:
            continue
        answer_ids = set(r.get("answer_session_ids", []))
        b_ranked = _unique_session_ids(b.get("ranked", []))
        c_ranked = _unique_session_ids(r.get("ranked", []))
        b_hit = bool(set(b_ranked[:5]) & answer_ids)
        c_hit = bool(set(c_ranked[:5]) & answer_ids)
        if b_hit and not c_hit:
            lost += 1
            messages.append(f"    LOST: {qid} ({r.get('question_type', '')})")
        elif not b_hit and c_hit:
            gained += 1
            messages.append(f"    GAINED: {qid} ({r.get('question_type', '')})")

    if gained or lost:
        messages.insert(0, f"  LME R@5 flips: +{gained} gained, -{lost} lost")
    return messages


def _track_locomo_flips(
    baseline_results: list[dict[str, Any]],
    current_results: list[dict[str, Any]],
) -> list[str]:
    """Track LoCoMo questions that flipped perfect↔imperfect at R@10."""
    from benchmarks.locomo.run import evidence_to_session_ids

    # Key by (sample_id, question) since LoCoMo has no unique question ID
    def _key(r: dict[str, Any]) -> str:
        return f"{r.get('sample_id', '')}|{r.get('question', '')[:80]}"

    baseline_by_key = {_key(r): r for r in baseline_results}
    messages: list[str] = []
    gained = 0
    lost = 0

    for r in current_results:
        k = _key(r)
        b = baseline_by_key.get(k)
        if b is None:
            continue
        ev = evidence_to_session_ids(r.get("evidence", []))
        if not ev:
            continue
        b_hit = ev <= set(b.get("retrieved_ids", [])[:10])
        c_hit = ev <= set(r.get("retrieved_ids", [])[:10])
        if b_hit and not c_hit:
            lost += 1
        elif not b_hit and c_hit:
            gained += 1

    if gained or lost:
        messages.append(f"  LoCoMo R@10 flips: +{gained} gained, -{lost} lost")
    return messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run regression gate or create baseline."""
    create_baseline = "--create-baseline" in sys.argv

    # Load data
    lme_path = _DATA_DIR / "longmemeval" / "data" / "longmemeval_s_cleaned.json"
    locomo_path = _DATA_DIR / "locomo" / "data" / "locomo10.json"

    if not lme_path.exists() or not locomo_path.exists():
        print("ERROR: Missing benchmark data files.", file=sys.stderr)
        print(f"  LME: {lme_path} (exists={lme_path.exists()})", file=sys.stderr)
        print(
            f"  LoCoMo: {locomo_path} (exists={locomo_path.exists()})", file=sys.stderr
        )
        sys.exit(1)

    lme_data = json.loads(lme_path.read_text())
    locomo_data = json.loads(locomo_path.read_text())

    lme_subset = _select_lme_subset(lme_data)
    locomo_subset = _select_locomo_subset(locomo_data)

    print(f"\n{'=' * 60}", flush=True)
    print("  Myelin Regression Gate", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  LME subset:    {len(lme_subset)} questions", flush=True)
    locomo_qa = sum(len(s["qa"]) for s in locomo_subset)
    print(f"  LoCoMo subset: {len(locomo_subset)} convs, {locomo_qa} QA", flush=True)
    print(f"{'─' * 60}", flush=True)

    t0 = time.time()

    print("\nRunning LongMemEval subset...", flush=True)
    lme_results = _run_lme(lme_subset)
    lme_scores = _score_lme(lme_results)

    print("\nRunning LoCoMo subset...", flush=True)
    locomo_results = _run_locomo(locomo_subset)
    locomo_scores = _score_locomo(locomo_results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s", flush=True)

    if create_baseline:
        _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        (_BASELINE_DIR / "lme_results.json").write_text(
            json.dumps(lme_results, indent=2)
        )
        (_BASELINE_DIR / "lme_scores.json").write_text(json.dumps(lme_scores, indent=2))
        (_BASELINE_DIR / "locomo_results.json").write_text(
            json.dumps(locomo_results, indent=2)
        )
        (_BASELINE_DIR / "locomo_scores.json").write_text(
            json.dumps(locomo_scores, indent=2)
        )
        print(f"\nBaseline saved to {_BASELINE_DIR}/", flush=True)
        _print_scores(lme_scores, locomo_scores)
        return

    # Compare against baseline
    if not _BASELINE_DIR.exists():
        print("\nNo baseline found. Run with --create-baseline first.", file=sys.stderr)
        _print_scores(lme_scores, locomo_scores)
        sys.exit(1)

    b_lme_scores = json.loads((_BASELINE_DIR / "lme_scores.json").read_text())
    b_locomo_scores = json.loads((_BASELINE_DIR / "locomo_scores.json").read_text())
    b_lme_results = json.loads((_BASELINE_DIR / "lme_results.json").read_text())
    b_locomo_results = json.loads((_BASELINE_DIR / "locomo_results.json").read_text())

    print(f"\n{'─' * 60}", flush=True)
    print("  Regression Report", flush=True)
    print(f"{'─' * 60}", flush=True)

    lme_ok, lme_msgs = _diff_scores("LME", b_lme_scores, lme_scores, "by_type")
    loc_ok, loc_msgs = _diff_scores(
        "LoCoMo", b_locomo_scores, locomo_scores, "by_category"
    )
    flip_msgs = _track_lme_flips(b_lme_results, lme_results)
    loc_flip_msgs = _track_locomo_flips(b_locomo_results, locomo_results)

    for msg in lme_msgs + flip_msgs + loc_msgs + loc_flip_msgs:
        print(msg, flush=True)

    print(f"\n{'─' * 60}", flush=True)
    if lme_ok and loc_ok:
        print("  PASSED — no regressions detected", flush=True)
    else:
        print("  FAILED — regressions detected (>2pp drop in a category)", flush=True)
        sys.exit(1)


def _print_scores(lme: dict[str, Any], locomo: dict[str, Any]) -> None:
    """Print current scores summary."""
    print(f"\n  LME (n={lme['n']}): {lme['overall']}", flush=True)
    for t, m in sorted(lme.get("by_type", {}).items()):
        print(f"    {t}: {m}", flush=True)
    print(f"\n  LoCoMo (n={locomo['n']}): {locomo['overall']}", flush=True)
    for c, m in sorted(locomo.get("by_category", {}).items()):
        print(f"    {c}: {m}", flush=True)


if __name__ == "__main__":
    main()
