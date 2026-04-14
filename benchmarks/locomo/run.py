"""Run Myelin against LoCoMo and produce a results file for evaluation.

LoCoMo: 10 conversations x ~200 QA pairs = 1,986 questions across 5 categories.
Each conversation has 19-32 sessions between two speakers.

Usage:
    uv run python -m benchmarks.locomo.run \
        benchmarks/locomo/data/locomo10.json \
        benchmarks/locomo/output/myelin_locomo.json

    # With limited conversations for quick testing:
    uv run python -m benchmarks.locomo.run \
        benchmarks/locomo/data/locomo10.json \
        benchmarks/locomo/output/myelin_locomo.json --limit 1
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path
from typing import Any

# Load .env before any HF/torch imports
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from sentence_transformers import SentenceTransformer  # noqa: E402

from myelin.config import MyelinSettings  # noqa: E402
from myelin.models import MemoryMetadata  # noqa: E402
from myelin.reranker import Neocortex  # noqa: E402
from myelin.store.consolidation import extract_entities  # noqa: E402
from myelin.store.hippocampus import Hippocampus  # noqa: E402
from myelin.store.neocortex import SemanticNetwork  # noqa: E402

# LoCoMo QA category labels
CATEGORIES = {
    1: "single-hop",
    2: "temporal",
    3: "temporal-inference",
    4: "open-domain",
    5: "adversarial",
}


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------


def load_conversation_sessions(
    conversation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract sessions from a LoCoMo conversation dict.

    Returns list of {session_num, date, dialogs} dicts.
    """
    sessions: list[dict[str, Any]] = []
    session_num = 1
    while True:
        key = f"session_{session_num}"
        if key not in conversation:
            break
        sessions.append(
            {
                "session_num": session_num,
                "date": conversation.get(f"session_{session_num}_date_time", ""),
                "dialogs": conversation[key],
            }
        )
        session_num += 1
    return sessions


def flatten_session(session: dict[str, Any]) -> str:
    """Flatten a LoCoMo session dict into a single text block.

    Each dialog turn becomes 'Speaker: text'.
    """
    lines: list[str] = []
    for d in session["dialogs"]:
        speaker = d.get("speaker", "?")
        text = d.get("text", "")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def evidence_to_session_ids(evidence: list[str]) -> set[str]:
    """Convert LoCoMo evidence dialog IDs to session IDs.

    Evidence format: 'D3:5' means session 3, dialog 5.
    Session-level recall checks whether session_3 was retrieved.
    """
    sessions: set[str] = set()
    for eid in evidence:
        # D<session>:<dialog>
        if eid.startswith("D") and ":" in eid:
            sess_num = eid[1:].split(":")[0]
            sessions.add(f"session_{sess_num}")
    return sessions


def _build_semantic_network(
    session_texts: list[str],
    cfg: MyelinSettings,
) -> SemanticNetwork:
    """Build a SemanticNetwork from entity co-occurrence across sessions."""
    neo_path = Path(tempfile.mktemp(suffix=".db"))
    neo = SemanticNetwork(db_path=neo_path, cfg=cfg)
    for text in session_texts:
        entities = extract_entities(text)
        for e in entities:
            neo.add_entity(e, entity_type="auto")
        for a, b in combinations(entities, 2):
            neo.add_relationship(a, b, predicate="co_occurs", weight=1.0)
    return neo


# ------------------------------------------------------------------
# Instance runner
# ------------------------------------------------------------------


def run_conversation(
    sample: dict[str, Any],
    n_results: int,
    embedder: SentenceTransformer | None,
    reranker: Neocortex | None,
) -> list[dict[str, Any]]:
    """Run all QA pairs for a single LoCoMo conversation.

    For each conversation:
    1. Store all sessions in a fresh ephemeral Hippocampus
    2. Build a SemanticNetwork from entity co-occurrence
    3. For each QA pair, recall and record results

    Returns a list of result dicts, one per QA pair.
    """
    sample_id = sample.get("sample_id", "unknown")
    conversation = sample["conversation"]
    qa_pairs = sample["qa"]

    sessions = load_conversation_sessions(conversation)

    # Flatten all session texts for storage
    all_texts: list[str] = []
    for sess in sessions:
        all_texts.append(flatten_session(sess))

    # Build config and semantic network
    cfg = MyelinSettings(data_dir=Path(tempfile.mkdtemp(prefix="myelin_locomo_")))
    neo = _build_semantic_network(all_texts, cfg)

    # Create ephemeral Hippocampus
    hc = Hippocampus(
        cfg=cfg,
        embedder=embedder,
        ephemeral=True,
        reranker=reranker,
        semantic_network=neo,
    )

    # Store each session
    for sess, text in zip(sessions, all_texts):
        metadata = MemoryMetadata(
            scope=f"session_{sess['session_num']}",
            tags=[sess["date"]] if sess["date"] else [],
        )
        hc.store(text, metadata)

    n_stored = hc.count()

    # Run each QA pair
    results: list[dict[str, Any]] = []
    for qi, qa in enumerate(qa_pairs):
        if qi % 25 == 0:
            print(".", end="", flush=True)
        question = qa["question"]
        answer = qa.get("answer", qa.get("adversarial_answer", ""))
        category = qa["category"]
        evidence = qa.get("evidence", [])

        # Recall
        recall_results = hc.recall(question, n_results=n_results)

        # Build hypothesis
        if recall_results:
            hypothesis = "\n---\n".join(r.memory.content for r in recall_results)
        else:
            hypothesis = ""

        # Extract ranked session IDs from results
        ranked: list[dict[str, Any]] = []
        for r in recall_results:
            scope = r.memory.metadata.scope or ""
            sid = (
                scope.removeprefix("session_")
                if scope.startswith("session_")
                else scope
            )
            ranked.append(
                {
                    "session_id": f"session_{sid}" if sid else "",
                    "score": round(r.score, 4),
                }
            )

        results.append(
            {
                "sample_id": sample_id,
                "question": question,
                "answer": answer,
                "category": category,
                "evidence": evidence,
                "retrieved_ids": [r["session_id"] for r in ranked],
                "ranked": ranked,
                "hypothesis": hypothesis,
                "n_stored": n_stored,
            }
        )

    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main(
    data_file: str,
    output_file: str,
    n_results: int = 10,
    limit: int = 0,
) -> None:
    """Run the full LoCoMo benchmark."""
    print(f"\n{'=' * 60}", flush=True)
    print("  Myelin x LoCoMo Benchmark", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Load data
    with open(data_file) as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]

    print(f"  Data:          {Path(data_file).name}", flush=True)
    print(f"  Conversations: {len(data)}", flush=True)
    print(f"  Top-k:         {n_results}", flush=True)

    total_qa = sum(len(s["qa"]) for s in data)
    print(f"  Total QA:      {total_qa}", flush=True)
    print(f"{'─' * 60}", flush=True)

    # Load models once, reuse across conversations
    cfg = MyelinSettings(data_dir=Path(tempfile.mkdtemp()))
    print(f"\n  Loading embedding model: {cfg.embedding_model}", flush=True)
    embedder = SentenceTransformer(cfg.embedding_model)
    print(f"  Loading cross-encoder: {cfg.cross_encoder_model}", flush=True)
    reranker = Neocortex(model_name=cfg.cross_encoder_model)

    all_results: list[dict[str, Any]] = []
    start_time = time.time()

    for conv_idx, sample in enumerate(data):
        sample_id = sample.get("sample_id", f"conv-{conv_idx}")
        n_sess = sum(
            1
            for k in sample["conversation"]
            if k.startswith("session_") and not k.endswith("_date_time")
        )
        n_qa = len(sample["qa"])

        conv_start = time.time()
        print(
            f"  [{conv_idx + 1}/{len(data)}] {sample_id}: "
            f"{n_sess} sessions, {n_qa} QA — running...",
            end="",
            flush=True,
        )
        results = run_conversation(sample, n_results, embedder, reranker)
        conv_elapsed = time.time() - conv_start

        all_results.extend(results)

        # Quick stats for this conversation
        evidence_sets = [evidence_to_session_ids(r["evidence"]) for r in results]
        recalls = []
        for r, ev_set in zip(results, evidence_sets):
            if not ev_set:
                recalls.append(1.0)  # no evidence = vacuously true
                continue
            retrieved = set(r["retrieved_ids"])
            found = len(ev_set & retrieved)
            recalls.append(found / len(ev_set))

        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        perfect = sum(1 for r in recalls if r >= 1.0)

        print(
            f" recall={avg_recall:.3f} ({perfect}/{n_qa} perfect) "
            f"[{conv_elapsed:.1f}s]",
            flush=True,
        )

    elapsed = time.time() - start_time

    # Write results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(
        f"\n  Total time: {elapsed:.1f}s ({elapsed / max(len(all_results), 1):.2f}s/q)",
        flush=True,
    )
    print(f"  Results saved to: {output_file}", flush=True)
    print(f"{'=' * 60}\n", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Myelin x LoCoMo Benchmark")
    parser.add_argument("data_file", help="Path to locomo10.json")
    parser.add_argument("output_file", help="Output JSON file path")
    parser.add_argument(
        "--n-results",
        type=int,
        default=10,
        help="Top-k results per query (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to N conversations (0=all)",
    )
    args = parser.parse_args()

    if not Path(args.data_file).exists():
        print(f"Data file not found: {args.data_file}")
        print(
            "Download: git clone "
            "https://github.com/snap-research/locomo.git /tmp/locomo"
        )
        print("Then: cp /tmp/locomo/data/locomo10.json benchmarks/locomo/data/")
        sys.exit(1)

    main(args.data_file, args.output_file, args.n_results, args.limit)
