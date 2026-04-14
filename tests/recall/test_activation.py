"""Tests for the Hebbian reinforcement tracker."""

from myelin.models import Memory, RecallResult
from myelin.recall.activation import HebbianTracker


class TestHebbianTracker:
    def _make_result(self, id_: str, score: float) -> RecallResult:
        return RecallResult(memory=Memory(id=id_, content=f"memory {id_}"), score=score)

    def test_reinforce_creates_edges(self, hebbian_tracker: HebbianTracker) -> None:
        hebbian_tracker.reinforce(["a", "b", "c"])

        rows = list(hebbian_tracker.db.execute("SELECT * FROM co_access").fetchall())
        assert len(rows) == 3  # (a,b), (a,c), (b,c)

    def test_reinforce_increments_weight(self, hebbian_tracker: HebbianTracker) -> None:
        hebbian_tracker.reinforce(["a", "b"])
        hebbian_tracker.reinforce(["a", "b"])

        row = hebbian_tracker.db.execute(
            "SELECT weight FROM co_access WHERE id_a = ? AND id_b = ?", ["a", "b"]
        ).fetchone()
        assert row[0] > 0.1  # should be 0.2 (two reinforcements)

    def test_boost_reranks_results(self, hebbian_tracker: HebbianTracker) -> None:
        # Reinforce a-c many times so c gets a meaningful multiplicative boost
        for _ in range(20):
            hebbian_tracker.reinforce(["a", "c"])

        # c starts slightly below b — Hebbian co-access should tip the balance
        results = [
            self._make_result("a", 0.9),
            self._make_result("b", 0.85),
            self._make_result("c", 0.84),
        ]
        boosted = hebbian_tracker.boost(results)
        ids = [r.memory.id for r in boosted]
        # c should beat b thanks to Hebbian boost from co-access with a
        assert ids.index("c") < ids.index("b")

    def test_cleanup_removes_stale(self, hebbian_tracker: HebbianTracker) -> None:
        hebbian_tracker.reinforce(["a", "b", "c"])
        removed = hebbian_tracker.cleanup(valid_ids={"a", "b"})
        assert removed == 2  # (a,c) and (b,c) removed

        rows = list(hebbian_tracker.db.execute("SELECT * FROM co_access").fetchall())
        assert len(rows) == 1  # only (a,b) remains
