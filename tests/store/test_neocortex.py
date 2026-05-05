"""Tests for neocortex.py — semantic memory network."""

from pathlib import Path
from tempfile import TemporaryDirectory

from myelin.store.neocortex import SemanticNetwork


def _make_network() -> SemanticNetwork:
    """Create an in-memory semantic network for testing."""
    return SemanticNetwork(db_path=Path(":memory:"))


class TestEntityOperations:
    def test_add_and_get_entity(self) -> None:
        net = _make_network()
        net.add_entity("jwt", entity_type="technology")
        e = net.get_entity("jwt")
        assert e is not None
        assert e["name"] == "jwt"
        assert e["entity_type"] == "technology"

    def test_entity_case_insensitive(self) -> None:
        net = _make_network()
        net.add_entity("JWT")
        assert net.get_entity("jwt") is not None
        assert net.get_entity("JWT") is not None

    def test_entity_count(self) -> None:
        net = _make_network()
        assert net.entity_count() == 0
        net.add_entity("auth")
        net.add_entity("jwt")
        assert net.entity_count() == 2

    def test_add_entity_idempotent(self) -> None:
        net = _make_network()
        net.add_entity("auth")
        net.add_entity("auth")
        assert net.entity_count() == 1

    def test_specific_type_wins_over_concept(self) -> None:
        """A specific entity_type should overwrite the default 'concept' type."""
        net = _make_network()
        net.add_entity("alice")  # inserted as "concept" (default)
        net.add_entity("alice", entity_type="person")  # should update type
        entity = net.get_entity("alice")
        assert entity is not None
        assert entity["entity_type"] == "person"

    def test_concept_does_not_overwrite_specific_type(self) -> None:
        """A later 'concept' registration must not demote an explicit type."""
        net = _make_network()
        net.add_entity("kafka", entity_type="technology")
        net.add_entity("kafka")  # called by add_relationship internally
        entity = net.get_entity("kafka")
        assert entity is not None
        assert entity["entity_type"] == "technology"

        net = _make_network()
        assert net.get_entity("nope") is None


class TestRelationshipOperations:
    def test_add_and_get_relationship(self) -> None:
        net = _make_network()
        net.add_relationship("kai", "auth", predicate="works_on")
        rels = net.get_relationships("kai")
        assert len(rels) == 1
        assert rels[0]["subject"] == "kai"
        assert rels[0]["object"] == "auth"
        assert rels[0]["predicate"] == "works_on"

    def test_co_occurrence_strengthens(self) -> None:
        net = _make_network()
        net.add_relationship("auth", "jwt")
        net.add_relationship("auth", "jwt")
        rels = net.get_relationships("auth")
        assert len(rels) == 1
        assert rels[0]["weight"] == 2.0

    def test_relationship_bidirectional_lookup(self) -> None:
        net = _make_network()
        net.add_relationship("kai", "auth")
        # Looking up "auth" should also find the relationship
        rels = net.get_relationships("auth")
        assert len(rels) == 1

    def test_filter_by_predicate(self) -> None:
        net = _make_network()
        net.add_relationship("kai", "auth", predicate="works_on")
        net.add_relationship("kai", "jwt", predicate="co_occurs")
        rels = net.get_relationships("kai", predicate="works_on")
        assert len(rels) == 1
        assert rels[0]["object"] == "auth"

    def test_relationship_count(self) -> None:
        net = _make_network()
        assert net.relationship_count() == 0
        net.add_relationship("a", "b")
        net.add_relationship("b", "c")
        assert net.relationship_count() == 2

    def test_temporal_validity(self) -> None:
        net = _make_network()
        net.add_relationship("kai", "auth", valid_from="2025-01-01")
        rels = net.get_relationships("kai")
        assert rels[0]["valid_from"] == "2025-01-01"
        assert rels[0]["valid_to"] is None


class TestSpreadingActivation:
    def test_spread_finds_neighbors(self) -> None:
        net = _make_network()
        net.add_relationship("auth", "jwt", weight=2.0)
        net.add_relationship("auth", "oauth", weight=3.0)
        net.add_relationship("jwt", "security", weight=2.0)

        results = net.spread(["auth"], min_weight=0.1)
        names = [r[0] for r in results]
        assert "jwt" in names
        assert "oauth" in names

    def test_spread_empty_network(self) -> None:
        net = _make_network()
        results = net.spread(["auth"])
        assert results == []

    def test_spread_excludes_seeds(self) -> None:
        net = _make_network()
        net.add_relationship("auth", "jwt")
        results = net.spread(["auth"], min_weight=0.01)
        names = [r[0] for r in results]
        assert "auth" not in names

    def test_spread_respects_depth(self) -> None:
        net = _make_network()
        net.add_relationship("a", "b", weight=3.0)
        net.add_relationship("b", "c", weight=3.0)
        net.add_relationship("c", "d", weight=3.0)

        # Depth 2: seeds processed → neighbors discovered → 1-hop visited
        results_d2 = net.spread(["a"], max_depth=2, min_weight=0.01)
        names_d2 = [r[0] for r in results_d2]
        assert "b" in names_d2

        # Depth 3 should reach further
        results_d3 = net.spread(["a"], max_depth=3, min_weight=0.01)
        names_d3 = [r[0] for r in results_d3]
        assert "b" in names_d3
        assert "c" in names_d3

    def test_spread_top_k(self) -> None:
        net = _make_network()
        for i in range(20):
            net.add_relationship("root", f"node_{i}", weight=3.0)
        results = net.spread(["root"], top_k=5, min_weight=0.01)
        assert len(results) <= 5


class TestInvalidation:
    def test_invalidate_sets_valid_to(self) -> None:
        net = _make_network()
        net.add_relationship("kai", "auth", predicate="works_on")
        ok = net.invalidate("kai", "works_on", "auth", ended="2025-06-01")
        assert ok
        rels = net.get_relationships("kai")
        assert rels[0]["valid_to"] == "2025-06-01"

    def test_invalidate_nonexistent_returns_false(self) -> None:
        net = _make_network()
        ok = net.invalidate("nobody", "works_on", "nothing")
        assert not ok


class TestClear:
    def test_clear_removes_everything(self) -> None:
        net = _make_network()
        net.add_relationship("a", "b")
        net.add_relationship("c", "d")
        net.clear()
        assert net.entity_count() == 0
        assert net.relationship_count() == 0


class TestPersistence:
    def test_survives_reopen(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            net = SemanticNetwork(db_path=db_path)
            net.add_relationship("auth", "jwt")
            net.close()

            net2 = SemanticNetwork(db_path=db_path)
            assert net2.entity_count() == 2
            assert net2.relationship_count() == 1
            net2.close()
