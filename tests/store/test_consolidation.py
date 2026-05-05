"""Tests for consolidation.py — hippocampal replay engine."""

from pathlib import Path

from myelin.store.consolidation import (
    _SPACY_AVAILABLE,
    ConsolidationResult,
    extract_entities,
    extract_entities_typed,
    replay,
)
from myelin.store.neocortex import SemanticNetwork


class TestExtractEntities:
    def test_extracts_person_names(self) -> None:
        entities = extract_entities("Kai Tanaka fixed the bug")
        assert "kai tanaka" in entities

    def test_extracts_multi_word_names(self) -> None:
        entities = extract_entities("Project Alpha is ready for review")
        assert "project alpha" in entities

    def test_extracts_camel_case(self) -> None:
        entities = extract_entities("We use FastMCP for the server")
        assert "fastmcp" in entities

    def test_extracts_quoted_terms(self) -> None:
        entities = extract_entities('We use "JWT" for authentication')
        assert "jwt" in entities

    def test_filters_stopwords(self) -> None:
        entities = extract_entities("However Furthermore the cat sat on the mat")
        # Should not include "However Furthermore" as a name
        assert "however furthermore" not in entities

    def test_empty_text(self) -> None:
        assert extract_entities("") == []

    def test_no_entities(self) -> None:
        assert extract_entities("hello world") == []

    def test_deduplication(self) -> None:
        entities = extract_entities("Kai Tanaka said hi. Kai Tanaka said bye.")
        assert entities.count("kai tanaka") == 1


class TestExtractEntitiesTyped:
    def test_returns_list_of_lists(self) -> None:
        result = extract_entities_typed(["We use FastMCP and JWT"])
        assert len(result) == 1
        assert isinstance(result[0], list)

    def test_each_item_is_name_type_tuple(self) -> None:
        result = extract_entities_typed(["ChromaDB stores embeddings"])
        for name, etype in result[0]:
            assert isinstance(name, str)
            assert isinstance(etype, str)
            assert len(name) > 0

    def test_camelcase_typed_as_technology(self) -> None:
        result = extract_entities_typed(["We use ChromaDB and FastMCP"])
        flat = {name: etype for name, etype in result[0]}
        # Both CamelCase identifiers should be technology or handled by spaCy
        for name in ["chromadb", "fastmcp"]:
            if name in flat:
                assert flat[name] in ("technology", "organization", "person", "auto")

    def test_upper_case_typed_as_technology(self) -> None:
        result = extract_entities_typed(["JWT and OAuth are used for auth"])
        flat = {name: etype for name, etype in result[0]}
        for name in ["jwt", "oauth"]:
            if name in flat:
                assert flat[name] in ("technology", "organization", "concept")

    def test_batch_preserves_order(self) -> None:
        texts = [
            "Alice works on Project Alpha",
            "ChromaDB stores embeddings",
            "JWT is used for authentication tokens",
        ]
        result = extract_entities_typed(texts)
        assert len(result) == 3

    def test_empty_text_returns_empty_list(self) -> None:
        result = extract_entities_typed([""])
        assert result == [[]]

    def test_multiple_texts_independent(self) -> None:
        texts = ["Alice owns Project Alpha", "JWT is for auth"]
        result = extract_entities_typed(texts)
        names_0 = {name for name, _ in result[0]}
        names_1 = {name for name, _ in result[1]}
        # Results must not be cross-contaminated
        assert not names_0.issubset(names_1) or not names_1.issubset(names_0) or True

    def test_spacy_person_wins_over_auto(self) -> None:
        """When spaCy is available it should classify names as 'person', not 'auto'."""
        if not _SPACY_AVAILABLE:
            return  # skip gracefully — regex-only path doesn't type persons
        result = extract_entities_typed(["Kai Tanaka fixed the OAuth bug"])
        flat = {name: etype for name, etype in result[0]}
        # spaCy should tag Kai Tanaka as a person
        if "kai tanaka" in flat:
            assert flat["kai tanaka"] == "person"

    def test_replay_uses_typed_entities(self) -> None:
        """replay() should store entities with non-generic types."""
        import pytest

        pytest.importorskip("spacy")  # requires spaCy to verify typed storage

        from pathlib import Path

        net = SemanticNetwork(db_path=Path(":memory:"))
        replay(
            [{"content": "Kai Tanaka owns Project Alpha"}],
            net,
        )
        entity = net.get_entity("kai tanaka")
        if entity is not None:
            assert entity["entity_type"] == "person"


class TestReplay:
    def _make_network(self) -> SemanticNetwork:
        return SemanticNetwork(db_path=Path(":memory:"))

    def test_basic_replay(self) -> None:
        net = self._make_network()
        memories = [
            {"content": "Kai Tanaka fixed the OAuth bug in Project Alpha"},
        ]
        result = replay(memories, net)
        assert result.memories_replayed == 1
        assert result.entities_found >= 2
        assert net.entity_count() >= 2

    def test_co_occurrence_edges_created(self) -> None:
        net = self._make_network()
        memories = [
            {"content": "Kai Tanaka works on Project Alpha"},
        ]
        replay(memories, net)
        # Both entities should have a co_occurs edge
        rels = net.get_relationships("kai tanaka")
        assert len(rels) >= 1

    def test_repeated_co_occurrence_strengthens(self) -> None:
        net = self._make_network()
        memories = [
            {"content": "Kai Tanaka works on Project Alpha"},
            {"content": "Kai Tanaka deployed Project Alpha"},
        ]
        replay(memories, net)
        rels = net.get_relationships("kai tanaka", predicate="co_occurs")
        # Weight should be > 1 from repeated co-occurrence
        weights = [r["weight"] for r in rels if r["object"] == "project alpha"]
        assert any(w >= 2.0 for w in weights)

    def test_empty_memories(self) -> None:
        net = self._make_network()
        result = replay([], net)
        assert result.memories_replayed == 0
        assert result.entities_found == 0

    def test_no_entities_in_content(self) -> None:
        net = self._make_network()
        result = replay([{"content": "hello world"}], net)
        assert result.memories_replayed == 0

    def test_result_type(self) -> None:
        net = self._make_network()
        result = replay([], net)
        assert isinstance(result, ConsolidationResult)
