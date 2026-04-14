"""Tests for consolidation.py — hippocampal replay engine."""

from pathlib import Path

from myelin.store.consolidation import ConsolidationResult, extract_entities, replay
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
