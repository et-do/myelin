"""Tests for prefrontal.py — PFC schema-consistent encoding."""

from myelin.store.prefrontal import classify, classify_memory_type


class TestClassify:
    def test_decision_markers(self) -> None:
        match = classify("We decided to use JWT for all API authentication")
        assert match is not None
        assert match.schema_id == "decision"
        assert match.memory_type == "semantic"

    def test_preference_markers(self) -> None:
        text = "The team prefers small PRs. We should always use conventional commits."
        match = classify(text)
        assert match is not None
        assert match.schema_id == "preference"
        assert match.memory_type == "procedural"

    def test_procedure_markers(self) -> None:
        match = classify("Step 1: run the migration script. Step 2: verify the schema.")
        assert match is not None
        assert match.schema_id == "procedure"
        assert match.memory_type == "procedural"

    def test_plan_markers(self) -> None:
        match = classify("TODO: fix the auth bug by Friday")
        assert match is not None
        assert match.schema_id == "plan"
        assert match.memory_type == "prospective"

    def test_event_markers(self) -> None:
        match = classify("Yesterday we debugged the OAuth crash for three hours")
        assert match is not None
        assert match.schema_id == "event"
        assert match.memory_type == "episodic"

    def test_no_match_returns_none(self) -> None:
        match = classify("Hello world")
        assert match is None

    def test_best_match_wins(self) -> None:
        # Multiple schemas could match; highest confidence wins
        text = (
            "We decided to switch to RS256. Let's go with that. "
            "After discussion, agreed on RS256."
        )
        match = classify(text)
        assert match is not None
        assert match.schema_id == "decision"
        assert match.confidence > 0.3

    def test_case_insensitive(self) -> None:
        match = classify("WE DECIDED TO USE JWT")
        assert match is not None
        assert match.schema_id == "decision"


class TestClassifyMemoryType:
    def test_decision_returns_semantic(self) -> None:
        assert classify_memory_type("We decided to use Postgres") == "semantic"

    def test_preference_returns_procedural(self) -> None:
        assert classify_memory_type("Team prefers to use TypeScript") == "procedural"

    def test_plan_returns_prospective(self) -> None:
        assert classify_memory_type("TODO: fix the auth bug by Friday") == "prospective"

    def test_unknown_defaults_to_episodic(self) -> None:
        assert classify_memory_type("Hello world") == "episodic"
