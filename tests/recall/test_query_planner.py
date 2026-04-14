"""Tests for query_planner.py — PFC inhibitory gating for recall."""

from myelin.recall.query_planner import plan


class TestPlan:
    def test_decision_query_infers_semantic(self) -> None:
        p = plan("what did we decide about the database?")
        assert p.memory_type == "semantic"
        assert p.scope_hint == "database"

    def test_preference_query_infers_procedural(self) -> None:
        p = plan("what does the team prefer for code style?")
        assert p.memory_type == "procedural"

    def test_temporal_query_infers_episodic(self) -> None:
        p = plan("what happened last week with the deployment?")
        assert p.memory_type == "episodic"
        assert p.scope_hint is not None  # should detect "deployment"

    def test_plan_query_infers_prospective(self) -> None:
        p = plan("what should we do next for the API?")
        assert p.memory_type == "prospective"
        assert p.scope_hint == "api"

    def test_auth_scope_detected(self) -> None:
        p = plan("how does our OAuth implementation work?")
        assert p.scope_hint == "oauth"

    def test_database_scope_detected(self) -> None:
        p = plan("tell me about the postgres migration")
        assert p.scope_hint is not None
        assert "postgres" in p.scope_hint

    def test_no_signals_returns_empty_plan(self) -> None:
        p = plan("hello")
        assert p.memory_type is None
        assert p.scope_hint is None
        assert len(p.signals) == 0

    def test_security_scope(self) -> None:
        p = plan("what are our encryption settings?")
        assert p.scope_hint == "encryption"

    def test_deploy_scope(self) -> None:
        p = plan("how do we deploy to kubernetes?")
        assert p.scope_hint is not None

    def test_signals_dict_populated(self) -> None:
        p = plan("what did we decide about auth?")
        assert "type_semantic" in p.signals
        assert "scope_detected" in p.signals
