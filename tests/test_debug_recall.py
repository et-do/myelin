"""Tests for do_debug_recall() and the debug-recall CLI command."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from myelin.config import MyelinSettings
from myelin.server import configure, do_debug_recall, do_store


@pytest.fixture(autouse=True)
def _isolate(tmp_settings: MyelinSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    configure(tmp_settings)
    monkeypatch.setattr("myelin.cli.settings", tmp_settings)


# ---------------------------------------------------------------------------
# do_debug_recall — unit tests
# ---------------------------------------------------------------------------


class TestDoDebugRecallEmpty:
    def test_returns_structure_on_empty_store(self) -> None:
        result = do_debug_recall("what auth approach did we pick?")
        assert result["query"] == "what auth approach did we pick?"
        assert result["memory_count"] == 0
        assert result["results"] == []
        assert "query_plan" in result
        assert "gate_check" in result
        assert "filters_applied" in result

    def test_query_plan_infers_memory_type(self) -> None:
        result = do_debug_recall("what did we decide about auth?")
        assert result["query_plan"]["memory_type"] == "semantic"

    def test_query_plan_infers_scope(self) -> None:
        result = do_debug_recall("how do we run database migrations?")
        qp = result["query_plan"]
        assert qp["scope_hint"] is not None
        assert "database" in qp["scope_hint"] or "db" in qp["scope_hint"]

    def test_no_filter_when_plan_empty(self) -> None:
        result = do_debug_recall("something about blue elephants")
        fa = result["filters_applied"]
        assert fa["project"] is None
        assert fa["language"] is None

    def test_gate_check_short_query(self) -> None:
        result = do_debug_recall("hi")
        gc = result["gate_check"]
        assert gc["would_store"] is False
        assert "too short" in gc["reason"]

    def test_gate_check_normal_query(self) -> None:
        result = do_debug_recall("what database approach did we choose for auth?")
        gc = result["gate_check"]
        # long enough to pass length check (no existing memories for dedup)
        assert gc["would_store"] is True
        assert gc["reason"] == "ok"


class TestDoDebugRecallWithMemories:
    def test_result_count_respects_n(self) -> None:
        for i in range(5):
            do_store(
                f"The project uses PostgreSQL for persistent storage of user data {i}"
            )
        result = do_debug_recall("database storage", n_results=3)
        assert len(result["results"]) <= 3

    def test_results_have_required_fields(self) -> None:
        do_store("We decided to use JWT with RS256 for the authentication service")
        result = do_debug_recall("JWT authentication decision", n_results=5)
        assert len(result["results"]) >= 1
        r = result["results"][0]
        assert "rank" in r
        assert "id" in r
        assert "final_score" in r
        assert "bi_encoder_sim" in r
        assert "ce_score" in r
        assert "hebbian_weight" in r
        assert "memory_type" in r
        assert "access_count" in r
        assert "content_preview" in r

    def test_rank_is_sequential(self) -> None:
        for i in range(3):
            do_store(
                "Memory about authentication and JWT tokens "
                f"approach {i} used in the service"
            )
        result = do_debug_recall("authentication JWT")
        ranks = [r["rank"] for r in result["results"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_bi_encoder_sim_is_in_range(self) -> None:
        do_store("We decided to use PostgreSQL as the primary database for the service")
        result = do_debug_recall("database PostgreSQL decision", n_results=5)
        for r in result["results"]:
            if r["bi_encoder_sim"] is not None:
                assert 0.0 <= r["bi_encoder_sim"] <= 1.1  # slight headroom for float

    def test_ce_score_present_when_reranker_enabled(self) -> None:
        do_store("We decided to use JWT with RS256 for the authentication service")
        result = do_debug_recall("JWT auth decision", n_results=5)
        assert len(result["results"]) >= 1
        # CE score should be present (reranker is enabled by default)
        r1 = result["results"][0]
        assert r1["ce_score"] is not None

    def test_hebbian_weight_zero_for_fresh_memories(self) -> None:
        do_store("We decided to use JWT with RS256 for the authentication service")
        do_store("Redis is used for session caching in the production environment")
        result = do_debug_recall("authentication caching")
        # Freshly stored memories have never been co-recalled, so weight is 0
        for r in result["results"]:
            assert r["hebbian_weight"] == 0.0

    def test_content_preview_truncated(self) -> None:
        long_content = "A" * 300 + " suffix that makes it long"
        do_store(long_content)
        result = do_debug_recall("A" * 50)
        if result["results"]:
            preview = result["results"][0]["content_preview"]
            assert len(preview) <= 155  # 150 chars + "..."

    def test_memory_count_reflects_store(self) -> None:
        do_store("First memory about deployment and CI/CD pipelines")
        do_store("Second memory about authentication and JWT tokens")
        result = do_debug_recall("deployment")
        assert result["memory_count"] >= 2

    def test_explicit_filters_appear_in_output(self) -> None:
        do_store(
            "JWT RS256 authentication decision",
            project="myapp",
            scope="auth",
        )
        result = do_debug_recall("JWT authentication", project="myapp", scope="auth")
        fa = result["filters_applied"]
        assert fa["project"] == "myapp"
        assert fa["scope"] == "auth"
        # When explicit scope is given, auto_scope should be None
        assert fa["auto_scope"] is None

    def test_auto_filters_when_no_explicit(self) -> None:
        do_store("We use Redis for session caching in production")
        result = do_debug_recall("how does authentication work?")
        fa = result["filters_applied"]
        # auto filters are set from query plan (may be None if not inferred)
        assert "auto_memory_type" in fa
        assert "auto_scope" in fa


# ---------------------------------------------------------------------------
# Hebbian weight appears after co-recall
# ---------------------------------------------------------------------------


class TestDebugRecallHebbianWeight:
    def test_hebbian_weight_positive_after_co_recall(self) -> None:
        """After two memories are co-recalled, Hebbian weight should show > 0."""
        from myelin.server import do_recall

        do_store("We use JWT RS256 tokens for authentication in the service")
        do_store("The auth service validates tokens on every API request")

        # Co-recall to build Hebbian links
        do_recall("JWT authentication token validation")
        do_recall("JWT authentication token validation")

        # Now debug-recall should show non-zero Hebbian weight
        result = do_debug_recall("JWT authentication", n_results=5)
        weights = [r["hebbian_weight"] for r in result["results"]]
        # At least one result should have non-zero weight after co-recall
        assert any(w > 0 for w in weights), (
            f"expected non-zero Hebbian weight, got {weights}"
        )


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


class TestCmdDebugRecall:
    def test_formatted_output(
        self, capsys: pytest.CaptureFixture[str], tmp_settings: MyelinSettings
    ) -> None:
        configure(tmp_settings)
        do_store("We decided to use JWT with RS256 for the authentication service")
        with patch("sys.argv", ["myelin", "debug-recall", "JWT auth decision"]):
            from myelin.cli import main

            main()
        out = capsys.readouterr().out
        assert "QUERY PLAN" in out
        assert "AMYGDALA GATE" in out
        assert "RESULTS" in out

    def test_json_flag(
        self, capsys: pytest.CaptureFixture[str], tmp_settings: MyelinSettings
    ) -> None:
        configure(tmp_settings)
        do_store("We decided to use JWT with RS256 for the authentication service")
        with patch(
            "sys.argv", ["myelin", "debug-recall", "JWT auth decision", "--json"]
        ):
            from myelin.cli import main

            main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "query" in data
        assert "results" in data
        assert "query_plan" in data

    def test_no_results_does_not_crash(
        self, capsys: pytest.CaptureFixture[str], tmp_settings: MyelinSettings
    ) -> None:
        configure(tmp_settings)
        with patch("sys.argv", ["myelin", "debug-recall", "anything at all here"]):
            from myelin.cli import main

            main()
        out = capsys.readouterr().out
        assert "No results" in out

    def test_n_flag_limits_results(
        self, capsys: pytest.CaptureFixture[str], tmp_settings: MyelinSettings
    ) -> None:
        configure(tmp_settings)
        for i in range(5):
            do_store(f"JWT authentication approach used by the service number {i}")
        with patch(
            "sys.argv", ["myelin", "debug-recall", "JWT auth", "-n", "2", "--json"]
        ):
            from myelin.cli import main

            main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data["results"]) <= 2

    def test_project_filter_flag(
        self, capsys: pytest.CaptureFixture[str], tmp_settings: MyelinSettings
    ) -> None:
        configure(tmp_settings)
        do_store("JWT RS256 auth approach", project="myapp")
        with patch(
            "sys.argv",
            [
                "myelin",
                "debug-recall",
                "JWT authentication",
                "--project",
                "myapp",
                "--json",
            ],
        ):
            from myelin.cli import main

            main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["filters_applied"]["project"] == "myapp"
