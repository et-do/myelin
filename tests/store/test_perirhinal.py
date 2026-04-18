"""Tests for perirhinal.py — gist extraction and summary index."""

from __future__ import annotations

from myelin.store.perirhinal import SummaryIndex, summarise


class TestSummarise:
    def test_extracts_signal_sentences(self) -> None:
        text = (
            "user: Hi there!\n"
            "assistant: Hello! How can I help?\n"
            "user: I need to configure JWT authentication for my API.\n"
            "assistant: You should use RS256 for production. "
            "Always rotate keys every 90 days. "
            "Set the token expiry to 15 minutes.\n"
            "user: Thanks!"
        )
        summary = summarise(text, max_sentences=3)
        # Should capture the high-signal content, not greetings
        assert "JWT" in summary or "RS256" in summary or "token" in summary.lower()
        assert "hello" not in summary.lower()

    def test_short_text_returned_intact(self) -> None:
        text = "The server uses PostgreSQL for persistent storage."
        summary = summarise(text)
        assert "PostgreSQL" in summary

    def test_empty_text(self) -> None:
        assert summarise("") == ""
        assert summarise("   ") == ""

    def test_strips_role_prefixes(self) -> None:
        text = "user: What is Docker?\nassistant: Docker is a container platform."
        summary = summarise(text)
        # Role prefixes should be stripped
        assert not summary.startswith("user:")
        assert not summary.startswith("assistant:")

    def test_respects_max_sentences(self) -> None:
        # Build text with many sentences
        sentences = [f"Sentence number {i} is about topic {i}." for i in range(20)]
        text = " ".join(sentences)
        summary = summarise(text, max_sentences=3)
        # Should have at most 3 sentences
        # Count by period-space splits as a rough heuristic
        parts = [s.strip() for s in summary.split(". ") if s.strip()]
        assert len(parts) <= 4  # Allow some slack from splitting

    def test_filters_greeting_only_text(self) -> None:
        text = "user: Hi\nassistant: Hello\nuser: Thanks\nassistant: Sure"
        summary = summarise(text)
        # All content is filler — should be empty or very short
        assert len(summary) < 50

    def test_conversation_with_facts(self) -> None:
        text = (
            "user: What's my email?\n"
            "assistant: Your email address is kai@example.com.\n"
            "user: And my birthday?\n"
            "assistant: Your birthday is March 15, 1990."
        )
        summary = summarise(text)
        assert "email" in summary.lower() or "birthday" in summary.lower()


class TestSummaryIndex:
    _counter: int = 0

    def _make_index(self) -> SummaryIndex:
        import chromadb
        from sentence_transformers import SentenceTransformer

        TestSummaryIndex._counter += 1
        client = chromadb.Client()
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return SummaryIndex(client, embedder, prefix=f"t{self._counter}_")

    def test_add_and_search(self) -> None:
        idx = self._make_index()
        idx.add("session_1", "JWT authentication setup with RS256 keys")
        idx.add("session_2", "PostgreSQL database migration procedures")
        idx.add("session_3", "Docker container deployment pipeline")

        embedder_for_query = __import__(
            "sentence_transformers", fromlist=["SentenceTransformer"]
        ).SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = embedder_for_query.encode("How do I set up JWT?").tolist()

        results = idx.search(query_emb, n_results=2)
        assert len(results) >= 1
        parent_ids = [r[0] for r in results]
        assert "session_1" in parent_ids

    def test_empty_index_returns_empty(self) -> None:
        idx = self._make_index()
        embedder = __import__(
            "sentence_transformers", fromlist=["SentenceTransformer"]
        ).SentenceTransformer("all-MiniLM-L6-v2")
        query_emb = embedder.encode("anything").tolist()
        results = idx.search(query_emb, n_results=5)
        assert results == []

    def test_count(self) -> None:
        idx = self._make_index()
        assert idx.count() == 0
        idx.add("s1", "some summary text")
        assert idx.count() == 1

    def test_upsert_updates(self) -> None:
        idx = self._make_index()
        idx.add("s1", "original summary")
        idx.add("s1", "updated summary about Docker")
        assert idx.count() == 1

    def test_empty_summary_skipped(self) -> None:
        idx = self._make_index()
        idx.add("s1", "")
        idx.add("s2", "   ")
        assert idx.count() == 0

    def test_metadata_stored(self) -> None:
        idx = self._make_index()
        idx.add("s1", "test summary", metadata={"project": "myelin"})
        assert idx.count() == 1

    def test_filler_sentence_filtered_when_long_enough_to_pass_length_check(
        self,
    ) -> None:
        """Filler sentences >=15 chars are removed even after passing length check.

        Covers perirhinal.py line 172 — the ``continue`` inside the filler regex
        branch that previously had no test exercising it.
        """
        # "Sounds good!!!!!" is 16 chars (>= 15) so it passes the length gate,
        # but matches the filler regex and must be discarded.
        text = "The system uses JWT tokens for authentication. Sounds good!!!!!"
        result = summarise(text)
        assert "jwt" in result.lower() or "authentication" in result.lower()
        assert "sounds good" not in result.lower()


class TestSummaryIndexDelete:
    def test_delete_nonexistent_does_not_raise(self) -> None:
        """SummaryIndex.delete on a missing ID must silently no-op.

        Covers perirhinal.py lines 255-258 — the try/except guard inside delete().
        """
        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.Client()
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        idx = SummaryIndex(client, embedder, collection_name="test_del_noexist")
        idx.delete("does-not-exist")  # must not raise
