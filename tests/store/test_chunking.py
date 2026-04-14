"""Tests for the chunking module — pattern separation for long content."""

from __future__ import annotations

from myelin.store.chunking import (
    _topic_shifted,
    _turn_keywords,
    chunk,
    chunk_conversation,
    chunk_text,
    is_conversation,
)


class TestIsConversation:
    def test_detects_user_assistant_pattern(self) -> None:
        text = "user: hello\nassistant: hi there"
        assert is_conversation(text) is True

    def test_detects_human_ai_pattern(self) -> None:
        text = "human: question\nai: answer"
        assert is_conversation(text) is True

    def test_rejects_plain_text(self) -> None:
        text = "This is just a paragraph about databases."
        assert is_conversation(text) is False

    def test_rejects_single_role(self) -> None:
        text = "user: only one role marker here"
        assert is_conversation(text) is False

    def test_case_insensitive(self) -> None:
        text = "User: hello\nAssistant: hi"
        assert is_conversation(text) is True


class TestChunkConversation:
    def test_splits_into_exchange_pairs(self) -> None:
        text = (
            "user: What is JWT?\n"
            "assistant: JSON Web Token is a standard for auth.\n"
            "user: How do I refresh tokens?\n"
            "assistant: Use a refresh token endpoint."
        )
        chunks = chunk_conversation(text, max_chars=120)
        assert len(chunks) >= 2
        # Each chunk should have at least one role marker
        for c in chunks:
            assert "user:" in c.lower() or "assistant:" in c.lower()

    def test_keeps_short_conversation_intact(self) -> None:
        text = "user: hello\nassistant: hi"
        chunks = chunk_conversation(text, max_chars=5000)
        assert len(chunks) == 1
        assert "user:" in chunks[0].lower()
        assert "assistant:" in chunks[0].lower()

    def test_handles_empty_text(self) -> None:
        assert chunk_conversation("") == []
        assert chunk_conversation("   ") == []

    def test_handles_no_role_markers(self) -> None:
        text = "Just plain text without roles"
        chunks = chunk_conversation(text, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_handles_long_single_turn(self) -> None:
        text = "user: " + "x" * 2000 + "\nassistant: short"
        chunks = chunk_conversation(text, max_chars=500)
        assert len(chunks) >= 2


class TestChunkText:
    def test_splits_at_paragraph_boundaries(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, max_chars=40, overlap_chars=0)
        assert len(chunks) >= 2

    def test_keeps_short_text_intact(self) -> None:
        text = "Short text."
        chunks = chunk_text(text, max_chars=5000)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_overlap_carries_context(self) -> None:
        para_a = "A" * 100
        para_b = "B" * 100
        para_c = "C" * 100
        text = f"{para_a}\n\n{para_b}\n\n{para_c}"
        chunks = chunk_text(text, max_chars=150, overlap_chars=50)
        # Second chunk should start with tail of previous content
        if len(chunks) >= 2:
            assert chunks[1].startswith("A" * 50) or "B" in chunks[1]

    def test_handles_empty_text(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   \n\n   ") == []

    def test_single_long_paragraph(self) -> None:
        text = "word " * 500  # ~2500 chars in one paragraph
        chunks = chunk_text(text, max_chars=500, overlap_chars=100)
        # With no paragraph breaks, falls back to single chunk
        assert len(chunks) >= 1


class TestChunkAutoDetect:
    def test_short_content_returns_single(self) -> None:
        chunks = chunk("Short content here.", max_chars=1000)
        assert len(chunks) == 1
        assert chunks[0] == "Short content here."

    def test_empty_returns_empty(self) -> None:
        assert chunk("") == []
        assert chunk("   ") == []

    def test_long_conversation_uses_conversation_chunking(self) -> None:
        turns = []
        for i in range(20):
            turns.append(f"user: Question number {i} about topic {i}")
            turns.append(f"assistant: Answer number {i} with details about {i}")
        text = "\n".join(turns)
        chunks = chunk(text, max_chars=200)
        assert len(chunks) > 1
        # Conversation chunks should contain role markers
        assert any("user:" in c.lower() for c in chunks)

    def test_long_prose_uses_text_chunking(self) -> None:
        paragraphs = [f"Paragraph {i} " * 50 for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunk(text, max_chars=500, overlap_chars=100)
        assert len(chunks) > 1


class TestTopicShift:
    def test_keywords_extracted(self) -> None:
        kw = _turn_keywords("We use PostgreSQL for the database and Redis for caching")
        assert "postgresql" in kw or "database" in kw or "redis" in kw

    def test_keywords_filter_stops(self) -> None:
        kw = _turn_keywords("I think we should also just really use the thing")
        assert "think" not in kw
        assert "should" not in kw

    def test_same_topic_not_shifted(self) -> None:
        a = "user: How do I configure PostgreSQL replication?"
        b = "assistant: Set up streaming replication in PostgreSQL."
        assert _topic_shifted(a, b) is False

    def test_different_topic_is_shifted(self) -> None:
        a = (
            "user: Configure PostgreSQL database replication and backup.\n"
            "assistant: Set up streaming replication with wal_level=replica."
        )
        b = (
            "user: What's my flight departure time tomorrow morning?\n"
            "assistant: Your flight departs at 7:30 AM from terminal B."
        )
        assert _topic_shifted(a, b) is True

    def test_topic_shift_splits_conversation(self) -> None:
        text = (
            "user: How do I configure PostgreSQL replication?\n"
            "assistant: Set wal_level to replica and configure primary_conninfo. "
            "PostgreSQL streaming replication is reliable for production databases.\n"
            "user: What's the best Italian restaurant downtown?\n"
            "assistant: Try Bella Napoli on Main Street, excellent pasta and pizza."
        )
        # With topic shift detection
        chunks_with = chunk_conversation(text, max_chars=5000, detect_topic_shift=True)
        # Without topic shift detection
        chunks_without = chunk_conversation(
            text, max_chars=5000, detect_topic_shift=False
        )
        # Topic shift should create more chunks
        assert len(chunks_with) > len(chunks_without)

    def test_no_shift_on_short_turns(self) -> None:
        """Short turns don't have enough keywords for shift detection."""
        text = "user: Hi\nassistant: Hello\nuser: Bye\nassistant: Goodbye"
        chunks = chunk_conversation(text, max_chars=5000, detect_topic_shift=True)
        # Should remain as one chunk — not enough content for keyword extraction
        assert len(chunks) == 1
