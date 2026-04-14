"""Tests for entorhinal.py — context coordinate system."""

from __future__ import annotations

from myelin.store.entorhinal import (
    assign_region,
    detect_query_speakers,
    extract_keywords,
    extract_speakers,
    topic_overlap,
)

# -------------------------------------------------------------------
# Keyword extraction (LEC pathway)
# -------------------------------------------------------------------


class TestExtractKeywords:
    def test_extracts_content_words(self) -> None:
        kw = extract_keywords("The auth service uses JWT tokens for API calls")
        assert "auth" in kw or "jwt" in kw or "tokens" in kw

    def test_filters_stop_words(self) -> None:
        kw = extract_keywords("I think this is a very good thing overall")
        # All content is stop words — should get nothing or very little
        assert len(kw) <= 2

    def test_respects_top_n(self) -> None:
        text = "alpha beta gamma delta epsilon zeta eta theta"
        kw = extract_keywords(text, top_n=3)
        assert len(kw) == 3

    def test_empty_text(self) -> None:
        assert extract_keywords("") == []


# -------------------------------------------------------------------
# Region classification (MEC pathway)
# -------------------------------------------------------------------


class TestAssignRegion:
    def test_technology_region(self) -> None:
        text = "Deploy the Docker container to the Kubernetes cluster"
        assert assign_region(text) == "technology"

    def test_security_region(self) -> None:
        text = "Configure OAuth authentication with JWT tokens"
        assert assign_region(text) == "security"

    def test_health_region(self) -> None:
        text = "Schedule a doctor appointment for the blood test"
        assert assign_region(text) == "health"

    def test_no_region_on_ambiguous(self) -> None:
        text = "I went to the park yesterday"
        assert assign_region(text) is None

    def test_requires_minimum_hits(self) -> None:
        # Single keyword match shouldn't assign a region
        text = "I saw a movie"
        assert assign_region(text) is None


# -------------------------------------------------------------------
# Topic overlap
# -------------------------------------------------------------------


class TestTopicOverlap:
    def test_perfect_overlap(self) -> None:
        assert topic_overlap(["a", "b"], ["a", "b"]) == 1.0

    def test_no_overlap(self) -> None:
        assert topic_overlap(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self) -> None:
        score = topic_overlap(["a", "b", "c"], ["b", "c", "d"])
        assert 0.4 < score < 0.6  # Jaccard = 2/4 = 0.5

    def test_empty_lists(self) -> None:
        assert topic_overlap([], ["a"]) == 0.0
        assert topic_overlap(["a"], []) == 0.0


# -------------------------------------------------------------------
# Speaker extraction (source monitoring — "who" pathway)
# -------------------------------------------------------------------


class TestExtractSpeakers:
    def test_detects_named_speakers(self) -> None:
        text = "Alice: Hi there!\nBob: Hello Alice.\nAlice: How are you?"
        speakers = extract_speakers(text)
        assert "Alice" in speakers
        assert "Bob" in speakers

    def test_excludes_generic_roles(self) -> None:
        text = "user: Hello\nassistant: Hi\nAlice: Good morning"
        speakers = extract_speakers(text)
        assert "Alice" in speakers
        assert "user" not in [s.lower() for s in speakers]
        assert "assistant" not in [s.lower() for s in speakers]

    def test_orders_by_frequency(self) -> None:
        text = "Alice: one\nBob: two\nAlice: three\nAlice: four\nBob: five"
        speakers = extract_speakers(text)
        assert speakers[0] == "Alice"  # 3 turns vs Bob's 2

    def test_empty_text(self) -> None:
        assert extract_speakers("") == []

    def test_no_speakers(self) -> None:
        text = "This is a plain paragraph with no speaker labels."
        assert extract_speakers(text) == []

    def test_multiword_speaker_name(self) -> None:
        text = "Dr. Smith: Your results look fine.\nAlice: Thanks doctor."
        speakers = extract_speakers(text)
        assert any("Smith" in s for s in speakers)


# -------------------------------------------------------------------
# Query speaker detection
# -------------------------------------------------------------------


class TestDetectQuerySpeakers:
    def test_finds_mentioned_speaker(self) -> None:
        known = ["Alice", "Bob", "Charlie"]
        result = detect_query_speakers("What did Alice say about the project?", known)
        assert result == ["Alice"]

    def test_case_insensitive(self) -> None:
        known = ["Alice"]
        result = detect_query_speakers("what did alice mention?", known)
        assert result == ["Alice"]

    def test_multiple_speakers(self) -> None:
        known = ["Alice", "Bob"]
        result = detect_query_speakers("Did Alice and Bob agree?", known)
        assert "Alice" in result
        assert "Bob" in result

    def test_no_match(self) -> None:
        known = ["Alice", "Bob"]
        result = detect_query_speakers("What was discussed?", known)
        assert result == []

    def test_empty_known_list(self) -> None:
        result = detect_query_speakers("What did Alice say?", [])
        assert result == []
