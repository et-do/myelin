"""Tests for hippocampal time cells — temporal expression parsing."""

from datetime import datetime

import pytest

from myelin.recall.time_cells import (
    parse_session_date,
    parse_temporal_reference,
    recency_boost,
)

# Reference date: Wednesday May 10, 2023
REF = datetime(2023, 5, 10)


class TestParseTemporalReference:
    def test_days_ago_numeric(self) -> None:
        result = parse_temporal_reference("What did I do 5 days ago?", REF)
        assert result is not None
        start, end = result
        # 5 days ago = May 5, ±1 day buffer
        assert start == datetime(2023, 5, 4)
        assert end == datetime(2023, 5, 6)

    def test_days_ago_word(self) -> None:
        result = parse_temporal_reference("What happened ten days ago?", REF)
        assert result is not None
        start, end = result
        # 10 days ago = April 30
        assert start == datetime(2023, 4, 29)
        assert end == datetime(2023, 5, 1)

    def test_weeks_ago(self) -> None:
        result = parse_temporal_reference("I mentioned an event two weeks ago", REF)
        assert result is not None
        start, end = result
        # 2 weeks ago = April 26, ±3 days
        assert start == datetime(2023, 4, 23)
        assert end == datetime(2023, 4, 29)

    def test_months_ago(self) -> None:
        result = parse_temporal_reference("What did I do four months ago?", REF)
        assert result is not None
        start, end = result
        # ~120 days ago = ~Jan 10, ±7 days
        assert start == datetime(2023, 1, 3)
        assert end == datetime(2023, 1, 17)

    def test_last_weekday(self) -> None:
        # REF is Wednesday May 10
        result = parse_temporal_reference("Who did I meet last Tuesday?", REF)
        assert result is not None
        start, end = result
        # Last Tuesday from Wednesday = May 9, ±1 day
        assert start == datetime(2023, 5, 8)
        assert end == datetime(2023, 5, 10)

    def test_last_weekday_same_day(self) -> None:
        # "last Wednesday" when today is Wednesday = 7 days back
        result = parse_temporal_reference("last Wednesday meeting", REF)
        assert result is not None
        start, end = result
        assert start == datetime(2023, 5, 2)
        assert end == datetime(2023, 5, 4)

    def test_no_temporal_reference(self) -> None:
        result = parse_temporal_reference("What is my favorite color?", REF)
        assert result is None

    def test_case_insensitive(self) -> None:
        result = parse_temporal_reference("LAST MONDAY event", REF)
        assert result is not None

    def test_singular_unit(self) -> None:
        result = parse_temporal_reference("1 week ago I went hiking", REF)
        assert result is not None
        start, end = result
        # 1 week ago = May 3, ±3 days
        assert start == datetime(2023, 4, 30)
        assert end == datetime(2023, 5, 6)


class TestParseSessionDate:
    def test_standard_format(self) -> None:
        result = parse_session_date("2023/05/20 (Sat) 02:21")
        assert result == datetime(2023, 5, 20)

    def test_invalid_string(self) -> None:
        result = parse_session_date("not a date")
        assert result is None

    def test_empty_string(self) -> None:
        result = parse_session_date("")
        assert result is None


class TestNamedMonth:
    def test_in_april(self) -> None:
        # REF is May 10 2023; "in April" → April 2023
        result = parse_temporal_reference("What did I do in April?", REF)
        assert result is not None
        start, end = result
        assert start == datetime(2023, 4, 1)
        assert end == datetime(2023, 4, 30)

    def test_during_march(self) -> None:
        result = parse_temporal_reference("during March I started a project", REF)
        assert result is not None
        start, end = result
        assert start == datetime(2023, 3, 1)
        assert end == datetime(2023, 3, 31)

    def test_last_january(self) -> None:
        result = parse_temporal_reference("last January I went skiing", REF)
        assert result is not None
        start, end = result
        assert start == datetime(2023, 1, 1)
        assert end == datetime(2023, 1, 31)


class TestRecencyBoost:
    def test_same_day(self) -> None:
        ref = datetime(2023, 5, 10).toordinal()
        assert recency_boost(ref, ref, 180) == pytest.approx(1.0)

    def test_half_life_decay(self) -> None:
        ref = datetime(2023, 5, 10).toordinal()
        sess = ref - 180  # exactly one half-life ago
        assert recency_boost(sess, ref, 180) == pytest.approx(0.5)

    def test_two_half_lives(self) -> None:
        ref = datetime(2023, 5, 10).toordinal()
        sess = ref - 360
        assert recency_boost(sess, ref, 180) == pytest.approx(0.25)

    def test_disabled_when_zero(self) -> None:
        ref = datetime(2023, 5, 10).toordinal()
        assert recency_boost(ref, ref, 0) == 0.0

    def test_future_session_clamped(self) -> None:
        ref = datetime(2023, 5, 10).toordinal()
        sess = ref + 10  # future — clamped to 0 age
        assert recency_boost(sess, ref, 180) == pytest.approx(1.0)

    def test_in_future_month_wraps_year(self) -> None:
        # "in June" when REF is May → previous year (June >= May)
        result = parse_temporal_reference("What happened in June?", REF)
        assert result is not None
        start, end = result
        assert start == datetime(2022, 6, 1)
        assert end == datetime(2022, 6, 30)

    def test_in_december_wraps(self) -> None:
        # December >= May → previous year; December has 31 days
        result = parse_temporal_reference("in December I traveled", REF)
        assert result is not None
        start, end = result
        assert start == datetime(2022, 12, 1)
        assert end == datetime(2022, 12, 31)

    def test_in_may_same_month_wraps(self) -> None:
        # "in May" when REF is May → previous year (May >= May)
        result = parse_temporal_reference("in May I moved", REF)
        assert result is not None
        start, end = result
        assert start == datetime(2022, 5, 1)
        assert end == datetime(2022, 5, 31)
