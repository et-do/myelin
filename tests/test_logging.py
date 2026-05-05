"""Tests for structured logging, warm-up, and graceful shutdown."""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest

from myelin.config import MyelinSettings
from myelin.log import JSONFormatter, request_id, setup_logging
from myelin.mcp import (
    _track,
    configure,
    shutdown,
    warm_up,
)
from myelin.store.hippocampus import Hippocampus

# ------------------------------------------------------------------
# JSONFormatter
# ------------------------------------------------------------------


class TestJSONFormatter:
    def test_basic_format(self) -> None:
        fmt = JSONFormatter()
        record = logging.LogRecord(
            "myelin.test", logging.INFO, "", 0, "hello world", (), None
        )
        line = fmt.format(record)
        obj = json.loads(line)
        assert obj["level"] == "INFO"
        assert obj["logger"] == "myelin.test"
        assert obj["msg"] == "hello world"
        assert "ts" in obj

    def test_includes_request_id(self) -> None:
        fmt = JSONFormatter()
        token = request_id.set("abc123")
        try:
            record = logging.LogRecord(
                "myelin.test", logging.INFO, "", 0, "msg", (), None
            )
            obj = json.loads(fmt.format(record))
            assert obj["request_id"] == "abc123"
        finally:
            request_id.reset(token)

    def test_omits_request_id_when_empty(self) -> None:
        fmt = JSONFormatter()
        record = logging.LogRecord("myelin.test", logging.INFO, "", 0, "msg", (), None)
        obj = json.loads(fmt.format(record))
        assert "request_id" not in obj

    def test_includes_extra_fields(self) -> None:
        fmt = JSONFormatter()
        record = logging.LogRecord("myelin.test", logging.INFO, "", 0, "msg", (), None)
        record.duration_ms = 42.5  # type: ignore[attr-defined]
        obj = json.loads(fmt.format(record))
        assert obj["duration_ms"] == 42.5

    def test_includes_exception(self) -> None:
        fmt = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                "myelin.test",
                logging.ERROR,
                "",
                0,
                "fail",
                (),
                sys.exc_info(),
            )
        obj = json.loads(fmt.format(record))
        assert "exc" in obj
        assert "boom" in obj["exc"]


# ------------------------------------------------------------------
# setup_logging
# ------------------------------------------------------------------


class TestSetupLogging:
    def test_configures_myelin_logger(self) -> None:
        logger = logging.getLogger("myelin")
        # Remove any pre-existing handlers from other tests
        logger.handlers.clear()

        setup_logging(level=logging.DEBUG)
        try:
            assert len(logger.handlers) == 1
            assert isinstance(logger.handlers[0].formatter, JSONFormatter)
            assert logger.level == logging.DEBUG
        finally:
            logger.handlers.clear()

    def test_idempotent(self) -> None:
        logger = logging.getLogger("myelin")
        logger.handlers.clear()

        setup_logging()
        setup_logging()  # second call should be a no-op
        try:
            assert len(logger.handlers) == 1
        finally:
            logger.handlers.clear()


# ------------------------------------------------------------------
# _track context manager
# ------------------------------------------------------------------


class TestTrack:
    def test_sets_request_id(self) -> None:
        captured: list[str] = []

        with _track("test_op"):
            captured.append(request_id.get(""))

        assert captured[0] != ""
        assert len(captured[0]) == 12

        # request_id is reset after the context manager
        assert request_id.get("") == ""

    def test_propagates_exceptions(self) -> None:
        with pytest.raises(ValueError, match="oops"):
            with _track("failing_op"):
                raise ValueError("oops")

        # request_id is still reset even after exception
        assert request_id.get("") == ""


# ------------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------------


class TestWarmUp:
    def test_hippocampus_warm_up(self, hippocampus: Hippocampus) -> None:
        """warm_up() runs without error on a fresh Hippocampus."""
        hippocampus.warm_up()

    def test_server_warm_up(self, tmp_settings: MyelinSettings) -> None:
        """Server-level warm_up() initialises singletons."""
        configure(tmp_settings)
        warm_up()
        # After warm_up, hippocampus should be initialized
        import myelin.mcp as _mcp_mod

        assert _mcp_mod._hippocampus is not None


# ------------------------------------------------------------------
# Graceful shutdown
# ------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_closes_stores(self, tmp_settings: MyelinSettings) -> None:
        """shutdown() closes all initialised stores."""
        configure(tmp_settings)

        # Force initialization of all stores
        from myelin.mcp import do_recall, do_status, do_store

        do_store("Test memory for shutdown verification here")
        do_recall("test")  # initialises _hebbian
        do_status()  # initialises _neocortex + _thalamus

        import myelin.mcp as srv

        assert srv._hebbian is not None
        assert srv._neocortex is not None
        assert srv._thalamus is not None

        # Patch close methods to verify they're called
        with (
            patch.object(srv._hebbian, "close", wraps=srv._hebbian.close) as mock_heb,
            patch.object(
                srv._neocortex, "close", wraps=srv._neocortex.close
            ) as mock_neo,
            patch.object(
                srv._thalamus, "close", wraps=srv._thalamus.close
            ) as mock_thal,
        ):
            shutdown()
            mock_heb.assert_called_once()
            mock_neo.assert_called_once()
            mock_thal.assert_called_once()

        # Decay timer should also be cleared
        assert srv._decay_timer is None

    def test_shutdown_stops_running_decay_timer(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """shutdown() stops the decay timer if it is running."""
        cfg = MyelinSettings(data_dir=tmp_path / ".myelin", decay_interval_hours=24.0)  # type: ignore[arg-type]
        configure(cfg)

        import myelin.mcp as srv

        timer = srv._get_decay_timer()
        timer.start()
        assert timer.is_running

        shutdown()
        assert not timer.is_running
        assert srv._decay_timer is None

    def test_shutdown_tolerates_uninitialized(self) -> None:
        """shutdown() is safe to call when nothing is initialized."""
        configure(MyelinSettings(data_dir="/tmp/myelin_test_empty"))
        shutdown()  # should not raise


# ------------------------------------------------------------------
# Config: log_level validation
# ------------------------------------------------------------------


class TestLogLevelConfig:
    def test_valid_levels(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            cfg = MyelinSettings(log_level=level)
            assert cfg.log_level == level

    def test_case_insensitive(self) -> None:
        cfg = MyelinSettings(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_invalid_level_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            MyelinSettings(log_level="TRACE")
