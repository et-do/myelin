"""Structured JSON logging with request-ID tracking."""

from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar
from typing import Any

request_id: ContextVar[str] = ContextVar("request_id", default="")

# Attribute names present on a baseline LogRecord — anything *not* in this
# set was injected via ``extra=`` and should appear in the JSON output.
_BASE_RECORD_ATTRS: frozenset[str] = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)),
) | {"message"}


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = request_id.get("")
        if rid:
            obj["request_id"] = rid
        if record.exc_info and record.exc_info[1]:
            obj["exc"] = self.formatException(record.exc_info)
        # Include caller-supplied extra fields.
        for key in record.__dict__:
            if key not in _BASE_RECORD_ATTRS and key not in obj:
                obj[key] = record.__dict__[key]
        return json.dumps(obj, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the ``myelin`` logger hierarchy with JSON output to stderr."""
    root = logging.getLogger("myelin")
    if root.handlers:
        return  # already configured
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)
    root.propagate = False


# Noisy third-party loggers that emit warnings/info during model loading.
_NOISY_LOGGERS = (
    "transformers.modeling_utils",
    "safetensors",
    "sentence_transformers",
    "sentence_transformers.models",
    "sentence_transformers.cross_encoder",
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "httpx",
    "chromadb",
    "chromadb.telemetry",
    "torch",
    "onnxruntime",
    "PIL",
    "mcp",
    "fastmcp",
)


def suppress_noisy_loggers() -> None:
    """Silence verbose ML-library loggers and set env vars to hide progress bars."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
