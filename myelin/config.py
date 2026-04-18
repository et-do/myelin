"""Configuration — data directory and runtime settings."""

from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings


class MyelinSettings(BaseSettings):
    model_config = {"env_prefix": "MYELIN_"}

    data_dir: Path = Path.home() / ".myelin"
    embedding_model: str = "all-MiniLM-L6-v2"
    log_level: str = "INFO"

    # Recall defaults
    default_n_results: int = 5

    # Chunking — pattern separation for long content
    chunk_max_chars: int = 1000  # MiniLM-L6-v2 handles ~256 tokens ≈ 1000 chars
    chunk_overlap_chars: int = 200  # overlap for text (not conversation) chunks

    # Decay — simple TTL pruning
    max_idle_days: int = 90
    min_access_count: int = 2
    max_idle_days_absolute: int = 365  # hard cap — even popular memories decay

    # Hebbian reinforcement
    hebbian_delta: float = 0.1
    hebbian_scale: float = 0.1  # logarithmic boost scale factor

    # Amygdala gate — input filtering
    min_content_length: int = 20
    dedup_similarity_threshold: float = 0.95

    # Thalamus — working memory buffer
    thalamus_recency_limit: int = 20

    # Entorhinal cortex — context coordinate re-ranking
    entorhinal_boost: float = 0.3

    # Source monitoring — speaker-based retrieval boost ("who" coordinate)
    speaker_boost: float = 0.2  # multiplier when query mentions a stored speaker

    # Perirhinal cortex — gist-guided session familiarity boost
    perirhinal_boost: float = 0.5  # multiplier for gist-matched sessions
    perirhinal_top_k: int = 10  # how many gist-matched sessions to consider

    # Lateral inhibition — session-level diversity in recall results
    lateral_k: int = 1  # max chunks per scope in final results (0 = off)
    recall_over_factor: int = 8  # over-retrieval multiplier for re-ranking

    # Hippocampal time cells — temporal context boosting
    temporal_boost: float = 0.6  # boost for sessions matching temporal refs
    recency_half_life_days: int = 180  # soft recency gradient half-life (0 = off)

    # Multi-probe recall — query reformulation with merged candidate pool
    multiprobe: bool = True  # enable multi-probe recall

    # Session evidence aggregation — boost sessions with multiple chunks in pool
    session_aggregation_boost: float = 0.15  # log bonus per extra chunk (0 = off)

    # Spreading activation — entity-graph post-retrieval boost
    spreading_activation: bool = True  # enable if semantic network available
    spreading_boost: float = 0.15  # multiplier per related entity match
    spreading_max_depth: int = 2  # hops in entity graph
    spreading_top_k: int = 10  # max related entities to consider

    # Neocortex — cross-encoder deliberative re-ranking
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    neocortex_rerank: bool = True  # enable cross-encoder re-ranking
    neocortex_weight: float = 0.6  # blending weight (0=bi-encoder only, 1=CE only)

    # Storage cap — hard limit on total memories; 0 = disabled
    # When exceeded after a store, least-recently-used memories are evicted.
    # Pinned memories (thalamic relay) are never evicted.
    max_memories: int = 0

    # Auto-decay timer — run decay sweep every N hours in the background; 0 = disabled
    decay_interval_hours: float = 0.0

    # Auto-consolidation — replay after N stores (0 = disabled)
    consolidation_interval: int = 50

    @field_validator("neocortex_weight", "dedup_similarity_threshold")
    @classmethod
    def _unit_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            msg = f"must be between 0.0 and 1.0, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("chunk_overlap_chars")
    @classmethod
    def _overlap_lt_max(cls, v: int, info: Any) -> int:
        max_chars = info.data.get("chunk_max_chars", 1000)
        if v >= max_chars:
            msg = (
                f"chunk_overlap_chars ({v}) must be "
                f"less than chunk_max_chars ({max_chars})"
            )
            raise ValueError(msg)
        return v

    @field_validator(
        "default_n_results",
        "chunk_max_chars",
        "max_idle_days",
        "min_access_count",
        "max_idle_days_absolute",
        "recall_over_factor",
        "perirhinal_top_k",
        "spreading_max_depth",
        "spreading_top_k",
        "thalamus_recency_limit",
        "min_content_length",
    )
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            msg = f"must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("consolidation_interval", "max_memories")
    @classmethod
    def _non_negative_int(cls, v: int) -> int:
        if v < 0:
            msg = f"must be >= 0, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("decay_interval_hours")
    @classmethod
    def _non_negative_decay_interval(cls, v: float) -> float:
        if v < 0:
            msg = f"must be >= 0, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            msg = f"must be one of {sorted(valid)}, got {v!r}"
            raise ValueError(msg)
        return upper

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.chmod(0o700)
        (self.data_dir / "chroma").mkdir(exist_ok=True)


settings = MyelinSettings()
