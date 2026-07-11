"""Application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv
from typing import Final

from dotenv import load_dotenv

VALID_PINECONE_METRICS: Final[frozenset[str]] = frozenset(
    {"cosine", "dotproduct", "euclidean"}
)

DEFAULT_PINECONE_NAMESPACE: Final[str] = "haystack-team-chat-homework"
DEFAULT_PINECONE_DIMENSION: Final[int] = 1536
DEFAULT_PINECONE_METRIC: Final[str] = "cosine"
DEFAULT_RETRIEVAL_TOP_K: Final[int] = 50


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings."""

    telegram_bot_token: str
    openai_api_key: str
    api_base_url: str | None
    openai_model: str
    embedding_model: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    pinecone_dimension: int
    pinecone_metric: str
    retrieval_top_k: int


def _require_non_empty_str(name: str) -> str:
    raw = getenv(name)
    if raw is None:
        raise ConfigurationError(f"Missing required environment variable: {name}")
    value = raw.strip()
    if not value:
        raise ConfigurationError(f"Environment variable {name} must not be empty")
    return value


def _optional_non_empty_str(name: str, *, default: str) -> str:
    raw = getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    return value


def _optional_base_url(name: str) -> str | None:
    raw = getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return value


def _require_positive_int(name: str, *, default: int | None = None) -> int:
    raw = getenv(name)
    if raw is None:
        if default is None:
            raise ConfigurationError(f"Missing required environment variable: {name}")
        return default

    stripped = raw.strip()
    if not stripped:
        if default is None:
            raise ConfigurationError(f"Environment variable {name} must not be empty")
        return default

    try:
        value = int(stripped)
    except ValueError as exc:
        raise ConfigurationError(
            f"Environment variable {name} must be a positive integer"
        ) from exc

    if value <= 0:
        raise ConfigurationError(
            f"Environment variable {name} must be a positive integer"
        )
    return value


def _require_metric(name: str, *, default: str) -> str:
    raw = getenv(name)
    if raw is None or not raw.strip():
        value = default
    else:
        value = raw.strip()

    if value not in VALID_PINECONE_METRICS:
        allowed = ", ".join(sorted(VALID_PINECONE_METRICS))
        raise ConfigurationError(
            f"Environment variable {name} must be one of: {allowed}"
        )
    return value


def load_settings(*, dotenv_path: str | None = ".env") -> Settings:
    """Load and validate settings from environment variables."""
    if dotenv_path is not None:
        load_dotenv(dotenv_path)

    return Settings(
        telegram_bot_token=_require_non_empty_str("TELEGRAM_BOT_TOKEN"),
        openai_api_key=_require_non_empty_str("OPENAI_API_KEY"),
        api_base_url=_optional_base_url("OPENAI_BASE_URL"),
        openai_model=_require_non_empty_str("OPENAI_MODEL"),
        embedding_model=_require_non_empty_str("EMBEDDING_MODEL"),
        pinecone_api_key=_require_non_empty_str("PINECONE_API_KEY"),
        pinecone_index_name=_require_non_empty_str("PINECONE_INDEX_NAME"),
        pinecone_namespace=_optional_non_empty_str(
            "PINECONE_NAMESPACE", default=DEFAULT_PINECONE_NAMESPACE
        ),
        pinecone_dimension=_require_positive_int(
            "PINECONE_DIMENSION", default=DEFAULT_PINECONE_DIMENSION
        ),
        pinecone_metric=_require_metric(
            "PINECONE_METRIC", default=DEFAULT_PINECONE_METRIC
        ),
        retrieval_top_k=_require_positive_int(
            "RETRIEVAL_TOP_K", default=DEFAULT_RETRIEVAL_TOP_K
        ),
    )
