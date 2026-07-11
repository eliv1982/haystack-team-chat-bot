"""Control-plane validation for an existing Pinecone index."""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv

from pinecone import Pinecone

from config import Settings


class PineconePreflightError(Exception):
    """Raised when Pinecone index preflight validation fails."""


class PineconeIndexNotFoundError(PineconePreflightError):
    """Raised when the configured Pinecone index does not exist."""


class PineconeIndexConfigMismatchError(PineconePreflightError):
    """Raised when the Pinecone index configuration does not match settings."""


class PineconeIndexNotReadyError(PineconePreflightError):
    """Raised when the Pinecone index exists but is not ready."""


@dataclass(frozen=True, slots=True)
class PineconeIndexInfo:
    """Safe metadata about an existing Pinecone index."""

    name: str
    dimension: int
    metric: str
    ready: bool
    status: str | None


def _normalize_metric(metric: str) -> str:
    return metric.strip().lower()


def validate_existing_pinecone_index(settings: Settings) -> PineconeIndexInfo:
    """Verify that the configured Pinecone index exists and matches settings."""
    api_key = getenv("PINECONE_API_KEY")
    if api_key is None or not api_key.strip():
        raise PineconePreflightError("Missing required environment variable: PINECONE_API_KEY")

    index_name = settings.pinecone_index_name

    try:
        client = Pinecone(api_key=api_key.strip())
        if not client.indexes.exists(index_name):
            raise PineconeIndexNotFoundError(f"Pinecone index not found: {index_name}")

        index_model = client.indexes.describe(index_name)
    except PineconePreflightError:
        raise
    except Exception as exc:
        raise PineconePreflightError("Failed to validate Pinecone index") from exc

    dimension = index_model.dimension
    if dimension is None:
        raise PineconeIndexConfigMismatchError(
            f"PINECONE_DIMENSION mismatch: expected {settings.pinecone_dimension}, actual None"
        )

    actual_metric = _normalize_metric(index_model.metric)
    expected_metric = _normalize_metric(settings.pinecone_metric)
    if dimension != settings.pinecone_dimension:
        raise PineconeIndexConfigMismatchError(
            "PINECONE_DIMENSION mismatch: "
            f"expected {settings.pinecone_dimension}, actual {dimension}"
        )
    if actual_metric != expected_metric:
        raise PineconeIndexConfigMismatchError(
            f"PINECONE_METRIC mismatch: expected {expected_metric}, actual {actual_metric}"
        )

    ready = index_model.status.ready
    status = index_model.status.state
    if not ready:
        raise PineconeIndexNotReadyError(f"Pinecone index not ready: status={status}")

    return PineconeIndexInfo(
        name=index_name,
        dimension=dimension,
        metric=actual_metric,
        ready=ready,
        status=status,
    )
