"""Tests for Pinecone index preflight validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config import Settings
from pinecone_preflight import (
    PineconeIndexConfigMismatchError,
    PineconeIndexInfo,
    PineconeIndexNotFoundError,
    PineconeIndexNotReadyError,
    PineconePreflightError,
    validate_existing_pinecone_index,
)


def _mock_index_model(
    *,
    dimension: int = 1536,
    metric: str = "cosine",
    ready: bool = True,
    state: str = "Ready",
) -> MagicMock:
    index_model = MagicMock()
    index_model.dimension = dimension
    index_model.metric = metric
    index_model.status.ready = ready
    index_model.status.state = state
    return index_model


@patch("pinecone_preflight.Pinecone")
def test_validate_existing_ready_index(
    mock_pinecone_cls: MagicMock,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    mock_client = MagicMock()
    mock_pinecone_cls.return_value = mock_client
    mock_client.indexes.exists.return_value = True
    mock_client.indexes.describe.return_value = _mock_index_model()

    info = validate_existing_pinecone_index(settings)

    assert info == PineconeIndexInfo(
        name="test-index",
        dimension=1536,
        metric="cosine",
        ready=True,
        status="Ready",
    )
    mock_client.indexes.exists.assert_called_once_with("test-index")
    mock_client.indexes.describe.assert_called_once_with("test-index")
    mock_client.indexes.create.assert_not_called()
    mock_client.indexes.delete.assert_not_called()
    mock_client.create_index.assert_not_called()
    mock_client.delete_index.assert_not_called()


@patch("pinecone_preflight.Pinecone")
def test_missing_index_raises_not_found(
    mock_pinecone_cls: MagicMock,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    mock_client = MagicMock()
    mock_pinecone_cls.return_value = mock_client
    mock_client.indexes.exists.return_value = False

    with pytest.raises(PineconeIndexNotFoundError, match="test-index"):
        validate_existing_pinecone_index(settings)


@patch("pinecone_preflight.Pinecone")
def test_dimension_mismatch_raises_config_error(
    mock_pinecone_cls: MagicMock,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    mock_client = MagicMock()
    mock_pinecone_cls.return_value = mock_client
    mock_client.indexes.exists.return_value = True
    mock_client.indexes.describe.return_value = _mock_index_model(dimension=768)

    with pytest.raises(PineconeIndexConfigMismatchError, match="PINECONE_DIMENSION mismatch"):
        validate_existing_pinecone_index(settings)


@patch("pinecone_preflight.Pinecone")
def test_metric_mismatch_raises_config_error(
    mock_pinecone_cls: MagicMock,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    mock_client = MagicMock()
    mock_pinecone_cls.return_value = mock_client
    mock_client.indexes.exists.return_value = True
    mock_client.indexes.describe.return_value = _mock_index_model(metric="euclidean")

    with pytest.raises(PineconeIndexConfigMismatchError, match="PINECONE_METRIC mismatch"):
        validate_existing_pinecone_index(settings)


@patch("pinecone_preflight.Pinecone")
def test_not_ready_index_raises_error(
    mock_pinecone_cls: MagicMock,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    mock_client = MagicMock()
    mock_pinecone_cls.return_value = mock_client
    mock_client.indexes.exists.return_value = True
    mock_client.indexes.describe.return_value = _mock_index_model(ready=False, state="Initializing")

    with pytest.raises(PineconeIndexNotReadyError, match="not ready"):
        validate_existing_pinecone_index(settings)


@patch("pinecone_preflight.Pinecone")
def test_sdk_exception_is_chained(
    mock_pinecone_cls: MagicMock,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    mock_pinecone_cls.side_effect = RuntimeError("sdk unavailable")

    with pytest.raises(PineconePreflightError, match="Failed to validate Pinecone index") as exc_info:
        validate_existing_pinecone_index(settings)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@patch("pinecone_preflight.Pinecone")
def test_exception_message_does_not_contain_secret(
    mock_pinecone_cls: MagicMock,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_value = "super-secret-pinecone-key-value"
    monkeypatch.setenv("PINECONE_API_KEY", secret_value)
    mock_pinecone_cls.side_effect = RuntimeError("network failure")

    with pytest.raises(PineconePreflightError) as exc_info:
        validate_existing_pinecone_index(settings)

    message = str(exc_info.value)
    assert secret_value not in message
