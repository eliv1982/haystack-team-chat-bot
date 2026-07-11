"""Tests for Pinecone document store factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from haystack.utils import Secret

from config import Settings
from document_store import create_pinecone_document_store


@pytest.fixture
def settings() -> Settings:
    return Settings(
        telegram_bot_token="test-telegram-token",
        openai_api_key="test-openai-key",
        api_base_url="https://api.example.com/v1",
        openai_model="test-chat-model",
        embedding_model="test-embedding-model",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="test-index",
        pinecone_namespace="custom-namespace",
        pinecone_dimension=1536,
        pinecone_metric="dotproduct",
        retrieval_top_k=50,
    )


@patch("document_store.PineconeDocumentStore")
def test_create_pinecone_document_store_passes_expected_arguments(
    mock_store_cls: MagicMock,
    settings: Settings,
) -> None:
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    store = create_pinecone_document_store(settings)

    assert store is mock_store
    mock_store_cls.assert_called_once_with(
        api_key=Secret.from_env_var("PINECONE_API_KEY"),
        index="test-index",
        namespace="custom-namespace",
        dimension=1536,
        metric="dotproduct",
        show_progress=False,
    )


@patch("document_store.PineconeDocumentStore")
def test_factory_uses_env_based_secret_not_raw_token(
    mock_store_cls: MagicMock,
    settings: Settings,
) -> None:
    mock_store_cls.return_value = MagicMock()

    create_pinecone_document_store(settings)

    call_kwargs = mock_store_cls.call_args.kwargs
    secret = call_kwargs["api_key"]
    secret_dict = secret.to_dict()

    assert secret_dict == {
        "type": "env_var",
        "env_vars": ["PINECONE_API_KEY"],
        "strict": True,
    }
    assert settings.pinecone_api_key not in repr(call_kwargs)


@patch("document_store.PineconeDocumentStore")
def test_constructor_exception_is_preserved(
    mock_store_cls: MagicMock,
    settings: Settings,
) -> None:
    mock_store_cls.side_effect = ValueError("invalid pinecone configuration")

    with pytest.raises(ValueError, match="invalid pinecone configuration"):
        create_pinecone_document_store(settings)


def test_factory_does_not_create_store_on_import() -> None:
    with patch("document_store.PineconeDocumentStore") as mock_store_cls:
        import document_store

        assert callable(document_store.create_pinecone_document_store)

    mock_store_cls.assert_not_called()
