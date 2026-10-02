"""Tests for Pinecone document store factory."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from haystack.utils import Secret

from config import Settings
from document_store import create_pinecone_document_store


@pytest.fixture
def custom_settings(settings: Settings) -> Settings:
    return replace(settings, pinecone_namespace="custom-namespace", pinecone_metric="dotproduct")


@patch("document_store.PineconeDocumentStore")
def test_create_pinecone_document_store_passes_expected_arguments(
    mock_store_cls: MagicMock,
    custom_settings: Settings,
) -> None:
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    store = create_pinecone_document_store(custom_settings)

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
    custom_settings: Settings,
) -> None:
    mock_store_cls.return_value = MagicMock()

    create_pinecone_document_store(custom_settings)

    call_kwargs = mock_store_cls.call_args.kwargs
    secret = call_kwargs["api_key"]
    secret_dict = secret.to_dict()

    assert secret_dict == {
        "type": "env_var",
        "env_vars": ["PINECONE_API_KEY"],
        "strict": True,
    }
    assert custom_settings.pinecone_api_key not in repr(call_kwargs)
