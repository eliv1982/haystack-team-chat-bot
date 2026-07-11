"""Tests for query/retrieval pipeline construction."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from config import Settings
from pipelines import (
    _build_openai_embedder_kwargs,
    build_summarization_pipeline,
    create_query_pipeline,
)


@pytest.fixture
def direct_openai_settings() -> Settings:
    return Settings(
        telegram_bot_token="test-telegram-token",
        openai_api_key="test-openai-key",
        api_base_url=None,
        openai_model="test-chat-model",
        embedding_model="test-embedding-model",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="test-index",
        pinecone_namespace="haystack-team-chat-homework",
        pinecone_dimension=1536,
        pinecone_metric="cosine",
        retrieval_top_k=25,
    )


@pytest.fixture
def pinecone_document_store() -> PineconeDocumentStore:
    return PineconeDocumentStore(
        api_key=Secret.from_env_var("PINECONE_API_KEY"),
        index="test-index",
        namespace="haystack-team-chat-homework",
        dimension=1536,
        metric="cosine",
        show_progress=False,
    )


@pytest.fixture
def openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


def test_query_pipeline_has_expected_components(
    settings: Settings,
    pinecone_document_store: PineconeDocumentStore,
    openai_env: None,
) -> None:
    pipeline = create_query_pipeline(settings, pinecone_document_store)

    assert set(pipeline.graph.nodes) == {"text_embedder", "retriever"}


def test_query_pipeline_connection_is_correct(
    settings: Settings,
    pinecone_document_store: PineconeDocumentStore,
    openai_env: None,
) -> None:
    pipeline = create_query_pipeline(settings, pinecone_document_store)

    assert pipeline.to_dict()["connections"] == [
        {
            "sender": "text_embedder.embedding",
            "receiver": "retriever.query_embedding",
        }
    ]


def test_openai_embedder_kwargs_omit_api_base_url_for_direct_openai(
    direct_openai_settings: Settings,
) -> None:
    kwargs = _build_openai_embedder_kwargs(direct_openai_settings)

    assert "api_base_url" not in kwargs
    assert kwargs["model"] == direct_openai_settings.embedding_model


def test_openai_embedder_kwargs_include_custom_api_base_url(
    settings: Settings,
) -> None:
    kwargs = _build_openai_embedder_kwargs(settings)

    assert kwargs["api_base_url"] == settings.api_base_url


def test_query_pipeline_text_embedder_uses_model_without_custom_base_url_for_direct_openai(
    direct_openai_settings: Settings,
    pinecone_document_store: PineconeDocumentStore,
    openai_env: None,
) -> None:
    pipeline = create_query_pipeline(direct_openai_settings, pinecone_document_store)

    embedder = pipeline.get_component("text_embedder")
    assert embedder.model == direct_openai_settings.embedding_model
    assert embedder.api_base_url is None
    assert embedder.api_key.to_dict() == Secret.from_env_var("OPENAI_API_KEY").to_dict()


def test_query_pipeline_text_embedder_uses_model_and_api_base_url(
    settings: Settings,
    pinecone_document_store: PineconeDocumentStore,
    openai_env: None,
) -> None:
    pipeline = create_query_pipeline(settings, pinecone_document_store)

    embedder = pipeline.get_component("text_embedder")
    assert embedder.model == settings.embedding_model
    assert embedder.api_base_url == settings.api_base_url


def test_query_pipeline_retriever_uses_document_store_and_top_k(
    settings: Settings,
    pinecone_document_store: PineconeDocumentStore,
    openai_env: None,
) -> None:
    pipeline = create_query_pipeline(settings, pinecone_document_store)

    retriever = pipeline.get_component("retriever")
    assert retriever.document_store is pinecone_document_store
    assert retriever.top_k == settings.retrieval_top_k


def test_query_pipeline_serializes_without_custom_base_url_for_direct_openai(
    direct_openai_settings: Settings,
    pinecone_document_store: PineconeDocumentStore,
    openai_env: None,
) -> None:
    pipeline = create_query_pipeline(direct_openai_settings, pinecone_document_store)
    serialized = json.dumps(pipeline.to_dict())

    assert "api.example.com" not in serialized
    assert "test-openai-key" not in serialized
    assert '"env_vars": ["OPENAI_API_KEY"]' in serialized


def test_query_pipeline_serializes_without_exposing_dummy_secret_values(
    settings: Settings,
    pinecone_document_store: PineconeDocumentStore,
    openai_env: None,
) -> None:
    pipeline = create_query_pipeline(settings, pinecone_document_store)
    serialized = json.dumps(pipeline.to_dict())

    assert "test-openai-key" not in serialized
    assert '"env_vars": ["OPENAI_API_KEY"]' in serialized


@patch("pinecone.Pinecone")
def test_query_pipeline_creation_does_not_call_network(
    mock_pinecone_cls: MagicMock,
    settings: Settings,
    pinecone_document_store: PineconeDocumentStore,
    openai_env: None,
) -> None:
    create_query_pipeline(settings, pinecone_document_store)

    mock_pinecone_cls.assert_not_called()


def test_summarization_pipeline_remains_not_implemented(settings: Settings) -> None:
    with pytest.raises(NotImplementedError, match="Summarization pipeline"):
        build_summarization_pipeline(settings)
