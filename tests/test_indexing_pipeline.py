"""Tests for indexing pipeline construction and indexing service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from config import Settings
from indexing_service import IndexingService, IndexingServiceError
from models import ChatMessage
from pipelines import (
    _build_document_embedder_kwargs,
    build_retrieval_pipeline,
    build_summarization_pipeline,
    create_indexing_pipeline,
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
        retrieval_top_k=50,
    )


@pytest.fixture
def openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


def _second_message() -> ChatMessage:
    return ChatMessage(
        chat_id=-1001234567890,
        message_id=43,
        user_id=8,
        session_id="chat:-1001234567890",
        author_name="Bob",
        username=None,
        text="Second message",
        sent_at=datetime(2024, 1, 15, 12, 31, tzinfo=timezone.utc),
    )


def test_indexing_pipeline_has_expected_components(
    settings: Settings,
    openai_env: None,
) -> None:
    document_store = InMemoryDocumentStore()
    pipeline = create_indexing_pipeline(settings, document_store)

    assert set(pipeline.graph.nodes) == {"document_embedder", "writer"}


def test_document_embedder_kwargs_omit_api_base_url_for_direct_openai(
    direct_openai_settings: Settings,
) -> None:
    kwargs = _build_document_embedder_kwargs(direct_openai_settings)

    assert "api_base_url" not in kwargs
    assert kwargs["model"] == direct_openai_settings.embedding_model
    assert kwargs["progress_bar"] is False


def test_document_embedder_kwargs_include_custom_api_base_url(
    settings: Settings,
) -> None:
    kwargs = _build_document_embedder_kwargs(settings)

    assert kwargs["api_base_url"] == settings.api_base_url


def test_indexing_pipeline_embedder_uses_model_without_custom_base_url_for_direct_openai(
    direct_openai_settings: Settings,
    openai_env: None,
) -> None:
    pipeline = create_indexing_pipeline(direct_openai_settings, InMemoryDocumentStore())

    embedder = pipeline.get_component("document_embedder")
    assert embedder.model == direct_openai_settings.embedding_model
    assert embedder.api_base_url is None
    assert embedder.progress_bar is False


def test_indexing_pipeline_embedder_uses_model_and_api_base_url(
    settings: Settings,
    openai_env: None,
) -> None:
    document_store = InMemoryDocumentStore()
    pipeline = create_indexing_pipeline(settings, document_store)

    embedder = pipeline.get_component("document_embedder")
    assert embedder.model == settings.embedding_model
    assert embedder.api_base_url == settings.api_base_url
    assert embedder.progress_bar is False
    assert embedder.api_key.to_dict() == Secret.from_env_var("OPENAI_API_KEY").to_dict()


def test_indexing_pipeline_writer_uses_store_and_overwrite_policy(
    settings: Settings,
    openai_env: None,
) -> None:
    document_store = InMemoryDocumentStore()
    pipeline = create_indexing_pipeline(settings, document_store)

    writer = pipeline.get_component("writer")
    assert writer.document_store is document_store
    assert writer.policy is DuplicatePolicy.OVERWRITE


def test_indexing_pipeline_connection_is_correct(
    settings: Settings,
    openai_env: None,
) -> None:
    pipeline = create_indexing_pipeline(settings, InMemoryDocumentStore())
    pipeline_dict = pipeline.to_dict()

    assert pipeline_dict["connections"] == [
        {
            "sender": "document_embedder.documents",
            "receiver": "writer.documents",
        }
    ]


def test_indexing_pipeline_serializes_without_custom_base_url_for_direct_openai(
    direct_openai_settings: Settings,
    openai_env: None,
) -> None:
    pipeline = create_indexing_pipeline(direct_openai_settings, InMemoryDocumentStore())
    serialized = json.dumps(pipeline.to_dict())

    assert "api.example.com" not in serialized
    assert "test-openai-key" not in serialized
    assert '"env_vars": ["OPENAI_API_KEY"]' in serialized


def test_indexing_pipeline_serializes_without_exposing_dummy_secret_values(
    settings: Settings,
    openai_env: None,
) -> None:
    pipeline = create_indexing_pipeline(settings, InMemoryDocumentStore())
    serialized = json.dumps(pipeline.to_dict())

    assert "test-openai-key" not in serialized
    assert '"env_vars": ["OPENAI_API_KEY"]' in serialized


def test_retrieval_and_summarization_are_not_implemented(settings: Settings) -> None:
    with pytest.raises(NotImplementedError, match="Retrieval pipeline"):
        build_retrieval_pipeline(settings)

    with pytest.raises(NotImplementedError, match="Summarization pipeline"):
        build_summarization_pipeline(settings)


def test_indexing_service_rejects_empty_input() -> None:
    service = IndexingService(pipeline=MagicMock(spec=Pipeline))

    with pytest.raises(IndexingServiceError, match="must not be empty"):
        service.index_messages([])

    service._pipeline.run.assert_not_called()


def test_indexing_service_runs_pipeline_once_for_batch(
    sample_message: ChatMessage,
) -> None:
    pipeline = MagicMock(spec=Pipeline)
    pipeline.run.return_value = {"writer": {"documents_written": 2}}
    service = IndexingService(pipeline=pipeline)
    messages = [sample_message, _second_message()]

    written = service.index_messages(messages)

    assert written == 2
    pipeline.run.assert_called_once()
    run_args, run_kwargs = pipeline.run.call_args
    documents = run_args[0]["document_embedder"]["documents"]
    assert len(documents) == 2
    assert run_kwargs["include_outputs_from"] == {"writer"}


def test_indexing_service_returns_zero_documents_written(
    sample_message: ChatMessage,
) -> None:
    pipeline = MagicMock(spec=Pipeline)
    pipeline.run.return_value = {"writer": {"documents_written": 0}}
    service = IndexingService(pipeline=pipeline)

    assert service.index_messages([sample_message]) == 0


@pytest.mark.parametrize(
    "pipeline_result",
    [
        {},
        {"writer": {}},
        {"writer": {"documents_written": -1}},
        {"writer": {"documents_written": True}},
        {"writer": {"documents_written": "2"}},
        {"writer": "bad"},
    ],
)
def test_indexing_service_rejects_malformed_pipeline_results(
    sample_message: ChatMessage,
    pipeline_result: dict[str, object],
) -> None:
    pipeline = MagicMock(spec=Pipeline)
    pipeline.run.return_value = pipeline_result
    service = IndexingService(pipeline=pipeline)

    with pytest.raises(IndexingServiceError):
        service.index_messages([sample_message])


def test_indexing_service_does_not_swallow_pipeline_errors(
    sample_message: ChatMessage,
) -> None:
    pipeline = MagicMock(spec=Pipeline)
    pipeline.run.side_effect = RuntimeError("pipeline failed")
    service = IndexingService(pipeline=pipeline)

    with pytest.raises(RuntimeError, match="pipeline failed"):
        service.index_messages([sample_message])
