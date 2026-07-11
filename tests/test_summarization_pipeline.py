"""Tests for summarization pipeline construction."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from haystack.utils import Secret

from config import Settings
from pipelines import (
    _build_openai_chat_generator_kwargs,
    build_summarization_pipeline,
    create_indexing_pipeline,
    create_query_pipeline,
    create_summarization_pipeline,
)
from summarization_prompt import SUMMARIZATION_PROMPT_TEMPLATE


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
def openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


def test_summarization_pipeline_has_expected_components(
    settings: Settings,
    openai_env: None,
) -> None:
    pipeline = create_summarization_pipeline(settings)

    assert set(pipeline.graph.nodes) == {"prompt_builder", "llm"}


def test_summarization_pipeline_connection_is_correct(
    settings: Settings,
    openai_env: None,
) -> None:
    pipeline = create_summarization_pipeline(settings)

    assert pipeline.to_dict()["connections"] == [
        {
            "sender": "prompt_builder.prompt",
            "receiver": "llm.messages",
        }
    ]


def test_summarization_prompt_builder_uses_expected_template(
    settings: Settings,
    openai_env: None,
) -> None:
    pipeline = create_summarization_pipeline(settings)

    prompt_builder = pipeline.get_component("prompt_builder")
    assert prompt_builder.template == list(SUMMARIZATION_PROMPT_TEMPLATE)
    assert prompt_builder.required_variables == ["documents", "instruction"]


def test_openai_chat_generator_kwargs_omit_api_base_url_for_direct_openai(
    direct_openai_settings: Settings,
) -> None:
    kwargs = _build_openai_chat_generator_kwargs(direct_openai_settings)

    assert "api_base_url" not in kwargs
    assert kwargs["model"] == direct_openai_settings.openai_model


def test_openai_chat_generator_kwargs_include_custom_api_base_url(
    settings: Settings,
) -> None:
    kwargs = _build_openai_chat_generator_kwargs(settings)

    assert kwargs["api_base_url"] == settings.api_base_url


def test_summarization_pipeline_llm_uses_model_without_custom_base_url_for_direct_openai(
    direct_openai_settings: Settings,
    openai_env: None,
) -> None:
    pipeline = create_summarization_pipeline(direct_openai_settings)

    llm = pipeline.get_component("llm")
    assert llm.model == direct_openai_settings.openai_model
    assert llm.api_base_url is None
    assert llm.api_key.to_dict() == Secret.from_env_var("OPENAI_API_KEY").to_dict()


def test_summarization_pipeline_llm_uses_model_and_api_base_url(
    settings: Settings,
    openai_env: None,
) -> None:
    pipeline = create_summarization_pipeline(settings)

    llm = pipeline.get_component("llm")
    assert llm.model == settings.openai_model
    assert llm.api_base_url == settings.api_base_url


def test_summarization_pipeline_serializes_without_exposing_dummy_secret_values(
    settings: Settings,
    openai_env: None,
) -> None:
    pipeline = create_summarization_pipeline(settings)
    serialized = json.dumps(pipeline.to_dict())

    assert "test-openai-key" not in serialized
    assert '"env_vars": ["OPENAI_API_KEY"]' in serialized


@patch("haystack.components.generators.chat.openai.OpenAI")
def test_summarization_pipeline_creation_does_not_call_network(
    mock_openai_cls: MagicMock,
    direct_openai_settings: Settings,
    openai_env: None,
) -> None:
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    create_summarization_pipeline(direct_openai_settings)

    mock_client.chat.completions.create.assert_not_called()


def test_indexing_and_query_pipelines_still_create(
    settings: Settings,
    openai_env: None,
) -> None:
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

    indexing_pipeline = create_indexing_pipeline(settings, InMemoryDocumentStore())
    query_pipeline = create_query_pipeline(
        settings,
        PineconeDocumentStore(
            api_key=Secret.from_env_var("PINECONE_API_KEY"),
            index="test-index",
            namespace="haystack-team-chat-homework",
            dimension=1536,
            metric="cosine",
            show_progress=False,
        ),
    )

    assert set(indexing_pipeline.graph.nodes) == {"document_embedder", "writer"}
    assert set(query_pipeline.graph.nodes) == {"text_embedder", "retriever"}


def test_pipeline_factories_create_independent_component_instances(
    settings: Settings,
    openai_env: None,
) -> None:
    first = create_summarization_pipeline(settings)
    second = create_summarization_pipeline(settings)

    assert first.get_component("llm") is not second.get_component("llm")
    assert first.get_component("prompt_builder") is not second.get_component("prompt_builder")


def test_legacy_build_summarization_pipeline_remains_not_implemented(settings: Settings) -> None:
    with pytest.raises(NotImplementedError, match="Summarization pipeline"):
        build_summarization_pipeline(settings)


def test_summarization_pipeline_import_does_not_call_network() -> None:
    with patch("haystack.components.generators.chat.openai.OpenAI") as mock_openai_cls:
        import pipelines

        assert callable(pipelines.create_summarization_pipeline)

    mock_openai_cls.assert_not_called()
