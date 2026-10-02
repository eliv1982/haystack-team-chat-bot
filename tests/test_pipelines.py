"""Construction of the production Haystack pipelines (nothing is run here).

Running them end to end, with only the OpenAI client faked, is covered by
test_indexing_failure.py, test_session_summarization.py and test_runtime.py.
"""

from __future__ import annotations

import json

import pytest
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from config import Settings
from pipelines import create_indexing_pipeline, create_query_pipeline, create_summarization_pipeline
from summarization_prompt import SUMMARIZATION_PROMPT_TEMPLATE

OPENAI_KEY_SECRET = Secret.from_env_var("OPENAI_API_KEY").to_dict()


@pytest.fixture(params=["custom-base-url", "direct-openai"])
def openai_settings(
    request: pytest.FixtureRequest,
    settings: Settings,
    direct_openai_settings: Settings,
    openai_env: None,
) -> Settings:
    return settings if request.param == "custom-base-url" else direct_openai_settings


@pytest.fixture
def pinecone_store() -> PineconeDocumentStore:
    # Constructing the store does not contact Pinecone; the index is opened lazily.
    return PineconeDocumentStore(
        api_key=Secret.from_env_var("PINECONE_API_KEY"),
        index="test-index",
        namespace="haystack-team-chat-homework",
        dimension=1536,
        metric="cosine",
        show_progress=False,
    )


def _build_pipeline(name: str, settings: Settings, store: PineconeDocumentStore) -> Pipeline:
    if name == "indexing":
        return create_indexing_pipeline(settings, InMemoryDocumentStore())
    if name == "query":
        return create_query_pipeline(settings, store)
    return create_summarization_pipeline(settings)


# --- indexing ------------------------------------------------------------------


def test_indexing_pipeline_embeds_then_writes_with_overwrite(
    settings: Settings, openai_env: None
) -> None:
    store = InMemoryDocumentStore()

    pipeline = create_indexing_pipeline(settings, store)

    assert set(pipeline.graph.nodes) == {"document_embedder", "writer"}
    assert pipeline.to_dict()["connections"] == [
        {"sender": "document_embedder.documents", "receiver": "writer.documents"}
    ]
    writer = pipeline.get_component("writer")
    assert writer.document_store is store
    assert writer.policy is DuplicatePolicy.OVERWRITE  # re-indexing a message is idempotent


def test_indexing_embedder_follows_the_settings_and_fails_loudly(openai_settings: Settings) -> None:
    embedder = create_indexing_pipeline(openai_settings, InMemoryDocumentStore()).get_component(
        "document_embedder"
    )

    assert embedder.model == openai_settings.embedding_model
    assert embedder.api_base_url == openai_settings.api_base_url
    assert embedder.api_key.to_dict() == OPENAI_KEY_SECRET
    assert embedder.progress_bar is False
    # By default Haystack logs a failed embedding and carries on without a vector.
    assert embedder.raise_on_failure is True


# --- query ---------------------------------------------------------------------


def test_query_pipeline_embeds_the_query_then_retrieves(
    settings: Settings, pinecone_store: PineconeDocumentStore, openai_env: None
) -> None:
    pipeline = create_query_pipeline(settings, pinecone_store)

    assert set(pipeline.graph.nodes) == {"text_embedder", "retriever"}
    assert pipeline.to_dict()["connections"] == [
        {"sender": "text_embedder.embedding", "receiver": "retriever.query_embedding"}
    ]
    retriever = pipeline.get_component("retriever")
    assert retriever.document_store is pinecone_store
    assert retriever.top_k == settings.retrieval_top_k


def test_query_text_embedder_follows_the_settings(
    openai_settings: Settings, pinecone_store: PineconeDocumentStore
) -> None:
    embedder = create_query_pipeline(openai_settings, pinecone_store).get_component("text_embedder")

    assert embedder.model == openai_settings.embedding_model
    assert embedder.api_base_url == openai_settings.api_base_url
    assert embedder.api_key.to_dict() == OPENAI_KEY_SECRET


# --- summarization -------------------------------------------------------------


def test_summarization_pipeline_builds_the_prompt_then_calls_the_model(
    settings: Settings, openai_env: None
) -> None:
    pipeline = create_summarization_pipeline(settings)

    assert set(pipeline.graph.nodes) == {"prompt_builder", "llm"}
    assert pipeline.to_dict()["connections"] == [
        {"sender": "prompt_builder.prompt", "receiver": "llm.messages"}
    ]
    prompt_builder = pipeline.get_component("prompt_builder")
    assert prompt_builder.template == list(SUMMARIZATION_PROMPT_TEMPLATE)
    assert prompt_builder.required_variables == ["documents", "instruction"]


def test_summarization_llm_follows_the_settings(openai_settings: Settings) -> None:
    llm = create_summarization_pipeline(openai_settings).get_component("llm")

    assert llm.model == openai_settings.openai_model
    assert llm.api_base_url == openai_settings.api_base_url
    assert llm.api_key.to_dict() == OPENAI_KEY_SECRET


# --- all pipelines -------------------------------------------------------------


@pytest.mark.parametrize("name", ["indexing", "query", "summarization"])
def test_serialized_pipelines_name_the_credential_but_never_contain_it(
    name: str,
    settings: Settings,
    pinecone_store: PineconeDocumentStore,
    openai_env: None,
) -> None:
    serialized = json.dumps(_build_pipeline(name, settings, pinecone_store).to_dict())

    assert settings.openai_api_key not in serialized
    assert '"env_vars": ["OPENAI_API_KEY"]' in serialized


@pytest.mark.parametrize("name", ["indexing", "query", "summarization"])
def test_pipelines_for_direct_openai_serialize_no_custom_endpoint(
    name: str,
    direct_openai_settings: Settings,
    pinecone_store: PineconeDocumentStore,
    openai_env: None,
) -> None:
    serialized = json.dumps(_build_pipeline(name, direct_openai_settings, pinecone_store).to_dict())

    assert "api.example.com" not in serialized
