"""Regression tests: a failed embedding must fail indexing instead of being stored.

The production indexing pipeline runs unchanged; only the OpenAI client behind its
embedder is faked. The failure reaches the caller as ``IndexingProviderError``, chained
to Haystack's ``PipelineRuntimeError`` and the OpenAI error; the exception-boundary
tests (test_provider_failure_boundary.py) cover what is not translated.
"""

from __future__ import annotations

import httpx
import openai
import pytest
from haystack.core.errors import PipelineRuntimeError
from haystack.document_stores.in_memory import InMemoryDocumentStore

from config import Settings
from fakes import GROUP_CHAT_ID as CHAT_ID
from fakes import FakeOpenAIClient, telegram_message
from indexing_service import IndexingProviderError, IndexingService
from models import ChatMessage
from pipelines import create_indexing_pipeline
from session_store import InMemorySessionStore
from telegram_application import TelegramApplicationService


def _provider_errors() -> list[openai.APIError]:
    request = httpx.Request("POST", "https://api.example.com/v1/embeddings")
    return [
        openai.APIConnectionError(request=request),
        openai.APITimeoutError(request=request),
        openai.RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=request),
            body=None,
        ),
    ]


@pytest.fixture
def store() -> InMemoryDocumentStore:
    return InMemoryDocumentStore()


@pytest.fixture(params=_provider_errors(), ids=lambda error: type(error).__name__)
def failing_pipeline(
    request: pytest.FixtureRequest,
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    store: InMemoryDocumentStore,
):
    """The production indexing pipeline whose OpenAI embeddings call always fails."""
    fake_openai.embedding_error = request.param
    return create_indexing_pipeline(settings, store)


def _application(pipeline: object) -> tuple[TelegramApplicationService, InMemorySessionStore]:
    session_store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    application_service = TelegramApplicationService(
        session_store=session_store,
        indexing_service=IndexingService(pipeline),  # type: ignore[arg-type]
    )
    return application_service, session_store


def test_embedding_failure_raises_and_nothing_is_written(
    failing_pipeline: object,
    store: InMemoryDocumentStore,
    sample_message: ChatMessage,
) -> None:
    with pytest.raises(IndexingProviderError) as excinfo:
        IndexingService(failing_pipeline).index_messages([sample_message])  # type: ignore[arg-type]

    pipeline_error = excinfo.value.__cause__
    assert isinstance(pipeline_error, PipelineRuntimeError)
    assert isinstance(pipeline_error.__cause__, openai.APIError)
    assert store.count_documents() == 0


def test_embedding_failure_does_not_count_the_message_in_the_session(
    failing_pipeline: object,
    store: InMemoryDocumentStore,
) -> None:
    application_service, session_store = _application(failing_pipeline)
    application_service.start_listening(telegram_message("/start_listening", message_id=41))

    with pytest.raises(IndexingProviderError):
        application_service.record_text_message(telegram_message("Hello, team!", message_id=42))

    active = session_store.get_active_session(CHAT_ID)
    assert active is not None
    assert active.message_count == 0
    assert store.count_documents() == 0


def test_successful_embedding_still_writes_and_counts_the_message(
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    store: InMemoryDocumentStore,
) -> None:
    application_service, _ = _application(create_indexing_pipeline(settings, store))
    application_service.start_listening(telegram_message("/start_listening", message_id=41))

    session = application_service.record_text_message(telegram_message("Hello, team!", message_id=42))

    assert session is not None
    assert session.message_count == 1
    stored = store.filter_documents()
    assert len(stored) == 1
    assert stored[0].embedding == list(fake_openai.vector)
    assert fake_openai.embedding_inputs == [[stored[0].content]]
