"""Regression tests: a failed embedding must fail indexing instead of being stored."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from haystack.core.errors import PipelineRuntimeError
from haystack.document_stores.in_memory import InMemoryDocumentStore
from telebot.types import Chat, Message, User

from config import Settings
from indexing_service import IndexingService
from models import ChatMessage
from pipelines import create_indexing_pipeline
from session_store import InMemorySessionStore
from telegram_application import TelegramApplicationService

CHAT_ID = -1001234567890


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


@pytest.fixture(autouse=True)
def openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


@pytest.fixture(params=_provider_errors(), ids=lambda error: type(error).__name__)
def failing_pipeline(
    request: pytest.FixtureRequest,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, InMemoryDocumentStore]:
    """The production indexing pipeline whose OpenAI embeddings call always fails."""
    store = InMemoryDocumentStore()
    pipeline = create_indexing_pipeline(settings, store)
    client = MagicMock()
    client.embeddings.create.side_effect = request.param
    monkeypatch.setattr(pipeline.get_component("document_embedder"), "client", client)
    return pipeline, store


def _chat_message() -> ChatMessage:
    return ChatMessage(
        chat_id=CHAT_ID,
        message_id=42,
        user_id=7,
        session_id="session-1",
        author_name="Alice",
        username="alice",
        text="Hello, team!",
        sent_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
    )


def _telegram_message() -> Message:
    return Message(
        message_id=42,
        from_user=User(id=7, is_bot=False, first_name="Alice", username="alice"),
        date=int(datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc).timestamp()),
        chat=Chat(id=CHAT_ID, type="supergroup", title="Team Chat"),
        content_type="text",
        options={"text": "Hello, team!"},
        json_string="{}",
    )


def test_embedding_failure_raises_and_nothing_is_written(
    failing_pipeline: tuple[object, InMemoryDocumentStore],
) -> None:
    pipeline, store = failing_pipeline

    with pytest.raises(PipelineRuntimeError) as excinfo:
        IndexingService(pipeline).index_messages([_chat_message()])

    assert isinstance(excinfo.value.__cause__, openai.APIError)
    assert store.count_documents() == 0


def test_embedding_failure_does_not_count_the_message_in_the_session(
    failing_pipeline: tuple[object, InMemoryDocumentStore],
) -> None:
    pipeline, store = failing_pipeline
    session_store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    application_service = TelegramApplicationService(
        session_store=session_store,
        indexing_service=IndexingService(pipeline),
    )
    start = _telegram_message()
    application_service.start_listening(start)

    with pytest.raises(PipelineRuntimeError):
        application_service.record_text_message(_telegram_message())

    active = session_store.get_active_session(CHAT_ID)
    assert active is not None
    assert active.message_count == 0
    assert store.count_documents() == 0


def test_successful_embedding_still_writes_and_counts_the_message(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryDocumentStore()
    pipeline = create_indexing_pipeline(settings, store)
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    response.model = "test-embedding-model"
    response.usage = MagicMock()
    response.usage.__iter__ = lambda self: iter([("prompt_tokens", 1), ("total_tokens", 1)])
    client = MagicMock()
    client.embeddings.create.return_value = response
    monkeypatch.setattr(pipeline.get_component("document_embedder"), "client", client)
    session_store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    application_service = TelegramApplicationService(
        session_store=session_store,
        indexing_service=IndexingService(pipeline),
    )
    application_service.start_listening(_telegram_message())

    session = application_service.record_text_message(_telegram_message())

    assert session is not None
    assert session.message_count == 1
    stored = store.filter_documents()
    assert len(stored) == 1
    assert stored[0].embedding == [0.1, 0.2, 0.3]
