"""End-to-end offline tests: a whole session reaches the summarizer, complete and in order.

Only the OpenAI client and the vector store's answers are faked. The production
summarization pipeline (prompt building, request shaping, response parsing), document
loading, validation, the completeness gate, ordering, the bounded retry, the Telegram
summary application and the handler are the real code paths.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import httpx
import openai
import pytest
import telebot
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from pinecone.exceptions import ServiceException

from config import Settings
from documents import chat_message_to_document
from fakes import GROUP_CHAT_ID as CHAT_ID
from fakes import FakeOpenAIClient, telegram_message
from models import ChatMessage, SummarizationRequest
from pipelines import create_summarization_pipeline
from session_documents import (
    SESSION_DOCUMENT_LIMIT,
    SessionDocumentService,
    SessionIncompleteError,
    SessionInconsistentError,
    SessionTooLargeError,
)
from session_store import InMemorySessionStore
from summarization_service import NoSummarizationContextError, SummarizationService
from telegram_handlers import (
    _SUMMARY_INCOMPLETE_REPLY,
    _SUMMARY_INTERNAL_ERROR_REPLY,
    register_telegram_handlers,
)
from telegram_summary_application import TelegramSummaryApplicationService

OTHER_CHAT_ID = -1009999999999
SESSION_ID = "s1"
BASE_TIME = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
RETRY_DELAY = 0.5


class _PineconeLikeStore(InMemoryDocumentStore):
    """Caps every filter query like PineconeDocumentStore and counts the queries."""

    def __init__(self) -> None:
        super().__init__()
        self.queries = 0

    def filter_documents(self, filters=None):  # type: ignore[no-untyped-def]
        self.queries += 1
        return super().filter_documents(filters=filters)[:SESSION_DOCUMENT_LIMIT]


class _LaggingStore(_PineconeLikeStore):
    """Hides the newest documents of the session for the first few queries.

    Models Pinecone's eventual consistency: freshly indexed messages are written but
    not yet searchable.
    """

    def __init__(self, *, hidden: int, lagging_queries: int) -> None:
        super().__init__()
        self._hidden = hidden
        self._lagging_queries = lagging_queries

    def filter_documents(self, filters=None):  # type: ignore[no-untyped-def]
        documents = super().filter_documents(filters=filters)
        if self.queries <= self._lagging_queries:
            documents = sorted(
                documents, key=lambda d: (d.meta["sent_at"], int(d.meta["message_id"]))
            )
            documents = documents[: len(documents) - self._hidden]
        return documents


def _chat_message(
    index: int,
    *,
    chat_id: int = CHAT_ID,
    session_id: str = SESSION_ID,
    label: str = "msg",
    message_id: int | None = None,
) -> ChatMessage:
    # Telegram message ids are unique within a chat, so messages of different
    # sessions of the same chat must not share one.
    return ChatMessage(
        chat_id=chat_id,
        message_id=message_id if message_id is not None else 1_000 + index,
        user_id=7,
        session_id=session_id,
        author_name="Alice",
        username=None,
        text=f"{label}-{index:04d}",
        sent_at=BASE_TIME + timedelta(seconds=index),
    )


def _fill(store: InMemoryDocumentStore, messages: list[ChatMessage]) -> InMemoryDocumentStore:
    documents: list[Document] = [chat_message_to_document(message) for message in messages]
    random.Random(3).shuffle(documents)
    store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
    return store


def _store(messages: list[ChatMessage]) -> _PineconeLikeStore:
    store = _PineconeLikeStore()
    _fill(store, messages)
    return store


@pytest.fixture
def llm(fake_openai: FakeOpenAIClient) -> FakeOpenAIClient:
    """The OpenAI client behind the production summarization pipeline."""
    return fake_openai


@pytest.fixture
def make_service(settings: Settings, llm: FakeOpenAIClient) -> Callable[..., SummarizationService]:
    def build(store: InMemoryDocumentStore) -> SummarizationService:
        return SummarizationService(
            SessionDocumentService(store), create_summarization_pipeline(settings)
        )

    return build


def _request(
    expected: int,
    *,
    session_id: str = SESSION_ID,
    chat_id: int = CHAT_ID,
) -> SummarizationRequest:
    return SummarizationRequest(
        instruction="Подведи итог",
        chat_id=chat_id,
        session_id=session_id,
        expected_message_count=expected,
    )


# --- the service: whole session, exact count, chronological order --------------


def test_a_session_of_137_registered_messages_is_summarized_from_all_137(
    make_service: Callable[..., SummarizationService], llm: FakeOpenAIClient
) -> None:
    service = make_service(_store([_chat_message(index) for index in range(137)]))

    result = service.summarize(_request(137))

    prompt = llm.prompt_text()
    for index in range(137):
        assert f"msg-{index:04d}" in prompt
    assert len(result.source_document_ids) == 137
    assert result.text == llm.reply


def test_messages_reach_the_prompt_in_chronological_order(
    make_service: Callable[..., SummarizationService], llm: FakeOpenAIClient
) -> None:
    messages = [_chat_message(index) for index in range(60)]
    service = make_service(_store(messages))

    result = service.summarize(_request(60))

    prompt = llm.prompt_text()
    positions = [prompt.index(f"msg-{index:04d}") for index in range(60)]
    assert positions == sorted(positions)
    assert result.source_document_ids == tuple(
        chat_message_to_document(message).id for message in messages
    )


def test_137_registered_but_136_visible_never_reaches_the_model(
    make_service: Callable[..., SummarizationService], llm: FakeOpenAIClient
) -> None:
    service = make_service(_store([_chat_message(index) for index in range(136)]))

    with pytest.raises(SessionIncompleteError) as excinfo:
        service.summarize(_request(137))

    assert (excinfo.value.expected, excinfo.value.fetched) == (137, 136)
    assert llm.chat_requests == []


def test_137_registered_but_138_visible_never_reaches_the_model(
    make_service: Callable[..., SummarizationService], llm: FakeOpenAIClient
) -> None:
    service = make_service(_store([_chat_message(index) for index in range(138)]))

    with pytest.raises(SessionInconsistentError):
        service.summarize(_request(137))

    assert llm.chat_requests == []


def test_other_sessions_and_chats_never_reach_the_prompt(
    make_service: Callable[..., SummarizationService], llm: FakeOpenAIClient
) -> None:
    target = [_chat_message(index) for index in range(55)]
    other_session = [
        _chat_message(index, session_id="s2", label="OTHER-SESSION", message_id=5_000 + index)
        for index in range(40)
    ]
    other_chat = [
        _chat_message(index, chat_id=OTHER_CHAT_ID, label="OTHER-CHAT") for index in range(40)
    ]
    service = make_service(_store(target + other_session + other_chat))

    result = service.summarize(_request(55))

    prompt = llm.prompt_text()
    assert "OTHER-SESSION" not in prompt
    assert "OTHER-CHAT" not in prompt
    assert len(result.source_document_ids) == 55


def test_foreign_documents_cannot_make_up_for_missing_session_documents(
    make_service: Callable[..., SummarizationService], llm: FakeOpenAIClient
) -> None:
    target = [_chat_message(index) for index in range(50)]
    foreign = [
        _chat_message(index, session_id="s2", label="OTHER", message_id=5_000 + index)
        for index in range(20)
    ]
    service = make_service(_store(target + foreign))

    with pytest.raises(SessionIncompleteError):
        service.summarize(_request(60))  # 50 + 20 documents exist, but only 50 are s1's

    assert llm.chat_requests == []


def test_a_session_registered_with_the_store_limit_is_refused_and_the_model_never_called(
    make_service: Callable[..., SummarizationService], llm: FakeOpenAIClient
) -> None:
    messages = [_chat_message(index) for index in range(SESSION_DOCUMENT_LIMIT + 10)]
    store = _store(messages)
    service = make_service(store)

    with pytest.raises(SessionTooLargeError):
        service.summarize(_request(SESSION_DOCUMENT_LIMIT))

    assert llm.chat_requests == []
    assert store.queries == 0


def test_a_store_result_at_its_cap_for_a_session_registered_with_999_fails_closed(
    make_service: Callable[..., SummarizationService], llm: FakeOpenAIClient
) -> None:
    messages = [_chat_message(index) for index in range(SESSION_DOCUMENT_LIMIT + 10)]
    service = make_service(_store(messages))

    with pytest.raises(SessionInconsistentError):
        service.summarize(_request(SESSION_DOCUMENT_LIMIT - 1))

    assert llm.chat_requests == []


def test_a_session_registered_with_999_messages_is_summarized_completely(
    make_service: Callable[..., SummarizationService], llm: FakeOpenAIClient
) -> None:
    count = SESSION_DOCUMENT_LIMIT - 1
    service = make_service(_store([_chat_message(index) for index in range(count)]))

    result = service.summarize(_request(count))

    assert len(result.source_document_ids) == count == 999
    prompt = llm.prompt_text()
    assert f"msg-{0:04d}" in prompt
    assert f"msg-{count - 1:04d}" in prompt


def test_an_empty_session_that_registered_nothing_is_reported_and_the_model_never_called(
    make_service: Callable[..., SummarizationService], llm: FakeOpenAIClient
) -> None:
    service = make_service(_store([_chat_message(0, session_id="s2", message_id=9_000)]))

    with pytest.raises(NoSummarizationContextError):
        service.summarize(_request(0))

    assert llm.chat_requests == []


# --- through the Telegram summary application: retry and refusal ---------------


def _session_store(registered: int) -> InMemorySessionStore:
    session_store = InMemorySessionStore(session_id_factory=lambda: SESSION_ID)
    session_store.start_session(
        chat_id=CHAT_ID,
        started_at=BASE_TIME,
        started_by_user_id=7,
        started_by_name="Alice",
    )
    for _ in range(registered):
        session_store.record_message(CHAT_ID)
    return session_store


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def make_application(
    make_service: Callable[..., SummarizationService], sleeps: list[float]
) -> Callable[..., TelegramSummaryApplicationService]:
    def build(
        store: InMemoryDocumentStore, session_store: InMemorySessionStore
    ) -> TelegramSummaryApplicationService:
        return TelegramSummaryApplicationService(
            session_store=session_store,
            summarization_service=make_service(store),
            max_attempts=3,
            retry_delay_seconds=RETRY_DELAY,
            sleep=sleeps.append,
        )

    return build


def test_summary_command_summarizes_the_whole_active_session_and_not_the_previous_one(
    make_application: Callable[..., TelegramSummaryApplicationService],
    llm: FakeOpenAIClient,
    sleeps: list[float],
) -> None:
    session_ids = iter(["old-session", "new-session"])
    session_store = InMemorySessionStore(session_id_factory=lambda: next(session_ids))
    old = [
        _chat_message(index, session_id="old-session", label="OLD", message_id=2_000 + index)
        for index in range(70)
    ]
    new = [
        _chat_message(100 + index, session_id="new-session", label="NEW") for index in range(90)
    ]
    for _ in range(2):
        session_store.start_session(
            chat_id=CHAT_ID, started_at=BASE_TIME, started_by_user_id=7, started_by_name="Alice"
        )
        if session_store.get_active_session(CHAT_ID).session_id == "old-session":
            for _ in range(70):
                session_store.record_message(CHAT_ID)
            session_store.stop_session(CHAT_ID)
        else:
            for _ in range(90):
                session_store.record_message(CHAT_ID)
    application = make_application(_store(old + new), session_store)

    result = application.summarize_discussion(telegram_message("/summary"))

    prompt = llm.prompt_text()
    assert "OLD-" not in prompt
    assert all(f"NEW-{100 + index:04d}" in prompt for index in range(90))
    assert len(result.source_document_ids) == 90
    assert sleeps == []


def test_index_lag_of_one_message_is_recovered_by_the_bounded_retry(
    make_application: Callable[..., TelegramSummaryApplicationService],
    llm: FakeOpenAIClient,
    sleeps: list[float],
) -> None:
    messages = [_chat_message(index) for index in range(137)]
    store = _fill(_LaggingStore(hidden=1, lagging_queries=1), messages)
    application = make_application(store, _session_store(137))

    result = application.summarize_discussion(telegram_message("/summary"))

    assert len(result.source_document_ids) == 137
    assert store.queries == 2
    assert sleeps == [RETRY_DELAY]
    assert "msg-0136" in llm.prompt_text()  # the lagging newest message is in the summary


def test_persistent_index_lag_is_refused_after_the_bounded_retry_with_zero_model_calls(
    make_application: Callable[..., TelegramSummaryApplicationService],
    llm: FakeOpenAIClient,
    sleeps: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    messages = [_chat_message(index) for index in range(137)]
    store = _fill(_LaggingStore(hidden=1, lagging_queries=99), messages)
    application = make_application(store, _session_store(137))
    caplog.set_level(logging.INFO)

    with pytest.raises(SessionIncompleteError) as excinfo:
        application.summarize_discussion(telegram_message("/summary"))

    assert (excinfo.value.expected, excinfo.value.fetched) == (137, 136)
    assert store.queries == 3
    assert sleeps == [RETRY_DELAY, RETRY_DELAY]
    assert llm.chat_requests == []
    assert "Summary completed" not in caplog.text


def test_more_documents_than_registered_fail_closed_without_retry_or_model_call(
    make_application: Callable[..., TelegramSummaryApplicationService],
    llm: FakeOpenAIClient,
    sleeps: list[float],
) -> None:
    store = _store([_chat_message(index) for index in range(138)])
    application = make_application(store, _session_store(137))

    with pytest.raises(SessionInconsistentError):
        application.summarize_discussion(telegram_message("/summary"))

    assert store.queries == 1
    assert sleeps == []
    assert llm.chat_requests == []


def test_message_recorded_while_waiting_for_the_index_is_not_mistaken_for_an_extra_document(
    make_application: Callable[..., TelegramSummaryApplicationService],
    llm: FakeOpenAIClient,
) -> None:
    # 137 messages are registered but one is still invisible. During the wait a 138th
    # message is indexed and registered; the retry must expect 138, not fail on 138 > 137.
    session_store = _session_store(137)

    class _StoreWithArrival(_PineconeLikeStore):
        def filter_documents(self, filters=None):  # type: ignore[no-untyped-def]
            documents = super().filter_documents(filters=filters)
            if self.queries == 1:
                # first query: 136 visible, and a new message is recorded afterwards
                session_store.record_message(CHAT_ID)
                ordered = sorted(documents, key=lambda d: int(d.meta["message_id"]))
                return ordered[:136]
            return documents

    store = _StoreWithArrival()
    _fill(store, [_chat_message(index) for index in range(138)])
    application = make_application(store, session_store)

    result = application.summarize_discussion(telegram_message("/summary"))

    assert len(result.source_document_ids) == 138
    assert store.queries == 2
    assert len(llm.chat_requests) == 1


# --- through the handler: what the user is told --------------------------------


def _dispatch(
    bot: telebot.TeleBot, application: TelegramSummaryApplicationService
) -> MagicMock:
    register_telegram_handlers(bot, MagicMock(), application)
    return bot.send_message


def test_user_is_told_the_full_session_is_not_yet_available_and_can_retry(
    telegram_bot: telebot.TeleBot,
    make_application: Callable[..., TelegramSummaryApplicationService],
    llm: FakeOpenAIClient,
) -> None:
    messages = [_chat_message(index) for index in range(137)]
    store = _fill(_LaggingStore(hidden=1, lagging_queries=99), messages)
    sent = _dispatch(telegram_bot, make_application(store, _session_store(137)))

    telegram_bot.process_new_messages([telegram_message("/summary")])  # must not raise

    sent.assert_called_once_with(CHAT_ID, _SUMMARY_INCOMPLETE_REPLY)
    assert "/summary" in _SUMMARY_INCOMPLETE_REPLY
    assert "Неполный итог бот не формирует" in _SUMMARY_INCOMPLETE_REPLY
    assert llm.chat_requests == []


def test_the_user_retry_succeeds_once_the_index_has_caught_up(
    telegram_bot: telebot.TeleBot,
    make_application: Callable[..., TelegramSummaryApplicationService],
    llm: FakeOpenAIClient,
) -> None:
    messages = [_chat_message(index) for index in range(137)]
    store = _fill(_LaggingStore(hidden=1, lagging_queries=3), messages)  # lags the whole first /summary
    sent = _dispatch(telegram_bot, make_application(store, _session_store(137)))

    telegram_bot.process_new_messages([telegram_message("/summary", message_id=1)])
    sent.assert_called_once_with(CHAT_ID, _SUMMARY_INCOMPLETE_REPLY)
    assert llm.chat_requests == []
    sent.reset_mock()

    telegram_bot.process_new_messages([telegram_message("/summary", message_id=2)])

    sent.assert_called_once_with(CHAT_ID, llm.reply)
    assert len(llm.chat_requests) == 1


def test_an_inconsistent_session_gets_the_generic_error_not_the_retry_hint(
    telegram_bot: telebot.TeleBot,
    make_application: Callable[..., TelegramSummaryApplicationService],
    llm: FakeOpenAIClient,
) -> None:
    store = _store([_chat_message(index) for index in range(138)])
    sent = _dispatch(telegram_bot, make_application(store, _session_store(137)))

    telegram_bot.process_new_messages([telegram_message("/summary")])  # must not raise

    sent.assert_called_once_with(CHAT_ID, _SUMMARY_INTERNAL_ERROR_REPLY)
    assert llm.chat_requests == []


class _FailingStore(_PineconeLikeStore):
    """A vector store whose queries raise ``error``, as the Pinecone SDK would."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    def filter_documents(self, filters=None):  # type: ignore[no-untyped-def]
        raise self._error


def test_an_llm_outage_gets_the_generic_error_and_never_raises(
    telegram_bot: telebot.TeleBot,
    make_application: Callable[..., TelegramSummaryApplicationService],
    llm: FakeOpenAIClient,
) -> None:
    llm.chat_error = openai.APIConnectionError(
        request=httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    )
    store = _store([_chat_message(index) for index in range(3)])
    sent = _dispatch(telegram_bot, make_application(store, _session_store(3)))

    telegram_bot.process_new_messages([telegram_message("/summary")])  # must not raise

    sent.assert_called_once_with(CHAT_ID, _SUMMARY_INTERNAL_ERROR_REPLY)


def test_a_vector_store_outage_gets_the_generic_error_and_never_raises(
    telegram_bot: telebot.TeleBot,
    make_application: Callable[..., TelegramSummaryApplicationService],
    llm: FakeOpenAIClient,
) -> None:
    store = _FailingStore(ServiceException("pinecone unavailable", status_code=503))
    sent = _dispatch(telegram_bot, make_application(store, _session_store(3)))

    telegram_bot.process_new_messages([telegram_message("/summary")])  # must not raise

    sent.assert_called_once_with(CHAT_ID, _SUMMARY_INTERNAL_ERROR_REPLY)
    assert llm.chat_requests == []


def test_a_programming_defect_while_loading_the_session_propagates(
    telegram_bot: telebot.TeleBot,
    make_application: Callable[..., TelegramSummaryApplicationService],
    llm: FakeOpenAIClient,
) -> None:
    store = _FailingStore(TypeError("programming defect"))
    sent = _dispatch(telegram_bot, make_application(store, _session_store(3)))

    with pytest.raises(TypeError, match="programming defect"):
        telegram_bot.process_new_messages([telegram_message("/summary")])

    sent.assert_not_called()
    assert llm.chat_requests == []
