"""End-to-end offline tests: a whole session reaches the summarizer, complete and in order.

Only the LLM and the vector store's answers are faked. Document loading, validation,
the completeness gate, ordering, prompt building, the bounded retry, the Telegram
summary application and the handler are the production code paths.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import telebot
from haystack import Document, Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses.chat_message import ChatMessage as HaystackChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from telebot.types import Chat, Message, User

from documents import chat_message_to_document
from models import ChatMessage, SummarizationRequest
from session_documents import (
    SESSION_DOCUMENT_LIMIT,
    SessionDocumentService,
    SessionIncompleteError,
    SessionInconsistentError,
    SessionTooLargeError,
)
from session_store import InMemorySessionStore
from summarization_prompt import SUMMARIZATION_PROMPT_TEMPLATE
from summarization_service import NoSummarizationContextError, SummarizationService
from telegram_handlers import (
    _SUMMARY_INCOMPLETE_REPLY,
    _SUMMARY_INTERNAL_ERROR_REPLY,
    register_telegram_handlers,
)
from telegram_summary_application import TelegramSummaryApplicationService

CHAT_ID = -1001234567890
OTHER_CHAT_ID = -1009999999999
SESSION_ID = "s1"
BASE_TIME = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
RETRY_DELAY = 0.5


@component
class _RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[list[HaystackChatMessage]] = []

    @component.output_types(replies=list[HaystackChatMessage])
    def run(self, messages: list[HaystackChatMessage]) -> dict[str, list[HaystackChatMessage]]:
        self.calls.append(messages)
        return {"replies": [HaystackChatMessage.from_assistant("Тема\nИтог")]}


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


def _service(store: InMemoryDocumentStore) -> tuple[SummarizationService, _RecordingLLM]:
    llm = _RecordingLLM()
    pipeline = Pipeline()
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=list(SUMMARIZATION_PROMPT_TEMPLATE),
            required_variables=["documents", "instruction"],
        ),
    )
    pipeline.add_component("llm", llm)
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return SummarizationService(SessionDocumentService(store), pipeline), llm


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


def _prompt_text(llm: _RecordingLLM) -> str:
    assert len(llm.calls) == 1
    return "\n".join(message.text or "" for message in llm.calls[0])


# --- the service: whole session, exact count, chronological order --------------


def test_a_session_of_137_registered_messages_is_summarized_from_all_137() -> None:
    messages = [_chat_message(index) for index in range(137)]
    service, llm = _service(_store(messages))

    result = service.summarize(_request(137))

    prompt = _prompt_text(llm)
    for index in range(137):
        assert f"msg-{index:04d}" in prompt
    assert len(result.source_document_ids) == 137


def test_messages_reach_the_prompt_in_chronological_order() -> None:
    messages = [_chat_message(index) for index in range(60)]
    service, llm = _service(_store(messages))

    result = service.summarize(_request(60))

    prompt = _prompt_text(llm)
    positions = [prompt.index(f"msg-{index:04d}") for index in range(60)]
    assert positions == sorted(positions)
    assert result.source_document_ids == tuple(
        chat_message_to_document(message).id for message in messages
    )


def test_137_registered_but_136_visible_never_reaches_the_model() -> None:
    service, llm = _service(_store([_chat_message(index) for index in range(136)]))

    with pytest.raises(SessionIncompleteError) as excinfo:
        service.summarize(_request(137))

    assert (excinfo.value.expected, excinfo.value.fetched) == (137, 136)
    assert llm.calls == []


def test_137_registered_but_138_visible_never_reaches_the_model() -> None:
    service, llm = _service(_store([_chat_message(index) for index in range(138)]))

    with pytest.raises(SessionInconsistentError):
        service.summarize(_request(137))

    assert llm.calls == []


def test_other_sessions_and_chats_never_reach_the_prompt() -> None:
    target = [_chat_message(index) for index in range(55)]
    other_session = [
        _chat_message(index, session_id="s2", label="OTHER-SESSION", message_id=5_000 + index)
        for index in range(40)
    ]
    other_chat = [
        _chat_message(index, chat_id=OTHER_CHAT_ID, label="OTHER-CHAT") for index in range(40)
    ]
    service, llm = _service(_store(target + other_session + other_chat))

    result = service.summarize(_request(55))

    prompt = _prompt_text(llm)
    assert "OTHER-SESSION" not in prompt
    assert "OTHER-CHAT" not in prompt
    assert len(result.source_document_ids) == 55


def test_foreign_documents_cannot_make_up_for_missing_session_documents() -> None:
    target = [_chat_message(index) for index in range(50)]
    foreign = [
        _chat_message(index, session_id="s2", label="OTHER", message_id=5_000 + index)
        for index in range(20)
    ]
    service, llm = _service(_store(target + foreign))

    with pytest.raises(SessionIncompleteError):
        service.summarize(_request(60))  # 50 + 20 documents exist, but only 50 are s1's

    assert llm.calls == []


def test_a_session_registered_with_the_store_limit_is_refused_and_the_model_never_called() -> None:
    messages = [_chat_message(index) for index in range(SESSION_DOCUMENT_LIMIT + 10)]
    store = _store(messages)
    service, llm = _service(store)

    with pytest.raises(SessionTooLargeError):
        service.summarize(_request(SESSION_DOCUMENT_LIMIT))

    assert llm.calls == []
    assert store.queries == 0


def test_a_store_result_at_its_cap_for_a_session_registered_with_999_fails_closed() -> None:
    messages = [_chat_message(index) for index in range(SESSION_DOCUMENT_LIMIT + 10)]
    service, llm = _service(_store(messages))

    with pytest.raises(SessionInconsistentError):
        service.summarize(_request(SESSION_DOCUMENT_LIMIT - 1))

    assert llm.calls == []


def test_a_session_registered_with_999_messages_is_summarized_completely() -> None:
    count = SESSION_DOCUMENT_LIMIT - 1
    service, llm = _service(_store([_chat_message(index) for index in range(count)]))

    result = service.summarize(_request(count))

    assert len(result.source_document_ids) == count == 999
    prompt = _prompt_text(llm)
    assert f"msg-{0:04d}" in prompt
    assert f"msg-{count - 1:04d}" in prompt


def test_an_empty_session_that_registered_nothing_is_reported_and_the_model_never_called() -> None:
    service, llm = _service(_store([_chat_message(0, session_id="s2", message_id=9_000)]))

    with pytest.raises(NoSummarizationContextError):
        service.summarize(_request(0))

    assert llm.calls == []


# --- through the Telegram summary application: retry and refusal ---------------


def _group_message(text: str = "/summary") -> Message:
    return Message(
        message_id=999_999,
        from_user=User(id=7, is_bot=False, first_name="Alice", username="alice"),
        date=int(BASE_TIME.timestamp()),
        chat=Chat(id=CHAT_ID, type="supergroup", title="Team Chat"),
        content_type="text",
        options={"text": text},
        json_string="{}",
    )


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


def _application(
    store: InMemoryDocumentStore,
    session_store: InMemorySessionStore,
    sleeps: list[float],
) -> tuple[TelegramSummaryApplicationService, _RecordingLLM]:
    service, llm = _service(store)
    application = TelegramSummaryApplicationService(
        session_store=session_store,
        summarization_service=service,
        max_attempts=3,
        retry_delay_seconds=RETRY_DELAY,
        sleep=sleeps.append,
    )
    return application, llm


def test_summary_command_summarizes_the_whole_active_session_and_not_the_previous_one() -> None:
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
    sleeps: list[float] = []
    application, llm = _application(_store(old + new), session_store, sleeps)

    result = application.summarize_discussion(_group_message())

    prompt = _prompt_text(llm)
    assert "OLD-" not in prompt
    assert all(f"NEW-{100 + index:04d}" in prompt for index in range(90))
    assert len(result.source_document_ids) == 90
    assert sleeps == []


def test_index_lag_of_one_message_is_recovered_by_the_bounded_retry() -> None:
    messages = [_chat_message(index) for index in range(137)]
    store = _fill(_LaggingStore(hidden=1, lagging_queries=1), messages)
    sleeps: list[float] = []
    application, llm = _application(store, _session_store(137), sleeps)

    result = application.summarize_discussion(_group_message())

    assert len(result.source_document_ids) == 137
    assert store.queries == 2
    assert sleeps == [RETRY_DELAY]
    prompt = _prompt_text(llm)
    assert "msg-0136" in prompt  # the lagging newest message is in the summary


def test_persistent_index_lag_is_refused_after_the_bounded_retry_with_zero_model_calls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    messages = [_chat_message(index) for index in range(137)]
    store = _fill(_LaggingStore(hidden=1, lagging_queries=99), messages)
    sleeps: list[float] = []
    application, llm = _application(store, _session_store(137), sleeps)
    caplog.set_level(logging.INFO)

    with pytest.raises(SessionIncompleteError) as excinfo:
        application.summarize_discussion(_group_message())

    assert (excinfo.value.expected, excinfo.value.fetched) == (137, 136)
    assert store.queries == 3
    assert sleeps == [RETRY_DELAY, RETRY_DELAY]
    assert llm.calls == []
    assert "Summary completed" not in caplog.text


def test_more_documents_than_registered_fail_closed_without_retry_or_model_call() -> None:
    messages = [_chat_message(index) for index in range(138)]
    store = _store(messages)
    sleeps: list[float] = []
    application, llm = _application(store, _session_store(137), sleeps)

    with pytest.raises(SessionInconsistentError):
        application.summarize_discussion(_group_message())

    assert store.queries == 1
    assert sleeps == []
    assert llm.calls == []


def test_message_recorded_while_waiting_for_the_index_is_not_mistaken_for_an_extra_document() -> None:
    # 137 messages are registered but one is still invisible. During the wait a 138th
    # message is indexed and registered; the retry must expect 138, not fail on 138 > 137.
    all_messages = [_chat_message(index) for index in range(138)]
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
    _fill(store, all_messages)
    sleeps: list[float] = []
    application, llm = _application(store, session_store, sleeps)

    result = application.summarize_discussion(_group_message())

    assert len(result.source_document_ids) == 138
    assert store.queries == 2
    assert len(llm.calls) == 1


# --- through the handler: what the user is told --------------------------------


def _dispatch(
    application: TelegramSummaryApplicationService,
) -> tuple[telebot.TeleBot, MagicMock]:
    bot = telebot.TeleBot("123456789:AAH-s3cretTokenValue_0123456789abcdefghi", threaded=False)
    bot.get_me = lambda: User(id=999, is_bot=True, first_name="This", username="ThisBot")  # type: ignore[method-assign]
    bot.reply_to = MagicMock()  # type: ignore[method-assign]
    bot.send_message = MagicMock()  # type: ignore[method-assign]
    register_telegram_handlers(bot, MagicMock(), application)
    return bot, bot.send_message


def test_user_is_told_the_full_session_is_not_yet_available_and_can_retry() -> None:
    messages = [_chat_message(index) for index in range(137)]
    store = _fill(_LaggingStore(hidden=1, lagging_queries=99), messages)
    sleeps: list[float] = []
    application, llm = _application(store, _session_store(137), sleeps)
    bot, sent = _dispatch(application)

    bot.process_new_messages([_group_message("/summary")])  # must not raise

    sent.assert_called_once_with(CHAT_ID, _SUMMARY_INCOMPLETE_REPLY)
    assert "/summary" in _SUMMARY_INCOMPLETE_REPLY
    assert "Неполный итог бот не формирует" in _SUMMARY_INCOMPLETE_REPLY
    assert llm.calls == []


def test_the_user_retry_succeeds_once_the_index_has_caught_up() -> None:
    messages = [_chat_message(index) for index in range(137)]
    store = _fill(_LaggingStore(hidden=1, lagging_queries=3), messages)  # lags the whole first /summary
    application, llm = _application(store, _session_store(137), [])
    bot, sent = _dispatch(application)

    bot.process_new_messages([_group_message("/summary")])
    sent.assert_called_once_with(CHAT_ID, _SUMMARY_INCOMPLETE_REPLY)
    assert llm.calls == []
    sent.reset_mock()

    bot.process_new_messages([_group_message("/summary")])

    sent.assert_called_once_with(CHAT_ID, "Тема\nИтог")
    assert len(llm.calls) == 1


def test_an_inconsistent_session_gets_the_generic_error_not_the_retry_hint() -> None:
    store = _store([_chat_message(index) for index in range(138)])
    application, llm = _application(store, _session_store(137), [])
    bot, sent = _dispatch(application)

    bot.process_new_messages([_group_message("/summary")])  # must not raise

    sent.assert_called_once_with(CHAT_ID, _SUMMARY_INTERNAL_ERROR_REPLY)
    assert llm.calls == []
