"""Tests for the Telegram summary application service."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from telebot.types import Chat, Message, User

from models import SummarizationResult
from session_store import InMemorySessionStore, NoActiveSessionError
from summarization_service import NoSummarizationContextError, SummarizationServiceError
from telegram_application import UnsupportedTelegramChatError
from telegram_summary_application import (
    SUMMARIZATION_CONTEXT_QUERY,
    SUMMARIZATION_INSTRUCTION,
    InvalidSummaryRequestError,
    TelegramSummaryApplicationService,
)


def _utc_timestamp() -> int:
    return int(datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc).timestamp())


def _make_message(
    *,
    chat_id: int = -1001234567890,
    chat_type: str = "supergroup",
    message_id: int = 42,
    text: str = "Что думаешь?",
) -> Message:
    user = User(id=7, is_bot=False, first_name="Alice", username="alice")
    chat = Chat(id=chat_id, type=chat_type, title="Team Chat")
    return Message(
        message_id=message_id,
        from_user=user,
        date=_utc_timestamp(),
        chat=chat,
        content_type="text",
        options={"text": text},
        json_string="{}",
    )


@pytest.fixture
def session_store() -> InMemorySessionStore:
    return InMemorySessionStore(session_id_factory=lambda: "telegram-session-test")


@pytest.fixture
def summarization_service() -> MagicMock:
    service = MagicMock()
    service.summarize.return_value = SummarizationResult(
        text="Краткий итог обсуждения.",
        source_document_ids=("doc-1", "doc-2"),
    )
    return service


@pytest.fixture
def summary_service(
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> TelegramSummaryApplicationService:
    return TelegramSummaryApplicationService(
        session_store=session_store,
        summarization_service=summarization_service,
    )


def _start_session(
    session_store: InMemorySessionStore,
    *,
    chat_id: int = -1001234567890,
    message_count: int = 0,
) -> None:
    session_store.start_session(
        chat_id=chat_id,
        started_at=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    for _ in range(message_count):
        session_store.record_message(chat_id)


@pytest.mark.parametrize("chat_type", ["group", "supergroup"])
def test_summarize_success_in_supported_chats(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
    chat_type: str,
) -> None:
    _start_session(session_store, message_count=2)
    message = _make_message(chat_type=chat_type)

    result = summary_service.summarize_active_discussion(message)

    assert result.text == "Краткий итог обсуждения."
    assert result.source_document_ids == ("doc-1", "doc-2")
    assert session_store.get_active_session(-1001234567890) is not None
    assert session_store.get_active_session(-1001234567890).message_count == 2


def test_summarize_builds_exact_request(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store)
    message = _make_message()

    summary_service.summarize_active_discussion(message)

    summarization_service.summarize.assert_called_once()
    request = summarization_service.summarize.call_args.args[0]
    assert request.context_query == SUMMARIZATION_CONTEXT_QUERY
    assert request.instruction == SUMMARIZATION_INSTRUCTION
    assert request.chat_id == -1001234567890
    assert request.session_id == "telegram-session-test"


def test_summarize_calls_service_once(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store)

    summary_service.summarize_active_discussion(_make_message())

    summarization_service.summarize.assert_called_once()


def test_summarize_does_not_mutate_session_or_result(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store, message_count=1)
    before = session_store.get_active_session(-1001234567890)
    message = _make_message()
    original_text = message.text

    result = summary_service.summarize_active_discussion(message)

    after = session_store.get_active_session(-1001234567890)
    assert before == after
    assert message.text == original_text
    assert result.source_document_ids == ("doc-1", "doc-2")


def test_summarize_does_not_call_store_lifecycle_methods(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store)
    original_start = session_store.start_session
    original_record = session_store.record_message
    original_stop = session_store.stop_session
    calls = {"start": 0, "record": 0, "stop": 0}

    def counted_start(**kwargs: object) -> object:
        calls["start"] += 1
        return original_start(**kwargs)  # type: ignore[arg-type]

    def counted_record(chat_id: int) -> object:
        calls["record"] += 1
        return original_record(chat_id)

    def counted_stop(chat_id: int) -> object:
        calls["stop"] += 1
        return original_stop(chat_id)

    session_store.start_session = counted_start  # type: ignore[method-assign]
    session_store.record_message = counted_record  # type: ignore[method-assign]
    session_store.stop_session = counted_stop  # type: ignore[method-assign]

    summary_service.summarize_active_discussion(_make_message())

    assert calls == {"start": 0, "record": 0, "stop": 0}


@pytest.mark.parametrize("chat_type", ["private", "channel", "unknown"])
def test_summarize_rejects_unsupported_chat(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
    chat_type: str,
) -> None:
    _start_session(session_store)

    with pytest.raises(UnsupportedTelegramChatError):
        summary_service.summarize_active_discussion(_make_message(chat_type=chat_type))

    summarization_service.summarize.assert_not_called()


def test_summarize_rejects_non_trigger_text(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store)

    with pytest.raises(InvalidSummaryRequestError):
        summary_service.summarize_active_discussion(_make_message(text="Hello"))

    summarization_service.summarize.assert_not_called()


def test_summarize_without_active_session_propagates(
    summary_service: TelegramSummaryApplicationService,
    summarization_service: MagicMock,
) -> None:
    with pytest.raises(NoActiveSessionError):
        summary_service.summarize_active_discussion(_make_message())

    summarization_service.summarize.assert_not_called()


def test_summarize_no_context_error_propagates(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store)
    summarization_service.summarize.side_effect = NoSummarizationContextError("empty")

    with pytest.raises(NoSummarizationContextError):
        summary_service.summarize_active_discussion(_make_message())

    assert session_store.get_active_session(-1001234567890) is not None


def test_summarize_error_does_not_stop_session_or_retry(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store, message_count=2)
    summarization_service.summarize.side_effect = SummarizationServiceError("failed")

    with pytest.raises(SummarizationServiceError):
        summary_service.summarize_active_discussion(_make_message())

    assert session_store.get_active_session(-1001234567890).message_count == 2
    summarization_service.summarize.assert_called_once()
