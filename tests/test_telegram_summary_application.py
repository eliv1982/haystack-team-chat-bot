"""Tests for the Telegram summary application service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from telebot.types import Chat, Message, User

from models import SummarizationResult
from session_documents import (
    SessionIncompleteError,
    SessionInconsistentError,
    SessionTooLargeError,
)
from session_store import InMemorySessionStore
from summarization_service import NoSummarizationContextError, SummarizationServiceError
from telegram_application import UnsupportedTelegramChatError
from telegram_summary_application import (
    SUMMARIZATION_INSTRUCTION,
    SUMMARY_COMPLETENESS_ATTEMPTS,
    SUMMARY_COMPLETENESS_RETRY_DELAY_SECONDS,
    NoSummarizableSessionError,
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

    result = summary_service.summarize_discussion(message)

    assert result.text == "Краткий итог обсуждения."
    assert result.source_document_ids == ("doc-1", "doc-2")
    assert session_store.get_active_session(-1001234567890) is not None
    assert session_store.get_active_session(-1001234567890).message_count == 2


def test_active_session_has_priority_over_completed(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store, message_count=1)
    session_store.stop_session(-1001234567890)
    session_store.start_session(
        chat_id=-1001234567890,
        started_at=datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc),
        started_by_user_id=8,
        started_by_name="Bob",
    )
    session_store.record_message(-1001234567890)
    session_store.record_message(-1001234567890)

    summary_service.summarize_discussion(_make_message())

    request = summarization_service.summarize.call_args.args[0]
    active = session_store.get_active_session(-1001234567890)
    assert active is not None
    assert request.session_id == active.session_id


def test_completed_session_used_after_stop(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store, message_count=3)
    stopped = session_store.stop_session(-1001234567890)

    summary_service.summarize_discussion(_make_message(text="/summary"))

    request = summarization_service.summarize.call_args.args[0]
    assert request.session_id == stopped.session_id
    assert request.chat_id == -1001234567890


def test_no_active_or_completed_raises_no_summarizable_session(
    summary_service: TelegramSummaryApplicationService,
    summarization_service: MagicMock,
) -> None:
    with pytest.raises(NoSummarizableSessionError):
        summary_service.summarize_discussion(_make_message())

    summarization_service.summarize.assert_not_called()


def test_summarize_builds_exact_request(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store)
    message = _make_message()

    summary_service.summarize_discussion(message)

    summarization_service.summarize.assert_called_once()
    request = summarization_service.summarize.call_args.args[0]
    assert request.instruction == SUMMARIZATION_INSTRUCTION
    assert request.chat_id == -1001234567890
    assert request.session_id == "telegram-session-test"
    assert request.expected_message_count == 0


def test_summarize_calls_service_once(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store)

    summary_service.summarize_discussion(_make_message())

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

    result = summary_service.summarize_discussion(message)

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

    summary_service.summarize_discussion(_make_message())

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
        summary_service.summarize_discussion(_make_message(chat_type=chat_type))

    summarization_service.summarize.assert_not_called()


def test_summarize_no_context_error_propagates(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store)
    summarization_service.summarize.side_effect = NoSummarizationContextError("empty")

    with pytest.raises(NoSummarizationContextError):
        summary_service.summarize_discussion(_make_message())

    assert session_store.get_active_session(-1001234567890) is not None


def test_summarize_error_does_not_stop_session_or_retry(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store, message_count=2)
    summarization_service.summarize.side_effect = SummarizationServiceError("failed")

    with pytest.raises(SummarizationServiceError):
        summary_service.summarize_discussion(_make_message())

    assert session_store.get_active_session(-1001234567890).message_count == 2
    summarization_service.summarize.assert_called_once()


def test_summary_after_stop_uses_completed_session(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store, message_count=4)
    stopped = session_store.stop_session(-1001234567890)

    summary_service.summarize_discussion(_make_message(text="/summary"))

    request = summarization_service.summarize.call_args.args[0]
    assert request.session_id == stopped.session_id
    assert stopped.message_count == 4


def test_new_active_switches_summary_to_new_session(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store, message_count=1)
    session_store.stop_session(-1001234567890)
    session_store.start_session(
        chat_id=-1001234567890,
        started_at=datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc),
        started_by_user_id=8,
        started_by_name="Bob",
    )

    summary_service.summarize_discussion(_make_message())

    active = session_store.get_active_session(-1001234567890)
    request = summarization_service.summarize.call_args.args[0]
    assert active is not None
    assert request.session_id == active.session_id


def test_summary_logs_safe_observability_fields_only(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    _start_session(session_store, message_count=2)

    summary_service.summarize_discussion(_make_message())

    assert "Summary completed: message_count=2 source_count=2 session_state=active" in caplog.text
    assert "telegram-session-test" not in caplog.text
    assert "-1001234567890" not in caplog.text
    assert "doc-1" not in caplog.text


# --- completeness gate and bounded retry --------------------------------------


def _incomplete(expected: int = 137, fetched: int = 136) -> SessionIncompleteError:
    return SessionIncompleteError(expected=expected, fetched=fetched)


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def retrying_service(
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
    sleeps: list[float],
) -> TelegramSummaryApplicationService:
    return TelegramSummaryApplicationService(
        session_store=session_store,
        summarization_service=summarization_service,
        max_attempts=3,
        retry_delay_seconds=1.5,
        sleep=sleeps.append,
    )


def test_request_carries_the_session_message_count_as_the_expected_count(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store, message_count=137)

    summary_service.summarize_discussion(_make_message())

    request = summarization_service.summarize.call_args.args[0]
    assert request.expected_message_count == 137


def test_request_for_a_completed_session_uses_its_final_message_count(
    summary_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store, message_count=4)
    session_store.stop_session(-1001234567890)

    summary_service.summarize_discussion(_make_message())

    request = summarization_service.summarize.call_args.args[0]
    assert request.expected_message_count == 4


def test_complete_session_is_summarized_without_any_retry_or_wait(
    retrying_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
    sleeps: list[float],
) -> None:
    _start_session(session_store, message_count=137)

    result = retrying_service.summarize_discussion(_make_message())

    assert result.text == "Краткий итог обсуждения."
    summarization_service.summarize.assert_called_once()
    assert sleeps == []


def test_a_bounded_retry_recovers_from_index_lag_and_then_summarizes(
    retrying_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
    sleeps: list[float],
) -> None:
    _start_session(session_store, message_count=137)
    expected_result = SummarizationResult(text="Итог", source_document_ids=("doc-1",))
    summarization_service.summarize.side_effect = [_incomplete(), expected_result]

    result = retrying_service.summarize_discussion(_make_message())

    assert result is expected_result
    assert summarization_service.summarize.call_count == 2
    assert sleeps == [1.5]


def test_a_second_retry_may_still_recover(
    retrying_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
    sleeps: list[float],
) -> None:
    _start_session(session_store, message_count=137)
    expected_result = SummarizationResult(text="Итог", source_document_ids=("doc-1",))
    summarization_service.summarize.side_effect = [
        _incomplete(fetched=130),
        _incomplete(fetched=136),
        expected_result,
    ]

    result = retrying_service.summarize_discussion(_make_message())

    assert result is expected_result
    assert summarization_service.summarize.call_count == 3
    assert sleeps == [1.5, 1.5]


def test_persistent_shortfall_is_refused_after_exactly_the_bounded_attempts(
    retrying_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
    sleeps: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _start_session(session_store, message_count=137)
    summarization_service.summarize.side_effect = _incomplete()
    caplog.set_level(logging.INFO)

    with pytest.raises(SessionIncompleteError) as excinfo:
        retrying_service.summarize_discussion(_make_message())

    assert (excinfo.value.expected, excinfo.value.fetched) == (137, 136)
    assert summarization_service.summarize.call_count == 3
    assert sleeps == [1.5, 1.5]  # no wait after the final attempt
    assert "Summary completed" not in caplog.text
    assert "summary refused: message_count=137 visible=136 attempts=3" in caplog.text


def test_more_documents_than_counted_fails_at_once_without_retry_or_wait(
    retrying_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
    sleeps: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _start_session(session_store, message_count=137)
    summarization_service.summarize.side_effect = SessionInconsistentError(
        expected=137, fetched=138
    )
    caplog.set_level(logging.INFO)

    with pytest.raises(SessionInconsistentError):
        retrying_service.summarize_discussion(_make_message())

    summarization_service.summarize.assert_called_once()
    assert sleeps == []
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "message_count=137 documents=138" in error_records[0].getMessage()


@pytest.mark.parametrize(
    "error",
    [
        NoSummarizationContextError("empty"),
        SessionTooLargeError(1_000),
        RuntimeError("provider down"),
    ],
    ids=["no-context", "too-large", "provider-error"],
)
def test_only_an_incomplete_session_is_retried(
    retrying_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
    sleeps: list[float],
    error: Exception,
) -> None:
    _start_session(session_store, message_count=3)
    summarization_service.summarize.side_effect = error

    with pytest.raises(type(error)):
        retrying_service.summarize_discussion(_make_message())

    summarization_service.summarize.assert_called_once()
    assert sleeps == []


def test_every_attempt_re_reads_the_message_count(
    retrying_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
    sleeps: list[float],
) -> None:
    # A message recorded while waiting for the index must raise the expected count
    # instead of showing up as an "extra" document on the next attempt.
    _start_session(session_store, message_count=137)
    expected_result = SummarizationResult(text="Итог", source_document_ids=("doc-1",))
    seen_counts: list[int] = []

    def summarize(request: object) -> SummarizationResult:
        seen_counts.append(request.expected_message_count)  # type: ignore[attr-defined]
        if len(seen_counts) == 1:
            session_store.record_message(-1001234567890)  # arrives during the wait
            raise _incomplete()
        return expected_result

    summarization_service.summarize.side_effect = summarize

    result = retrying_service.summarize_discussion(_make_message())

    assert result is expected_result
    assert seen_counts == [137, 138]
    assert sleeps == [1.5]


def test_each_retry_resolves_the_session_again(
    retrying_service: TelegramSummaryApplicationService,
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
) -> None:
    _start_session(session_store, message_count=5)
    expected_result = SummarizationResult(text="Итог", source_document_ids=("doc-1",))
    requests: list[object] = []

    def summarize(request: object) -> SummarizationResult:
        requests.append(request)
        if len(requests) == 1:
            session_store.stop_session(-1001234567890)  # the session ends meanwhile
            raise _incomplete(expected=5, fetched=4)
        return expected_result

    summarization_service.summarize.side_effect = summarize

    retrying_service.summarize_discussion(_make_message())

    assert [r.session_id for r in requests] == ["telegram-session-test"] * 2  # type: ignore[attr-defined]
    assert [r.expected_message_count for r in requests] == [5, 5]  # type: ignore[attr-defined]


def test_a_single_attempt_configuration_never_retries(
    session_store: InMemorySessionStore,
    summarization_service: MagicMock,
    sleeps: list[float],
) -> None:
    service = TelegramSummaryApplicationService(
        session_store=session_store,
        summarization_service=summarization_service,
        max_attempts=1,
        sleep=sleeps.append,
    )
    _start_session(session_store, message_count=137)
    summarization_service.summarize.side_effect = _incomplete()

    with pytest.raises(SessionIncompleteError):
        service.summarize_discussion(_make_message())

    summarization_service.summarize.assert_called_once()
    assert sleeps == []


def test_default_retry_bounds_are_short_and_fixed() -> None:
    assert SUMMARY_COMPLETENESS_ATTEMPTS == 3
    assert SUMMARY_COMPLETENESS_RETRY_DELAY_SECONDS == 1.5
    total_wait = SUMMARY_COMPLETENESS_RETRY_DELAY_SECONDS * (SUMMARY_COMPLETENESS_ATTEMPTS - 1)
    assert total_wait <= 5


@pytest.mark.parametrize("max_attempts", [0, -1, True, 1.5, None])
def test_constructor_rejects_invalid_attempt_counts(max_attempts: object) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        TelegramSummaryApplicationService(
            session_store=MagicMock(),
            summarization_service=MagicMock(),
            max_attempts=max_attempts,  # type: ignore[arg-type]
        )


def test_constructor_rejects_a_negative_retry_delay() -> None:
    with pytest.raises(ValueError, match="retry_delay_seconds"):
        TelegramSummaryApplicationService(
            session_store=MagicMock(),
            summarization_service=MagicMock(),
            retry_delay_seconds=-0.1,
        )
