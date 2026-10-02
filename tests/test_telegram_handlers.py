"""Tests for the Telegram handlers: what each outcome of the application services
means for the user, and how failures stay contained.

Messages travel through a real TeleBot, so the handlers are reached the way they are
in production. The application services are mocks here because the point is to force
their outcomes and errors; test_telegram_dispatch.py covers routing and filtering with
the real services.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import requests
import telebot
from haystack.core.errors import PipelineRuntimeError
from pinecone import PineconeError, ServiceError
from pinecone.exceptions import PineconeApiTypeError
from telebot.apihelper import ApiTelegramException
from telebot.types import Message

from fakes import BOT_TOKEN, GROUP_CHAT_ID, FakeClock, telegram_message
from indexing_service import IndexingProviderError, IndexingServiceError
from models import InvalidChatMessageError, InvalidSummarizationRequestError, SummarizationResult
from retrieval_service import RetrievalServiceError
from session_documents import (
    SessionDocumentProviderError,
    SessionDocumentsError,
    SessionIncompleteError,
    SessionInconsistentError,
    SessionTooLargeError,
)
from session_store import (
    InvalidSessionStoreInputError,
    ListeningSession,
    NoActiveSessionError,
    SessionAlreadyActiveError,
)
from summarization_service import (
    NoSummarizationContextError,
    SummarizationProviderError,
    SummarizationResultError,
)
from telegram_adapter import TelegramAdapterError
from telegram_application import (
    DiscussionStatus,
    UnexpectedIndexingResultError,
    UnsupportedTelegramChatError,
)
from telegram_handlers import (
    _COMMAND_INTERNAL_ERROR_REPLY,
    _HELP_REPLY,
    _RECORD_FAILURE_NOTICE_INTERVAL_SECONDS,
    _RECORD_INTERNAL_ERROR_REPLY,
    _START_DUPLICATE_REPLY,
    _START_SUCCESS_REPLY,
    _STOP_NO_ACTIVE_REPLY,
    _STOP_SUCCESS_TEMPLATE,
    _SUMMARY_INCOMPLETE_REPLY,
    _SUMMARY_INTERNAL_ERROR_REPLY,
    _SUMMARY_NO_CONTEXT_REPLY,
    _SUMMARY_NO_SESSION_REPLY,
    _UNSUPPORTED_CHAT_REPLY,
    _split_for_telegram,
    register_telegram_handlers,
)
from telegram_summary_application import NoSummarizableSessionError

SECRET_CONTENT = "SENSITIVE-MESSAGE-CONTENT-4711"

SUMMARY_TRIGGERS = ["/summary", "Подведи итог"]


class _Harness:
    def __init__(self, bot: telebot.TeleBot) -> None:
        self.bot = bot
        self.clock = FakeClock(1_000.0)
        self.application = MagicMock()
        self.summary = MagicMock()
        register_telegram_handlers(bot, self.application, self.summary, clock=self.clock)

    def send(self, message: Message) -> None:
        self.bot.process_new_messages([message])

    @property
    def replies(self) -> MagicMock:
        return self.bot.reply_to

    @property
    def sent(self) -> MagicMock:
        return self.bot.send_message


@pytest.fixture
def harness(telegram_bot: telebot.TeleBot) -> _Harness:
    return _Harness(telegram_bot)


def _session(message_count: int) -> ListeningSession:
    return ListeningSession(
        chat_id=GROUP_CHAT_ID,
        session_id="session-1",
        started_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
        started_by_user_id=7,
        started_by_name="Alice",
        message_count=message_count,
    )


def _summary_result(text: str = "Итог обсуждения") -> SummarizationResult:
    return SummarizationResult(text=text, source_document_ids=("doc-1",))


def _provider_failure(detail: str = "component failed") -> IndexingProviderError:
    """A failed OpenAI or Pinecone request while indexing, as the indexing service reports it."""
    return IndexingProviderError(detail)


def _telegram_network_error() -> requests.exceptions.ConnectionError:
    return requests.exceptions.ConnectionError(
        f"HTTPSConnectionPool(host='api.telegram.org', port=443): url: /bot{BOT_TOKEN}/sendMessage"
    )


def _telegram_api_error() -> ApiTelegramException:
    return ApiTelegramException(
        "sendMessage",
        MagicMock(),
        {"error_code": 403, "description": "Forbidden: bot was kicked from the supergroup chat"},
    )


# --- listening commands --------------------------------------------------------


def test_start_listening_replies_with_the_instructions(harness: _Harness) -> None:
    message = telegram_message("/start_listening")

    harness.send(message)

    harness.application.start_listening.assert_called_once_with(message)
    harness.replies.assert_called_once_with(message, _START_SUCCESS_REPLY)


def test_stop_listening_reply_contains_the_message_count(harness: _Harness) -> None:
    message = telegram_message("/stop_listening")
    harness.application.stop_listening.return_value = _session(message_count=4)

    harness.send(message)

    harness.replies.assert_called_once_with(message, _STOP_SUCCESS_TEMPLATE.format(count=4))


@pytest.mark.parametrize(
    ("command", "service_method", "error", "expected_reply"),
    [
        ("/start_listening", "start_listening", SessionAlreadyActiveError("x"), _START_DUPLICATE_REPLY),
        ("/stop_listening", "stop_listening", NoActiveSessionError("x"), _STOP_NO_ACTIVE_REPLY),
        ("/start_listening", "start_listening", UnsupportedTelegramChatError("x"), _UNSUPPORTED_CHAT_REPLY),
        ("/stop_listening", "stop_listening", UnsupportedTelegramChatError("x"), _UNSUPPORTED_CHAT_REPLY),
        ("/status", "get_discussion_status", UnsupportedTelegramChatError("x"), _UNSUPPORTED_CHAT_REPLY),
    ],
    ids=["already-active", "nothing-to-stop", "start-in-private", "stop-in-private", "status-in-private"],
)
def test_expected_command_errors_get_their_own_reply(
    harness: _Harness,
    command: str,
    service_method: str,
    error: Exception,
    expected_reply: str,
) -> None:
    message = telegram_message(command)
    getattr(harness.application, service_method).side_effect = error

    harness.send(message)

    harness.replies.assert_called_once_with(message, expected_reply)


@pytest.mark.parametrize(
    ("command", "service_method"),
    [
        ("/start_listening", "start_listening"),
        ("/stop_listening", "stop_listening"),
        ("/status", "get_discussion_status"),
    ],
)
def test_unconvertible_telegram_message_in_a_command_gets_a_generic_reply(
    harness: _Harness,
    caplog: pytest.LogCaptureFixture,
    command: str,
    service_method: str,
) -> None:
    caplog.set_level(logging.INFO)
    getattr(harness.application, service_method).side_effect = TelegramAdapterError(
        "message.from_user is required"
    )
    message = telegram_message(command)

    harness.send(message)

    harness.replies.assert_called_once_with(message, _COMMAND_INTERNAL_ERROR_REPLY)
    assert "Handler failed" in caplog.text


# --- status and help -----------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected_fragments"),
    [
        (DiscussionStatus(_session(3), None), ("активна", "3")),
        (DiscussionStatus(None, _session(4)), ("Последняя завершенная сессия", "4")),
        (DiscussionStatus(None, None), ("Начните запись командой /start_listening.",)),
    ],
    ids=["active", "completed", "none"],
)
def test_status_reply_describes_the_session_state(
    harness: _Harness,
    status: DiscussionStatus,
    expected_fragments: tuple[str, ...],
) -> None:
    harness.application.get_discussion_status.return_value = status

    harness.send(telegram_message("/status"))

    reply = harness.replies.call_args.args[1]
    assert all(fragment in reply for fragment in expected_fragments)


def test_help_replies_with_the_command_list(harness: _Harness) -> None:
    message = telegram_message("/help")

    harness.send(message)

    harness.replies.assert_called_once_with(message, _HELP_REPLY)


# --- summary -------------------------------------------------------------------


@pytest.mark.parametrize("trigger", SUMMARY_TRIGGERS)
def test_summary_sends_the_exact_result_text(harness: _Harness, trigger: str) -> None:
    message = telegram_message(trigger)
    harness.summary.summarize_discussion.return_value = _summary_result("Итог обсуждения")

    harness.send(message)

    harness.summary.summarize_discussion.assert_called_once_with(message)
    harness.sent.assert_called_once_with(GROUP_CHAT_ID, "Итог обсуждения")


@pytest.mark.parametrize("trigger", SUMMARY_TRIGGERS)
@pytest.mark.parametrize(
    ("error", "expected_reply"),
    [
        (NoSummarizableSessionError("missing"), _SUMMARY_NO_SESSION_REPLY),
        (NoSummarizationContextError("empty"), _SUMMARY_NO_CONTEXT_REPLY),
        (SessionInconsistentError(expected=137, fetched=138), _SUMMARY_INTERNAL_ERROR_REPLY),
        (SummarizationProviderError("openai unavailable"), _SUMMARY_INTERNAL_ERROR_REPLY),
        (SessionDocumentProviderError("pinecone unavailable"), _SUMMARY_INTERNAL_ERROR_REPLY),
        (RetrievalServiceError("store returned garbage"), _SUMMARY_INTERNAL_ERROR_REPLY),
        (SummarizationResultError("llm reply was empty"), _SUMMARY_INTERNAL_ERROR_REPLY),
        (TelegramAdapterError("message.chat is required"), _SUMMARY_INTERNAL_ERROR_REPLY),
    ],
    ids=[
        "no-session",
        "no-context",
        "inconsistent-session",
        "openai-failure",
        "pinecone-failure",
        "invalid-store-output",
        "invalid-llm-output",
        "unconvertible-message",
    ],
)
def test_summary_errors_get_one_reply_and_never_reach_telebot(
    harness: _Harness,
    trigger: str,
    error: Exception,
    expected_reply: str,
) -> None:
    harness.summary.summarize_discussion.side_effect = error

    harness.send(telegram_message(trigger))  # must not raise

    harness.sent.assert_called_once_with(GROUP_CHAT_ID, expected_reply)


def test_summary_provider_failure_log_has_no_exception_text(
    harness: _Harness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    harness.summary.summarize_discussion.side_effect = SummarizationProviderError(
        f"openai said: {SECRET_CONTENT}"
    )

    harness.send(telegram_message("/summary"))

    assert "Handler failed: handler=summary" in caplog.text
    assert SECRET_CONTENT not in caplog.text


def test_inconsistent_session_is_logged_as_a_handler_failure(
    harness: _Harness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    harness.summary.summarize_discussion.side_effect = SessionInconsistentError(
        expected=137, fetched=138
    )

    harness.send(telegram_message("/summary"))

    assert "Handler failed: handler=summary" in caplog.text
    assert "SessionInconsistentError" in caplog.text


@pytest.mark.parametrize("trigger", SUMMARY_TRIGGERS)
def test_incomplete_session_tells_the_user_to_retry_and_never_claims_a_summary(
    harness: _Harness,
    caplog: pytest.LogCaptureFixture,
    trigger: str,
) -> None:
    caplog.set_level(logging.INFO)
    harness.summary.summarize_discussion.side_effect = SessionIncompleteError(
        expected=137, fetched=136
    )

    harness.send(telegram_message(trigger))  # must not raise

    harness.sent.assert_called_once_with(GROUP_CHAT_ID, _SUMMARY_INCOMPLETE_REPLY)
    assert "/summary" in _SUMMARY_INCOMPLETE_REPLY  # tells the user how to retry
    assert "Неполный итог бот не формирует" in _SUMMARY_INCOMPLETE_REPLY
    assert _SUMMARY_INCOMPLETE_REPLY != _SUMMARY_INTERNAL_ERROR_REPLY
    # Counts are logged by the application service, not duplicated as a handler failure.
    assert "Handler failed" not in caplog.text


def test_summary_of_an_oversized_session_is_refused_explicitly(harness: _Harness) -> None:
    harness.summary.summarize_discussion.side_effect = SessionTooLargeError(1_000)

    harness.send(telegram_message("/summary"))

    harness.sent.assert_called_once()
    chat_id, reply = harness.sent.call_args.args
    assert chat_id == GROUP_CHAT_ID
    assert "слишком много сообщений" in reply
    assert "999" in reply
    assert "/start_listening" in reply
    assert reply != _SUMMARY_INTERNAL_ERROR_REPLY


# --- recording text: failures are throttled and never raised -------------------


def test_recorded_text_is_passed_to_the_service_without_any_reply(harness: _Harness) -> None:
    message = telegram_message("Regular text")

    harness.send(message)

    harness.application.record_text_message.assert_called_once_with(message)
    harness.replies.assert_not_called()
    harness.sent.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        _provider_failure("provider down"),
        IndexingServiceError("pipeline result is missing writer.documents_written"),
        UnexpectedIndexingResultError("expected documents_written=1, got 0"),
        TelegramAdapterError("message.text must be a non-empty string"),
    ],
    ids=[
        "provider-failure",
        "invalid-pipeline-output",
        "unexpected-indexing-result",
        "unconvertible-message",
    ],
)
def test_expected_record_failure_is_handled_without_reraising_and_replies_once(
    harness: _Harness,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    caplog.set_level(logging.INFO)
    message = telegram_message("Regular text")
    harness.application.record_text_message.side_effect = error

    harness.send(message)  # must not raise into TeleBot

    harness.replies.assert_called_once_with(message, _RECORD_INTERNAL_ERROR_REPLY)
    assert "Handler failed: handler=record_text" in caplog.text


def test_record_failures_do_not_spam_the_chat_during_an_outage(harness: _Harness) -> None:
    harness.application.record_text_message.side_effect = _provider_failure("provider down")

    for index in range(25):
        harness.clock.now += 1.0
        harness.send(telegram_message(f"message {index}", message_id=100 + index))

    assert harness.replies.call_count == 1
    assert harness.application.record_text_message.call_count == 25


def test_record_failure_notice_is_repeated_after_the_interval(harness: _Harness) -> None:
    harness.application.record_text_message.side_effect = _provider_failure("provider down")

    harness.send(telegram_message("one", message_id=1))
    harness.clock.now += _RECORD_FAILURE_NOTICE_INTERVAL_SECONDS - 1
    harness.send(telegram_message("two", message_id=2))
    assert harness.replies.call_count == 1

    harness.clock.now += 1
    harness.send(telegram_message("three", message_id=3))
    assert harness.replies.call_count == 2


def test_record_failure_notices_are_throttled_per_chat(harness: _Harness) -> None:
    harness.application.record_text_message.side_effect = _provider_failure("provider down")

    harness.send(telegram_message("a", chat_id=-100, message_id=1))
    harness.send(telegram_message("b", chat_id=-200, message_id=2))
    harness.send(telegram_message("c", chat_id=-100, message_id=3))

    assert harness.replies.call_count == 2


def test_record_success_never_sends_a_notice(harness: _Harness) -> None:
    for index in range(5):
        harness.send(telegram_message(f"message {index}", message_id=index + 1))

    harness.replies.assert_not_called()


def test_each_registration_has_its_own_notice_throttle(telegram_bot: telebot.TeleBot) -> None:
    application = MagicMock()
    application.record_text_message.side_effect = _provider_failure("provider down")
    clock = FakeClock()
    second_bot = telebot.TeleBot(BOT_TOKEN, threaded=False)
    second_bot.reply_to = MagicMock()  # type: ignore[method-assign]

    for bot in (telegram_bot, second_bot):
        register_telegram_handlers(bot, application, MagicMock(), clock=clock)
        bot.process_new_messages([telegram_message("Plain text")])

    telegram_bot.reply_to.assert_called_once()
    second_bot.reply_to.assert_called_once()


def test_handler_failure_log_has_no_exception_text_content_or_token(
    harness: _Harness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    harness.application.record_text_message.side_effect = _provider_failure(
        f"upstream echoed {SECRET_CONTENT} and /bot{BOT_TOKEN}/getMe"
    )

    harness.send(telegram_message(SECRET_CONTENT))

    assert (
        f"Handler failed: handler=record_text chat_id={GROUP_CHAT_ID} error=IndexingProviderError"
        in caplog.text
    )
    assert SECRET_CONTENT not in caplog.text
    assert BOT_TOKEN not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


# --- programming defects are not operational failures --------------------------
#
# Handlers answer expected operational failures themselves. Anything else is a defect
# in this code base: it must reach TeleBot (which logs it and restarts polling), not
# be turned into a reply that tells the chat an outage happened.

# Every route into an application service: message text, service, method it calls.
_HANDLER_ROUTES = [
    pytest.param("/start_listening", "application", "start_listening", id="start"),
    pytest.param("/stop_listening", "application", "stop_listening", id="stop"),
    pytest.param("/status", "application", "get_discussion_status", id="status"),
    pytest.param("/summary", "summary", "summarize_discussion", id="summary-command"),
    pytest.param("Подведи итог", "summary", "summarize_discussion", id="summary-phrase"),
    pytest.param(SECRET_CONTENT, "application", "record_text_message", id="record-text"),
]

_PROGRAMMING_DEFECTS = [
    pytest.param(TypeError, "programming defect", id="TypeError"),
    pytest.param(AssertionError, "invariant bug", id="AssertionError"),
    pytest.param(RuntimeError, "unexpected runtime failure", id="RuntimeError"),
]


@pytest.mark.parametrize(("error_type", "detail"), _PROGRAMMING_DEFECTS)
@pytest.mark.parametrize(("text", "service", "method"), _HANDLER_ROUTES)
def test_programming_defects_propagate_out_of_every_handler(
    harness: _Harness,
    caplog: pytest.LogCaptureFixture,
    text: str,
    service: str,
    method: str,
    error_type: type[Exception],
    detail: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    error = error_type(f"{detail}: {SECRET_CONTENT} /bot{BOT_TOKEN}/getMe")
    getattr(getattr(harness, service), method).side_effect = error

    with pytest.raises(error_type) as raised:
        harness.send(telegram_message(text))

    assert raised.value is error
    harness.replies.assert_not_called()
    harness.sent.assert_not_called()
    # The handler neither reports the defect as a handled failure nor logs anything
    # derived from the exception text or the message content.
    assert "Handler failed" not in caplog.text
    assert SECRET_CONTENT not in caplog.text
    assert BOT_TOKEN not in caplog.text


@pytest.mark.parametrize(("error_type", "detail"), _PROGRAMMING_DEFECTS)
def test_programming_defect_while_recording_is_not_throttled_as_a_provider_outage(
    harness: _Harness,
    error_type: type[Exception],
    detail: str,
) -> None:
    harness.application.record_text_message.side_effect = error_type(detail)

    with pytest.raises(error_type):
        harness.send(telegram_message("first", message_id=1))
    harness.replies.assert_not_called()

    # The clock has not moved. Had the defect used up the chat's notice allowance, the
    # real outage that follows would be swallowed silently.
    harness.application.record_text_message.side_effect = _provider_failure()
    second = telegram_message("second", message_id=2)
    harness.send(second)

    harness.replies.assert_called_once_with(second, _RECORD_INTERNAL_ERROR_REPLY)


@pytest.mark.parametrize(
    ("text", "service", "method", "error"),
    [
        pytest.param(
            "/start_listening",
            "application",
            "start_listening",
            InvalidSessionStoreInputError("started_by_user_id must be a positive integer"),
            id="start-store-input-contract",
        ),
        pytest.param(
            "/stop_listening",
            "application",
            "stop_listening",
            InvalidSessionStoreInputError("chat_id must be an integer, not bool"),
            id="stop-store-input-contract",
        ),
        pytest.param(
            SECRET_CONTENT,
            "application",
            "record_text_message",
            InvalidChatMessageError("text must not be empty"),
            id="record-invalid-domain-message",
        ),
        pytest.param(
            SECRET_CONTENT,
            "application",
            "record_text_message",
            UnsupportedTelegramChatError("message.chat.type must be group or supergroup"),
            id="record-chat-type-filtered-before-the-handler",
        ),
        pytest.param(
            SECRET_CONTENT,
            "application",
            "record_text_message",
            NoActiveSessionError("session vanished while its chat lock was held"),
            id="record-session-vanished-under-lock",
        ),
        pytest.param(
            "/summary",
            "summary",
            "summarize_discussion",
            SessionDocumentsError("chat_id must be an integer"),
            id="summary-document-service-argument-contract",
        ),
        pytest.param(
            "/summary",
            "summary",
            "summarize_discussion",
            InvalidSummarizationRequestError("instruction must not be empty"),
            id="summary-invalid-request",
        ),
    ],
)
def test_contract_violations_next_to_expected_failures_still_propagate(
    harness: _Harness,
    text: str,
    service: str,
    method: str,
    error: Exception,
) -> None:
    """Sibling classes of the handled exceptions (ValueError, SessionDocumentsError, ...)
    are caller errors or broken invariants, so the catches must not be widened to them."""
    getattr(getattr(harness, service), method).side_effect = error

    with pytest.raises(type(error)):
        harness.send(telegram_message(text))

    harness.replies.assert_not_called()
    harness.sent.assert_not_called()


def _pipeline_error_caused_by(cause: Exception) -> PipelineRuntimeError:
    """What ``Pipeline.run()`` raises when a component raised ``cause``."""
    try:
        raise PipelineRuntimeError.from_exception("component", type(cause), cause) from cause
    except PipelineRuntimeError as error:
        return error


# Haystack's wrapper and the Pinecone SDK's errors are not this project's error types.
# The services translate the failed provider requests among them (and the handlers then
# answer the chat); whatever still arrives raw is a bug, or an outage the service failed
# to recognize, and must reach TeleBot rather than be absorbed here.
_RAW_FRAMEWORK_ERRORS = [
    pytest.param(
        lambda: PipelineRuntimeError("component", None, "wrapped"), id="PipelineRuntimeError"
    ),
    pytest.param(
        lambda: _pipeline_error_caused_by(TypeError("component bug")), id="pipeline-caused-by-TypeError"
    ),
    pytest.param(
        lambda: _pipeline_error_caused_by(AssertionError("component invariant")),
        id="pipeline-caused-by-AssertionError",
    ),
    pytest.param(
        lambda: _pipeline_error_caused_by(ServiceError()), id="pipeline-caused-by-provider-outage"
    ),
    pytest.param(lambda: PineconeError("sdk base error"), id="PineconeError"),
    pytest.param(lambda: PineconeApiTypeError("malformed argument"), id="PineconeApiTypeError"),
    pytest.param(lambda: ServiceError(), id="raw-ServiceError"),
]


@pytest.mark.parametrize("make_error", _RAW_FRAMEWORK_ERRORS)
@pytest.mark.parametrize(("text", "service", "method"), _HANDLER_ROUTES)
def test_raw_framework_and_sdk_errors_propagate_out_of_every_handler(
    harness: _Harness,
    caplog: pytest.LogCaptureFixture,
    text: str,
    service: str,
    method: str,
    make_error: Callable[[], Exception],
) -> None:
    caplog.set_level(logging.DEBUG)
    error = make_error()
    getattr(getattr(harness, service), method).side_effect = error

    with pytest.raises(type(error)) as raised:
        harness.send(telegram_message(text))

    assert raised.value is error
    harness.replies.assert_not_called()
    harness.sent.assert_not_called()
    assert "Handler failed" not in caplog.text


@pytest.mark.parametrize("make_error", _RAW_FRAMEWORK_ERRORS)
def test_raw_framework_error_while_recording_does_not_use_up_the_notice_allowance(
    harness: _Harness,
    make_error: Callable[[], Exception],
) -> None:
    error = make_error()
    harness.application.record_text_message.side_effect = error
    with pytest.raises(type(error)):
        harness.send(telegram_message("first", message_id=1))
    harness.replies.assert_not_called()

    harness.application.record_text_message.side_effect = _provider_failure()
    second = telegram_message("second", message_id=2)
    harness.send(second)

    harness.replies.assert_called_once_with(second, _RECORD_INTERNAL_ERROR_REPLY)


# --- Telegram delivery failures -------------------------------------------------


@pytest.mark.parametrize(
    "delivery_error",
    [_telegram_network_error, _telegram_api_error],
    ids=["network-error", "telegram-api-error"],
)
@pytest.mark.parametrize(
    "text",
    [
        "/start_listening",
        "/stop_listening",
        "/summary",
        "/status",
        "/help",
        "Подведи итог",
        "Plain text",
    ],
)
def test_telegram_delivery_failures_are_absorbed_without_leaking_the_token(
    harness: _Harness,
    caplog: pytest.LogCaptureFixture,
    delivery_error: object,
    text: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    harness.application.record_text_message.side_effect = _provider_failure("force a notice")
    harness.summary.summarize_discussion.return_value = _summary_result("Итог")
    harness.replies.side_effect = delivery_error()  # type: ignore[operator]
    harness.sent.side_effect = delivery_error()  # type: ignore[operator]

    harness.send(telegram_message(text))  # must not raise

    assert BOT_TOKEN not in caplog.text
    assert "Handler failed" in caplog.text


def test_programming_errors_in_the_reply_path_are_not_swallowed(harness: _Harness) -> None:
    harness.replies.side_effect = TypeError("bug in the handler code")

    with pytest.raises(TypeError, match="bug in the handler code"):
        harness.send(telegram_message("/help"))


# --- splitting long replies -----------------------------------------------------


@pytest.mark.parametrize("length", [4095, 4096])
def test_text_that_fits_one_message_is_not_split(length: int) -> None:
    text = ("слово " * 1000)[:length]

    assert _split_for_telegram(text) == [text]


def test_text_just_over_the_limit_is_split_without_empty_parts() -> None:
    text = "x" * 4097

    parts = _split_for_telegram(text)

    assert len(parts) == 2
    assert all(0 < len(part) <= 4096 for part in parts)
    assert "".join(parts) == text


def test_a_long_unbroken_token_is_hard_split_without_losing_characters() -> None:
    text = "я" * 10_000

    parts = _split_for_telegram(text)

    assert all(0 < len(part) <= 4096 for part in parts)
    assert "".join(parts) == text


def test_whitespace_only_parts_are_never_sent() -> None:
    text = "a" * 4095 + "\n" + "\n" * 4000 + "b" * 50

    parts = _split_for_telegram(text)

    assert all(part.strip() for part in parts)
    assert all(len(part) <= 4096 for part in parts)
    assert "".join(parts).replace("\n", "") == text.replace("\n", "")
