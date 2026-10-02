"""Tests for the Telegram handlers: what each outcome of the application services
means for the user, and how failures stay contained.

Messages travel through a real TeleBot, so the handlers are reached the way they are
in production. The application services are mocks here because the point is to force
their outcomes and errors; test_telegram_dispatch.py covers routing and filtering with
the real services.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import requests
import telebot
from telebot.apihelper import ApiTelegramException
from telebot.types import Message

from fakes import BOT_TOKEN, GROUP_CHAT_ID, FakeClock, telegram_message
from models import SummarizationResult
from session_documents import SessionIncompleteError, SessionInconsistentError, SessionTooLargeError
from session_store import ListeningSession, NoActiveSessionError, SessionAlreadyActiveError
from summarization_service import NoSummarizationContextError
from telegram_application import DiscussionStatus, UnsupportedTelegramChatError
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
def test_unexpected_command_failure_replies_and_does_not_reraise(
    harness: _Harness,
    caplog: pytest.LogCaptureFixture,
    command: str,
    service_method: str,
) -> None:
    caplog.set_level(logging.INFO)
    getattr(harness.application, service_method).side_effect = RuntimeError("unexpected")
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
        (RuntimeError("openai unavailable"), _SUMMARY_INTERNAL_ERROR_REPLY),
    ],
    ids=["no-session", "no-context", "inconsistent-session", "unexpected-error"],
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


def test_unexpected_summary_failure_log_has_no_exception_text(
    harness: _Harness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    harness.summary.summarize_discussion.side_effect = RuntimeError(f"openai said: {SECRET_CONTENT}")

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


def test_record_failure_is_handled_without_reraising_and_replies_once(harness: _Harness) -> None:
    message = telegram_message("Regular text")
    harness.application.record_text_message.side_effect = RuntimeError("provider down")

    harness.send(message)  # must not raise into TeleBot

    harness.replies.assert_called_once_with(message, _RECORD_INTERNAL_ERROR_REPLY)


def test_record_failures_do_not_spam_the_chat_during_an_outage(harness: _Harness) -> None:
    harness.application.record_text_message.side_effect = RuntimeError("provider down")

    for index in range(25):
        harness.clock.now += 1.0
        harness.send(telegram_message(f"message {index}", message_id=100 + index))

    assert harness.replies.call_count == 1
    assert harness.application.record_text_message.call_count == 25


def test_record_failure_notice_is_repeated_after_the_interval(harness: _Harness) -> None:
    harness.application.record_text_message.side_effect = RuntimeError("provider down")

    harness.send(telegram_message("one", message_id=1))
    harness.clock.now += _RECORD_FAILURE_NOTICE_INTERVAL_SECONDS - 1
    harness.send(telegram_message("two", message_id=2))
    assert harness.replies.call_count == 1

    harness.clock.now += 1
    harness.send(telegram_message("three", message_id=3))
    assert harness.replies.call_count == 2


def test_record_failure_notices_are_throttled_per_chat(harness: _Harness) -> None:
    harness.application.record_text_message.side_effect = RuntimeError("provider down")

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
    application.record_text_message.side_effect = RuntimeError("provider down")
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
    harness.application.record_text_message.side_effect = RuntimeError(
        f"upstream echoed {SECRET_CONTENT} and /bot{BOT_TOKEN}/getMe"
    )

    harness.send(telegram_message(SECRET_CONTENT))

    assert (
        f"Handler failed: handler=record_text chat_id={GROUP_CHAT_ID} error=RuntimeError"
        in caplog.text
    )
    assert SECRET_CONTENT not in caplog.text
    assert BOT_TOKEN not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


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
    harness.application.record_text_message.side_effect = RuntimeError("force a notice")
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
