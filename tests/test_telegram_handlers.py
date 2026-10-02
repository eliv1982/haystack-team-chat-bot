"""Tests for Telegram handler registration and callbacks."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import requests
from telebot.apihelper import ApiTelegramException
from telebot.types import Chat, Message, User

from models import SummarizationResult
from session_documents import SessionIncompleteError, SessionInconsistentError, SessionTooLargeError
from session_store import ListeningSession, SessionAlreadyActiveError
from summarization_service import NoSummarizationContextError
from telegram_application import DiscussionStatus
from telegram_handlers import (
    _COMMAND_INTERNAL_ERROR_REPLY,
    _HELP_REPLY,
    _RECORD_FAILURE_NOTICE_INTERVAL_SECONDS,
    _RECORD_INTERNAL_ERROR_REPLY,
    _START_SUCCESS_REPLY,
    _STOP_SUCCESS_TEMPLATE,
    _SUMMARY_INCOMPLETE_REPLY,
    _SUMMARY_INTERNAL_ERROR_REPLY,
    _is_non_command_non_summary_text_message,
    _is_summary_phrase_message,
    _split_for_telegram,
    register_telegram_handlers,
)
from telegram_summary_application import NoSummarizableSessionError


def _utc_timestamp() -> int:
    return int(datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc).timestamp())


def _make_message(
    *,
    chat_id: int = -1001234567890,
    chat_type: str = "supergroup",
    message_id: int = 42,
    user_id: int = 7,
    text: str = "Hello, team!",
) -> Message:
    user = User(id=user_id, is_bot=False, first_name="Alice", username="alice")
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
def bot() -> MagicMock:
    mock_bot = MagicMock()
    mock_bot.register_message_handler = MagicMock()
    return mock_bot


@pytest.fixture
def application_service() -> MagicMock:
    return MagicMock()


@pytest.fixture
def summary_application_service() -> MagicMock:
    return MagicMock()


def _register(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service, summary_application_service)


def _handler_at(index: int, bot: MagicMock) -> MagicMock:
    return bot.register_message_handler.call_args_list[index].args[0]


def test_register_message_handler_called_seven_times_in_order(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)

    assert bot.register_message_handler.call_count == 7
    calls = bot.register_message_handler.call_args_list
    assert calls[0].kwargs["commands"] == ["start_listening"]
    assert calls[1].kwargs["commands"] == ["stop_listening"]
    assert calls[2].kwargs["commands"] == ["summary"]
    assert calls[3].kwargs["commands"] == ["status"]
    assert calls[4].kwargs["commands"] == ["help"]
    assert "commands" not in calls[5].kwargs
    assert calls[5].kwargs["func"] is _is_summary_phrase_message
    assert "commands" not in calls[6].kwargs
    assert calls[6].kwargs["func"] is _is_non_command_non_summary_text_message


def test_registration_does_not_call_bot_api(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)

    bot.polling.assert_not_called()
    bot.infinity_polling.assert_not_called()
    bot.get_me.assert_not_called()


def test_start_callback_success_reply(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)
    message = _make_message(text="/start_listening")
    application_service.start_listening.return_value = MagicMock()

    _handler_at(0, bot)(message)

    bot.reply_to.assert_called_once_with(message, _START_SUCCESS_REPLY)


def test_start_callback_duplicate_reply(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)
    message = _make_message(text="/start_listening")
    application_service.start_listening.side_effect = SessionAlreadyActiveError("duplicate")

    _handler_at(0, bot)(message)

    bot.reply_to.assert_called_once_with(message, "Запись обсуждения уже идет.")


def test_stop_callback_success_reply_contains_count(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)
    message = _make_message(text="/stop_listening")
    stopped = ListeningSession(
        chat_id=-1001234567890,
        session_id="telegram-session-test",
        started_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
        started_by_user_id=7,
        started_by_name="Alice",
        message_count=4,
    )
    application_service.stop_listening.return_value = stopped

    _handler_at(1, bot)(message)

    bot.reply_to.assert_called_once_with(
        message,
        _STOP_SUCCESS_TEMPLATE.format(count=4),
    )


def test_summary_command_callback_sends_exact_result_text(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)
    message = _make_message(text="/summary")
    summary_application_service.summarize_discussion.return_value = SummarizationResult(
        text="Итог обсуждения",
        source_document_ids=("doc-1",),
    )

    _handler_at(2, bot)(message)

    summary_application_service.summarize_discussion.assert_called_once_with(message)
    bot.send_message.assert_called_once_with(message.chat.id, "Итог обсуждения")


def test_summary_phrase_callback_sends_exact_result_text(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)
    message = _make_message(text="Подведи итог")
    summary_application_service.summarize_discussion.return_value = SummarizationResult(
        text="Итог обсуждения",
        source_document_ids=("doc-1",),
    )

    _handler_at(5, bot)(message)

    summary_application_service.summarize_discussion.assert_called_once_with(message)
    bot.send_message.assert_called_once_with(message.chat.id, "Итог обсуждения")


def test_summary_callback_no_session_reply(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)
    message = _make_message(text="/summary")
    summary_application_service.summarize_discussion.side_effect = NoSummarizableSessionError(
        "missing"
    )

    _handler_at(2, bot)(message)

    bot.send_message.assert_called_once()
    assert "нет записанного обсуждения" in bot.send_message.call_args.args[1]


def test_summary_callback_no_context_reply(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)
    message = _make_message(text="/summary")
    summary_application_service.summarize_discussion.side_effect = (
        NoSummarizationContextError("empty")
    )

    _handler_at(2, bot)(message)

    bot.send_message.assert_called_once()
    assert "недостаточно сохраненных сообщений" in bot.send_message.call_args.args[1]


def test_status_active_reply(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)
    message = _make_message(text="/status")
    active = ListeningSession(
        chat_id=-1001234567890,
        session_id="session-1",
        started_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
        started_by_user_id=7,
        started_by_name="Alice",
        message_count=3,
    )
    application_service.get_discussion_status.return_value = DiscussionStatus(
        active_session=active,
        latest_completed_session=None,
    )

    _handler_at(3, bot)(message)

    reply = bot.reply_to.call_args.args[1]
    assert "активна" in reply
    assert "3" in reply


def test_status_completed_reply(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)
    message = _make_message(text="/status")
    completed = ListeningSession(
        chat_id=-1001234567890,
        session_id="session-1",
        started_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
        started_by_user_id=7,
        started_by_name="Alice",
        message_count=4,
    )
    application_service.get_discussion_status.return_value = DiscussionStatus(
        active_session=None,
        latest_completed_session=completed,
    )

    _handler_at(3, bot)(message)

    reply = bot.reply_to.call_args.args[1]
    assert "Последняя завершенная сессия" in reply
    assert "4" in reply


def test_status_no_sessions_reply(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)
    message = _make_message(text="/status")
    application_service.get_discussion_status.return_value = DiscussionStatus(
        active_session=None,
        latest_completed_session=None,
    )

    _handler_at(3, bot)(message)

    reply = bot.reply_to.call_args.args[1]
    assert "Начните запись командой /start_listening." in reply


def test_help_reply(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)
    message = _make_message(text="/help")

    _handler_at(4, bot)(message)

    bot.reply_to.assert_called_once_with(message, _HELP_REPLY)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello", True),
        ("/start_listening", False),
        ("/summary", False),
        ("Подведи итог", False),
        ("Что думаешь?", False),
        (None, False),
    ],
)
def test_ordinary_predicate(text: object, expected: bool) -> None:
    message = MagicMock()
    message.text = text
    assert _is_non_command_non_summary_text_message(message) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Что думаешь?", True),
        ("подведи итог", True),
        ("  Подведи   итог обсуждения  ", True),
        ("/summary", False),
        ("Hello", False),
        (None, False),
    ],
)
def test_summary_phrase_predicate(text: object, expected: bool) -> None:
    message = MagicMock()
    message.text = text
    assert _is_summary_phrase_message(message) is expected


def test_ordinary_callback_records_without_reply(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    _register(bot, application_service, summary_application_service)
    message = _make_message(text="Regular text")
    application_service.record_text_message.return_value = MagicMock()

    _handler_at(6, bot)(message)

    application_service.record_text_message.assert_called_once_with(message)
    bot.reply_to.assert_not_called()
    bot.send_message.assert_not_called()


# --- failure semantics -------------------------------------------------------

TOKEN = "123456789:AAH-s3cretTokenValue_0123456789abcdefghi"
SECRET_CONTENT = "SENSITIVE-MESSAGE-CONTENT-4711"


def _telegram_network_error() -> requests.exceptions.ConnectionError:
    return requests.exceptions.ConnectionError(
        f"HTTPSConnectionPool(host='api.telegram.org', port=443): url: /bot{TOKEN}/sendMessage"
    )


def _telegram_api_error() -> ApiTelegramException:
    return ApiTelegramException(
        "sendMessage",
        MagicMock(),
        {"error_code": 403, "description": "Forbidden: bot was kicked from the supergroup chat"},
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


def _register_with_clock(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    clock: _Clock,
) -> None:
    register_telegram_handlers(
        bot, application_service, summary_application_service, clock=clock
    )


def test_record_failure_is_handled_without_reraising_and_replies_once(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    clock: _Clock,
) -> None:
    _register_with_clock(bot, application_service, summary_application_service, clock)
    message = _make_message(text="Regular text")
    application_service.record_text_message.side_effect = RuntimeError("provider down")

    _handler_at(6, bot)(message)  # must not raise into TeleBot

    bot.reply_to.assert_called_once_with(message, _RECORD_INTERNAL_ERROR_REPLY)


def test_record_failures_do_not_spam_the_chat_during_an_outage(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    clock: _Clock,
) -> None:
    _register_with_clock(bot, application_service, summary_application_service, clock)
    application_service.record_text_message.side_effect = RuntimeError("provider down")
    handler = _handler_at(6, bot)

    for index in range(25):
        clock.now += 1.0
        handler(_make_message(message_id=100 + index, text=f"message {index}"))

    assert bot.reply_to.call_count == 1
    assert application_service.record_text_message.call_count == 25


def test_record_failure_notice_is_repeated_after_the_interval(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    clock: _Clock,
) -> None:
    _register_with_clock(bot, application_service, summary_application_service, clock)
    application_service.record_text_message.side_effect = RuntimeError("provider down")
    handler = _handler_at(6, bot)

    handler(_make_message(message_id=1))
    clock.now += _RECORD_FAILURE_NOTICE_INTERVAL_SECONDS - 1
    handler(_make_message(message_id=2))
    assert bot.reply_to.call_count == 1

    clock.now += 1
    handler(_make_message(message_id=3))
    assert bot.reply_to.call_count == 2


def test_record_failure_notices_are_throttled_per_chat(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    clock: _Clock,
) -> None:
    _register_with_clock(bot, application_service, summary_application_service, clock)
    application_service.record_text_message.side_effect = RuntimeError("provider down")
    handler = _handler_at(6, bot)

    handler(_make_message(chat_id=-100, message_id=1))
    handler(_make_message(chat_id=-200, message_id=2))
    handler(_make_message(chat_id=-100, message_id=3))

    assert bot.reply_to.call_count == 2


def test_record_success_never_sends_a_notice(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    clock: _Clock,
) -> None:
    _register_with_clock(bot, application_service, summary_application_service, clock)
    handler = _handler_at(6, bot)

    for index in range(5):
        handler(_make_message(message_id=index + 1))

    bot.reply_to.assert_not_called()


def test_each_registration_has_its_own_notice_throttle(
    application_service: MagicMock,
    summary_application_service: MagicMock,
    clock: _Clock,
) -> None:
    application_service.record_text_message.side_effect = RuntimeError("provider down")
    first_bot, second_bot = MagicMock(), MagicMock()
    _register_with_clock(first_bot, application_service, summary_application_service, clock)
    _register_with_clock(second_bot, application_service, summary_application_service, clock)

    _handler_at(6, first_bot)(_make_message())
    _handler_at(6, second_bot)(_make_message())

    first_bot.reply_to.assert_called_once()
    second_bot.reply_to.assert_called_once()


def test_handler_failure_log_has_no_exception_text_content_or_token(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    register_telegram_handlers(bot, application_service, summary_application_service)
    application_service.record_text_message.side_effect = RuntimeError(
        f"upstream echoed {SECRET_CONTENT} and /bot{TOKEN}/getMe"
    )

    _handler_at(6, bot)(_make_message(text=SECRET_CONTENT))

    assert (
        "Handler failed: handler=record_text chat_id=-1001234567890 error=RuntimeError"
        in caplog.text
    )
    assert SECRET_CONTENT not in caplog.text
    assert TOKEN not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize(
    ("handler_index", "text", "service_method"),
    [
        (0, "/start_listening", "start_listening"),
        (1, "/stop_listening", "stop_listening"),
        (3, "/status", "get_discussion_status"),
    ],
)
def test_unexpected_command_failure_replies_and_does_not_reraise(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    caplog: pytest.LogCaptureFixture,
    handler_index: int,
    text: str,
    service_method: str,
) -> None:
    caplog.set_level(logging.INFO)
    register_telegram_handlers(bot, application_service, summary_application_service)
    getattr(application_service, service_method).side_effect = RuntimeError("unexpected")
    message = _make_message(text=text)

    _handler_at(handler_index, bot)(message)

    bot.reply_to.assert_called_once_with(message, _COMMAND_INTERNAL_ERROR_REPLY)
    assert "Handler failed" in caplog.text


@pytest.mark.parametrize(("handler_index", "text"), [(2, "/summary"), (5, "Подведи итог")])
def test_summary_failure_replies_once_and_does_not_reraise(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    caplog: pytest.LogCaptureFixture,
    handler_index: int,
    text: str,
) -> None:
    caplog.set_level(logging.INFO)
    register_telegram_handlers(bot, application_service, summary_application_service)
    message = _make_message(text=text)
    summary_application_service.summarize_discussion.side_effect = RuntimeError(
        f"openai said: {SECRET_CONTENT}"
    )

    _handler_at(handler_index, bot)(message)

    bot.send_message.assert_called_once_with(message.chat.id, _SUMMARY_INTERNAL_ERROR_REPLY)
    assert "Handler failed: handler=summary" in caplog.text
    assert SECRET_CONTENT not in caplog.text


@pytest.mark.parametrize("handler_index, text", [(2, "/summary"), (5, "Подведи итог")])
def test_incomplete_session_tells_the_user_to_retry_and_never_claims_a_summary(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    caplog: pytest.LogCaptureFixture,
    handler_index: int,
    text: str,
) -> None:
    caplog.set_level(logging.INFO)
    register_telegram_handlers(bot, application_service, summary_application_service)
    message = _make_message(text=text)
    summary_application_service.summarize_discussion.side_effect = SessionIncompleteError(
        expected=137, fetched=136
    )

    _handler_at(handler_index, bot)(message)  # must not raise

    bot.send_message.assert_called_once_with(message.chat.id, _SUMMARY_INCOMPLETE_REPLY)
    assert "/summary" in _SUMMARY_INCOMPLETE_REPLY  # tells the user how to retry
    assert "Неполный итог бот не формирует" in _SUMMARY_INCOMPLETE_REPLY
    assert _SUMMARY_INCOMPLETE_REPLY != _SUMMARY_INTERNAL_ERROR_REPLY
    # Counts are logged by the application service, not duplicated as a handler failure.
    assert "Handler failed" not in caplog.text


def test_inconsistent_session_gets_the_generic_error_and_is_logged_as_a_handler_failure(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    register_telegram_handlers(bot, application_service, summary_application_service)
    message = _make_message(text="/summary")
    summary_application_service.summarize_discussion.side_effect = SessionInconsistentError(
        expected=137, fetched=138
    )

    _handler_at(2, bot)(message)  # must not raise

    bot.send_message.assert_called_once_with(message.chat.id, _SUMMARY_INTERNAL_ERROR_REPLY)
    assert "Handler failed: handler=summary" in caplog.text
    assert "SessionInconsistentError" in caplog.text


def test_summary_of_an_oversized_session_is_refused_explicitly(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service, summary_application_service)
    message = _make_message(text="/summary")
    summary_application_service.summarize_discussion.side_effect = SessionTooLargeError(1_000)

    _handler_at(2, bot)(message)

    bot.send_message.assert_called_once()
    chat_id, reply = bot.send_message.call_args.args
    assert chat_id == message.chat.id
    assert "слишком много сообщений" in reply
    assert "999" in reply
    assert "/start_listening" in reply
    assert reply != _SUMMARY_INTERNAL_ERROR_REPLY


@pytest.mark.parametrize(
    "delivery_error",
    [_telegram_network_error, _telegram_api_error],
    ids=["network-error", "telegram-api-error"],
)
@pytest.mark.parametrize(
    ("handler_index", "text"),
    [
        (0, "/start_listening"),
        (1, "/stop_listening"),
        (2, "/summary"),
        (3, "/status"),
        (4, "/help"),
        (5, "Подведи итог"),
        (6, "Plain text"),
    ],
)
def test_telegram_delivery_failures_are_absorbed_without_leaking_the_token(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    caplog: pytest.LogCaptureFixture,
    delivery_error: object,
    handler_index: int,
    text: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    register_telegram_handlers(bot, application_service, summary_application_service)
    application_service.record_text_message.side_effect = RuntimeError("force a notice")
    summary_application_service.summarize_discussion.return_value = SummarizationResult(
        text="Итог", source_document_ids=("doc-1",)
    )
    bot.reply_to.side_effect = delivery_error()  # type: ignore[operator]
    bot.send_message.side_effect = delivery_error()  # type: ignore[operator]

    _handler_at(handler_index, bot)(_make_message(text=text))  # must not raise

    assert TOKEN not in caplog.text
    assert "Handler failed" in caplog.text


def test_programming_errors_in_the_reply_path_are_not_swallowed(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service, summary_application_service)
    bot.reply_to.side_effect = TypeError("bug in the handler code")

    with pytest.raises(TypeError, match="bug in the handler code"):
        _handler_at(4, bot)(_make_message(text="/help"))


# --- commands are filtered by the bot they address ---------------------------


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_every_command_handler_filters_on_the_addressed_bot(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
    index: int,
) -> None:
    bot.user.username = "ThisBot"
    _register(bot, application_service, summary_application_service)
    command_filter = bot.register_message_handler.call_args_list[index].kwargs["func"]

    assert command_filter(_make_message(text="/anything")) is True
    assert command_filter(_make_message(text="/anything@ThisBot")) is True
    assert command_filter(_make_message(text="/anything@thisbot")) is True
    assert command_filter(_make_message(text="/anything@OtherBot")) is False


def test_command_filter_fails_closed_when_the_bot_username_is_unavailable(
    application_service: MagicMock,
    summary_application_service: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _BotWithoutUsername(MagicMock):
        @property
        def user(self) -> None:  # type: ignore[override]
            raise _telegram_network_error()

    caplog.set_level(logging.WARNING)
    bot = _BotWithoutUsername()
    _register(bot, application_service, summary_application_service)
    command_filter = bot.register_message_handler.call_args_list[2].kwargs["func"]

    assert command_filter(_make_message(text="/summary")) is True
    assert command_filter(_make_message(text="/summary@ThisBot")) is False
    assert TOKEN not in caplog.text


# --- long summaries are split ------------------------------------------------


def test_short_summary_is_sent_as_one_message(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service, summary_application_service)
    summary_application_service.summarize_discussion.return_value = SummarizationResult(
        text="Короткий итог", source_document_ids=("doc-1",)
    )
    message = _make_message(text="/summary")

    _handler_at(2, bot)(message)

    bot.send_message.assert_called_once_with(message.chat.id, "Короткий итог")


def test_summary_longer_than_one_telegram_message_is_sent_in_order_in_safe_parts(
    bot: MagicMock,
    application_service: MagicMock,
    summary_application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service, summary_application_service)
    paragraphs = [
        f"Раздел {index}\n" + ("Содержательное предложение. " * 40) for index in range(12)
    ]
    long_text = "\n\n".join(paragraphs)
    assert len(long_text) > 4096 * 2
    summary_application_service.summarize_discussion.return_value = SummarizationResult(
        text=long_text, source_document_ids=("doc-1",)
    )
    message = _make_message(text="/summary")

    _handler_at(2, bot)(message)

    parts = [call.args[1] for call in bot.send_message.call_args_list]
    assert len(parts) >= 3
    assert all(call.args[0] == message.chat.id for call in bot.send_message.call_args_list)
    assert all(0 < len(part) <= 4096 for part in parts)
    assert all(part.strip() for part in parts)
    # SummarizationResult trims its text; nothing else may be lost or reordered.
    assert "".join(parts) == long_text.strip()


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
