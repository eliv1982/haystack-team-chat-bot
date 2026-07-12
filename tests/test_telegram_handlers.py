"""Tests for Telegram handler registration and callbacks."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from telebot.types import Chat, Message, User

from models import SummarizationResult
from session_store import ListeningSession, SessionAlreadyActiveError
from summarization_service import NoSummarizationContextError
from telegram_application import DiscussionStatus
from telegram_handlers import (
    _HELP_REPLY,
    _START_SUCCESS_REPLY,
    _STOP_SUCCESS_TEMPLATE,
    _is_non_command_non_summary_text_message,
    _is_summary_phrase_message,
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
