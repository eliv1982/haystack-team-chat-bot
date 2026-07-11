"""Tests for Telegram handler registration and callbacks."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from telebot.types import Chat, Message, User

from session_store import ListeningSession, NoActiveSessionError, SessionAlreadyActiveError
from telegram_application import UnsupportedTelegramChatError
from telegram_handlers import _is_non_command_text_message, register_telegram_handlers


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


def test_register_message_handler_called_three_times_in_order(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)

    assert bot.register_message_handler.call_count == 3
    start_call, stop_call, ordinary_call = bot.register_message_handler.call_args_list

    assert start_call.kwargs["commands"] == ["start_listening"]
    assert stop_call.kwargs["commands"] == ["stop_listening"]
    assert "commands" not in ordinary_call.kwargs

    for call in (start_call, stop_call, ordinary_call):
        assert call.kwargs["content_types"] == ["text"]
        assert call.kwargs["chat_types"] == ["group", "supergroup"]

    assert ordinary_call.kwargs["func"] is _is_non_command_text_message


def test_registration_does_not_call_bot_api(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)

    bot.polling.assert_not_called()
    bot.infinity_polling.assert_not_called()
    bot.get_me.assert_not_called()


def _handler_at(index: int, bot: MagicMock) -> MagicMock:
    return bot.register_message_handler.call_args_list[index].args[0]


def test_start_callback_success_reply(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)
    message = _make_message(text="/start_listening")
    session = ListeningSession(
        chat_id=-1001234567890,
        session_id="telegram-session-test",
        started_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    application_service.start_listening.return_value = session

    _handler_at(0, bot)(message)

    application_service.start_listening.assert_called_once_with(message)
    bot.reply_to.assert_called_once()
    assert "начата" in bot.reply_to.call_args.args[1]


def test_start_callback_duplicate_reply(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)
    message = _make_message(text="/start_listening")
    application_service.start_listening.side_effect = SessionAlreadyActiveError("duplicate")

    _handler_at(0, bot)(message)

    bot.reply_to.assert_called_once_with(message, "Запись обсуждения уже идет.")


def test_start_callback_unsupported_chat_reply(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)
    message = _make_message(text="/start_listening")
    application_service.start_listening.side_effect = UnsupportedTelegramChatError("unsupported")

    _handler_at(0, bot)(message)

    bot.reply_to.assert_called_once_with(
        message,
        "Эта команда работает только в группах и супергруппах.",
    )


def test_start_command_is_not_indexed(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)
    message = _make_message(text="/start_listening")
    application_service.start_listening.return_value = MagicMock()

    _handler_at(0, bot)(message)

    application_service.record_text_message.assert_not_called()


def test_stop_callback_success_reply_contains_count(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)
    message = _make_message(text="/stop_listening")
    stopped = ListeningSession(
        chat_id=-1001234567890,
        session_id="telegram-session-test",
        started_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
        started_by_user_id=7,
        started_by_name="Alice",
        message_count=3,
    )
    application_service.stop_listening.return_value = stopped

    _handler_at(1, bot)(message)

    application_service.stop_listening.assert_called_once_with(message)
    bot.reply_to.assert_called_once_with(
        message,
        "Запись обсуждения остановлена. Сохранено сообщений: 3.",
    )


def test_stop_callback_zero_count(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)
    message = _make_message(text="/stop_listening")
    application_service.stop_listening.return_value = ListeningSession(
        chat_id=-1001234567890,
        session_id="telegram-session-test",
        started_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
        started_by_user_id=7,
        started_by_name="Alice",
        message_count=0,
    )

    _handler_at(1, bot)(message)

    bot.reply_to.assert_called_once_with(
        message,
        "Запись обсуждения остановлена. Сохранено сообщений: 0.",
    )


def test_stop_callback_no_active_reply(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)
    message = _make_message(text="/stop_listening")
    application_service.stop_listening.side_effect = NoActiveSessionError("missing")

    _handler_at(1, bot)(message)

    bot.reply_to.assert_called_once_with(message, "Активной записи обсуждения нет.")


def test_stop_does_not_call_summarization(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)
    message = _make_message(text="/stop_listening")
    application_service.stop_listening.return_value = MagicMock(message_count=0)

    _handler_at(1, bot)(message)

    application_service.stop_listening.assert_called_once()
    assert "summarize" not in application_service.method_calls


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello", True),
        ("/start_listening", False),
        ("  /stop_listening", False),
        (None, False),
    ],
)
def test_ordinary_predicate(text: object, expected: bool) -> None:
    message = MagicMock()
    message.text = text
    assert _is_non_command_text_message(message) is expected


def test_ordinary_callback_records_without_reply(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)
    message = _make_message(text="Regular text")
    application_service.record_text_message.return_value = MagicMock()

    _handler_at(2, bot)(message)

    application_service.record_text_message.assert_called_once_with(message)
    bot.reply_to.assert_not_called()


def test_ordinary_callback_none_result_does_not_reply(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)
    message = _make_message(text="Regular text")
    application_service.record_text_message.return_value = None

    _handler_at(2, bot)(message)

    bot.reply_to.assert_not_called()


def test_ordinary_callback_service_error_is_not_success(
    bot: MagicMock,
    application_service: MagicMock,
) -> None:
    register_telegram_handlers(bot, application_service)
    message = _make_message(text="Regular text")
    application_service.record_text_message.side_effect = RuntimeError("index failed")

    with pytest.raises(RuntimeError, match="index failed"):
        _handler_at(2, bot)(message)

    bot.reply_to.assert_called_once()
    assert "внутренней ошибки" in bot.reply_to.call_args.args[1]
