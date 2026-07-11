"""Tests for the Telegram message adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from telebot.types import Chat, Message, User

from models import ChatMessage
from telegram_adapter import TelegramAdapterError, telegram_text_message_to_chat_message


def _make_message(
    *,
    chat_id: int = -1001234567890,
    message_id: int = 42,
    user_id: int = 7,
    is_bot: bool = False,
    first_name: str | None = "Alice",
    last_name: str | None = "Smith",
    username: str | None = "alice",
    text: str = "Hello, team!",
    date: object = 1705320600,
) -> Message:
    user = User(
        id=user_id,
        is_bot=is_bot,
        first_name=first_name,
        last_name=last_name,
        username=username,
    )
    chat = Chat(id=chat_id, type="supergroup", title="Team Chat")
    return Message(
        message_id=message_id,
        from_user=user,
        date=date,
        chat=chat,
        content_type="text",
        options={"text": text},
        json_string="{}",
    )


def test_group_message_converts_to_chat_message() -> None:
    message = _make_message()
    result = telegram_text_message_to_chat_message(message, session_id="session-1")
    assert isinstance(result, ChatMessage)
    assert result.chat_id == -1001234567890
    assert result.message_id == 42
    assert result.user_id == 7
    assert result.session_id == "session-1"
    assert result.text == "Hello, team!"


def test_negative_chat_id_is_allowed() -> None:
    message = _make_message(chat_id=-100)
    result = telegram_text_message_to_chat_message(message, session_id="session-1")
    assert result.chat_id == -100


def test_author_name_from_first_and_last_name() -> None:
    message = _make_message(first_name="Alice", last_name="Smith", username="alice")
    result = telegram_text_message_to_chat_message(message, session_id="session-1")
    assert result.author_name == "Alice Smith"
    assert result.username == "alice"


def test_author_name_from_first_name_only() -> None:
    message = _make_message(first_name="Alice", last_name=None, username="alice")
    result = telegram_text_message_to_chat_message(message, session_id="session-1")
    assert result.author_name == "Alice"


def test_author_name_fallback_to_username() -> None:
    message = _make_message(first_name=None, last_name=None, username="alice")
    result = telegram_text_message_to_chat_message(message, session_id="session-1")
    assert result.author_name == "alice"
    assert result.username == "alice"


def test_author_name_fallback_to_user_id() -> None:
    message = _make_message(first_name=None, last_name=None, username=None, user_id=99)
    result = telegram_text_message_to_chat_message(message, session_id="session-1")
    assert result.author_name == "user-99"
    assert result.username is None


def test_username_is_trimmed_and_normalized() -> None:
    message = _make_message(first_name="Alice", last_name=None, username="  @alice  ")
    result = telegram_text_message_to_chat_message(message, session_id="session-1")
    assert result.username == "alice"


def test_unix_timestamp_converts_to_utc_datetime() -> None:
    message = _make_message(date=1705320600)
    result = telegram_text_message_to_chat_message(message, session_id="session-1")
    assert result.sent_at == datetime.fromtimestamp(1705320600, tz=timezone.utc)


def test_aware_datetime_converts_to_utc() -> None:
    aware = datetime(2024, 1, 15, 15, 30, tzinfo=timezone.utc)
    message = _make_message(date=aware)
    result = telegram_text_message_to_chat_message(message, session_id="session-1")
    assert result.sent_at == aware


def test_same_telegram_message_produces_same_domain_fields() -> None:
    message = _make_message()
    first = telegram_text_message_to_chat_message(message, session_id="session-1")
    second = telegram_text_message_to_chat_message(message, session_id="session-1")
    assert first == second


def test_message_none_is_rejected() -> None:
    with pytest.raises(TelegramAdapterError, match="message"):
        telegram_text_message_to_chat_message(None, session_id="session-1")  # type: ignore[arg-type]


def test_missing_chat_is_rejected() -> None:
    message = _make_message()
    message.chat = None
    with pytest.raises(TelegramAdapterError, match="message.chat"):
        telegram_text_message_to_chat_message(message, session_id="session-1")


def test_missing_chat_id_is_rejected() -> None:
    message = _make_message()
    message.chat.id = None
    with pytest.raises(TelegramAdapterError, match="message.chat.id"):
        telegram_text_message_to_chat_message(message, session_id="session-1")


def test_missing_message_id_is_rejected() -> None:
    message = _make_message()
    message.message_id = None
    with pytest.raises(TelegramAdapterError, match="message.message_id"):
        telegram_text_message_to_chat_message(message, session_id="session-1")


def test_missing_user_is_rejected() -> None:
    message = _make_message()
    message.from_user = None
    with pytest.raises(TelegramAdapterError, match="message.from_user"):
        telegram_text_message_to_chat_message(message, session_id="session-1")


def test_bot_user_is_rejected() -> None:
    message = _make_message(is_bot=True)
    with pytest.raises(TelegramAdapterError, match="bot"):
        telegram_text_message_to_chat_message(message, session_id="session-1")


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_is_rejected(text: str) -> None:
    message = _make_message(text=text)
    with pytest.raises(TelegramAdapterError, match="message.text"):
        telegram_text_message_to_chat_message(message, session_id="session-1")


@pytest.mark.parametrize("session_id", ["", "   "])
def test_empty_session_id_is_rejected(session_id: str) -> None:
    message = _make_message()
    with pytest.raises(TelegramAdapterError, match="session_id"):
        telegram_text_message_to_chat_message(message, session_id=session_id)


def test_missing_date_is_rejected() -> None:
    message = _make_message()
    message.date = None
    with pytest.raises(TelegramAdapterError, match="date"):
        telegram_text_message_to_chat_message(message, session_id="session-1")


def test_bool_date_is_rejected() -> None:
    message = _make_message()
    message.date = True
    with pytest.raises(TelegramAdapterError, match="date"):
        telegram_text_message_to_chat_message(message, session_id="session-1")


def test_negative_timestamp_is_rejected() -> None:
    message = _make_message(date=-1)
    with pytest.raises(TelegramAdapterError, match="date"):
        telegram_text_message_to_chat_message(message, session_id="session-1")


def test_naive_datetime_is_rejected() -> None:
    message = _make_message(date=datetime(2024, 1, 15, 12, 0))
    with pytest.raises(TelegramAdapterError, match="date"):
        telegram_text_message_to_chat_message(message, session_id="session-1")


def test_unsupported_date_type_is_rejected() -> None:
    message = _make_message()
    message.date = "1705320600"
    with pytest.raises(TelegramAdapterError, match="date"):
        telegram_text_message_to_chat_message(message, session_id="session-1")


def test_adapter_does_not_mutate_input_message() -> None:
    message = _make_message(username="  @alice  ", text="  Hello  ")
    original_text = message.text
    original_username = message.from_user.username
    telegram_text_message_to_chat_message(message, session_id="  session-1  ")
    assert message.text == original_text
    assert message.from_user.username == original_username


def test_adapter_does_not_call_api_methods() -> None:
    message = MagicMock()
    message.chat = SimpleNamespace(id=-100)
    message.message_id = 1
    message.from_user = SimpleNamespace(
        id=7,
        is_bot=False,
        first_name="Alice",
        last_name=None,
        username="alice",
    )
    message.text = "Hello"
    message.date = 1705320600

    telegram_text_message_to_chat_message(message, session_id="session-1")

    for method_name in ("get_me", "send_message", "get_updates"):
        assert not getattr(message, method_name, MagicMock()).called
