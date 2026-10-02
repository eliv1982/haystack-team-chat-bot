"""Tests for the Telegram message adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from telebot.types import Chat, Message, User

from models import ChatMessage
from telegram_adapter import (
    TelegramAdapterError,
    command_target_username,
    is_command_addressed_to,
    is_summary_command_text,
    is_summary_phrase_text,
    is_summary_request_text,
    telegram_text_message_to_chat_message,
    unsupported_sender_reason,
)


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


@pytest.mark.parametrize(
    "text",
    [
        "Что думаешь?",
        "что думаешь?",
        "  ЧТО   ДУМАЕШЬ?  ",
        "Подведи итог",
        "  подведи   итог обсуждения  ",
    ],
)
def test_is_summary_phrase_text_accepts_normalized_phrases(text: str) -> None:
    assert is_summary_phrase_text(text) is True
    assert is_summary_request_text(text) is True


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "   ",
        "Что думаешь",
        "Что думаешь??",
        "А что думаешь?",
        "/summary",
        "/summary@botname",
        "Prefix Что думаешь?",
        "Что думаешь? suffix",
        "Подведи итог сейчас",
        123,
    ],
)
def test_is_summary_phrase_text_rejects_non_exact_phrases(text: object) -> None:
    assert is_summary_phrase_text(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "/summary",
        "/summary@team_bot",
        "  /summary@team_bot  ",
    ],
)
def test_is_summary_command_text_accepts_summary_command(text: str) -> None:
    assert is_summary_command_text(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "/summary extra",
        "/summary@bot extra",
        "Подведи итог",
        "/start_listening",
        None,
    ],
)
def test_is_summary_command_text_rejects_non_command_forms(text: object) -> None:
    assert is_summary_command_text(text) is False


def test_is_summary_request_text_does_not_mutate_input() -> None:
    original = "  Что   думаешь?  "
    text = original
    is_summary_request_text(text)
    assert text == original


# --- commands addressed to a specific bot ------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/summary", None),
        ("/summary@ThisBot", "ThisBot"),
        ("/summary@OtherBot", "OtherBot"),
        ("/summary@ThisBot extra args", "ThisBot"),
        ("/summary extra@args", None),
        ("/summary@", ""),
        ("hello@ThisBot", None),
        ("", None),
        (None, None),
    ],
)
def test_command_target_username(text: str | None, expected: str | None) -> None:
    assert command_target_username(text) == expected


@pytest.mark.parametrize(
    ("text", "bot_username", "expected"),
    [
        ("/summary", "ThisBot", True),
        ("/summary@ThisBot", "ThisBot", True),
        ("/summary@thisbot", "ThisBot", True),
        ("/summary@THISBOT", "thisbot", True),
        ("/summary@ThisBot", "@ThisBot", True),
        ("/summary@ThisBot arg", "ThisBot", True),
        ("/summary@OtherBot", "ThisBot", False),
        ("/summary@ThisBotExtra", "ThisBot", False),
        ("/summary@Thi", "ThisBot", False),
        ("/summary@", "ThisBot", False),
        ("/summary@ThisBot@x", "ThisBot", False),
        # Own username unknown: only commands that name no bot are accepted.
        ("/summary", None, True),
        ("/summary@ThisBot", None, False),
        ("/summary@ThisBot", "", False),
        # Not commands at all.
        ("summary", "ThisBot", False),
        ("Подведи итог", "ThisBot", False),
        ("", "ThisBot", False),
        (None, "ThisBot", False),
    ],
)
def test_is_command_addressed_to(text: str | None, bot_username: str | None, expected: bool) -> None:
    assert is_command_addressed_to(text, bot_username) is expected


# --- senders that are not group participants ----------------------------------


def test_unsupported_sender_reason_is_none_for_a_regular_participant() -> None:
    assert unsupported_sender_reason(_make_message()) is None


def test_unsupported_sender_reason_flags_bot_users() -> None:
    assert unsupported_sender_reason(_make_message(is_bot=True)) == "bot_user"


def test_unsupported_sender_reason_flags_sender_chat_even_with_a_human_looking_sender() -> None:
    message = _make_message(user_id=777000, first_name="Telegram", last_name=None, username=None)
    message.sender_chat = Chat(id=-1007654321000, type="channel", title="Announcements")

    assert unsupported_sender_reason(message) == "sender_chat"


def test_unsupported_sender_reason_flags_anonymous_admin_payload() -> None:
    message = _make_message(
        user_id=1087968824,
        is_bot=True,
        first_name="Group",
        last_name=None,
        username="GroupAnonymousBot",
    )
    message.sender_chat = Chat(id=-1001234567890, type="supergroup", title="Team Chat")

    assert unsupported_sender_reason(message) == "sender_chat"


def test_adapter_still_refuses_to_convert_unsupported_senders() -> None:
    # Defense in depth: even if a caller forgets to filter, no identity is invented.
    message = _make_message(user_id=1087968824, is_bot=True, first_name="Group")
    with pytest.raises(TelegramAdapterError):
        telegram_text_message_to_chat_message(message, session_id="session-1")
