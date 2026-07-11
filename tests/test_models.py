"""Tests for the ChatMessage domain model."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from models import ChatMessage, InvalidChatMessageError


def test_valid_chat_message(sample_message: ChatMessage) -> None:
    assert sample_message.chat_id == -1001234567890
    assert sample_message.message_id == 42
    assert sample_message.user_id == 7
    assert sample_message.session_id == "chat:-1001234567890"
    assert sample_message.author_name == "Alice"
    assert sample_message.username == "alice"
    assert sample_message.text == "Hello, team!"
    assert sample_message.sent_at.tzinfo is not None


@pytest.mark.parametrize("field_name", ["chat_id", "message_id", "user_id"])
def test_bool_instead_of_integer_id_is_rejected(field_name: str) -> None:
    values = {
        "chat_id": -100,
        "message_id": 1,
        "user_id": 1,
        "session_id": "session-1",
        "author_name": "Alice",
        "username": None,
        "text": "Hello",
        "sent_at": datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
    }
    values[field_name] = True

    with pytest.raises(InvalidChatMessageError, match=field_name):
        ChatMessage(**values)


@pytest.mark.parametrize("message_id", [0, -1])
def test_zero_or_negative_message_id_is_rejected(message_id: int) -> None:
    with pytest.raises(InvalidChatMessageError, match="message_id"):
        ChatMessage(
            chat_id=-100,
            message_id=message_id,
            user_id=1,
            session_id="session-1",
            author_name="Alice",
            username=None,
            text="Hello",
            sent_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize("user_id", [0, -1])
def test_zero_or_negative_user_id_is_rejected(user_id: int) -> None:
    with pytest.raises(InvalidChatMessageError, match="user_id"):
        ChatMessage(
            chat_id=-100,
            message_id=1,
            user_id=user_id,
            session_id="session-1",
            author_name="Alice",
            username=None,
            text="Hello",
            sent_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
        )


def test_negative_group_chat_id_is_allowed() -> None:
    message = ChatMessage(
        chat_id=-1001234567890,
        message_id=1,
        user_id=1,
        session_id="session-1",
        author_name="Alice",
        username=None,
        text="Hello",
        sent_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
    )

    assert message.chat_id == -1001234567890


@pytest.mark.parametrize("field_name", ["session_id", "author_name", "text"])
def test_empty_required_text_fields_are_rejected(field_name: str) -> None:
    values = {
        "chat_id": -100,
        "message_id": 1,
        "user_id": 1,
        "session_id": "session-1",
        "author_name": "Alice",
        "username": None,
        "text": "Hello",
        "sent_at": datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
    }
    values[field_name] = "   "

    with pytest.raises(InvalidChatMessageError, match=field_name):
        ChatMessage(**values)


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(InvalidChatMessageError, match="sent_at"):
        ChatMessage(
            chat_id=-100,
            message_id=1,
            user_id=1,
            session_id="session-1",
            author_name="Alice",
            username=None,
            text="Hello",
            sent_at=datetime(2024, 1, 15, 12, 30),
        )


def test_whitespace_username_becomes_none() -> None:
    message = ChatMessage(
        chat_id=-100,
        message_id=1,
        user_id=1,
        session_id="session-1",
        author_name="Alice",
        username="   ",
        text="Hello",
        sent_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
    )

    assert message.username is None
