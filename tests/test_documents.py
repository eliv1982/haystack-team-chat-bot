"""Tests for chat message to Haystack document conversion."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from documents import DOCUMENT_ID_VERSION, chat_message_to_document
from models import ChatMessage


def _make_message(**overrides: object) -> ChatMessage:
    values = {
        "chat_id": -1001234567890,
        "message_id": 42,
        "user_id": 7,
        "session_id": "chat:-1001234567890",
        "author_name": "Alice",
        "username": "alice",
        "text": "Hello, team!",
        "sent_at": datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return ChatMessage(**values)


def _expected_document_id(chat_id: int, message_id: int) -> str:
    identifier = f"{DOCUMENT_ID_VERSION}:{chat_id}:{message_id}"
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()


def test_same_message_identity_produces_same_document_id() -> None:
    message_a = _make_message()
    message_b = _make_message(text="Different text")

    document_a = chat_message_to_document(message_a)
    document_b = chat_message_to_document(message_b)

    assert document_a.id == document_b.id


@pytest.mark.parametrize(
    "changed_identity",
    [{"message_id": 43}, {"chat_id": -1009999999999}],
    ids=["other-message", "same-message-id-in-another-chat"],
)
def test_a_different_message_identity_gives_a_different_document_id(
    changed_identity: dict[str, int],
) -> None:
    base = chat_message_to_document(_make_message())
    changed = chat_message_to_document(_make_message(**changed_identity))

    assert base.id != changed.id


def test_document_id_matches_expected_sha256() -> None:
    message = _make_message()
    document = chat_message_to_document(message)

    assert document.id == _expected_document_id(message.chat_id, message.message_id)


def test_content_contains_timestamp_author_and_text() -> None:
    document = chat_message_to_document(_make_message())

    assert document.content == "[2024-01-15T12:30:00+00:00] Alice (@alice): Hello, team!"


def test_content_omits_username_when_missing() -> None:
    document = chat_message_to_document(_make_message(username=None))

    assert document.content == "[2024-01-15T12:30:00+00:00] Alice: Hello, team!"


def test_sent_at_is_normalized_to_utc_in_metadata() -> None:
    from datetime import timedelta

    message = _make_message(
        sent_at=datetime(2024, 1, 15, 15, 30, tzinfo=timezone(timedelta(hours=3))),
    )
    document = chat_message_to_document(message)

    assert document.meta["sent_at"] == "2024-01-15T12:30:00+00:00"


def test_metadata_contains_expected_values_and_safe_types() -> None:
    document = chat_message_to_document(_make_message())

    assert document.meta == {
        "source": "telegram",
        "schema_version": 1,
        "chat_id": "-1001234567890",
        "message_id": "42",
        "user_id": "7",
        "session_id": "chat:-1001234567890",
        "author_name": "Alice",
        "username": "alice",
        "sent_at": "2024-01-15T12:30:00+00:00",
    }
    for value in document.meta.values():
        assert isinstance(value, (str, int, float, bool))


def test_missing_username_is_not_stored_in_metadata() -> None:
    document = chat_message_to_document(_make_message(username=None))

    assert "username" not in document.meta
