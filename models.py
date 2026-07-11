"""Domain models for Telegram chat messages and related data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


class InvalidChatMessageError(ValueError):
    """Raised when a chat message has invalid field values."""


class InvalidRetrievalRequestError(ValueError):
    """Raised when a retrieval request has invalid field values."""


def _validate_int_field(
    value: object,
    name: str,
    *,
    positive: bool,
    error_type: type[ValueError] = InvalidChatMessageError,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{name} must be an integer, not bool")
    if positive and value <= 0:
        raise error_type(f"{name} must be a positive integer")
    return value


def _validate_non_empty_text(
    value: str,
    name: str,
    *,
    error_type: type[ValueError] = InvalidChatMessageError,
) -> str:
    normalized = value.strip()
    if not normalized:
        raise error_type(f"{name} must not be empty")
    return normalized


def _validate_timezone_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise InvalidChatMessageError(f"{name} must be timezone-aware")
    return value


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single message from a Telegram group chat."""

    chat_id: int
    message_id: int
    user_id: int
    session_id: str
    author_name: str
    username: str | None
    text: str
    sent_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "chat_id", _validate_int_field(self.chat_id, "chat_id", positive=False))
        object.__setattr__(
            self, "message_id", _validate_int_field(self.message_id, "message_id", positive=True)
        )
        object.__setattr__(self, "user_id", _validate_int_field(self.user_id, "user_id", positive=True))
        object.__setattr__(self, "session_id", _validate_non_empty_text(self.session_id, "session_id"))
        object.__setattr__(self, "author_name", _validate_non_empty_text(self.author_name, "author_name"))
        object.__setattr__(self, "text", _validate_non_empty_text(self.text, "text"))
        object.__setattr__(self, "sent_at", _validate_timezone_aware(self.sent_at, "sent_at"))

        if self.username is not None:
            normalized_username = self.username.strip()
            object.__setattr__(
                self,
                "username",
                normalized_username if normalized_username else None,
            )


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """A scoped retrieval query for a specific chat session."""

    query: str
    chat_id: int
    session_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query",
            _validate_non_empty_text(self.query, "query", error_type=InvalidRetrievalRequestError),
        )
        object.__setattr__(
            self,
            "chat_id",
            _validate_int_field(
                self.chat_id,
                "chat_id",
                positive=False,
                error_type=InvalidRetrievalRequestError,
            ),
        )
        object.__setattr__(
            self,
            "session_id",
            _validate_non_empty_text(
                self.session_id,
                "session_id",
                error_type=InvalidRetrievalRequestError,
            ),
        )
