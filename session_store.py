"""Thread-safe in-memory store for per-chat listening sessions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from threading import RLock
from typing import Callable, Protocol


class SessionStoreError(Exception):
    """Base exception for session store operations."""


class SessionAlreadyActiveError(SessionStoreError):
    """Raised when starting a session while one is already active."""


class NoActiveSessionError(SessionStoreError):
    """Raised when an operation requires an active session but none exists."""


class InvalidSessionStoreInputError(SessionStoreError, ValueError):
    """Raised when session store input is invalid."""


class InvalidListeningSessionError(ValueError):
    """Raised when a ListeningSession has invalid field values."""


def _validate_listening_session_id(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidListeningSessionError("session_id must not be empty")
    return normalized


def _validate_listening_started_at(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise InvalidListeningSessionError("started_at must be timezone-aware")
    return value


def _validate_listening_user_id(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidListeningSessionError(f"{name} must be an integer, not bool")
    if value <= 0:
        raise InvalidListeningSessionError(f"{name} must be a positive integer")
    return value


def _validate_listening_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidListeningSessionError("started_by_name must not be empty")
    return normalized


def _validate_listening_message_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidListeningSessionError("message_count must be an integer, not bool")
    if value < 0:
        raise InvalidListeningSessionError("message_count must be >= 0")
    return value


def _validate_store_chat_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidSessionStoreInputError("chat_id must be an integer, not bool")
    return value


def _validate_store_positive_user_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidSessionStoreInputError("started_by_user_id must be an integer, not bool")
    if value <= 0:
        raise InvalidSessionStoreInputError("started_by_user_id must be a positive integer")
    return value


def _validate_store_non_empty_str(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidSessionStoreInputError(f"{name} must not be empty")
    return normalized


def _validate_store_started_at(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise InvalidSessionStoreInputError("started_at must be timezone-aware")
    return value


def _default_session_id_factory() -> str:
    return f"telegram-session-{uuid.uuid4()}"


@dataclass(frozen=True, slots=True)
class ListeningSession:
    """Immutable snapshot of an active listening session for one chat."""

    chat_id: int
    session_id: str
    started_at: datetime
    started_by_user_id: int
    started_by_name: str
    message_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.chat_id, bool) or not isinstance(self.chat_id, int):
            raise InvalidListeningSessionError("chat_id must be an integer, not bool")
        object.__setattr__(self, "session_id", _validate_listening_session_id(self.session_id))
        object.__setattr__(self, "started_at", _validate_listening_started_at(self.started_at))
        object.__setattr__(
            self,
            "started_by_user_id",
            _validate_listening_user_id(self.started_by_user_id, "started_by_user_id"),
        )
        object.__setattr__(self, "started_by_name", _validate_listening_name(self.started_by_name))
        object.__setattr__(
            self,
            "message_count",
            _validate_listening_message_count(self.message_count),
        )


class SessionStore(Protocol):
    """Interface for managing per-chat listening sessions."""

    def start_session(
        self,
        *,
        chat_id: int,
        started_at: datetime,
        started_by_user_id: int,
        started_by_name: str,
    ) -> ListeningSession:
        """Start a new listening session for the given chat."""
        ...

    def get_active_session(self, chat_id: int) -> ListeningSession | None:
        """Return the active session for the chat, if any."""
        ...

    def record_message(self, chat_id: int) -> ListeningSession:
        """Increment the message counter for the active session."""
        ...

    def stop_session(self, chat_id: int) -> ListeningSession:
        """Stop the active session and return its final snapshot."""
        ...

    def get_latest_completed_session(self, chat_id: int) -> ListeningSession | None:
        """Return the latest completed session for the chat, if any."""
        ...


class InMemorySessionStore:
    """Thread-safe in-memory implementation of SessionStore."""

    def __init__(
        self,
        *,
        session_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._session_id_factory = session_id_factory or _default_session_id_factory
        self._active_sessions: dict[int, ListeningSession] = {}
        self._latest_completed_sessions: dict[int, ListeningSession] = {}
        self._lock = RLock()

    def start_session(
        self,
        *,
        chat_id: int,
        started_at: datetime,
        started_by_user_id: int,
        started_by_name: str,
    ) -> ListeningSession:
        validated_chat_id = _validate_store_chat_id(chat_id)
        validated_started_at = _validate_store_started_at(started_at)
        validated_user_id = _validate_store_positive_user_id(started_by_user_id)
        validated_name = _validate_store_non_empty_str(started_by_name, "started_by_name")

        session_id = self._session_id_factory()
        if not isinstance(session_id, str):
            raise InvalidSessionStoreInputError("session_id_factory must return a string")
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise InvalidSessionStoreInputError("session_id_factory must return a non-empty string")

        session = ListeningSession(
            chat_id=validated_chat_id,
            session_id=normalized_session_id,
            started_at=validated_started_at,
            started_by_user_id=validated_user_id,
            started_by_name=validated_name,
            message_count=0,
        )

        with self._lock:
            if validated_chat_id in self._active_sessions:
                raise SessionAlreadyActiveError(
                    f"An active session already exists for chat_id {validated_chat_id}"
                )
            self._active_sessions[validated_chat_id] = session
            return session

    def get_active_session(self, chat_id: int) -> ListeningSession | None:
        validated_chat_id = _validate_store_chat_id(chat_id)
        with self._lock:
            return self._active_sessions.get(validated_chat_id)

    def record_message(self, chat_id: int) -> ListeningSession:
        validated_chat_id = _validate_store_chat_id(chat_id)
        with self._lock:
            current = self._active_sessions.get(validated_chat_id)
            if current is None:
                raise NoActiveSessionError(
                    f"No active session exists for chat_id {validated_chat_id}"
                )
            updated = replace(current, message_count=current.message_count + 1)
            self._active_sessions[validated_chat_id] = updated
            return updated

    def stop_session(self, chat_id: int) -> ListeningSession:
        validated_chat_id = _validate_store_chat_id(chat_id)
        with self._lock:
            current = self._active_sessions.pop(validated_chat_id, None)
            if current is None:
                raise NoActiveSessionError(
                    f"No active session exists for chat_id {validated_chat_id}"
                )
            self._latest_completed_sessions[validated_chat_id] = current
            return current

    def get_latest_completed_session(self, chat_id: int) -> ListeningSession | None:
        validated_chat_id = _validate_store_chat_id(chat_id)
        with self._lock:
            return self._latest_completed_sessions.get(validated_chat_id)
