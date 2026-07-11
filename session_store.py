"""Session store placeholder for per-chat conversation state."""

from __future__ import annotations

from typing import Protocol

from models import ChatMessage


class SessionStore(Protocol):
    """Interface for storing and retrieving chat session messages."""

    def add_message(self, chat_id: int, message: ChatMessage) -> None:
        """Record a message for the given chat."""
        ...

    def get_messages(self, chat_id: int) -> list[ChatMessage]:
        """Return messages recorded for the given chat."""
        ...


class InMemorySessionStore:
    """In-memory session store placeholder. Not implemented in Stage 1."""

    def add_message(self, chat_id: int, message: ChatMessage) -> None:
        raise NotImplementedError("Session store is not implemented yet.")

    def get_messages(self, chat_id: int) -> list[ChatMessage]:
        raise NotImplementedError("Session store is not implemented yet.")
