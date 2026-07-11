"""Application service for Telegram listening session flow."""

from __future__ import annotations

from threading import RLock

import telebot.types

from indexing_service import IndexingService
from session_store import (
    ListeningSession,
    SessionStore,
)
from telegram_adapter import (
    GROUP_CHAT_TYPES,
    TelegramAdapterError,
    build_author_fields,
    is_telegram_command,
    normalize_sent_at,
    telegram_text_message_to_chat_message,
)


class TelegramApplicationError(Exception):
    """Base exception for Telegram application service errors."""


class UnsupportedTelegramChatError(TelegramApplicationError):
    """Raised when a message comes from an unsupported chat type."""


class UnexpectedIndexingResultError(TelegramApplicationError):
    """Raised when indexing does not write exactly one document."""


class _ChatLockRegistry:
    def __init__(self) -> None:
        self._registry_lock = RLock()
        self._chat_locks: dict[int, RLock] = {}

    def lock_for(self, chat_id: int) -> RLock:
        with self._registry_lock:
            chat_lock = self._chat_locks.get(chat_id)
            if chat_lock is None:
                chat_lock = RLock()
                self._chat_locks[chat_id] = chat_lock
            return chat_lock


class TelegramApplicationService:
    """Coordinates listening sessions, message adaptation, and indexing."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        indexing_service: IndexingService,
    ) -> None:
        self._session_store = session_store
        self._indexing_service = indexing_service
        self._chat_locks = _ChatLockRegistry()

    def start_listening(self, message: telebot.types.Message) -> ListeningSession:
        chat_id = self._require_group_chat_id(message)
        if message.from_user is None:
            raise TelegramAdapterError("message.from_user is required")

        author_name, _ = build_author_fields(message.from_user)
        started_at = normalize_sent_at(message.date)

        return self._session_store.start_session(
            chat_id=chat_id,
            started_at=started_at,
            started_by_user_id=message.from_user.id,
            started_by_name=author_name,
        )

    def record_text_message(
        self,
        message: telebot.types.Message,
    ) -> ListeningSession | None:
        chat_id = self._require_group_chat_id(message)

        if is_telegram_command(message.text):
            return None

        with self._chat_locks.lock_for(chat_id):
            active_session = self._session_store.get_active_session(chat_id)
            if active_session is None:
                return None

            chat_message = telegram_text_message_to_chat_message(
                message,
                session_id=active_session.session_id,
            )
            documents_written = self._indexing_service.index_messages([chat_message])
            if documents_written != 1:
                raise UnexpectedIndexingResultError(
                    f"expected documents_written=1, got {documents_written}"
                )
            return self._session_store.record_message(chat_id)

    def stop_listening(self, message: telebot.types.Message) -> ListeningSession:
        chat_id = self._require_group_chat_id(message)
        with self._chat_locks.lock_for(chat_id):
            return self._session_store.stop_session(chat_id)

    def _require_group_chat_id(self, message: telebot.types.Message) -> int:
        if message.chat is None:
            raise TelegramAdapterError("message.chat is required")
        if message.chat.type not in GROUP_CHAT_TYPES:
            raise UnsupportedTelegramChatError(
                "message.chat.type must be group or supergroup"
            )
        if message.chat.id is None:
            raise TelegramAdapterError("message.chat.id is required")
        return message.chat.id
