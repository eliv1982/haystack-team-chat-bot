"""Application service for Telegram summary requests."""

from __future__ import annotations

import logging
from typing import Literal

import telebot.types

from models import SummarizationRequest, SummarizationResult
from session_store import ListeningSession, SessionStore
from summarization_service import SummarizationService
from telegram_adapter import GROUP_CHAT_TYPES, TelegramAdapterError
from telegram_application import UnsupportedTelegramChatError

logger = logging.getLogger(__name__)

SessionState = Literal["active", "completed"]


class TelegramSummaryApplicationError(Exception):
    """Base exception for Telegram summary application errors."""


class NoSummarizableSessionError(TelegramSummaryApplicationError):
    """Raised when no active or completed session exists for summarization."""


SUMMARIZATION_CONTEXT_QUERY = (
    "Ключевые позиции участников обсуждения, принятые решения, следующие действия, "
    "ответственные, сроки и нерешенные вопросы"
)

SUMMARIZATION_INSTRUCTION = (
    "Подведи краткий итог текущего обсуждения на русском языке. "
    "Отдели ключевые позиции участников от принятых решений. "
    "Укажи следующие действия, ответственных и сроки только при наличии этих сведений "
    "в сообщениях. Отдельно перечисли нерешенные вопросы. "
    "В конце добавь краткую рекомендацию AI о следующем организационном шаге "
    "и явно обозначь ее как рекомендацию."
)


class TelegramSummaryApplicationService:
    """Coordinates summary requests for active or latest completed sessions."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        summarization_service: SummarizationService,
    ) -> None:
        self._session_store = session_store
        self._summarization_service = summarization_service

    def summarize_discussion(
        self,
        message: telebot.types.Message,
    ) -> SummarizationResult:
        chat_id = self._require_group_chat_id(message)
        session, session_state = self._resolve_session(chat_id)

        request = SummarizationRequest(
            context_query=SUMMARIZATION_CONTEXT_QUERY,
            instruction=SUMMARIZATION_INSTRUCTION,
            chat_id=chat_id,
            session_id=session.session_id,
        )
        result = self._summarization_service.summarize(request)
        logger.info(
            "Summary completed: message_count=%s source_count=%s session_state=%s",
            session.message_count,
            len(result.source_document_ids),
            session_state,
        )
        return result

    def _resolve_session(self, chat_id: int) -> tuple[ListeningSession, SessionState]:
        active_session = self._session_store.get_active_session(chat_id)
        if active_session is not None:
            return active_session, "active"

        completed_session = self._session_store.get_latest_completed_session(chat_id)
        if completed_session is not None:
            return completed_session, "completed"

        raise NoSummarizableSessionError(
            f"No active or completed session exists for chat_id {chat_id}"
        )

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
