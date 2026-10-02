"""Application service for Telegram summary requests."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Final, Literal

import telebot.types

from models import SummarizationRequest, SummarizationResult
from session_documents import SessionIncompleteError, SessionInconsistentError
from session_store import ListeningSession, SessionStore
from summarization_service import SummarizationService
from telegram_adapter import GROUP_CHAT_TYPES, TelegramAdapterError
from telegram_application import UnsupportedTelegramChatError

logger = logging.getLogger(__name__)

SessionState = Literal["active", "completed"]

# The vector index is eventually consistent: right after a message was indexed it may
# not be searchable yet. A summary whose documents are fewer than the session counted
# is therefore re-checked a couple of times before it is refused. This is the only
# retry in the summary flow and it is deliberately short and fixed.
SUMMARY_COMPLETENESS_ATTEMPTS: Final[int] = 3
SUMMARY_COMPLETENESS_RETRY_DELAY_SECONDS: Final[float] = 1.5


class TelegramSummaryApplicationError(Exception):
    """Base exception for Telegram summary application errors."""


class NoSummarizableSessionError(TelegramSummaryApplicationError):
    """Raised when no active or completed session exists for summarization."""


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
        max_attempts: int = SUMMARY_COMPLETENESS_ATTEMPTS,
        retry_delay_seconds: float = SUMMARY_COMPLETENESS_RETRY_DELAY_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must not be negative")
        self._session_store = session_store
        self._summarization_service = summarization_service
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds
        self._sleep = sleep

    def summarize_discussion(
        self,
        message: telebot.types.Message,
    ) -> SummarizationResult:
        """Summarize the whole session, or raise if its documents are not all available.

        The session registry's ``message_count`` is the number of messages that were
        successfully indexed, so the summarizer is only invoked when exactly that many
        session documents can be loaded:

        * fewer visible (``SessionIncompleteError``): checked again up to
          ``max_attempts`` times in total, waiting ``retry_delay_seconds`` in between,
          then raised. The count is re-read on every attempt so that messages recorded
          meanwhile are not mistaken for extra documents.
        * more documents than counted (``SessionInconsistentError``): raised at once.
          This is a data-consistency error, not index lag, and is never retried.
        """
        chat_id = self._require_group_chat_id(message)

        for attempt in range(1, self._max_attempts + 1):
            session, session_state = self._resolve_session(chat_id)
            request = SummarizationRequest(
                instruction=SUMMARIZATION_INSTRUCTION,
                chat_id=chat_id,
                session_id=session.session_id,
                expected_message_count=session.message_count,
            )
            try:
                result = self._summarization_service.summarize(request)
            except SessionIncompleteError as exc:
                if attempt == self._max_attempts:
                    logger.warning(
                        "Session still incomplete, summary refused: "
                        "message_count=%s visible=%s attempts=%s session_state=%s",
                        exc.expected,
                        exc.fetched,
                        attempt,
                        session_state,
                    )
                    raise
                logger.info(
                    "Session not fully visible yet, retrying: "
                    "message_count=%s visible=%s attempt=%s/%s",
                    exc.expected,
                    exc.fetched,
                    attempt,
                    self._max_attempts,
                )
                self._sleep(self._retry_delay_seconds)
                continue
            except SessionInconsistentError as exc:
                logger.error(
                    "Session has more documents than messages, summary refused: "
                    "message_count=%s documents=%s session_state=%s",
                    exc.expected,
                    exc.fetched,
                    session_state,
                )
                raise

            logger.info(
                "Summary completed: message_count=%s source_count=%s session_state=%s",
                session.message_count,
                len(result.source_document_ids),
                session_state,
            )
            return result

        raise AssertionError("unreachable: the last attempt returns or raises")

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
