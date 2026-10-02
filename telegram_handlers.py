"""Telegram message handler registration for listening flow."""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable

import requests
import telebot
import telebot.types
from telebot.apihelper import ApiException
from telebot.util import MAX_MESSAGE_LENGTH, smart_split

from error_reporting import describe_exception
from session_documents import SessionIncompleteError, SessionTooLargeError
from session_store import NoActiveSessionError, SessionAlreadyActiveError
from summarization_service import NoSummarizationContextError
from telegram_adapter import is_command_addressed_to, is_summary_phrase_text, is_telegram_command
from telegram_application import DiscussionStatus, TelegramApplicationService, UnsupportedTelegramChatError
from telegram_summary_application import (
    NoSummarizableSessionError,
    TelegramSummaryApplicationService,
)

logger = logging.getLogger(__name__)

# Telegram and network failures while answering. Expected during outages, rate
# limiting or when the bot lacks rights in a chat; telebot cannot do anything
# useful with them except restart polling, so handlers absorb them.
_DELIVERY_ERRORS = (ApiException, requests.RequestException)

# While recording keeps failing (for example during an OpenAI outage) the chat is
# told at most once per interval instead of once per ordinary message.
_RECORD_FAILURE_NOTICE_INTERVAL_SECONDS = 600.0

_START_SUCCESS_REPLY = (
    "Запись обсуждения начата.\n\n"
    "Бот сохраняет только текстовые сообщения участников до команды /stop_listening. "
    "Команды, запросы на итог, фото, файлы, голосовые сообщения и стикеры не сохраняются.\n\n"
    "Для промежуточного итога используйте /summary или фразу «Подведи итог»."
)
_START_DUPLICATE_REPLY = "Запись обсуждения уже идет."
_STOP_SUCCESS_TEMPLATE = (
    "Запись обсуждения остановлена. Сохранено сообщений: {count}.\n\n"
    "Итог последней завершенной сессии доступен по команде /summary."
)
_STOP_NO_ACTIVE_REPLY = "Активной записи обсуждения нет."
_UNSUPPORTED_CHAT_REPLY = "Эта команда работает только в группах и супергруппах."
_RECORD_INTERNAL_ERROR_REPLY = (
    "Не удалось сохранить сообщение из-за внутренней ошибки. "
    "Часть сообщений может не попасть в итог.\n\n"
    "Пока ошибка повторяется, это уведомление не будет отправляться чаще раза в 10 минут."
)
_COMMAND_INTERNAL_ERROR_REPLY = "Не удалось выполнить команду из-за внутренней ошибки."
_SUMMARY_NO_SESSION_REPLY = (
    "В этом чате пока нет записанного обсуждения.\n\n"
    "Начните запись командой /start_listening."
)
_SUMMARY_NO_CONTEXT_REPLY = (
    "В выбранной сессии пока недостаточно сохраненных сообщений для подведения итога."
)
_SUMMARY_UNSUPPORTED_CHAT_REPLY = "Эта функция работает только в группах и супергруппах."
_SUMMARY_INTERNAL_ERROR_REPLY = "Не удалось подвести итог обсуждения из-за внутренней ошибки."
_SUMMARY_TOO_LARGE_TEMPLATE = (
    "Не удалось подвести итог: в этой сессии слишком много сообщений. "
    "Полный итог возможен для сессий до {max_messages} сообщений, "
    "а неполный итог бот не формирует, чтобы не исказить обсуждение.\n\n"
    "Начните новое обсуждение командой /start_listening "
    "(если запись еще идет, сначала остановите ее командой /stop_listening)."
)
_SUMMARY_INCOMPLETE_REPLY = (
    "Итог не подготовлен: полная запись обсуждения пока недоступна для надежного итога — "
    "часть сохраненных сообщений еще не появилась в хранилище. "
    "Неполный итог бот не формирует.\n\n"
    "Повторите команду /summary через несколько секунд."
)
_STATUS_ACTIVE_TEMPLATE = (
    "Запись обсуждения активна. Сохранено сообщений: {count}.\n\n"
    "Для промежуточного итога используйте /summary."
)
_STATUS_COMPLETED_TEMPLATE = (
    "Активной записи нет. Последняя завершенная сессия содержит {count} сообщений.\n\n"
    "Для получения итога используйте /summary."
)
_STATUS_NO_SESSIONS_REPLY = (
    "Активной записи и завершенных сессий в этом чате нет.\n\n"
    "Начните запись командой /start_listening."
)
_HELP_REPLY = (
    "Команды бота:\n\n"
    "/start_listening — начать запись обсуждения\n"
    "/summary — подвести итог текущей или последней завершенной сессии\n"
    "/stop_listening — остановить запись\n"
    "/status — показать статус записи\n"
    "/help — показать эту инструкцию\n\n"
    "Во время активной записи бот сохраняет только текстовые сообщения участников. "
    "Команды, запросы на итог и вложения не сохраняются.\n\n"
    "Итог также можно запросить фразами «Подведи итог» или «Подведи итог обсуждения»."
)


def _is_summary_phrase_message(message: telebot.types.Message) -> bool:
    return is_summary_phrase_text(message.text)


def _is_non_command_non_summary_text_message(message: telebot.types.Message) -> bool:
    text = message.text
    if text is None:
        return False
    if is_telegram_command(text):
        return False
    return not is_summary_phrase_text(text)


def _own_username(bot: telebot.TeleBot) -> str | None:
    """Return this bot's username; telebot caches it after the first lookup."""
    try:
        return bot.user.username
    except _DELIVERY_ERRORS as exc:
        logger.warning("Bot username unavailable: error=%s", describe_exception(exc))
        return None


def _make_addressed_to_this_bot_filter(
    bot: telebot.TeleBot,
) -> Callable[[telebot.types.Message], bool]:
    """Build a filter that drops "/command@OtherBot", which telebot's commands filter accepts."""

    def addressed_to_this_bot(message: telebot.types.Message) -> bool:
        return is_command_addressed_to(message.text, _own_username(bot))

    return addressed_to_this_bot


class _NoticeThrottle:
    """Allow at most one notice per chat within a time interval."""

    def __init__(self, interval_seconds: float, clock: Callable[[], float]) -> None:
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._last_allowed: dict[int, float] = {}

    def allow(self, chat_id: int) -> bool:
        now = self._clock()
        with self._lock:
            last = self._last_allowed.get(chat_id)
            if last is not None and now - last < self._interval_seconds:
                return False
            self._last_allowed[chat_id] = now
            return True


def _log_handler_failure(
    handler_name: str,
    message: telebot.types.Message,
    exc: BaseException,
) -> None:
    """Log a handler failure without exception text, message content, or secrets."""
    logger.error(
        "Handler failed: handler=%s chat_id=%s error=%s",
        handler_name,
        message.chat.id,
        describe_exception(exc),
    )


def _absorb_delivery_errors(
    handler_name: str,
) -> Callable[
    [Callable[[telebot.types.Message], None]],
    Callable[[telebot.types.Message], None],
]:
    """Keep failed Telegram API calls inside the handler instead of restarting polling.

    A handler exception that reaches telebot makes it log a traceback, sleep, restart
    polling, and (with skip_pending) drop every update that arrived meanwhile. Only
    Telegram/network errors are absorbed here; programming errors still propagate.
    """

    def decorator(
        handler: Callable[[telebot.types.Message], None],
    ) -> Callable[[telebot.types.Message], None]:
        @functools.wraps(handler)
        def guarded(message: telebot.types.Message) -> None:
            try:
                handler(message)
            except _DELIVERY_ERRORS as exc:
                _log_handler_failure(f"{handler_name}.delivery", message, exc)

        return guarded

    return decorator


def _split_for_telegram(text: str) -> list[str]:
    """Split text into parts that each fit into one Telegram message."""
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]
    return [part for part in smart_split(text, MAX_MESSAGE_LENGTH) if part.strip()]


def register_telegram_handlers(
    bot: telebot.TeleBot,
    application_service: TelegramApplicationService,
    summary_application_service: TelegramSummaryApplicationService,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Register listening, summary, status, help, and text capture handlers on the bot."""
    addressed_to_this_bot = _make_addressed_to_this_bot_filter(bot)
    record_failure_notices = _NoticeThrottle(_RECORD_FAILURE_NOTICE_INTERVAL_SECONDS, clock)

    bot.register_message_handler(
        _make_start_listening_handler(bot, application_service),
        commands=["start_listening"],
        content_types=["text"],
        chat_types=["group", "supergroup"],
        func=addressed_to_this_bot,
    )
    bot.register_message_handler(
        _make_stop_listening_handler(bot, application_service),
        commands=["stop_listening"],
        content_types=["text"],
        chat_types=["group", "supergroup"],
        func=addressed_to_this_bot,
    )
    bot.register_message_handler(
        _make_summary_handler(bot, summary_application_service),
        commands=["summary"],
        content_types=["text"],
        chat_types=["group", "supergroup"],
        func=addressed_to_this_bot,
    )
    bot.register_message_handler(
        _make_status_handler(bot, application_service),
        commands=["status"],
        content_types=["text"],
        chat_types=["group", "supergroup"],
        func=addressed_to_this_bot,
    )
    bot.register_message_handler(
        _make_help_handler(bot),
        commands=["help"],
        content_types=["text"],
        chat_types=["group", "supergroup"],
        func=addressed_to_this_bot,
    )
    bot.register_message_handler(
        _make_summary_handler(bot, summary_application_service),
        content_types=["text"],
        chat_types=["group", "supergroup"],
        func=_is_summary_phrase_message,
    )
    bot.register_message_handler(
        _make_record_text_handler(bot, application_service, record_failure_notices),
        content_types=["text"],
        chat_types=["group", "supergroup"],
        func=_is_non_command_non_summary_text_message,
    )


def _make_start_listening_handler(
    bot: telebot.TeleBot,
    application_service: TelegramApplicationService,
) -> Callable[[telebot.types.Message], None]:
    @_absorb_delivery_errors("start_listening")
    def handler(message: telebot.types.Message) -> None:
        try:
            application_service.start_listening(message)
        except SessionAlreadyActiveError:
            bot.reply_to(message, _START_DUPLICATE_REPLY)
            return
        except UnsupportedTelegramChatError:
            bot.reply_to(message, _UNSUPPORTED_CHAT_REPLY)
            return
        except Exception as exc:
            _log_handler_failure("start_listening", message, exc)
            bot.reply_to(message, _COMMAND_INTERNAL_ERROR_REPLY)
            return

        bot.reply_to(message, _START_SUCCESS_REPLY)

    return handler


def _make_stop_listening_handler(
    bot: telebot.TeleBot,
    application_service: TelegramApplicationService,
) -> Callable[[telebot.types.Message], None]:
    @_absorb_delivery_errors("stop_listening")
    def handler(message: telebot.types.Message) -> None:
        try:
            stopped_session = application_service.stop_listening(message)
        except NoActiveSessionError:
            bot.reply_to(message, _STOP_NO_ACTIVE_REPLY)
            return
        except UnsupportedTelegramChatError:
            bot.reply_to(message, _UNSUPPORTED_CHAT_REPLY)
            return
        except Exception as exc:
            _log_handler_failure("stop_listening", message, exc)
            bot.reply_to(message, _COMMAND_INTERNAL_ERROR_REPLY)
            return

        bot.reply_to(
            message,
            _STOP_SUCCESS_TEMPLATE.format(count=stopped_session.message_count),
        )

    return handler


def _make_summary_handler(
    bot: telebot.TeleBot,
    summary_application_service: TelegramSummaryApplicationService,
) -> Callable[[telebot.types.Message], None]:
    @_absorb_delivery_errors("summary")
    def handler(message: telebot.types.Message) -> None:
        try:
            result = summary_application_service.summarize_discussion(message)
        except NoSummarizableSessionError:
            bot.send_message(message.chat.id, _SUMMARY_NO_SESSION_REPLY)
            return
        except NoSummarizationContextError:
            bot.send_message(message.chat.id, _SUMMARY_NO_CONTEXT_REPLY)
            return
        except SessionTooLargeError as exc:
            logger.warning(
                "Summary refused, session too large: chat_id=%s max_messages=%s",
                message.chat.id,
                exc.max_messages,
            )
            bot.send_message(
                message.chat.id,
                _SUMMARY_TOO_LARGE_TEMPLATE.format(max_messages=exc.max_messages),
            )
            return
        except SessionIncompleteError:
            # Already logged with counts by the summary application service.
            bot.send_message(message.chat.id, _SUMMARY_INCOMPLETE_REPLY)
            return
        except UnsupportedTelegramChatError:
            bot.send_message(message.chat.id, _SUMMARY_UNSUPPORTED_CHAT_REPLY)
            return
        except Exception as exc:
            _log_handler_failure("summary", message, exc)
            bot.send_message(message.chat.id, _SUMMARY_INTERNAL_ERROR_REPLY)
            return

        for part in _split_for_telegram(result.text):
            bot.send_message(message.chat.id, part)

    return handler


def _make_status_handler(
    bot: telebot.TeleBot,
    application_service: TelegramApplicationService,
) -> Callable[[telebot.types.Message], None]:
    @_absorb_delivery_errors("status")
    def handler(message: telebot.types.Message) -> None:
        try:
            status = application_service.get_discussion_status(message)
        except UnsupportedTelegramChatError:
            bot.reply_to(message, _UNSUPPORTED_CHAT_REPLY)
            return
        except Exception as exc:
            _log_handler_failure("status", message, exc)
            bot.reply_to(message, _COMMAND_INTERNAL_ERROR_REPLY)
            return

        bot.reply_to(message, _format_status_reply(status))

    return handler


def _make_help_handler(bot: telebot.TeleBot) -> Callable[[telebot.types.Message], None]:
    @_absorb_delivery_errors("help")
    def handler(message: telebot.types.Message) -> None:
        bot.reply_to(message, _HELP_REPLY)

    return handler


def _format_status_reply(status: DiscussionStatus) -> str:
    if status.active_session is not None:
        return _STATUS_ACTIVE_TEMPLATE.format(count=status.active_session.message_count)
    if status.latest_completed_session is not None:
        return _STATUS_COMPLETED_TEMPLATE.format(
            count=status.latest_completed_session.message_count
        )
    return _STATUS_NO_SESSIONS_REPLY


def _make_record_text_handler(
    bot: telebot.TeleBot,
    application_service: TelegramApplicationService,
    failure_notices: _NoticeThrottle,
) -> Callable[[telebot.types.Message], None]:
    @_absorb_delivery_errors("record_text")
    def handler(message: telebot.types.Message) -> None:
        try:
            application_service.record_text_message(message)
        except Exception as exc:
            _log_handler_failure("record_text", message, exc)
            if failure_notices.allow(message.chat.id):
                bot.reply_to(message, _RECORD_INTERNAL_ERROR_REPLY)

    return handler
