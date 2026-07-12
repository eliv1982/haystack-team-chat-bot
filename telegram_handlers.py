"""Telegram message handler registration for listening flow."""

from __future__ import annotations

import logging
from collections.abc import Callable

import telebot
import telebot.types

from session_store import NoActiveSessionError, SessionAlreadyActiveError
from summarization_service import NoSummarizationContextError
from telegram_adapter import is_summary_phrase_text, is_telegram_command
from telegram_application import DiscussionStatus, TelegramApplicationService, UnsupportedTelegramChatError
from telegram_summary_application import (
    NoSummarizableSessionError,
    TelegramSummaryApplicationService,
)

logger = logging.getLogger(__name__)

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
_RECORD_INTERNAL_ERROR_REPLY = "Не удалось сохранить сообщение из-за внутренней ошибки."
_SUMMARY_NO_SESSION_REPLY = (
    "В этом чате пока нет записанного обсуждения.\n\n"
    "Начните запись командой /start_listening."
)
_SUMMARY_NO_CONTEXT_REPLY = (
    "В выбранной сессии пока недостаточно сохраненных сообщений для подведения итога."
)
_SUMMARY_UNSUPPORTED_CHAT_REPLY = "Эта функция работает только в группах и супергруппах."
_SUMMARY_INTERNAL_ERROR_REPLY = "Не удалось подвести итог обсуждения из-за внутренней ошибки."
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


def register_telegram_handlers(
    bot: telebot.TeleBot,
    application_service: TelegramApplicationService,
    summary_application_service: TelegramSummaryApplicationService,
) -> None:
    """Register listening, summary, status, help, and text capture handlers on the bot."""
    bot.register_message_handler(
        _make_start_listening_handler(bot, application_service),
        commands=["start_listening"],
        content_types=["text"],
        chat_types=["group", "supergroup"],
    )
    bot.register_message_handler(
        _make_stop_listening_handler(bot, application_service),
        commands=["stop_listening"],
        content_types=["text"],
        chat_types=["group", "supergroup"],
    )
    bot.register_message_handler(
        _make_summary_handler(bot, summary_application_service),
        commands=["summary"],
        content_types=["text"],
        chat_types=["group", "supergroup"],
    )
    bot.register_message_handler(
        _make_status_handler(bot, application_service),
        commands=["status"],
        content_types=["text"],
        chat_types=["group", "supergroup"],
    )
    bot.register_message_handler(
        _make_help_handler(bot),
        commands=["help"],
        content_types=["text"],
        chat_types=["group", "supergroup"],
    )
    bot.register_message_handler(
        _make_summary_handler(bot, summary_application_service),
        content_types=["text"],
        chat_types=["group", "supergroup"],
        func=_is_summary_phrase_message,
    )
    bot.register_message_handler(
        _make_record_text_handler(bot, application_service),
        content_types=["text"],
        chat_types=["group", "supergroup"],
        func=_is_non_command_non_summary_text_message,
    )


def _make_start_listening_handler(
    bot: telebot.TeleBot,
    application_service: TelegramApplicationService,
) -> Callable[[telebot.types.Message], None]:
    def handler(message: telebot.types.Message) -> None:
        try:
            application_service.start_listening(message)
        except SessionAlreadyActiveError:
            bot.reply_to(message, _START_DUPLICATE_REPLY)
            return
        except UnsupportedTelegramChatError:
            bot.reply_to(message, _UNSUPPORTED_CHAT_REPLY)
            return
        except Exception:
            logger.exception("Failed to start listening session")
            raise

        bot.reply_to(message, _START_SUCCESS_REPLY)

    return handler


def _make_stop_listening_handler(
    bot: telebot.TeleBot,
    application_service: TelegramApplicationService,
) -> Callable[[telebot.types.Message], None]:
    def handler(message: telebot.types.Message) -> None:
        try:
            stopped_session = application_service.stop_listening(message)
        except NoActiveSessionError:
            bot.reply_to(message, _STOP_NO_ACTIVE_REPLY)
            return
        except UnsupportedTelegramChatError:
            bot.reply_to(message, _UNSUPPORTED_CHAT_REPLY)
            return
        except Exception:
            logger.exception("Failed to stop listening session")
            raise

        bot.reply_to(
            message,
            _STOP_SUCCESS_TEMPLATE.format(count=stopped_session.message_count),
        )

    return handler


def _make_summary_handler(
    bot: telebot.TeleBot,
    summary_application_service: TelegramSummaryApplicationService,
) -> Callable[[telebot.types.Message], None]:
    def handler(message: telebot.types.Message) -> None:
        try:
            result = summary_application_service.summarize_discussion(message)
        except NoSummarizableSessionError:
            bot.send_message(message.chat.id, _SUMMARY_NO_SESSION_REPLY)
            return
        except NoSummarizationContextError:
            bot.send_message(message.chat.id, _SUMMARY_NO_CONTEXT_REPLY)
            return
        except UnsupportedTelegramChatError:
            bot.send_message(message.chat.id, _SUMMARY_UNSUPPORTED_CHAT_REPLY)
            return
        except Exception:
            logger.exception("Failed to summarize discussion")
            bot.send_message(message.chat.id, _SUMMARY_INTERNAL_ERROR_REPLY)
            raise

        bot.send_message(message.chat.id, result.text)

    return handler


def _make_status_handler(
    bot: telebot.TeleBot,
    application_service: TelegramApplicationService,
) -> Callable[[telebot.types.Message], None]:
    def handler(message: telebot.types.Message) -> None:
        try:
            status = application_service.get_discussion_status(message)
        except UnsupportedTelegramChatError:
            bot.reply_to(message, _UNSUPPORTED_CHAT_REPLY)
            return
        except Exception:
            logger.exception("Failed to get discussion status")
            raise

        bot.reply_to(message, _format_status_reply(status))

    return handler


def _make_help_handler(bot: telebot.TeleBot) -> Callable[[telebot.types.Message], None]:
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
) -> Callable[[telebot.types.Message], None]:
    def handler(message: telebot.types.Message) -> None:
        try:
            application_service.record_text_message(message)
        except Exception:
            logger.exception("Failed to record text message")
            bot.reply_to(message, _RECORD_INTERNAL_ERROR_REPLY)
            raise

    return handler
