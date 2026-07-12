"""Telegram Bot API command menu configuration."""

from __future__ import annotations

import telebot
from telebot.types import BotCommand, BotCommandScopeAllGroupChats


class TelegramCommandMenuError(Exception):
    """Raised when Telegram command menu configuration fails."""


def configure_telegram_command_menu(bot: telebot.TeleBot) -> None:
    """Register the production command menu for all group and supergroup chats."""
    commands = [
        BotCommand("start_listening", "Начать запись обсуждения"),
        BotCommand("summary", "Подвести итог обсуждения"),
        BotCommand("stop_listening", "Остановить запись"),
        BotCommand("status", "Показать статус записи"),
        BotCommand("help", "Показать инструкцию"),
    ]
    scope = BotCommandScopeAllGroupChats()
    result = bot.set_my_commands(commands, scope=scope)
    if result is not True:
        raise TelegramCommandMenuError("Telegram command menu configuration failed")
