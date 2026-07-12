"""Tests for Telegram command menu configuration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from telebot.types import BotCommandScopeAllGroupChats

from telegram_commands import TelegramCommandMenuError, configure_telegram_command_menu


def test_configure_telegram_command_menu_registers_exact_five_commands() -> None:
    bot = MagicMock()
    bot.set_my_commands.return_value = True

    configure_telegram_command_menu(bot)

    bot.set_my_commands.assert_called_once()
    commands = bot.set_my_commands.call_args.args[0]
    scope = bot.set_my_commands.call_args.kwargs["scope"]
    assert len(commands) == 5
    assert commands[0].command == "start_listening"
    assert commands[0].description == "Начать запись обсуждения"
    assert commands[1].command == "summary"
    assert commands[1].description == "Подвести итог обсуждения"
    assert commands[2].command == "stop_listening"
    assert commands[2].description == "Остановить запись"
    assert commands[3].command == "status"
    assert commands[3].description == "Показать статус записи"
    assert commands[4].command == "help"
    assert commands[4].description == "Показать инструкцию"
    assert isinstance(scope, BotCommandScopeAllGroupChats)


def test_configure_telegram_command_menu_true_result_passes() -> None:
    bot = MagicMock()
    bot.set_my_commands.return_value = True

    configure_telegram_command_menu(bot)

    bot.set_my_commands.assert_called_once()


def test_configure_telegram_command_menu_false_result_raises() -> None:
    bot = MagicMock()
    bot.set_my_commands.return_value = False

    with pytest.raises(TelegramCommandMenuError, match="configuration failed"):
        configure_telegram_command_menu(bot)


def test_configure_telegram_command_menu_exception_propagates() -> None:
    bot = MagicMock()
    bot.set_my_commands.side_effect = RuntimeError("menu failed")

    with pytest.raises(RuntimeError, match="menu failed"):
        configure_telegram_command_menu(bot)


def test_configure_telegram_command_menu_does_not_call_other_bot_api() -> None:
    bot = MagicMock()
    bot.set_my_commands.return_value = True

    configure_telegram_command_menu(bot)

    bot.get_my_commands.assert_not_called()
    bot.delete_my_commands.assert_not_called()
    bot.set_chat_menu_button.assert_not_called()
    bot.get_me.assert_not_called()
