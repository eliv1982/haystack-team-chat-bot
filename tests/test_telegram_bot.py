"""Tests for the TeleBot factories (a real TeleBot is built; nothing is sent)."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
import telebot

from config import Settings
from telegram_bot import create_configured_telegram_bot, create_telegram_bot


@pytest.fixture
def created_bots() -> Iterator[list[telebot.TeleBot]]:
    bots: list[telebot.TeleBot] = []
    yield bots
    for bot in bots:
        bot.worker_pool.close()


def test_create_telegram_bot_builds_a_bot_for_the_configured_token(
    settings: Settings,
    created_bots: list[telebot.TeleBot],
) -> None:
    bot = create_telegram_bot(settings)
    created_bots.append(bot)

    assert isinstance(bot, telebot.TeleBot)
    assert bot.token == settings.telegram_bot_token
    assert not bot.message_handlers  # handler registration is a separate step


def test_create_configured_telegram_bot_registers_the_handlers(
    settings: Settings,
    created_bots: list[telebot.TeleBot],
) -> None:
    bot = create_configured_telegram_bot(settings, MagicMock(), MagicMock())
    created_bots.append(bot)

    assert bot.token == settings.telegram_bot_token
    assert bot.message_handlers  # what they do is covered by the dispatch tests
