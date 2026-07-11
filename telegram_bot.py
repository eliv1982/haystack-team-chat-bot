"""Telegram bot factory without polling or handler registration."""

from __future__ import annotations

import telebot

from config import Settings
from telegram_application import TelegramApplicationService
from telegram_handlers import register_telegram_handlers
from telegram_summary_application import TelegramSummaryApplicationService


def create_telegram_bot(settings: Settings) -> telebot.TeleBot:
    """Create a synchronous TeleBot instance from application settings."""
    return telebot.TeleBot(settings.telegram_bot_token)


def create_configured_telegram_bot(
    settings: Settings,
    application_service: TelegramApplicationService,
    summary_application_service: TelegramSummaryApplicationService,
) -> telebot.TeleBot:
    """Create a TeleBot with listening and summary handlers registered."""
    bot = create_telegram_bot(settings)
    register_telegram_handlers(bot, application_service, summary_application_service)
    return bot
