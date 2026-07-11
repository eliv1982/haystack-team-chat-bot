"""Telegram bot factory without polling or handler registration."""

from __future__ import annotations

import telebot

from config import Settings


def create_telegram_bot(settings: Settings) -> telebot.TeleBot:
    """Create a synchronous TeleBot instance from application settings."""
    return telebot.TeleBot(settings.telegram_bot_token)
