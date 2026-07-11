"""Tests for the TeleBot factory."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        telegram_bot_token="secret-telegram-token",
        openai_api_key="test-openai-key",
        api_base_url=None,
        openai_model="test-chat-model",
        embedding_model="test-embedding-model",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="test-index",
        pinecone_namespace="haystack-team-chat-homework",
        pinecone_dimension=1536,
        pinecone_metric="cosine",
        retrieval_top_k=50,
    )


def test_create_telegram_bot_calls_constructor_once(settings: Settings) -> None:
    mock_bot = MagicMock()
    with patch("telegram_bot.telebot.TeleBot", return_value=mock_bot) as constructor:
        from telegram_bot import create_telegram_bot

        bot = create_telegram_bot(settings)

    constructor.assert_called_once_with(settings.telegram_bot_token)
    assert bot is mock_bot


def test_create_telegram_bot_does_not_poll(settings: Settings) -> None:
    mock_bot = MagicMock()
    with patch("telegram_bot.telebot.TeleBot", return_value=mock_bot):
        from telegram_bot import create_telegram_bot

        create_telegram_bot(settings)

    mock_bot.polling.assert_not_called()
    mock_bot.infinity_polling.assert_not_called()
    mock_bot.get_me.assert_not_called()
    mock_bot.send_message.assert_not_called()


def test_create_telegram_bot_does_not_register_handlers(settings: Settings) -> None:
    mock_bot = MagicMock()
    with patch("telegram_bot.telebot.TeleBot", return_value=mock_bot):
        from telegram_bot import create_telegram_bot

        create_telegram_bot(settings)

    assert mock_bot.message_handler.call_count == 0
    assert mock_bot.callback_query_handler.call_count == 0


def test_importing_telegram_bot_does_not_create_bot() -> None:
    module_name = "telegram_bot"
    sys.modules.pop(module_name, None)
    with patch("telegram_bot.telebot.TeleBot") as constructor:
        importlib.import_module(module_name)
    constructor.assert_not_called()


def test_factory_errors_do_not_expose_token(settings: Settings) -> None:
    with patch("telegram_bot.telebot.TeleBot", side_effect=RuntimeError("constructor failed")):
        from telegram_bot import create_telegram_bot

        with pytest.raises(RuntimeError, match="constructor failed") as exc_info:
            create_telegram_bot(settings)

    assert settings.telegram_bot_token not in str(exc_info.value)


def test_create_configured_telegram_bot_registers_handlers_once(
    settings: Settings,
) -> None:
    mock_bot = MagicMock()
    application_service = MagicMock()
    with patch("telegram_bot.create_telegram_bot", return_value=mock_bot) as create_bot:
        with patch("telegram_bot.register_telegram_handlers") as register_handlers:
            from telegram_bot import create_configured_telegram_bot

            bot = create_configured_telegram_bot(settings, application_service)

    create_bot.assert_called_once_with(settings)
    register_handlers.assert_called_once_with(mock_bot, application_service)
    assert bot is mock_bot


def test_create_configured_telegram_bot_does_not_poll_or_call_bot_api(
    settings: Settings,
) -> None:
    mock_bot = MagicMock()
    application_service = MagicMock()
    with patch("telegram_bot.create_telegram_bot", return_value=mock_bot):
        with patch("telegram_bot.register_telegram_handlers"):
            from telegram_bot import create_configured_telegram_bot

            create_configured_telegram_bot(settings, application_service)

    mock_bot.polling.assert_not_called()
    mock_bot.infinity_polling.assert_not_called()
    mock_bot.get_me.assert_not_called()
    mock_bot.send_message.assert_not_called()


def test_importing_telegram_bot_does_not_create_global_dependencies() -> None:
    module_name = "telegram_bot"
    sys.modules.pop(module_name, None)
    with patch("telegram_bot.telebot.TeleBot") as constructor:
        with patch("telegram_bot.register_telegram_handlers"):
            module = importlib.import_module(module_name)
    constructor.assert_not_called()
    assert not hasattr(module, "bot")
    assert not hasattr(module, "store")
    assert not hasattr(module, "application_service")
