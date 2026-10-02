"""Tests for bot.py production entry point."""

from __future__ import annotations

import ast
import dataclasses
import io
import logging
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import telebot

import bot
import runtime
from config import ConfigurationError
from pinecone_preflight import PineconeIndexNotFoundError

# Shaped like a real Telegram bot token: "<bot id>:<secret>".
TELEGRAM_TOKEN = "123456789:AAH-s3cretTokenValue_0123456789abcdefghi"
TOKEN_SECRET_PART = TELEGRAM_TOKEN.split(":", 1)[1]


@pytest.fixture
def runtime_components() -> runtime.RuntimeComponents:
    return runtime.RuntimeComponents(
        document_store=MagicMock(name="document_store"),
        session_store=MagicMock(name="session_store"),
        indexing_service=MagicMock(name="indexing_service"),
        session_document_service=MagicMock(name="session_document_service"),
        summarization_service=MagicMock(name="summarization_service"),
        telegram_application_service=MagicMock(name="telegram_application_service"),
        telegram_summary_application_service=MagicMock(
            name="telegram_summary_application_service"
        ),
        bot=MagicMock(name="bot"),
    )


def test_import_bot_does_not_start_main() -> None:
    bot_source = Path(__file__).resolve().parent.parent / "bot.py"
    bot_tree = ast.parse(bot_source.read_text(encoding="utf-8"))
    forbidden_calls = {"configure_logging", "build_runtime", "run_polling", "main"}

    for node in bot_tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if isinstance(call.func, ast.Name) and call.func.id in forbidden_calls:
            pytest.fail(f"bot.py must not call {call.func.id} at import time")


def test_import_bot_does_not_create_global_runtime() -> None:
    assert not hasattr(bot, "runtime")
    assert not hasattr(bot, "bot")
    assert not hasattr(bot, "store")


def test_main_wires_startup_once(runtime_components: runtime.RuntimeComponents) -> None:
    with (
        patch("bot.configure_logging") as configure_logging,
        patch("bot.build_runtime", return_value=runtime_components) as build_runtime,
        patch("bot.run_polling") as run_polling,
    ):
        from bot import main

        main()

    configure_logging.assert_called_once_with()
    build_runtime.assert_called_once_with()
    run_polling.assert_called_once_with(runtime_components.bot)


def test_main_build_failure_does_not_start_polling(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR)
    with (
        patch("bot.configure_logging"),
        patch("bot.build_runtime", side_effect=RuntimeError("build failed")),
        patch("bot.run_polling") as run_polling,
    ):
        from bot import main

        with pytest.raises(SystemExit) as excinfo:
            main()

    assert excinfo.value.code == 1
    run_polling.assert_not_called()
    assert "Startup failed: RuntimeError" in caplog.text


def test_main_polling_failure_does_not_rebuild(runtime_components: runtime.RuntimeComponents) -> None:
    with (
        patch("bot.configure_logging"),
        patch("bot.build_runtime", return_value=runtime_components) as build_runtime,
        patch("bot.run_polling", side_effect=RuntimeError("polling failed")),
    ):
        from bot import main

        with pytest.raises(SystemExit) as excinfo:
            main()

    assert excinfo.value.code == 1
    build_runtime.assert_called_once_with()


def test_main_keyboard_interrupt_is_handled_gracefully(
    runtime_components: runtime.RuntimeComponents,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    with (
        patch("bot.configure_logging"),
        patch("bot.build_runtime", return_value=runtime_components),
        patch("bot.run_polling", side_effect=KeyboardInterrupt()),
    ):
        from bot import main

        main()

    assert "Shutdown requested" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        ConfigurationError("Missing required environment variable: TELEGRAM_BOT_TOKEN"),
        PineconeIndexNotFoundError("Pinecone index not found: team-chat"),
    ],
)
def test_main_logs_message_of_project_authored_startup_errors(
    error: Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    with (
        patch("bot.configure_logging"),
        patch("bot.build_runtime", side_effect=error),
    ):
        with pytest.raises(SystemExit) as excinfo:
            bot.main()

    assert excinfo.value.code == 1
    assert str(error) in caplog.text


def _telegram_network_error() -> requests.exceptions.ConnectionError:
    # Mirrors a real requests error: the message embeds the full request URL,
    # which for the Telegram API contains the bot token.
    return requests.exceptions.ConnectionError(
        "HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries exceeded "
        f"with url: /bot{TELEGRAM_TOKEN}/setMyCommands"
    )


def test_main_does_not_expose_token_from_telegram_network_error_at_startup(
    runtime_components: runtime.RuntimeComponents,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command menu call fails inside telebot with the token in the error text.

    Nothing is stubbed between ``main()`` and the failing HTTP call: the real
    run_polling, command-menu configuration and telebot request code all run.
    """
    telebot_log = io.StringIO()
    telebot_handler = logging.StreamHandler(telebot_log)
    telebot_logger = logging.getLogger("TeleBot")
    previous_level = telebot_logger.level
    telebot_logger.addHandler(telebot_handler)
    telebot_logger.setLevel(logging.DEBUG)

    def failing_request(*args: object, **kwargs: object) -> None:
        raise _telegram_network_error()

    monkeypatch.setattr(requests.Session, "request", failing_request)
    real_bot = telebot.TeleBot(TELEGRAM_TOKEN, threaded=False)
    components = dataclasses.replace(runtime_components, bot=real_bot)
    caplog.set_level(logging.DEBUG)

    try:
        with (
            patch("bot.configure_logging"),
            patch("bot.build_runtime", return_value=components),
        ):
            with pytest.raises(SystemExit) as excinfo:
                bot.main()
    finally:
        telebot_logger.removeHandler(telebot_handler)
        telebot_logger.setLevel(previous_level)

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    rendered_traceback = "".join(
        traceback.format_exception(type(excinfo.value), excinfo.value, excinfo.value.__traceback__)
    )
    everything = "\n".join(
        [caplog.text, captured.out, captured.err, telebot_log.getvalue(), rendered_traceback]
    )
    assert TELEGRAM_TOKEN not in everything
    assert TOKEN_SECRET_PART not in everything
    assert "ConnectionError" in caplog.text


def test_main_does_not_expose_token_from_unexpected_build_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    with (
        patch("bot.configure_logging"),
        patch("bot.build_runtime", side_effect=_telegram_network_error()),
    ):
        with pytest.raises(SystemExit) as excinfo:
            bot.main()

    rendered_traceback = "".join(
        traceback.format_exception(type(excinfo.value), excinfo.value, excinfo.value.__traceback__)
    )
    assert TELEGRAM_TOKEN not in caplog.text + rendered_traceback
    assert TOKEN_SECRET_PART not in caplog.text + rendered_traceback
    assert "Startup failed: ConnectionError" in caplog.text
