"""Tests for bot.py production entry point."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import bot
import runtime


@pytest.fixture
def runtime_components() -> runtime.RuntimeComponents:
    return runtime.RuntimeComponents(
        document_store=MagicMock(name="document_store"),
        session_store=MagicMock(name="session_store"),
        indexing_service=MagicMock(name="indexing_service"),
        retrieval_service=MagicMock(name="retrieval_service"),
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


def test_main_build_failure_does_not_start_polling() -> None:
    with (
        patch("bot.configure_logging"),
        patch("bot.build_runtime", side_effect=RuntimeError("build failed")),
        patch("bot.run_polling") as run_polling,
    ):
        from bot import main

        with pytest.raises(RuntimeError, match="build failed"):
            main()

    run_polling.assert_not_called()


def test_main_polling_failure_does_not_rebuild(runtime_components: runtime.RuntimeComponents) -> None:
    with (
        patch("bot.configure_logging"),
        patch("bot.build_runtime", return_value=runtime_components) as build_runtime,
        patch("bot.run_polling", side_effect=RuntimeError("polling failed")),
    ):
        from bot import main

        with pytest.raises(RuntimeError, match="polling failed"):
            main()

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


def test_main_error_logging_does_not_expose_secrets(
    runtime_components: runtime.RuntimeComponents,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    with (
        patch("bot.configure_logging"),
        patch(
            "bot.build_runtime",
            side_effect=RuntimeError("build failed"),
        ),
        patch("bot.run_polling") as run_polling,
    ):
        from bot import main

        with pytest.raises(RuntimeError):
            main()

    run_polling.assert_not_called()
    assert "secret-telegram-token" not in caplog.text
