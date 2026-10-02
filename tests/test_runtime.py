"""Tests for production runtime composition and polling boundary."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from haystack.document_stores.in_memory import InMemoryDocumentStore
from telebot.types import Chat, Message, User

import runtime
from config import Settings
from fakes import FakeOpenAIClient
from pinecone_preflight import PineconeIndexInfo, PineconeIndexNotFoundError

CHAT_ID = -1001234567890


@pytest.fixture
def bot() -> MagicMock:
    return MagicMock(name="bot")


def _group_message(text: str, *, message_id: int) -> Message:
    return Message(
        message_id=message_id,
        from_user=User(id=7, is_bot=False, first_name="Alice", username="alice"),
        date=1_705_320_600 + message_id,
        chat=Chat(id=CHAT_ID, type="supergroup", title="Team Chat"),
        content_type="text",
        options={"text": text},
        json_string="{}",
    )


@pytest.fixture
def assembled(
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    monkeypatch: pytest.MonkeyPatch,
):
    """The production runtime, assembled for real.

    Only the three external boundaries are fake: the OpenAI client, the vector store
    (Haystack's in-memory store) and the Telegram Bot API (``get_me`` and the two
    send methods; any other API call would hit the offline guard).
    """
    store = InMemoryDocumentStore()
    components = runtime.assemble_runtime(settings=settings, document_store=store)
    bot = components.bot
    bot.threaded = False  # run handlers inline, so the test sees their effects
    monkeypatch.setattr(
        bot, "get_me", lambda: User(id=999, is_bot=True, first_name="This", username="ThisBot")
    )
    bot.reply_to = MagicMock()  # type: ignore[method-assign]
    bot.send_message = MagicMock()  # type: ignore[method-assign]
    yield components, store
    bot.worker_pool.close()


def test_assembled_runtime_records_a_discussion_and_summarizes_exactly_it(
    assembled: tuple[runtime.RuntimeComponents, InMemoryDocumentStore],
    fake_openai: FakeOpenAIClient,
) -> None:
    components, store = assembled
    bot = components.bot

    for message_id, text in enumerate(
        ["/start_listening", "Let's ship on Tuesday", "Agreed, Tuesday works", "/summary"], start=1
    ):
        bot.process_new_messages([_group_message(text, message_id=message_id)])

    # One session store is shared by recording and summarizing: both messages were
    # counted, indexed through the real indexing pipeline, and the summary was built
    # from exactly those two, oldest first.
    assert components.session_store.get_active_session(CHAT_ID).message_count == 2
    assert store.count_documents() == 2
    assert len(fake_openai.embedding_inputs) == 2
    prompt = fake_openai.prompt_text()
    assert prompt.index("Let's ship on Tuesday") < prompt.index("Agreed, Tuesday works")
    bot.send_message.assert_called_once_with(CHAT_ID, fake_openai.reply)


def test_assembling_the_runtime_makes_no_telegram_api_call_and_does_not_poll(
    assembled: tuple[runtime.RuntimeComponents, InMemoryDocumentStore],
) -> None:
    # assemble_runtime already ran under the offline guard with a real TeleBot: a
    # getMe, setMyCommands or polling request would have raised. Nothing was sent.
    components, store = assembled

    components.bot.send_message.assert_not_called()
    components.bot.reply_to.assert_not_called()
    assert store.count_documents() == 0


def test_build_runtime_order_and_wiring(settings: Settings) -> None:
    preflight = PineconeIndexInfo(
        name="test-index",
        dimension=1536,
        metric="cosine",
        ready=True,
        status="Ready",
    )
    document_store = MagicMock(name="document_store")
    runtime_components = MagicMock(name="runtime_components")
    events: list[str] = []

    def _load_settings() -> Settings:
        events.append("load_settings")
        return settings

    def _validate(existing_settings: Settings) -> PineconeIndexInfo:
        events.append("validate_preflight")
        assert existing_settings is settings
        return preflight

    def _create_store(existing_settings: Settings) -> MagicMock:
        events.append("create_document_store")
        assert existing_settings is settings
        return document_store

    def _assemble(**_: object) -> MagicMock:
        events.append("assemble_runtime")
        return runtime_components

    with (
        patch("runtime.load_settings", side_effect=_load_settings) as mock_load,
        patch("runtime.validate_existing_pinecone_index", side_effect=_validate) as mock_validate,
        patch("runtime.create_pinecone_document_store", side_effect=_create_store) as mock_create,
        patch("runtime.assemble_runtime", side_effect=_assemble) as mock_assemble,
    ):
        result = runtime.build_runtime()

    assert result is runtime_components
    assert events == [
        "load_settings",
        "validate_preflight",
        "create_document_store",
        "assemble_runtime",
    ]
    mock_load.assert_called_once_with()
    mock_validate.assert_called_once_with(settings)
    mock_create.assert_called_once_with(settings)
    mock_assemble.assert_called_once_with(settings=settings, document_store=document_store)


def test_build_runtime_preflight_failure_skips_store_and_assembly(settings: Settings) -> None:
    with (
        patch("runtime.load_settings", return_value=settings),
        patch(
            "runtime.validate_existing_pinecone_index",
            side_effect=PineconeIndexNotFoundError("Pinecone index not found: test-index"),
        ),
        patch("runtime.create_pinecone_document_store") as mock_create,
        patch("runtime.assemble_runtime") as mock_assemble,
    ):
        with pytest.raises(PineconeIndexNotFoundError, match="Pinecone index not found"):
            runtime.build_runtime()

    mock_create.assert_not_called()
    mock_assemble.assert_not_called()


def test_build_runtime_store_failure_skips_assembly(settings: Settings) -> None:
    preflight = PineconeIndexInfo(
        name="test-index",
        dimension=1536,
        metric="cosine",
        ready=True,
        status="Ready",
    )
    with (
        patch("runtime.load_settings", return_value=settings),
        patch("runtime.validate_existing_pinecone_index", return_value=preflight),
        patch(
            "runtime.create_pinecone_document_store",
            side_effect=RuntimeError("store creation failed"),
        ),
        patch("runtime.assemble_runtime") as mock_assemble,
    ):
        with pytest.raises(RuntimeError, match="store creation failed"):
            runtime.build_runtime()

    mock_assemble.assert_not_called()


def test_run_polling_configures_menu_before_polling(bot: MagicMock) -> None:
    events: list[str] = []

    def _menu_configured(_bot: MagicMock) -> None:
        events.append("menu")

    def _polling(**kwargs: object) -> None:
        events.append("polling")

    bot.infinity_polling.side_effect = _polling
    with patch("runtime.configure_telegram_command_menu", side_effect=_menu_configured):
        runtime.run_polling(bot)

    assert events == ["menu", "polling"]
    bot.infinity_polling.assert_called_once_with(**runtime.POLLING_KWARGS)
    bot.stop_polling.assert_called_once_with()


def test_run_polling_menu_failure_blocks_polling(bot: MagicMock) -> None:
    from telegram_commands import TelegramCommandMenuError

    with patch(
        "runtime.configure_telegram_command_menu",
        side_effect=TelegramCommandMenuError("configuration failed"),
    ):
        with pytest.raises(TelegramCommandMenuError, match="configuration failed"):
            runtime.run_polling(bot)

    bot.infinity_polling.assert_not_called()
    bot.stop_polling.assert_called_once_with()


@pytest.mark.parametrize("failing_call", ["command_menu", "polling"])
def test_run_polling_failures_are_logged_without_the_exception_text(
    bot: MagicMock,
    caplog: pytest.LogCaptureFixture,
    failing_call: str,
) -> None:
    caplog.set_level(logging.INFO)
    token = "123456789:AAH-s3cretTokenValue_0123456789abcdefghi"
    error = ConnectionError(f"HTTPSConnectionPool: url: /bot{token}/{failing_call}")
    if failing_call == "polling":
        bot.infinity_polling.side_effect = error
        menu = MagicMock()
    else:
        menu = MagicMock(side_effect=error)

    with patch("runtime.configure_telegram_command_menu", menu):
        with pytest.raises(ConnectionError):
            runtime.run_polling(bot)

    assert "Telegram startup or polling failed: ConnectionError" in caplog.text
    assert token not in caplog.text
    assert bot.infinity_polling.called is (failing_call == "polling")
    bot.stop_polling.assert_called_once_with()


def test_run_polling_keyboard_interrupt_performs_cleanup(bot: MagicMock) -> None:
    bot.infinity_polling.side_effect = KeyboardInterrupt()

    with patch("runtime.configure_telegram_command_menu"):
        runtime.run_polling(bot)

    bot.infinity_polling.assert_called_once_with(**runtime.POLLING_KWARGS)
    bot.stop_polling.assert_called_once_with()


def test_run_polling_unexpected_error_is_not_swallowed(bot: MagicMock) -> None:
    bot.infinity_polling.side_effect = RuntimeError("polling failed")

    with patch("runtime.configure_telegram_command_menu"):
        with pytest.raises(RuntimeError, match="polling failed"):
            runtime.run_polling(bot)

    bot.infinity_polling.assert_called_once_with(**runtime.POLLING_KWARGS)
    bot.stop_polling.assert_called_once_with()


def test_build_runtime_logs_safe_fields_only(
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    preflight = PineconeIndexInfo(
        name="test-index",
        dimension=1536,
        metric="cosine",
        ready=True,
        status="Ready",
    )
    caplog.set_level(logging.INFO)

    with (
        patch("runtime.load_settings", return_value=settings),
        patch("runtime.validate_existing_pinecone_index", return_value=preflight),
        patch("runtime.create_pinecone_document_store", return_value=MagicMock()),
        patch("runtime.assemble_runtime", return_value=MagicMock(spec=runtime.RuntimeComponents)),
    ):
        runtime.build_runtime()

    assert settings.telegram_bot_token not in caplog.text
    assert settings.openai_api_key not in caplog.text
    assert settings.pinecone_api_key not in caplog.text
    assert "test-index" in caplog.text
    assert settings.pinecone_namespace in caplog.text
    assert settings.openai_model in caplog.text
