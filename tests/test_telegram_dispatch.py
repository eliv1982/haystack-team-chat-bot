"""Dispatch tests through a real TeleBot instance (no network).

TeleBot's own handler filtering decides what runs, so these tests cover what unit
tests of the individual handler callbacks cannot: commands addressed to other bots,
anonymous-admin payloads, and handler failures never being raised into TeleBot.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import httpx
import openai
import pytest
import telebot
from haystack.document_stores.in_memory import InMemoryDocumentStore
from telebot.apihelper import ApiTelegramException
from telebot.types import Chat, Message, User

from config import Settings
from indexing_service import IndexingService
from pipelines import create_indexing_pipeline
from session_store import InMemorySessionStore
from telegram_application import TelegramApplicationService
from telegram_handlers import (
    _RECORD_FAILURE_NOTICE_INTERVAL_SECONDS,
    _RECORD_INTERNAL_ERROR_REPLY,
    register_telegram_handlers,
)
from telegram_summary_application import TelegramSummaryApplicationService

TOKEN = "123456789:AAH-s3cretTokenValue_0123456789abcdefghi"
CHAT_ID = -1001234567890
BOT_USERNAME = "ThisBot"
SENT_AT = int(datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc).timestamp())


def _message(
    text: str,
    *,
    message_id: int = 1,
    chat_id: int = CHAT_ID,
    user_id: int = 7,
    is_bot: bool = False,
    first_name: str = "Alice",
    username: str | None = "alice",
    sender_chat: Chat | None = None,
) -> Message:
    options: dict[str, object] = {"text": text}
    if sender_chat is not None:
        options["sender_chat"] = sender_chat
    return Message(
        message_id=message_id,
        from_user=User(id=user_id, is_bot=is_bot, first_name=first_name, username=username),
        date=SENT_AT,
        chat=Chat(id=chat_id, type="supergroup", title="Team Chat"),
        content_type="text",
        options=options,
        json_string="{}",
    )


def _anonymous_admin(text: str, *, message_id: int = 1) -> Message:
    return _message(
        text,
        message_id=message_id,
        user_id=1087968824,
        is_bot=True,
        first_name="Group",
        username="GroupAnonymousBot",
        sender_chat=Chat(id=CHAT_ID, type="supergroup", title="Team Chat"),
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 5_000.0

    def __call__(self) -> float:
        return self.now


class _Harness:
    def __init__(self, bot: telebot.TeleBot, clock: _Clock) -> None:
        self.bot = bot
        self.clock = clock
        self.session_store = InMemorySessionStore(session_id_factory=lambda: "session-1")
        self.indexing_service = MagicMock()
        self.indexing_service.index_messages.return_value = 1
        self.summarization_service = MagicMock()
        self.summarization_service.summarize.side_effect = AssertionError("not expected")

    def register(self, indexing_service: object | None = None) -> None:
        application_service = TelegramApplicationService(
            session_store=self.session_store,
            indexing_service=indexing_service or self.indexing_service,
        )
        summary_service = TelegramSummaryApplicationService(
            session_store=self.session_store,
            summarization_service=self.summarization_service,
        )
        register_telegram_handlers(self.bot, application_service, summary_service, clock=self.clock)

    def send(self, message: Message) -> None:
        self.bot.process_new_messages([message])

    @property
    def replies(self) -> MagicMock:
        return self.bot.reply_to

    @property
    def sent(self) -> MagicMock:
        return self.bot.send_message


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    bot = telebot.TeleBot(TOKEN, threaded=False)
    monkeypatch.setattr(
        bot,
        "get_me",
        lambda: User(id=999, is_bot=True, first_name="This", username=BOT_USERNAME),
    )
    bot.reply_to = MagicMock()  # type: ignore[method-assign]
    bot.send_message = MagicMock()  # type: ignore[method-assign]
    built = _Harness(bot, _Clock())
    built.register()
    return built


# --- commands for other bots are ignored --------------------------------------


@pytest.mark.parametrize(
    "command",
    ["/start_listening", "/start_listening@ThisBot", "/start_listening@thisbot"],
)
def test_start_listening_for_this_bot_starts_a_session(harness: _Harness, command: str) -> None:
    harness.send(_message(command))

    assert harness.session_store.get_active_session(CHAT_ID) is not None
    harness.replies.assert_called_once()


def test_start_listening_for_another_bot_is_ignored(harness: _Harness) -> None:
    harness.send(_message("/start_listening@OtherBot"))

    assert harness.session_store.get_active_session(CHAT_ID) is None
    harness.replies.assert_not_called()
    harness.sent.assert_not_called()


@pytest.mark.parametrize("command", ["/stop_listening", "/stop_listening@ThisBot"])
def test_stop_listening_for_this_bot_stops_the_session(harness: _Harness, command: str) -> None:
    harness.send(_message("/start_listening"))
    harness.replies.reset_mock()

    harness.send(_message(command, message_id=2))

    assert harness.session_store.get_active_session(CHAT_ID) is None
    harness.replies.assert_called_once()


def test_stop_listening_for_another_bot_does_not_stop_the_session(harness: _Harness) -> None:
    harness.send(_message("/start_listening"))
    harness.replies.reset_mock()

    harness.send(_message("/stop_listening@OtherBot", message_id=2))

    assert harness.session_store.get_active_session(CHAT_ID) is not None
    harness.replies.assert_not_called()


@pytest.mark.parametrize("command", ["/summary", "/summary@ThisBot", "/summary@THISBOT"])
def test_summary_for_this_bot_is_answered(harness: _Harness, command: str) -> None:
    harness.send(_message(command))

    # No session exists, so the handler answers that there is nothing to summarize.
    harness.sent.assert_called_once()
    assert "нет записанного обсуждения" in harness.sent.call_args.args[1]


def test_summary_for_another_bot_is_ignored(harness: _Harness) -> None:
    harness.send(_message("/start_listening"))
    harness.sent.reset_mock()

    harness.send(_message("/summary@OtherBot", message_id=2))

    harness.sent.assert_not_called()
    harness.summarization_service.summarize.assert_not_called()


@pytest.mark.parametrize("command", ["/status", "/help"])
def test_status_and_help_for_another_bot_are_ignored(harness: _Harness, command: str) -> None:
    harness.send(_message(f"{command}@OtherBot"))

    harness.replies.assert_not_called()


@pytest.mark.parametrize("command", ["/status", "/help"])
def test_status_and_help_for_this_bot_are_answered(harness: _Harness, command: str) -> None:
    harness.send(_message(command))
    harness.send(_message(f"{command}@ThisBot", message_id=2))

    assert harness.replies.call_count == 2


def test_commands_for_other_bots_are_never_recorded_as_discussion_text(harness: _Harness) -> None:
    harness.send(_message("/start_listening"))

    harness.send(_message("/anything@OtherBot please", message_id=2))
    harness.send(_message("/summary@OtherBot", message_id=3))
    harness.send(_message("/unknown", message_id=4))

    harness.indexing_service.index_messages.assert_not_called()


def test_ordinary_text_mentioning_another_bot_is_still_recorded(harness: _Harness) -> None:
    harness.send(_message("/start_listening"))

    harness.send(_message("let's ask @OtherBot about it", message_id=2))

    harness.indexing_service.index_messages.assert_called_once()


# --- anonymous admins / on-behalf-of-a-chat senders ---------------------------


def test_anonymous_admin_text_is_ignored_without_error_or_reply(harness: _Harness) -> None:
    harness.send(_message("/start_listening"))
    harness.replies.reset_mock()

    harness.send(_anonymous_admin("an anonymous remark", message_id=2))  # must not raise

    harness.indexing_service.index_messages.assert_not_called()
    harness.replies.assert_not_called()
    assert harness.session_store.get_active_session(CHAT_ID).message_count == 0


def test_anonymous_admin_commands_still_work(harness: _Harness) -> None:
    harness.send(_anonymous_admin("/start_listening"))
    assert harness.session_store.get_active_session(CHAT_ID) is not None

    harness.sent.reset_mock()
    harness.send(_anonymous_admin("/help", message_id=2))
    harness.replies.assert_called()


def test_participants_are_still_recorded_around_an_anonymous_admin(harness: _Harness) -> None:
    harness.send(_message("/start_listening"))

    harness.send(_message("first", message_id=2))
    harness.send(_anonymous_admin("anonymous", message_id=3))
    harness.send(_message("second", message_id=4))

    assert harness.indexing_service.index_messages.call_count == 2
    assert harness.session_store.get_active_session(CHAT_ID).message_count == 2


# --- handler failures never reach TeleBot ------------------------------------


@pytest.fixture
def failing_embedding_harness(
    harness: _Harness,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> _Harness:
    """Harness whose real indexing pipeline fails at the OpenAI embeddings call."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    store = InMemoryDocumentStore()
    pipeline = create_indexing_pipeline(settings, store)
    client = MagicMock()
    client.embeddings.create.side_effect = openai.APIConnectionError(
        request=httpx.Request("POST", "https://api.example.com/v1/embeddings")
    )
    monkeypatch.setattr(pipeline.get_component("document_embedder"), "client", client)
    # Re-register on a fresh bot-side registry using the failing real pipeline.
    harness.bot.message_handlers.clear()
    harness.register(indexing_service=IndexingService(pipeline))
    harness.document_store = store  # type: ignore[attr-defined]
    harness.embeddings_client = client  # type: ignore[attr-defined]
    return harness


def test_provider_outage_never_raises_into_telebot_and_does_not_count_messages(
    failing_embedding_harness: _Harness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = failing_embedding_harness
    caplog.set_level(logging.INFO)
    harness.send(_message("/start_listening"))
    harness.replies.reset_mock()

    for index in range(10):
        harness.clock.now += 5
        harness.send(_message(f"message {index}", message_id=10 + index))  # must not raise

    assert harness.session_store.get_active_session(CHAT_ID).message_count == 0
    assert harness.document_store.count_documents() == 0  # type: ignore[attr-defined]
    # One notice for the whole outage window, not one per message.
    harness.replies.assert_called_once()
    assert harness.replies.call_args.args[1] == _RECORD_INTERNAL_ERROR_REPLY
    assert caplog.text.count("Handler failed: handler=record_text") == 10
    assert "PipelineRuntimeError" in caplog.text
    assert "message 3" not in caplog.text


def test_outage_notice_is_repeated_only_after_the_interval_and_recovery_resumes_counting(
    failing_embedding_harness: _Harness,
) -> None:
    harness = failing_embedding_harness
    harness.send(_message("/start_listening"))
    harness.replies.reset_mock()

    harness.send(_message("one", message_id=10))
    harness.clock.now += _RECORD_FAILURE_NOTICE_INTERVAL_SECONDS + 1
    harness.send(_message("two", message_id=11))
    assert harness.replies.call_count == 2

    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    response.model = "test-embedding-model"
    response.usage = MagicMock()
    response.usage.__iter__ = lambda self: iter([("prompt_tokens", 1), ("total_tokens", 1)])
    harness.embeddings_client.embeddings.create.side_effect = None  # type: ignore[attr-defined]
    harness.embeddings_client.embeddings.create.return_value = response  # type: ignore[attr-defined]

    harness.send(_message("three", message_id=12))

    assert harness.replies.call_count == 2
    assert harness.session_store.get_active_session(CHAT_ID).message_count == 1
    assert harness.document_store.count_documents() == 1  # type: ignore[attr-defined]


def test_summary_failure_never_raises_into_telebot(harness: _Harness) -> None:
    harness.send(_message("/start_listening"))
    harness.summarization_service.summarize.side_effect = RuntimeError("openai unavailable")
    harness.sent.reset_mock()

    harness.send(_message("/summary", message_id=2))  # must not raise

    harness.sent.assert_called_once()
    assert "внутренней ошибки" in harness.sent.call_args.args[1]


def test_failed_telegram_reply_never_raises_into_telebot(
    harness: _Harness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    harness.replies.side_effect = ApiTelegramException(
        "sendMessage",
        MagicMock(),
        {"error_code": 403, "description": "Forbidden: bot was kicked from the supergroup chat"},
    )

    harness.send(_message("/help"))  # must not raise

    assert "Handler failed: handler=help.delivery" in caplog.text
    assert "ApiTelegramException" in caplog.text


def test_long_summary_is_delivered_in_several_telegram_safe_messages(harness: _Harness) -> None:
    from models import SummarizationResult

    harness.send(_message("/start_listening"))
    long_summary = "\n\n".join(f"Пункт {index}. " + "Текст итога. " * 60 for index in range(20))
    harness.summarization_service.summarize.side_effect = None
    harness.summarization_service.summarize.return_value = SummarizationResult(
        text=long_summary, source_document_ids=("doc-1",)
    )
    harness.sent.reset_mock()

    harness.send(_message("/summary", message_id=2))

    parts = [call.args[1] for call in harness.sent.call_args_list]
    assert len(parts) >= 2
    assert all(0 < len(part) <= 4096 for part in parts)
    assert "".join(parts) == long_summary.strip()
