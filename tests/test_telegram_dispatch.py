"""Dispatch tests through a real TeleBot instance (no network).

TeleBot's own handler filtering decides what runs, so these tests cover what unit
tests of the individual handler callbacks cannot: routing by message kind, commands
addressed to other bots, anonymous-admin payloads, chats the bot must ignore, and
handler failures never being raised into TeleBot.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import httpx
import openai
import pytest
import requests
import telebot
from haystack.document_stores.in_memory import InMemoryDocumentStore
from pinecone.exceptions import ServiceException
from telebot.apihelper import ApiTelegramException
from telebot.types import Message

from config import Settings
from fakes import (
    BOT_TOKEN,
    GROUP_CHAT_ID,
    FakeClock,
    FakeOpenAIClient,
    anonymous_admin_message,
    telegram_message,
)
from indexing_service import IndexingProviderError, IndexingService
from models import SummarizationResult
from pipelines import create_indexing_pipeline
from session_store import InMemorySessionStore
from summarization_service import SummarizationProviderError
from telegram_application import TelegramApplicationService
from telegram_handlers import (
    _RECORD_FAILURE_NOTICE_INTERVAL_SECONDS,
    _RECORD_INTERNAL_ERROR_REPLY,
    register_telegram_handlers,
)
from telegram_summary_application import TelegramSummaryApplicationService


class _Harness:
    """Real application services behind real handlers; only indexing/summarizing are mocks."""

    def __init__(self, bot: telebot.TeleBot) -> None:
        self.bot = bot
        self.clock = FakeClock()
        self.session_store = InMemorySessionStore(session_id_factory=lambda: "session-1")
        self.indexing_service = MagicMock()
        self.indexing_service.index_messages.return_value = 1
        self.summarization_service = MagicMock()
        self.summarization_service.summarize.side_effect = AssertionError("not expected")
        self.register()

    def register(self, indexing_service: object | None = None) -> None:
        self.bot.message_handlers.clear()
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

    def active_session(self):  # type: ignore[no-untyped-def]
        return self.session_store.get_active_session(GROUP_CHAT_ID)


@pytest.fixture
def harness(telegram_bot: telebot.TeleBot) -> _Harness:
    return _Harness(telegram_bot)


def _summary_result(text: str = "Итог") -> SummarizationResult:
    return SummarizationResult(text=text, source_document_ids=("doc-1",))


# --- commands for other bots are ignored --------------------------------------


@pytest.mark.parametrize(
    "command",
    ["/start_listening", "/start_listening@ThisBot", "/start_listening@thisbot"],
)
def test_start_listening_for_this_bot_starts_a_session(harness: _Harness, command: str) -> None:
    harness.send(telegram_message(command))

    assert harness.active_session() is not None
    harness.replies.assert_called_once()


def test_start_listening_for_another_bot_is_ignored(harness: _Harness) -> None:
    harness.send(telegram_message("/start_listening@OtherBot"))

    assert harness.active_session() is None
    harness.replies.assert_not_called()
    harness.sent.assert_not_called()


@pytest.mark.parametrize("command", ["/stop_listening", "/stop_listening@ThisBot"])
def test_stop_listening_for_this_bot_stops_the_session(harness: _Harness, command: str) -> None:
    harness.send(telegram_message("/start_listening"))
    harness.replies.reset_mock()

    harness.send(telegram_message(command, message_id=2))

    assert harness.active_session() is None
    harness.replies.assert_called_once()


def test_stop_listening_for_another_bot_does_not_stop_the_session(harness: _Harness) -> None:
    harness.send(telegram_message("/start_listening"))
    harness.replies.reset_mock()

    harness.send(telegram_message("/stop_listening@OtherBot", message_id=2))

    assert harness.active_session() is not None
    harness.replies.assert_not_called()


@pytest.mark.parametrize("command", ["/summary", "/summary@ThisBot", "/summary@THISBOT"])
def test_summary_for_this_bot_is_answered(harness: _Harness, command: str) -> None:
    harness.send(telegram_message(command))

    # No session exists, so the handler answers that there is nothing to summarize.
    harness.sent.assert_called_once()
    assert "нет записанного обсуждения" in harness.sent.call_args.args[1]


def test_summary_for_another_bot_is_ignored(harness: _Harness) -> None:
    harness.send(telegram_message("/start_listening"))
    harness.sent.reset_mock()

    harness.send(telegram_message("/summary@OtherBot", message_id=2))

    harness.sent.assert_not_called()
    harness.summarization_service.summarize.assert_not_called()


@pytest.mark.parametrize("command", ["/status", "/help"])
def test_status_and_help_for_another_bot_are_ignored(harness: _Harness, command: str) -> None:
    harness.send(telegram_message(f"{command}@OtherBot"))

    harness.replies.assert_not_called()


@pytest.mark.parametrize("command", ["/status", "/help"])
def test_status_and_help_for_this_bot_are_answered(harness: _Harness, command: str) -> None:
    harness.send(telegram_message(command))
    harness.send(telegram_message(f"{command}@ThisBot", message_id=2))

    assert harness.replies.call_count == 2


def test_commands_for_other_bots_are_never_recorded_as_discussion_text(harness: _Harness) -> None:
    harness.send(telegram_message("/start_listening"))

    harness.send(telegram_message("/anything@OtherBot please", message_id=2))
    harness.send(telegram_message("/summary@OtherBot", message_id=3))
    harness.send(telegram_message("/unknown", message_id=4))

    harness.indexing_service.index_messages.assert_not_called()


def test_ordinary_text_mentioning_another_bot_is_still_recorded(harness: _Harness) -> None:
    harness.send(telegram_message("/start_listening"))

    harness.send(telegram_message("let's ask @OtherBot about it", message_id=2))

    harness.indexing_service.index_messages.assert_called_once()


def test_command_filter_fails_closed_when_the_bot_username_is_unavailable(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def get_me_fails() -> None:
        raise requests.exceptions.ConnectionError(f"url: /bot{BOT_TOKEN}/getMe")

    monkeypatch.setattr(harness.bot, "get_me", get_me_fails)
    caplog.set_level(logging.WARNING)

    harness.send(telegram_message("/summary", message_id=1))  # names no bot: still answered
    harness.send(telegram_message("/summary@ThisBot", message_id=2))  # cannot be verified: ignored

    harness.sent.assert_called_once()
    assert BOT_TOKEN not in caplog.text


# --- routing by message kind ---------------------------------------------------


@pytest.mark.parametrize("text", ["Подведи итог", "  подведи   итог обсуждения  ", "Что думаешь?"])
def test_summary_phrases_request_a_summary_and_are_never_recorded(
    harness: _Harness, text: str
) -> None:
    harness.send(telegram_message("/start_listening"))
    harness.summarization_service.summarize.side_effect = None
    harness.summarization_service.summarize.return_value = _summary_result("Итог")
    harness.sent.reset_mock()

    harness.send(telegram_message(text, message_id=2))

    harness.sent.assert_called_once_with(GROUP_CHAT_ID, "Итог")
    harness.indexing_service.index_messages.assert_not_called()
    assert harness.active_session().message_count == 0


def test_text_that_only_resembles_a_summary_phrase_is_recorded_not_summarized(
    harness: _Harness,
) -> None:
    harness.send(telegram_message("/start_listening"))

    harness.send(telegram_message("А что думаешь?", message_id=2))

    harness.indexing_service.index_messages.assert_called_once()
    harness.summarization_service.summarize.assert_not_called()
    assert harness.active_session().message_count == 1


@pytest.mark.parametrize("chat_type", ["private", "channel"])
def test_messages_outside_groups_are_ignored_entirely(harness: _Harness, chat_type: str) -> None:
    for message_id, text in enumerate(
        ["/start_listening", "/help", "/summary", "Подведи итог", "hello"], start=1
    ):
        harness.send(telegram_message(text, message_id=message_id, chat_type=chat_type))

    assert harness.active_session() is None
    harness.replies.assert_not_called()
    harness.sent.assert_not_called()
    harness.indexing_service.index_messages.assert_not_called()


# --- anonymous admins / on-behalf-of-a-chat senders ---------------------------


def test_anonymous_admin_text_is_ignored_without_error_or_reply(harness: _Harness) -> None:
    harness.send(telegram_message("/start_listening"))
    harness.replies.reset_mock()

    harness.send(anonymous_admin_message("an anonymous remark", message_id=2))  # must not raise

    harness.indexing_service.index_messages.assert_not_called()
    harness.replies.assert_not_called()
    assert harness.active_session().message_count == 0


def test_anonymous_admin_commands_still_work(harness: _Harness) -> None:
    harness.send(anonymous_admin_message("/start_listening"))
    assert harness.active_session() is not None

    harness.sent.reset_mock()
    harness.send(anonymous_admin_message("/help", message_id=2))
    harness.replies.assert_called()


def test_participants_are_still_recorded_around_an_anonymous_admin(harness: _Harness) -> None:
    harness.send(telegram_message("/start_listening"))

    harness.send(telegram_message("first", message_id=2))
    harness.send(anonymous_admin_message("anonymous", message_id=3))
    harness.send(telegram_message("second", message_id=4))

    assert harness.indexing_service.index_messages.call_count == 2
    assert harness.active_session().message_count == 2


# --- handler failures never reach TeleBot ------------------------------------


@pytest.fixture
def failing_embedding_harness(
    harness: _Harness,
    settings: Settings,
    fake_openai: FakeOpenAIClient,
) -> tuple[_Harness, InMemoryDocumentStore]:
    """Harness whose real indexing pipeline fails at the OpenAI embeddings call."""
    store = InMemoryDocumentStore()
    fake_openai.embedding_error = openai.APIConnectionError(
        request=httpx.Request("POST", "https://api.example.com/v1/embeddings")
    )
    harness.register(IndexingService(create_indexing_pipeline(settings, store)))
    return harness, store


def test_provider_outage_never_raises_into_telebot_and_does_not_count_messages(
    failing_embedding_harness: tuple[_Harness, InMemoryDocumentStore],
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness, store = failing_embedding_harness
    caplog.set_level(logging.INFO)
    harness.send(telegram_message("/start_listening"))
    harness.replies.reset_mock()

    for index in range(10):
        harness.clock.now += 5
        harness.send(telegram_message(f"message {index}", message_id=10 + index))  # must not raise

    assert harness.active_session().message_count == 0
    assert store.count_documents() == 0
    # One notice for the whole outage window, not one per message.
    harness.replies.assert_called_once()
    assert harness.replies.call_args.args[1] == _RECORD_INTERNAL_ERROR_REPLY
    assert caplog.text.count("Handler failed: handler=record_text") == 10
    assert "IndexingProviderError <- PipelineRuntimeError <- " in caplog.text
    assert "message 3" not in caplog.text


def test_outage_notice_is_repeated_only_after_the_interval_and_recovery_resumes_counting(
    failing_embedding_harness: tuple[_Harness, InMemoryDocumentStore],
    fake_openai: FakeOpenAIClient,
) -> None:
    harness, store = failing_embedding_harness
    harness.send(telegram_message("/start_listening"))
    harness.replies.reset_mock()

    harness.send(telegram_message("one", message_id=10))
    harness.clock.now += _RECORD_FAILURE_NOTICE_INTERVAL_SECONDS + 1
    harness.send(telegram_message("two", message_id=11))
    assert harness.replies.call_count == 2

    fake_openai.embedding_error = None  # the provider recovers
    harness.send(telegram_message("three", message_id=12))

    assert harness.replies.call_count == 2
    assert harness.active_session().message_count == 1
    assert store.count_documents() == 1


def test_vector_store_write_outage_is_handled_like_a_provider_outage(
    harness: _Harness,
    settings: Settings,
    fake_openai: FakeOpenAIClient,
) -> None:
    class _UnavailableStore(InMemoryDocumentStore):
        def write_documents(self, *args: object, **kwargs: object) -> int:
            raise ServiceException("pinecone unavailable", status_code=503)

    harness.register(IndexingService(create_indexing_pipeline(settings, _UnavailableStore())))
    harness.send(telegram_message("/start_listening"))
    harness.replies.reset_mock()

    harness.send(telegram_message("hello", message_id=2))  # must not raise

    assert harness.active_session().message_count == 0
    harness.replies.assert_called_once()
    assert harness.replies.call_args.args[1] == _RECORD_INTERNAL_ERROR_REPLY


def test_summary_failure_never_raises_into_telebot(harness: _Harness) -> None:
    harness.send(telegram_message("/start_listening"))
    harness.summarization_service.summarize.side_effect = SummarizationProviderError(
        "summarization provider request failed"
    )
    harness.sent.reset_mock()

    harness.send(telegram_message("/summary", message_id=2))  # must not raise

    harness.sent.assert_called_once()
    assert "внутренней ошибки" in harness.sent.call_args.args[1]


# --- programming defects reach TeleBot through the real application services --


def test_programming_defect_while_indexing_propagates_and_counts_nothing(harness: _Harness) -> None:
    harness.send(telegram_message("/start_listening"))
    harness.replies.reset_mock()
    harness.indexing_service.index_messages.side_effect = TypeError("programming defect")

    with pytest.raises(TypeError, match="programming defect"):
        harness.send(telegram_message("hello", message_id=2))

    assert harness.active_session().message_count == 0
    harness.replies.assert_not_called()

    # The chat's notice allowance is untouched: an actual outage is still announced once.
    harness.indexing_service.index_messages.side_effect = IndexingProviderError(
        "indexing provider request failed"
    )
    harness.send(telegram_message("hello again", message_id=3))
    harness.replies.assert_called_once()
    assert harness.replies.call_args.args[1] == _RECORD_INTERNAL_ERROR_REPLY


def test_programming_defect_while_summarizing_propagates(harness: _Harness) -> None:
    harness.send(telegram_message("/start_listening"))
    harness.summarization_service.summarize.side_effect = AssertionError("invariant bug")
    harness.sent.reset_mock()

    with pytest.raises(AssertionError, match="invariant bug"):
        harness.send(telegram_message("/summary", message_id=2))

    harness.sent.assert_not_called()


@pytest.mark.parametrize(
    ("command", "store_method"),
    [
        ("/start_listening", "start_session"),
        ("/stop_listening", "stop_session"),
        ("/status", "get_active_session"),
    ],
)
def test_programming_defect_in_the_session_store_propagates(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    store_method: str,
) -> None:
    def defective(*args: object, **kwargs: object) -> None:
        raise TypeError("programming defect")

    monkeypatch.setattr(harness.session_store, store_method, defective)

    with pytest.raises(TypeError, match="programming defect"):
        harness.send(telegram_message(command))

    harness.replies.assert_not_called()


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

    harness.send(telegram_message("/help"))  # must not raise

    assert "Handler failed: handler=help.delivery" in caplog.text
    assert "ApiTelegramException" in caplog.text


def test_long_summary_is_delivered_in_several_telegram_safe_messages(harness: _Harness) -> None:
    harness.send(telegram_message("/start_listening"))
    long_summary = "\n\n".join(f"Пункт {index}. " + "Текст итога. " * 60 for index in range(20))
    harness.summarization_service.summarize.side_effect = None
    harness.summarization_service.summarize.return_value = _summary_result(long_summary)
    harness.sent.reset_mock()

    harness.send(telegram_message("/summary", message_id=2))

    parts = [call.args[1] for call in harness.sent.call_args_list]
    assert len(parts) >= 3
    assert all(call.args[0] == GROUP_CHAT_ID for call in harness.sent.call_args_list)
    assert all(0 < len(part) <= 4096 for part in parts)
    assert all(part.strip() for part in parts)
    # SummarizationResult trims its text; nothing else may be lost or reordered.
    assert "".join(parts) == long_summary.strip()
