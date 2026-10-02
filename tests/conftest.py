"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import telebot
from telebot.types import User

from config import Settings
from fakes import BOT_TOKEN, BOT_USERNAME, FakeOpenAIClient
from models import ChatMessage

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def pytest_configure(config: pytest.Config) -> None:
    # Haystack reads this when it is first imported, which happens during collection,
    # after this hook. Automated runs never send telemetry, whatever the caller set.
    os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to reach a remote host.

    The suite must never contact OpenAI, Pinecone or Telegram. Tests replace those
    boundaries with fakes; this makes a forgotten one an error instead of a request.
    Loopback and unix-socket addresses stay allowed.
    """
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: object, *args: object, **kwargs: object):
        if isinstance(address, tuple) and address and address[0] not in _LOOPBACK_HOSTS:
            raise RuntimeError(f"offline tests must not connect to {address[0]!r}")
        return real_connect(self, address, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_getaddrinfo(host: object, *args: object, **kwargs: object):
        if host is not None and host not in _LOOPBACK_HOSTS:
            raise RuntimeError(f"offline tests must not resolve {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        telegram_bot_token="123456789:test-telegram-token",
        openai_api_key="test-openai-key",
        api_base_url="https://api.example.com/v1",
        openai_model="test-chat-model",
        embedding_model="test-embedding-model",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="test-index",
        pinecone_namespace="haystack-team-chat-homework",
        pinecone_dimension=1536,
        pinecone_metric="cosine",
        retrieval_top_k=50,
    )


@pytest.fixture
def direct_openai_settings(settings: Settings) -> Settings:
    """Settings for the direct OpenAI API: no custom base URL."""
    return replace(settings, api_base_url=None)


@pytest.fixture
def openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipelines read the OpenAI key from the environment when they are built."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


_OPENAI_CLIENT_USERS = (
    "haystack.components.embedders.openai_document_embedder",
    "haystack.components.embedders.openai_text_embedder",
    "haystack.components.generators.chat.openai",
)


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch, openai_env: None) -> FakeOpenAIClient:
    """Every OpenAI client that Haystack builds during the test is this one fake.

    Pipelines are created exactly as in production; only the HTTP client behind the
    embedder and chat generator is replaced.
    """
    client = FakeOpenAIClient()
    for module in _OPENAI_CLIENT_USERS:
        monkeypatch.setattr(f"{module}.OpenAI", lambda **_: client)
    return client


@pytest.fixture
def telegram_bot(monkeypatch: pytest.MonkeyPatch) -> Iterator[telebot.TeleBot]:
    """A real TeleBot whose Bot API access is replaced by mocks.

    ``process_new_messages`` runs TeleBot's own handler filtering and dispatch inline.
    ``get_me`` answers locally; ``reply_to`` and ``send_message`` record what the bot
    would have sent. Any other Bot API call reaches the offline guard and fails.
    """
    bot = telebot.TeleBot(BOT_TOKEN, threaded=False)
    monkeypatch.setattr(
        bot, "get_me", lambda: User(id=999, is_bot=True, first_name="This", username=BOT_USERNAME)
    )
    bot.reply_to = MagicMock()  # type: ignore[method-assign]
    bot.send_message = MagicMock()  # type: ignore[method-assign]
    yield bot


@pytest.fixture
def sample_message() -> ChatMessage:
    return ChatMessage(
        chat_id=-1001234567890,
        message_id=42,
        user_id=7,
        session_id="chat:-1001234567890",
        author_name="Alice",
        username="alice",
        text="Hello, team!",
        sent_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
    )
