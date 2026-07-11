"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from config import Settings
from models import ChatMessage


@pytest.fixture
def settings() -> Settings:
    return Settings(
        telegram_bot_token="test-telegram-token",
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
