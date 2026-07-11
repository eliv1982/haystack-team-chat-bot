"""Adapters between domain chat messages and Haystack documents."""

from __future__ import annotations

import hashlib
from datetime import timezone

from haystack import Document

from models import ChatMessage

DOCUMENT_ID_VERSION = "telegram-message:v1"


def _build_document_id(chat_id: int, message_id: int) -> str:
    identifier = f"{DOCUMENT_ID_VERSION}:{chat_id}:{message_id}"
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()


def _format_content(message: ChatMessage) -> str:
    sent_at_utc = message.sent_at.astimezone(timezone.utc)
    timestamp = sent_at_utc.isoformat()
    if message.username is not None:
        author_display = f"{message.author_name} (@{message.username})"
    else:
        author_display = message.author_name
    return f"[{timestamp}] {author_display}: {message.text}"


def _build_metadata(message: ChatMessage) -> dict[str, str | int]:
    sent_at_utc = message.sent_at.astimezone(timezone.utc).isoformat()
    metadata: dict[str, str | int] = {
        "source": "telegram",
        "schema_version": 1,
        "chat_id": str(message.chat_id),
        "message_id": str(message.message_id),
        "user_id": str(message.user_id),
        "session_id": message.session_id,
        "author_name": message.author_name,
        "sent_at": sent_at_utc,
    }
    if message.username is not None:
        metadata["username"] = message.username
    return metadata


def chat_message_to_document(message: ChatMessage) -> Document:
    """Convert a domain chat message into a Haystack document."""
    return Document(
        id=_build_document_id(message.chat_id, message.message_id),
        content=_format_content(message),
        meta=_build_metadata(message),
    )
