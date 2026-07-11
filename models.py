"""Domain models for Telegram chat messages and related data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single message from a Telegram group chat."""

    message_id: int
    chat_id: int
    user_id: int
    text: str
    timestamp: datetime
