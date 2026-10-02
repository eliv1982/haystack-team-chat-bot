"""Pure adapter from Telegram message types to domain models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

import telebot.types

from models import ChatMessage

GROUP_CHAT_TYPES: Final[frozenset[str]] = frozenset({"group", "supergroup"})
_SUMMARY_PHRASE_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "что думаешь?",
        "подведи итог",
        "подведи итог обсуждения",
    }
)


class TelegramAdapterError(ValueError):
    """Raised when a Telegram message cannot be converted to a domain model."""


def normalize_trigger_text(text: str) -> str:
    """Normalize user text for exact summary phrase matching."""
    return " ".join(text.strip().split()).casefold()


def normalize_username(username: str | None) -> str | None:
    if username is None:
        return None
    normalized = username.strip().lstrip("@")
    return normalized if normalized else None


def build_author_fields(user: telebot.types.User) -> tuple[str, str | None]:
    username = normalize_username(user.username)

    name_parts: list[str] = []
    if user.first_name:
        name_parts.append(user.first_name.strip())
    if user.last_name:
        name_parts.append(user.last_name.strip())
    full_name = " ".join(part for part in name_parts if part).strip()
    if full_name:
        return full_name, username

    if username:
        return username, username

    return f"user-{user.id}", username


def normalize_sent_at(date: object) -> datetime:
    if date is None:
        raise TelegramAdapterError("date is required")
    if isinstance(date, bool):
        raise TelegramAdapterError("date must be an integer or timezone-aware datetime, not bool")
    if isinstance(date, int):
        if date < 0:
            raise TelegramAdapterError("date must not be a negative Unix timestamp")
        return datetime.fromtimestamp(date, tz=timezone.utc)
    if isinstance(date, datetime):
        if date.tzinfo is None or date.tzinfo.utcoffset(date) is None:
            raise TelegramAdapterError("date must be timezone-aware")
        return date.astimezone(timezone.utc)
    raise TelegramAdapterError(f"date has unsupported type: {type(date).__name__}")


def is_telegram_command(text: str | None) -> bool:
    if text is None:
        return False
    return text.lstrip().startswith("/")


def command_target_username(text: str | None) -> str | None:
    """Return the bot a command is addressed to: "/cmd@Bot args" -> "Bot".

    Returns None when the text is not a command or names no bot ("/cmd"). A command
    with an empty target ("/cmd@") yields an empty string.
    """
    if text is None or not is_telegram_command(text):
        return None
    token = text.split(maxsplit=1)[0]
    _, separator, target = token.partition("@")
    return target if separator else None


def is_command_addressed_to(text: str | None, bot_username: str | None) -> bool:
    """Return True for "/cmd" and "/cmd@<bot_username>"; False for other bots' commands.

    Usernames are compared case-insensitively. If the bot's own username is unknown,
    only commands that name no bot are accepted.
    """
    if not is_telegram_command(text):
        return False
    target = command_target_username(text)
    if target is None:
        return True
    normalized_target = normalize_username(target)
    normalized_own = normalize_username(bot_username)
    if normalized_target is None or normalized_own is None:
        return False
    return normalized_target.casefold() == normalized_own.casefold()


def unsupported_sender_reason(message: telebot.types.Message) -> str | None:
    """Return why a message cannot be attributed to a group participant, else None.

    Messages sent on behalf of a chat (anonymous group admins, a linked channel,
    "send as" identities) carry ``sender_chat`` and a fake or service ``from`` user,
    so no real participant identity exists. Bot users are not participants either.
    """
    if message.sender_chat is not None:
        return "sender_chat"
    if message.from_user is not None and message.from_user.is_bot:
        return "bot_user"
    return None


def is_summary_phrase_text(text: object) -> bool:
    if not isinstance(text, str):
        return False
    normalized = normalize_trigger_text(text)
    if not normalized:
        return False
    return normalized in _SUMMARY_PHRASE_ALIASES


def require_group_chat_id(message: telebot.types.Message) -> int:
    if message.chat is None:
        raise TelegramAdapterError("message.chat is required")
    if message.chat.type not in GROUP_CHAT_TYPES:
        raise TelegramAdapterError("message.chat.type must be group or supergroup")
    if message.chat.id is None:
        raise TelegramAdapterError("message.chat.id is required")
    return message.chat.id


def telegram_text_message_to_chat_message(
    message: telebot.types.Message,
    *,
    session_id: str,
) -> ChatMessage:
    """Convert a Telegram text message into a validated domain ChatMessage."""
    if message is None:
        raise TelegramAdapterError("message must not be None")

    normalized_session_id = session_id.strip()
    if not normalized_session_id:
        raise TelegramAdapterError("session_id must not be empty")

    if message.chat is None:
        raise TelegramAdapterError("message.chat is required")
    if message.chat.id is None:
        raise TelegramAdapterError("message.chat.id is required")
    if message.message_id is None:
        raise TelegramAdapterError("message.message_id is required")
    if message.from_user is None:
        raise TelegramAdapterError("message.from_user is required")
    if message.from_user.is_bot:
        raise TelegramAdapterError("messages from bot users are not supported")

    text = message.text
    if not isinstance(text, str) or not text.strip():
        raise TelegramAdapterError("message.text must be a non-empty string")

    author_name, username = build_author_fields(message.from_user)
    sent_at = normalize_sent_at(message.date)

    return ChatMessage(
        chat_id=message.chat.id,
        message_id=message.message_id,
        user_id=message.from_user.id,
        session_id=normalized_session_id,
        author_name=author_name,
        username=username,
        text=text.strip(),
        sent_at=sent_at,
    )
