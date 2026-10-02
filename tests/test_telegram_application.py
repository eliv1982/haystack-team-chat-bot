"""Tests for the Telegram application service."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from telebot.types import Chat, Message, User

from indexing_service import IndexingServiceError
from session_store import (
    InMemorySessionStore,
    NoActiveSessionError,
    SessionAlreadyActiveError,
)
from telegram_adapter import TelegramAdapterError
from telegram_application import (
    TelegramApplicationService,
    UnexpectedIndexingResultError,
    UnsupportedTelegramChatError,
)


def _utc_timestamp() -> int:
    return int(datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc).timestamp())


def _make_message(
    *,
    chat_id: int = -1001234567890,
    chat_type: str = "supergroup",
    message_id: int = 42,
    user_id: int = 7,
    is_bot: bool = False,
    first_name: str | None = "Alice",
    last_name: str | None = "Smith",
    username: str | None = "alice",
    text: str = "Hello, team!",
    date: object | None = None,
    sender_chat: Chat | None = None,
) -> Message:
    user = User(
        id=user_id,
        is_bot=is_bot,
        first_name=first_name,
        last_name=last_name,
        username=username,
    )
    chat = Chat(id=chat_id, type=chat_type, title="Team Chat")
    options: dict[str, object] = {"text": text}
    if sender_chat is not None:
        options["sender_chat"] = sender_chat
    return Message(
        message_id=message_id,
        from_user=user,
        date=date if date is not None else _utc_timestamp(),
        chat=chat,
        content_type="text",
        options=options,
        json_string="{}",
    )


def _anonymous_admin_message(*, chat_id: int = -1001234567890, text: str = "Anonymous hello") -> Message:
    """Payload Telegram sends for a group admin posting anonymously.

    ``from`` is the fake GroupAnonymousBot user and ``sender_chat`` is the group itself.
    """
    return _make_message(
        chat_id=chat_id,
        message_id=77,
        user_id=1087968824,
        is_bot=True,
        first_name="Group",
        last_name=None,
        username="GroupAnonymousBot",
        text=text,
        sender_chat=Chat(id=chat_id, type="supergroup", title="Team Chat"),
    )


def _linked_channel_message(*, chat_id: int = -1001234567890) -> Message:
    """Payload for a linked channel post forwarded into the discussion group."""
    return _make_message(
        chat_id=chat_id,
        message_id=78,
        user_id=777000,
        is_bot=False,
        first_name="Telegram",
        last_name=None,
        username=None,
        text="Channel announcement",
        sender_chat=Chat(id=-1007654321000, type="channel", title="Announcements"),
    )


@pytest.fixture
def session_store() -> InMemorySessionStore:
    return InMemorySessionStore(session_id_factory=lambda: "telegram-session-test")


@pytest.fixture
def indexing_service() -> MagicMock:
    service = MagicMock()
    service.index_messages.return_value = 1
    return service


@pytest.fixture
def application_service(
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
) -> TelegramApplicationService:
    return TelegramApplicationService(
        session_store=session_store,
        indexing_service=indexing_service,
    )


@pytest.mark.parametrize("chat_type", ["group", "supergroup"])
def test_start_listening_success(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
    chat_type: str,
) -> None:
    message = _make_message(chat_type=chat_type)

    session = application_service.start_listening(message)

    assert session.session_id == "telegram-session-test"
    assert session.started_by_user_id == 7
    assert session.started_by_name == "Alice Smith"
    assert session.started_at == datetime.fromtimestamp(_utc_timestamp(), tz=timezone.utc)
    assert session_store.get_active_session(message.chat.id) == session
    indexing_service.index_messages.assert_not_called()


def test_start_listening_calls_store_once(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
) -> None:
    message = _make_message()
    original_start = session_store.start_session
    call_count = {"value": 0}

    def counted_start(**kwargs: object) -> object:
        call_count["value"] += 1
        return original_start(**kwargs)  # type: ignore[arg-type]

    session_store.start_session = counted_start  # type: ignore[method-assign]
    application_service.start_listening(message)
    assert call_count["value"] == 1


def test_start_listening_uses_full_name_not_username(
    application_service: TelegramApplicationService,
) -> None:
    message = _make_message(first_name="Alice", last_name="Smith", username="alice")

    session = application_service.start_listening(message)

    assert session.started_by_name == "Alice Smith"


@pytest.mark.parametrize("chat_type", ["private", "channel", "unknown"])
def test_start_listening_rejects_unsupported_chat(
    application_service: TelegramApplicationService,
    chat_type: str,
) -> None:
    message = _make_message(chat_type=chat_type)

    with pytest.raises(UnsupportedTelegramChatError):
        application_service.start_listening(message)


def test_start_listening_duplicate_start_propagates(
    application_service: TelegramApplicationService,
) -> None:
    message = _make_message()
    application_service.start_listening(message)

    with pytest.raises(SessionAlreadyActiveError):
        application_service.start_listening(message)


def test_record_without_active_session_returns_none(
    application_service: TelegramApplicationService,
    indexing_service: MagicMock,
) -> None:
    message = _make_message()

    result = application_service.record_text_message(message)

    assert result is None
    indexing_service.index_messages.assert_not_called()


def test_record_indexes_one_message_and_increments_count(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
) -> None:
    start_message = _make_message(text="/start_listening")
    application_service.start_listening(start_message)
    record_message = _make_message(text="Discussion point", message_id=43)

    updated = application_service.record_text_message(record_message)

    assert updated is not None
    assert updated.message_count == 1
    indexing_service.index_messages.assert_called_once()
    indexed_batch = indexing_service.index_messages.call_args.args[0]
    assert len(indexed_batch) == 1
    assert indexed_batch[0].session_id == "telegram-session-test"
    assert indexed_batch[0].text == "Discussion point"


def test_record_passes_exact_active_session_id(
    application_service: TelegramApplicationService,
    indexing_service: MagicMock,
) -> None:
    application_service.start_listening(_make_message(text="/start_listening"))
    application_service.record_text_message(_make_message(text="One", message_id=43))

    indexed_message = indexing_service.index_messages.call_args.args[0][0]
    assert indexed_message.session_id == "telegram-session-test"


@pytest.mark.parametrize("text", ["/stop_listening", "  /start_listening"])
def test_record_ignores_commands(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
    text: str,
) -> None:
    application_service.start_listening(_make_message(text="/start_listening"))

    result = application_service.record_text_message(_make_message(text=text, message_id=43))

    assert result is None
    assert session_store.get_active_session(-1001234567890).message_count == 0
    indexing_service.index_messages.assert_not_called()


def test_record_indexing_failure_does_not_increment_count(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
) -> None:
    application_service.start_listening(_make_message(text="/start_listening"))
    indexing_service.index_messages.side_effect = IndexingServiceError("pipeline failed")

    with pytest.raises(IndexingServiceError):
        application_service.record_text_message(_make_message(text="Fail", message_id=43))

    assert session_store.get_active_session(-1001234567890).message_count == 0


def test_record_unexpected_indexing_result_raises(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
) -> None:
    application_service.start_listening(_make_message(text="/start_listening"))
    indexing_service.index_messages.return_value = 0

    with pytest.raises(UnexpectedIndexingResultError):
        application_service.record_text_message(_make_message(text="Fail", message_id=43))

    assert session_store.get_active_session(-1001234567890).message_count == 0


def test_record_adapter_failure_does_not_index_or_increment(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
) -> None:
    application_service.start_listening(_make_message(text="/start_listening"))
    malformed = _make_message(text="Bad date", message_id=43, date=-5)

    with pytest.raises(TelegramAdapterError):
        application_service.record_text_message(malformed)

    indexing_service.index_messages.assert_not_called()
    assert session_store.get_active_session(-1001234567890).message_count == 0


def test_record_ignores_bot_user_messages_without_error(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
) -> None:
    application_service.start_listening(_make_message(text="/start_listening"))

    result = application_service.record_text_message(
        _make_message(text="From bot", message_id=43, is_bot=True)
    )

    assert result is None
    indexing_service.index_messages.assert_not_called()
    assert session_store.get_active_session(-1001234567890).message_count == 0


@pytest.mark.parametrize(
    "message_factory",
    [_anonymous_admin_message, _linked_channel_message],
    ids=["anonymous-admin", "linked-channel"],
)
def test_record_ignores_messages_sent_on_behalf_of_a_chat(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
    message_factory: object,
) -> None:
    application_service.start_listening(_make_message(text="/start_listening"))

    result = application_service.record_text_message(message_factory())  # type: ignore[operator]

    assert result is None
    indexing_service.index_messages.assert_not_called()
    assert session_store.get_active_session(-1001234567890).message_count == 0


def test_anonymous_admin_message_does_not_disturb_recording_of_real_participants(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
) -> None:
    application_service.start_listening(_make_message(text="/start_listening"))

    application_service.record_text_message(_make_message(message_id=50, text="Before"))
    application_service.record_text_message(_anonymous_admin_message())
    application_service.record_text_message(_make_message(message_id=51, text="After"))

    assert indexing_service.index_messages.call_count == 2
    assert session_store.get_active_session(-1001234567890).message_count == 2


def test_anonymous_admin_can_still_start_a_session(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
) -> None:
    # Only recording is skipped for on-behalf-of-a-chat senders; commands keep working.
    session = application_service.start_listening(
        _anonymous_admin_message(text="/start_listening")
    )

    assert session.started_by_name == "Group"
    assert session_store.get_active_session(-1001234567890) == session


def test_ignored_sender_does_not_use_the_indexing_service_when_not_listening(
    application_service: TelegramApplicationService,
    indexing_service: MagicMock,
) -> None:
    assert application_service.record_text_message(_anonymous_admin_message()) is None
    indexing_service.index_messages.assert_not_called()


def test_record_uses_independent_sessions_per_chat(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
) -> None:
    application_service.start_listening(_make_message(chat_id=-100, text="/start_listening"))
    application_service.start_listening(_make_message(chat_id=-200, text="/start_listening"))

    application_service.record_text_message(_make_message(chat_id=-100, text="A", message_id=1))
    application_service.record_text_message(_make_message(chat_id=-200, text="B", message_id=2))
    application_service.record_text_message(_make_message(chat_id=-200, text="C", message_id=3))

    assert session_store.get_active_session(-100).message_count == 1
    assert session_store.get_active_session(-200).message_count == 2
    assert indexing_service.index_messages.call_count == 3


def test_stop_returns_final_snapshot_without_indexing(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
) -> None:
    application_service.start_listening(_make_message(text="/start_listening"))
    application_service.record_text_message(_make_message(text="One", message_id=43))
    stop_message = _make_message(text="/stop_listening", message_id=44)

    stopped = application_service.stop_listening(stop_message)

    assert stopped.message_count == 1
    assert session_store.get_active_session(-1001234567890) is None
    assert indexing_service.index_messages.call_count == 1


def test_stop_without_active_session_propagates(
    application_service: TelegramApplicationService,
) -> None:
    with pytest.raises(NoActiveSessionError):
        application_service.stop_listening(_make_message(text="/stop_listening"))


def test_stop_one_chat_does_not_affect_other(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
) -> None:
    application_service.start_listening(_make_message(chat_id=-100, text="/start_listening"))
    application_service.start_listening(_make_message(chat_id=-200, text="/start_listening"))

    application_service.stop_listening(_make_message(chat_id=-100, text="/stop_listening"))

    assert session_store.get_active_session(-100) is None
    assert session_store.get_active_session(-200) is not None


class _BlockingIndexingService:
    def __init__(self, blocked_chat_id: int) -> None:
        self.blocked_chat_id = blocked_chat_id
        self.entered = threading.Event()
        self.release = threading.Event()

    def index_messages(self, messages: list[object]) -> int:
        if messages[0].chat_id == self.blocked_chat_id:  # type: ignore[attr-defined]
            self.entered.set()
            assert self.release.wait(timeout=5)
        return 1


def test_record_stop_race_is_serialized_for_same_chat(
    session_store: InMemorySessionStore,
) -> None:
    blocked_chat_id = -1001234567890
    indexing_service = _BlockingIndexingService(blocked_chat_id)
    application_service = TelegramApplicationService(
        session_store=session_store,
        indexing_service=indexing_service,  # type: ignore[arg-type]
    )
    application_service.start_listening(_make_message(text="/start_listening"))

    record_message = _make_message(text="During stop", message_id=43)
    stop_message = _make_message(text="/stop_listening", message_id=44)
    errors: list[BaseException] = []
    results: dict[str, object] = {}

    def record_worker() -> None:
        try:
            results["record"] = application_service.record_text_message(record_message)
        except BaseException as exc:
            errors.append(exc)

    def stop_worker() -> None:
        try:
            assert indexing_service.entered.wait(timeout=5)
            results["stop"] = application_service.stop_listening(stop_message)
        except BaseException as exc:
            errors.append(exc)

    record_thread = threading.Thread(target=record_worker)
    stop_thread = threading.Thread(target=stop_worker)

    record_thread.start()
    assert indexing_service.entered.wait(timeout=5)

    stop_thread.start()
    stop_thread.join(timeout=0.05)
    assert stop_thread.is_alive()

    indexing_service.release.set()
    record_thread.join(timeout=5)
    stop_thread.join(timeout=5)

    assert not errors
    assert results["record"] is not None
    assert results["record"].message_count == 1  # type: ignore[attr-defined]
    assert results["stop"].message_count == 1  # type: ignore[attr-defined]


def test_slow_record_in_one_chat_does_not_block_stop_in_another(
    session_store: InMemorySessionStore,
) -> None:
    indexing_service = _BlockingIndexingService(blocked_chat_id=-100)
    application_service = TelegramApplicationService(
        session_store=session_store,
        indexing_service=indexing_service,  # type: ignore[arg-type]
    )
    application_service.start_listening(_make_message(chat_id=-100, text="/start_listening"))
    application_service.start_listening(_make_message(chat_id=-200, text="/start_listening"))

    errors: list[BaseException] = []

    def record_chat_a() -> None:
        try:
            application_service.record_text_message(
                _make_message(chat_id=-100, text="Blocked", message_id=43)
            )
        except BaseException as exc:
            errors.append(exc)

    record_thread = threading.Thread(target=record_chat_a)
    record_thread.start()
    assert indexing_service.entered.wait(timeout=5)

    stopped_b = application_service.stop_listening(
        _make_message(chat_id=-200, text="/stop_listening", message_id=44)
    )

    assert stopped_b.message_count == 0
    assert session_store.get_active_session(-200) is None
    assert session_store.get_active_session(-100) is not None

    indexing_service.release.set()
    record_thread.join(timeout=5)
    assert not errors


@pytest.mark.parametrize(
    "text",
    [
        "Что думаешь?",
        "  что   думаешь?  ",
        "Подведи итог",
        "  Подведи   итог обсуждения  ",
    ],
)
def test_record_ignores_summary_phrase_aliases(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
    text: str,
) -> None:
    application_service.start_listening(_make_message(text="/start_listening"))

    result = application_service.record_text_message(_make_message(text=text, message_id=43))

    assert result is None
    assert session_store.get_active_session(-1001234567890).message_count == 0
    indexing_service.index_messages.assert_not_called()


def test_record_similar_non_trigger_text_still_indexes(
    application_service: TelegramApplicationService,
    session_store: InMemorySessionStore,
    indexing_service: MagicMock,
) -> None:
    application_service.start_listening(_make_message(text="/start_listening"))

    updated = application_service.record_text_message(
        _make_message(text="А что думаешь?", message_id=43)
    )

    assert updated is not None
    assert updated.message_count == 1
    indexing_service.index_messages.assert_called_once()
